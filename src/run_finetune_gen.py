from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange
import json
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                                  BartConfig, BartTokenizer, BertTokenizer, BartForConditionalGeneration)

from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration

import transformers

# transformers.logging.set_verbosity_info()
transformers.logging.set_verbosity_warning()
import sys
sys.path.append('../')
from utils import TrainDataset, count_parameters, set_seed, save_model

from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftConfig, PeftModel
from peft.peft_model import PeftModelForSeq2SeqLM

peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
import peft

import subprocess
from nlgeval import compute_metrics

logger = logging.getLogger(__name__)



MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer),
    'T5': (T5Config, T5ForConditionalGeneration, T5Tokenizer)
}



def load_and_cache_examples(args, tokenizer, evaluate=False, max_q = None, max_a = None):
    if not evaluate:
        type_path = f'train{args.train_name}'
        logger.info("Loading training data from %s", type_path)
    else:
        type_path = 'dev'   # DIFF HERE
    dataset = TrainDataset(tokenizer, data_dir=args.data_dir, type_path=type_path, max_source_length=args.max_src_len, max_target_length=args.max_tgt_len)
    return dataset



def load_binary_class_examples(args, tokenizer, evaluate=False):
    # train和生成任务保持一致，生成<bos> label <eos>
    if not evaluate:
        type_path = f'train{args.train_name}'
        logger.info("Loading training data from %s", type_path)
        dataset = TrainDataset(tokenizer, data_dir=args.data_dir, type_path=type_path, max_source_length=args.max_src_len, max_target_length=args.max_tgt_len)
    else:
        pass
    return dataset


def model_step(batch: dict, model, tokenizer):
    source_ids, source_mask, target_ids = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
    # import pdb; pdb.set_trace()
    if isinstance(model, BartForConditionalGeneration):
    # if isinstance(model, NoPosBartForConditionalGeneration):
        decoder_input_ids = target_ids[:, :-1].contiguous()
        lm_labels = target_ids[:, 1:].clone()   # DIFF HERE
        lm_labels[lm_labels == tokenizer.pad_token_id] = -100
    elif isinstance(model, GPT2LMHeadModel):
        decoder_input_ids = target_ids[:, :-1].contiguous()
        lm_labels = target_ids.clone()[:, 1:].clone()
        lm_labels[lm_labels == tokenizer.pad_token_id] = -100
    elif isinstance(model, PeftModelForSeq2SeqLM):
        decoder_input_ids = target_ids[:, :-1].contiguous()
        lm_labels = target_ids.clone()[:, 1:].clone()
        lm_labels[lm_labels == tokenizer.pad_token_id] = -100
    else:
        decoder_input_ids = None

    assert decoder_input_ids is not None
    nb_tokens = (lm_labels != -100).sum(dim=1)
    ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
    
    
    if isinstance(model, BartForConditionalGeneration):
        # import pdb; pdb.set_trace()
        outputs = model(
            source_ids,
            attention_mask=source_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        )
    elif isinstance(model, GPT2LMHeadModel):
        outputs = model(decoder_input_ids)
    
    elif isinstance(model, PeftModelForSeq2SeqLM):
        outputs = model(input_ids=source_ids,
                        attention_mask=source_mask,
                        decoder_input_ids=decoder_input_ids,
                        use_cache=False
                        )
        
        
    token_wise_loss = ce_loss(outputs['logits'].view(-1, outputs['logits'].size(-1)), lm_labels.view(-1))
    token_wise_loss = token_wise_loss.view(lm_labels.shape)
    token_wise_loss = token_wise_loss.sum(dim=1)
    token_wise_loss = token_wise_loss/nb_tokens
    loss = token_wise_loss.mean(dim=0)
    return (loss,)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'runs'))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)


    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    print (args.save_steps)
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0], ncols=50)
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    # best_loss = 99999
    best_bleu = -1
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0], ncols=50)
        for step, batch in enumerate(epoch_iterator):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(args.device)
            model.train()
            outputs = model_step(batch, model, tokenizer)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        if results['bleu_4'] > best_bleu:
                            logger.info("Saving model checkpoint to %s", args.output_dir)
                            if isinstance(model, PeftModelForSeq2SeqLM):
                                peft_model_id = f"{args.output_dir}_{peft_config.peft_type}_{peft_config.task_type}"
                                model.save_pretrained(peft_model_id)
                                ckpt = f"{peft_model_id}/adapter_model.bin"
                            else:
                                save_model(model, tokenizer, args.output_dir)
                            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
                            best_bleu = results['bleu_4']
                            command1 = 'cp ' + args.output_dir + 'eval_generation.txt ' + args.output_dir + 'best_eval_generation.txt '
                            command2 = 'cp ' + args.output_dir + 'merge_result_file ' + args.output_dir + 'best_merge_result_file '
                            subprocess.check_output(command1, shell=True)
                            subprocess.check_output(command2, shell=True)
                            
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
                    if isinstance(model, PeftModelForSeq2SeqLM):
                        peft_model_id = f"{args.output_dir}_{peft_config.peft_type}_{peft_config.task_type}"
                        model.save_pretrained(peft_model_id)
                        ckpt = f"{peft_model_id}/adapter_model.bin"
                    else:
                        save_model(model, tokenizer, args.output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.local_rank in [-1, 0] and args.logging_steps > 0:
            if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                results = evaluate(args, model, tokenizer)
                for key, value in results.items():
                    tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                if results['bleu_4'] > best_bleu:
                    logger.info("Saving model checkpoint to %s", args.output_dir)
                    if isinstance(model, PeftModelForSeq2SeqLM):
                        peft_model_id = f"{args.output_dir}_{peft_config.peft_type}_{peft_config.task_type}"
                        model.save_pretrained(peft_model_id)
                        ckpt = f"{peft_model_id}/adapter_model.bin"
                    else:
                        save_model(model, tokenizer, args.output_dir)
                    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
                    best_bleu = results['bleu_4']
                    command1 = 'cp ' + args.output_dir + 'eval_generation.txt ' + args.output_dir + 'best_eval_generation.txt '
                    command2 = 'cp ' + args.output_dir + 'merge_result_file ' + args.output_dir + 'best_merge_result_file '
                    subprocess.check_output(command1, shell=True)
                    subprocess.check_output(command2, shell=True)
            tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
            tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
            logging_loss = tr_loss
        if args.local_rank in [-1, 0] and args.save_steps > 0:
            checkpoint_prefix = 'checkpoint'
            output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)
            if isinstance(model, PeftModelForSeq2SeqLM):
                peft_model_id = f"{args.output_dir}_{peft_config.peft_type}_{peft_config.task_type}"
                model.save_pretrained(peft_model_id)
                ckpt = f"{peft_model_id}/adapter_model.bin"
            else:
                save_model(model, tokenizer, args.output_dir)
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", max_q = None, max_a = None, dynamics=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True, max_q=max_q, max_a=max_a)
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=eval_dataset.collate_fn)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    preds = []
    
    decoder_start_token = 0
    # if isinstance(tokenizer, T5Tokenizer):
    #     decoder_start_token = 0
    # else:
    #     decoder_start_token = tokenizer.bos_token_id


    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(args.device)
        with torch.no_grad():
            outputs = model_step(batch, model, tokenizer)
            lm_loss = outputs[0]
            eval_loss += lm_loss.item()
            generated_ids = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=True, num_beams=5, 
				length_penalty=0.6, max_length=32, repetition_penalty=2.0, decoder_start_token_id=decoder_start_token)    # DIFF HERE
        nb_eval_steps += 1
        gen_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) # clean_up_tokenization_spaces = True 清楚空格
        
        # gen_text = [t if t.endswith('？') else t+' ？' for t in gen_text]
        gen_text = [' '.join(t.strip().split(" ")) for t in gen_text]
        preds += gen_text


    refs_path = args.data_dir + "dev.target"
    # refs_path = args.data_dir + "dev.target_space"
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))


    written_file = os.path.join(eval_output_dir, prefix, "eval_generation.txt")

    with open(written_file, "w") as f:
        for i in range(len(preds)):
            f.write(preds[i])
            f.write("\n")

    nlg_res = compute_metrics(hypothesis=written_file, references=[refs_path,])
    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    merge_file = eval_output_dir + 'merge_result_file'
    merge_written_refs(written_file, refs_path, merge_file)
    

    bleu_1 = nlg_res['Bleu_1']
    bleu_2 = nlg_res['Bleu_2']
    bleu_3 = nlg_res['Bleu_3']
    bleu_4 = nlg_res['Bleu_4']
    rouge_l = nlg_res['ROUGE_L']
    Metetor = nlg_res['METEOR']

    result = {
        "eval_loss": eval_loss,
        "perplexity": perplexity,
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_3": bleu_3,
        "bleu_4": bleu_4,
        "rouge_l": rouge_l,
        "meteor": Metetor   
    }

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    

    return result

def evaluate_binary_classification(args, model, tokenizer, prefix="", max_q = None, max_a = None, dynamics=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    # eval_dataset = load_and_
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=eval_dataset.collate_fn)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    preds = []
    
    decoder_start_token = 0
    # if isinstance(tokenizer, T5Tokenizer):
    #     decoder_start_token = 0
    # else:
    #     decoder_start_token = tokenizer.bos_token_id


    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(args.device)
        with torch.no_grad():
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            bs_size = input_ids.shape[0]
            decoder_start_token = 0
            decoder_input_ids = torch.tensor([[decoder_start_token] for _ in range(bs_size)])
            
            outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, use_cache=False, return_dict=True)
            logits = outputs[0].squeeze(1)
            res = torch.argmax(logits[:, [4273, 150]], dim=-1) # [bs]
            res = res.unsqueeze(-1) #[bs, 1]
            
            
            # print(input_ids)
            # print(decoder_input_ids)
            
            
            # outputs = model_step(batch, model, tokenizer)
            # lm_loss = outputs[0]
            eval_loss += lm_loss.item()
            
            generated_ids = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=True, num_beams=5, 
				length_penalty=0.6, max_length=1, repetition_penalty=2.0, decoder_start_token_id=decoder_start_token)    # DIFF HERE
        nb_eval_steps += 1
        gen_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) # clean_up_tokenization_spaces = True 清楚空格
        
        # gen_text = [t if t.endswith('？') else t+' ？' for t in gen_text]
        gen_text = [' '.join(t.strip().split(" ")) for t in gen_text]
        preds += gen_text


    refs_path = args.data_dir + "dev.target"
    # refs_path = args.data_dir + "dev.target_space"
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))


    written_file = os.path.join(eval_output_dir, prefix, "eval_generation.txt")

    with open(written_file, "w") as f:
        for i in range(len(preds)):
            f.write(preds[i])
            f.write("\n")

    nlg_res = compute_metrics(hypothesis=written_file, references=[refs_path,])
    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    merge_file = eval_output_dir + 'merge_result_file'
    merge_written_refs(written_file, refs_path, merge_file)
    

    bleu_1 = nlg_res['Bleu_1']
    bleu_2 = nlg_res['Bleu_2']
    bleu_3 = nlg_res['Bleu_3']
    bleu_4 = nlg_res['Bleu_4']
    rouge_l = nlg_res['ROUGE_L']
    Metetor = nlg_res['METEOR']

    result = {
        "eval_loss": eval_loss,
        "perplexity": perplexity,
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_3": bleu_3,
        "bleu_4": bleu_4,
        "rouge_l": rouge_l,
        "meteor": Metetor   
    }

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    

    return result

def merge_written_refs(written_file, ref_file, merge_file):
    with open(written_file, 'r', encoding='utf-8') as reader1:
        with open(ref_file, 'r', encoding='utf-8') as reader2:
            written_data = [''.join(line.strip().split()) for line in reader1.readlines()]
            ref_data = [''.join(line.strip().split()) for line in reader2.readlines()]
    assert len(written_data) == len(ref_data)
    
    if 'nonpara' in merge_file:
        label = '0'
    else:
        label = '1'
    
    with open(merge_file, 'w', encoding='utf-8') as writer:
        for origin, generated in zip(written_data, ref_data):
            writer.write(origin + '\t' + generated + '\t' + label + '\n')
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir.")
    parser.add_argument("--dataset", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_name", default='', type=str, 
                        help="The training file name.")

    ## Other parameters
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--max_src_len", default=32, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--max_tgt_len", default=32, type=int,
                        help="Optional target sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--lang_type", default='cn', type=str, required=True)
    
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--load_output_model", action='store_true',
                        help="Whether to load model from output directory.")

    parser.add_argument("--per_gpu_train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()


    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    if args.do_train:
        handler = logging.FileHandler(os.path.join(args.output_dir, 'train.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        os.system("cp run_finetune_gen.py %s" % os.path.join(args.output_dir, 'run_finetune_gen.py'))
        os.system("cp utils.py %s" % os.path.join(args.output_dir, 'utils.py'))
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    print('dataset:', args.dataset)
    if args.dataset == "LCQMC":
        tokenizer_class = BertTokenizer
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, cache_dir=args.cache_dir if args.cache_dir else None, 
    task_specific_params={'no_pos':True})   # DIFF HERE
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    
    if args.model_type == "T5":
        model = get_peft_model(model, peft_config)
    
    model.to(args.device)
    print (count_parameters(model))
    # import pdb; pdb.set_trace()
    #add padding token to gpt2
    if args.model_type == 'gpt2':
        special_tokens_dict = {'pad_token': '<PAD>'}

        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print('We have added', num_added_toks, 'tokens')
        model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.

        assert tokenizer.pad_token == '<PAD>'
        #add padding token to gpt2

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache   
        if args.local_rank == 0:
            torch.distributed.barrier()
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            #if args.load_output_model: # with this, it will just evaluate last model instead of best one
            if isinstance(model_class, T5ForConditionalGeneration):
                config = PeftConfig.from_pretrained(checkpoint)
                model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
                model = PeftModel.from_pretrained(model, checkpoint)
                tokenizer = T5Tokenizer.from_pretrained(checkpoint)
            else:
                model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
    return results

if __name__ == "__main__":
    main()

