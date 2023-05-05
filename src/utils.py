import json
import linecache
import os
from collections import Counter
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List
import re
import torch
from torch.utils.data import Dataset

from transformers import BartTokenizer, GPT2Tokenizer, BertTokenizer, T5Tokenizer
import subprocess
import random
import numpy as np

logger = getLogger(__name__)

blue_file_path = './multi-bleu-detok.perl'
def bleu_score(refs_path, hyps_path, num_refs=1):
    ref_files = []
    for i in range(num_refs):
        if num_refs == 1:
            ref_files.append(refs_path)
        else:
            ref_files.append(refs_path + str(i))
    command = 'perl {0} {1} < {2}'.format(blue_file_path, ' '.join(ref_files), hyps_path)

    result = subprocess.check_output(command, shell=True)

    try:
        bleu = float(re.findall('BLEU = (.+?),', str(result))[0])
    except:
        print('ERROR ON COMPUTING METEOR. MAKE SURE YOU HAVE PERL INSTALLED GLOBALLY ON YOUR MACHINE.')
        bleu = -1
    print('FINISHING TO COMPUTE BLEU...')
    return bleu


def count_parameters(model):
    	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_model(model, tokenizer, location):
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    tokenizer.save_pretrained(location)
    model_to_save.save_pretrained(location)
    


class TrainDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length=512,
        max_target_length=512,
        type_path="train",
        n_obs=None,
        prefix="",
    ):
        super().__init__()
        self.type_path = type_path
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:

        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = self.prefix + linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        if isinstance(self.tokenizer, BartTokenizer):
            source_inputs = self.tokenizer(source_line, add_prefix_space=True)
            target_inputs = self.tokenizer(tgt_line, add_prefix_space=True)
        elif isinstance(self.tokenizer, BertTokenizer):
            source_inputs = self.tokenizer(source_line)
            target_inputs = self.tokenizer(tgt_line)
        elif isinstance(self.tokenizer, T5Tokenizer):
            source_inputs = self.tokenizer(source_line)
            target_inputs = self.tokenizer(tgt_line)
        
        # import pdb; pdb.set_trace()
        source_ids = source_inputs["input_ids"]
        target_ids = target_inputs["input_ids"]
        src_mask = source_inputs["attention_mask"]
        qid = self.type_path+str(index)
        
        if isinstance(self.tokenizer, T5Tokenizer):
            decoder_start_ids = 0
            target_ids = [decoder_start_ids] + target_ids
        
        return {
            "qid": qid,
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        max_src_len = max([len(x['input_ids']) for x in batch])
        max_tgt_len = max([len(x['decoder_input_ids']) for x in batch])
        input_ids = []
        masks = []
        target_ids = []
        for x in batch:
            input_ids.append(x['input_ids']+[pad_token_id]*(max_src_len-len(x['input_ids'])))
            masks.append(x['attention_mask']+[0]*(max_src_len-len(x['attention_mask'])))
            target_ids.append(x['decoder_input_ids']+[pad_token_id]*(max_tgt_len-len(x['decoder_input_ids'])))
        source_ids = torch.tensor(input_ids, dtype=torch.long)
        source_mask = torch.tensor(masks, dtype=torch.long)
        y = torch.tensor(target_ids, dtype=torch.long)
        qids = [x['qid'] for x in batch]
        if self.prefix == '':
            batch = {
                "qids": qids,
                "input_ids": source_ids,
                "attention_mask": source_mask,
                "decoder_input_ids": y,
            }
            return batch
        else:
            decoder_input_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[lm_labels == pad_token_id] = -100
            lm_labels[lm_labels == self.tokenizer.trigger_token_id] = -100
            decoder_trigger_mask = decoder_input_ids.eq(self.tokenizer.trigger_token_id)
            trigger_mask = source_ids.eq(self.tokenizer.trigger_token_id)

            batch = {
                "qids": qids,
                "input_ids": source_ids,
                "attention_mask": source_mask,
                "decoder_input_ids": decoder_input_ids,
                "labels": lm_labels,
                "trigger_mask": trigger_mask, 
                "decoder_trigger_mask": decoder_trigger_mask,
            }
            return batch


# # 用于gen模型判断二分类任务的数据读取
class TestClassDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length=512,
        max_target_length=512,
        type_path="train",
        n_obs=None,
        prefix="",
    ):
        super().__init__()
        self.type_path = type_path
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:

        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = self.prefix + linecache.getline(str(self.tgt_file), index).rstrip("\n")
        labels = [1 if line == 'yes' else 0 for line in tgt_line]
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        if isinstance(self.tokenizer, BartTokenizer):
            source_inputs = self.tokenizer(source_line, add_prefix_space=True)
            decoder_start_ids = None
        elif isinstance(self.tokenizer, BertTokenizer):
            source_inputs = self.tokenizer(source_line)
            decoder_start_ids = None
        elif isinstance(self.tokenizer, T5Tokenizer):
            source_inputs = self.tokenizer(source_line)
            decoder_start_ids = 0
        
        # import pdb; pdb.set_trace()
        source_ids = source_inputs["input_ids"]
        decoder_input_ids = [decoder_start_ids]
        src_mask = source_inputs["attention_mask"]
        qid = self.type_path+str(index)
        

        
        return {
            "qid": qid,
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        max_src_len = max([len(x['input_ids']) for x in batch])
        max_tgt_len = max([len(x['decoder_input_ids']) for x in batch])
        input_ids = []
        masks = []
        decoder_input_ids = []
        for x in batch:
            input_ids.append(x['input_ids']+[pad_token_id]*(max_src_len-len(x['input_ids'])))
            masks.append(x['attention_mask']+[0]*(max_src_len-len(x['attention_mask'])))
            decoder_input_ids.append(x['decoder_input_ids']+[pad_token_id]*(max_tgt_len-len(x['decoder_input_ids'])))
            
        source_ids = torch.tensor(input_ids, dtype=torch.long)
        source_mask = torch.tensor(masks, dtype=torch.long)
        decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long)
        labels = torch.tensor(x['labels'], dtype=torch.long)
        qids = [x['qid'] for x in batch]
        if self.prefix == '':
            batch = {
                "qids": qids,
                "input_ids": source_ids,
                "attention_mask": source_mask,
                "decoder_input_ids": decoder_input_ids,
                "labels": labels
            }
            return batch
        else:
            print('something you dont write')
        # else:
        #     decoder_input_ids = y[:, :-1].contiguous()
        #     lm_labels = y[:, 1:].clone()
        #     lm_labels[lm_labels == pad_token_id] = -100
        #     lm_labels[lm_labels == self.tokenizer.trigger_token_id] = -100
        #     decoder_trigger_mask = decoder_input_ids.eq(self.tokenizer.trigger_token_id)
        #     trigger_mask = source_ids.eq(self.tokenizer.trigger_token_id)

        #     batch = {
        #         "qids": qids,
        #         "input_ids": source_ids,
        #         "attention_mask": source_mask,
        #         "decoder_input_ids": decoder_input_ids,
        #         "labels": lm_labels,
        #         "trigger_mask": trigger_mask, 
        #         "decoder_trigger_mask": decoder_trigger_mask,
        #     }
            return batch



class GenDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length=512,
        prefix="",
    ):
        super().__init__()
        self.src_file = data_dir
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"


        if isinstance(self.tokenizer, BartTokenizer):
            source_inputs = self.tokenizer(source_line, add_prefix_space=True)
        elif isinstance(self.tokenizer, BertTokenizer):
            source_inputs = self.tokenizer(source_line)

        
        source_ids = source_inputs["input_ids"]
        src_mask = source_inputs["attention_mask"]
        qid = 'dev' + str(index)
        return {
            "qid": qid,
            "input_ids": source_ids,
            "attention_mask": src_mask
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        max_src_len = max([len(x['input_ids']) for x in batch])

        input_ids = []
        masks = []

        for x in batch:
            input_ids.append(x['input_ids']+[pad_token_id]*(max_src_len-len(x['input_ids'])))
            masks.append(x['attention_mask']+[0]*(max_src_len-len(x['attention_mask'])))

        source_ids = torch.tensor(input_ids, dtype=torch.long)
        source_mask = torch.tensor(masks, dtype=torch.long)
        qids = [x['qid'] for x in batch]
        if self.prefix == '':
            batch = {
                "qids": qids,
                "input_ids": source_ids,
                "attention_mask": source_mask,
            }
            return batch
        else:
            return None
            # lm_labels = y[:, 1:].clone()
            # lm_labels[lm_labels == pad_token_id] = -100
            # lm_labels[lm_labels == self.tokenizer.trigger_token_id] = -100
            # decoder_trigger_mask = decoder_input_ids.eq(self.tokenizer.trigger_token_id)
            # trigger_mask = source_ids.eq(self.tokenizer.trigger_token_id)

            # batch = {
            #     "qids": qids,
            #     "input_ids": source_ids,
            #     "attention_mask": source_mask,
            #     "decoder_input_ids": decoder_input_ids,
            #     "labels": lm_labels,
            #     "trigger_mask": trigger_mask, 
            #     "decoder_trigger_mask": decoder_trigger_mask,
            # }
            # return batch



def accuracy(all_logits, all_labels):
    # import pdb; pdb.set_trace()
    all_predict = (all_logits > 0) + 0
    results = (all_predict == all_labels)
    acc = results.sum() / len(all_predict)
    return acc


def precision(all_logits, all_labels):
    all_pred = (all_logits > 0) + 0
    TP = ((all_pred == 1) & (all_labels == 1)).sum()
    FP = ((all_pred == 1) & (all_labels == 0)).sum()
    precision = TP / (TP + FP)
    return precision


def recall(all_logits, all_labels):
    all_pred = (all_logits > 0) + 0
    TP = ((all_pred == 1) & (all_labels == 1)).sum()
    FN = ((all_pred == 0) & (all_labels == 1)).sum()
    recall = TP / (TP + FN)
    return recall


def f1_score(all_logits, all_labels):
    all_pred = (all_logits > 0) + 0
    TP = ((all_pred == 1) & (all_labels == 1)).sum()
    FP = ((all_pred == 1) & (all_labels == 0)).sum()
    FN = ((all_pred == 0) & (all_labels == 1)).sum()
    # TN = ((all_pred == 0) & (all_labels == 0)).sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * precision * recall) / (precision + recall)
    return F1