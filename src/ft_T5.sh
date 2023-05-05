# For CommonGen
model_path="google/flan-t5-xl"
model_type="T5"
time2=$(date "+%Y%m%d%H%M%S")
dataset="quora"
lang_type='en'  # cn or en
gen_type="para"

data_file="../Data/"$dataset"/genQP/midones/"$gen_type"/"
# data_file="../Data/quora/pi/genpi"
per_gpu_train_batch_size=16
seed=15213
epoch=2
learning_rate=2e-5
accumulation=2
output_dir="result/"$dataset"/"$gen_type"/finetuned-"$model_path/"bs"$per_gpu_train_batch_size"_accumulation"$accumulation"_epoch"$epoch"_lr"$learning_rate"_seed"$seed/


CUDA_VISIBLE_DEVICES=$1 python run_finetune_gen.py \
    --model_type=$model_type \
    --dataset=$dataset \
    --lang_type $lang_type \
    --model_name_or_path=$model_path \
    --data_dir $data_file \
    --do_train --do_eval --evaluate_during_training \
    --per_gpu_train_batch_size=$per_gpu_train_batch_size \
    --per_gpu_eval_batch_size=$per_gpu_train_batch_size \
    --gradient_accumulation_steps=1 \
    --logging_steps 500 \
    --output_dir $output_dir \
    --num_train_epochs $epoch \
    --warmup_steps 500 \
    --save_steps -1 --learning_rate $learning_rate \
    --adam_epsilon 1e-6 --seed $seed \
    --overwrite_output_dir \
    --gradient_accumulation_steps $accumulation
    # For min-overlap --train_name _min-overlap --max_steps 8424
    # For random --train_name _random17K --max_steps 8424