model_name="plllama-7b"
train_dataset_names="super_glue-boolq" # "mmlu-train" "gsm8k-train" "piqa-train" "hellaswag-train"
eval_tasks="boolq"  # "mmlu" "gsm8k_yaml" "piqa" "hellaswag"
lora_config="pl"


source_max_len=900
gradient_checkpointing=False 

bs_gpu_accu=$((32*8))

gpu_count=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
learning_rate=5e-3
per_device_eval_batch_size=16
batch_size=8
deepspeed_file="./new_ds_config.json"
gradient_accumulation_steps=$((bs_gpu_accu/batch_size/gpu_count))

deepspeed exp_private_lora.py \
    --deepspeed $deepspeed_file \
    --is_baseline False \
    --baseline_use_lora False \
    --peft_lora_config_ind $lora_config \
    --learning_rate $learning_rate \
    --model_name $model_name \
    --num_train_epochs 6 \
    --train_dataset_names $train_dataset_names \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --eval_tasks $eval_tasks \
    --evaluation_strategy "no" \
    --do_eval True \
    --eval_steps 2 \
    --do_init_eval True \
    --do_train \
    --overwrite_output_dir \
    --bf16 True \
    --report_to "tensorboard" \
    --save_strategy "no" \
    --gradient_checkpointing $gradient_checkpointing \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --source_max_len $source_max_len \
    --logging_steps 1 \
    --disable_tqdm False \
    --remove_unused_columns False \
    --train_on_source False \
    --warmup_steps 20 \
    --output_dir "run"

