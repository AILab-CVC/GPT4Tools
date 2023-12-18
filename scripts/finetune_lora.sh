
CACHE_DIR="./cache" # revised to your cache directory
DATA_PATH="datasets/gpt4tools_71k.json" # datasets
DEEPSPEED_CONFIG="scripts/zero2.json"
OUTPUT_DIR="outputs/vicuna-13b-v1.5-gpt4tools"
MODEL="lmsys/vicuna-13b-v1.5"

deepspeed train.py \
    --model_name_or_path $MODEL \
    --deepspeed $DEEPSPEED_CONFIG \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path $DATA_PATH \
    --bf16 True \
    --tf32 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --cache_dir $CACHE_DIR \
    --report_to 'tensorboard' \
    --gradient_checkpointing True
