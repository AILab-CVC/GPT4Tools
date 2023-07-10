CHUNKS=$1
BASE_MODEL=$2
LORA_MODEL=$3
ANN_PATH=$4
SAVE_NAME=$5

for IDX in {0..7}; do
    CUDA_VISIBLE_DEVICES=$IDX python3 inference.py \
        --base_model $BASE_MODEL \
        --lora_model $LORA_MODEL \
        --ann_path $ANN_PATH \
        --save_name $SAVE_NAME\
        --llm_device 'cuda' \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

echo "Evaluating..."

python3 evaluate_result.py --ann_path $ANN_PATH --save_name $SAVE_NAME --num-chunks $CHUNKS