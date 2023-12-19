
CACHE_DIR="./cache" # revised to your cache directory
export CACHE_DIR

BASE_MODEL="lmsys/vicuna-7b-v1.5"
LORA_MODEL="./outputs/vicuna-7b-v1.5-gpt4tools"  # the path to save gpt4tools lora weights

python gpt4tools_demo.py \
	--base_model $BASE_MODEL \
	--lora_model $LORA_MODEL \
	--llm_device "cpu" \
	--load "ImageCaptioning_cuda:0" \
    --cache-dir $CACHE_DIR \
    --server-port 29509 \
	--share
