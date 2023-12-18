### Inference and Evaluation
* Using 8 GPUs (recommendation)

```
bash scripts/batch_inference.sh 8  <path_to_vicuna_with_tokenizer> <path_to_lora_weights> <your_annotation_path> <name_to_save>
```

* Using 1 GPU

```
python3 inference.py --base_model <path_to_vicuna_with_tokenizer> \
    --lora_model <path_to_lora_weights> \
    --ann_path <your_annotation_path> \
	--save_name <name_to_save> \
	--llm_device 'cuda'
```

then  

```
python3 evaluate_result.py --ann_path <your_annotation_path> \
	--save_name <name_to_save>
```

* Inference using GPT-3.5

```
python3 inference_chatgpt.py --ann_path <your_annotation_path> \
	--save_name <name_to_save> \
	--model 'davinci'
```
The openai api_key should be set in the env (OPENAI_API_KEY).

* ```your_annotation_path``` is 'your_path/gpt4tools_val_seen.json' or 'your_path/gpt4tools_test_unseen.json'.
