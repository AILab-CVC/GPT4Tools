### Old Weights
GPT4Tools is based on the Vicuna, we release the LoRA weights of GPT4Tools to comply with the LLaMA model license. You can merge our LoRA weights with the Vicuna weights to obtain the GPT4Tools weights.

Steps:
1. Get the original LLaMA weights in the Hugging Face format from [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
2. Using the [FastChat](https://github.com/lm-sys/FastChat/blob/main/README.md) to get Vicuna weigths by applying [the delta weights](https://huggingface.co/lmsys), more details please check [here](https://github.com/lm-sys/FastChat#vicuna-weights).
3. Get the LoRA weights of GPT4Tools ([Hugging Face](https://huggingface.co/stevengrove/gpt4tools-vicuna-13b-lora), [OneDrive](https://1drv.ms/f/s!AqPQkBZ4aeVnhRzM69NOXLyG8cTY?e=apmpyQ), or [Google Driver](https://drive.google.com/drive/folders/1ebUINGR0QzNL-4hoKl19-6D_5rfeWTPD?usp=share_link)).