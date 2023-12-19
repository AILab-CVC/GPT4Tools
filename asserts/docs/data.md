# Dataset
The data collection process is illustrated below: \
We fed GPT-3.5 with captions from 3K images and descriptions of 22 visual tasks. This produced 66K instructions, each corresponding to a specific visual task and a visual foundation model (tool). Subsequently, we eliminated duplicate instructions and retained 41K sound instructions. To teach the model to utilize tools in a predefined manner, we followed the prompt format used in Visual ChatGPT and converted these instructions into a conversational format. Concurrently, we generated negative data without tool usage by randomly sampling 3K instructions from [`alpaca_gpt4_data`](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) and converting them to the defined format. Using the generated 71K instructions, we finetuned the Vicuna using LoRA and got our GPT4Tools, which can automatically decide, control, and utilize distinct tools in a conversation.

Each sample follows the below format:
```
{
    'instruction': xxx,
    'input': xxx,
    'output': xxx,
}
```

### Download
| Data file name | **Size** | OneDrive| Google Driver|
|:------------------|:--------:| :--------: | :---------:|
| gpt4tools_71k.json    | 229 MB   | [link](https://1drv.ms/u/s!AqPQkBZ4aeVnhRdryHC9b1NtWJpZ?e=ZHBCqd) | [link](https://drive.google.com/file/d/1JKIT-Or1of7TJuWvmrJpPoOx0cLdcWry/view?usp=share_link)|
| gpt4tools_val_seen.json    | --   | [link](https://1drv.ms/u/s!AqPQkBZ4aeVnhT1DPh5qZtSoZjtC?e=bDALfB) | [link](https://drive.google.com/file/d/1nDl7zhtQSx-L12K7151DfQD-XTqh_uzc/view?usp=sharing)|
| gpt4tools_test_unseen.json    | --   | [link](https://1drv.ms/u/s!AqPQkBZ4aeVnhTz3dCV77Ps6abzQ?e=ex4ojQ) | [link](https://drive.google.com/file/d/1BHm0HEwYaVdMRYZiDdECy8ozyix607PH/view?usp=sharing)|

* ```gpt4tools_71k.json``` contains 71K instruction-following data we used for fine-tuning the GPT4Tools model. 

* ```gpt4tools_val_seen.json``` is the manually cleaned instruction data used for validation, which includes instructions related to tools of ```gpt4tools_71k.json```.

* ```gpt4tools_test_unseen.json``` cleaned instruction data used for testing, including instructions related to some tools that are absented in ```gpt4tools_71k.json```.


### Generation

During generation using GPT-3.5, the openai api_key should be set in the env (OPENAI_API_KEY).

* Raw Data Generation
```
python3 gpt4tools/data/get_instruction.py \
        --caption-path <your_caption_data_path> \
	    --instruction-path <instruction_data_path> 
```

* Cleaning, and Instructional Data Consutruction
```
python3 gpt4tools/data/generate_annoations.py \
        --input-path <instruction_data_path> \
        --output-path <annotations_path> \
	    --caption-path <your_caption_data_path> \
	    --alpaca-path <your_alpaca_instruction_path> \
	    --filter \
	    --complement \
	    --insert-alpaca
```
