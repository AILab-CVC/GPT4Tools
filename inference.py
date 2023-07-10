from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM,
                          LlamaForCausalLM, LlamaTokenizer, GenerationConfig)
import torch
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel
import json
import os
from tqdm import tqdm
import argparse
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--base_model', type=str, required=True, help='folder path to the vicuna with tokenizer')
    parser.add_argument('--lora_model', type=str, default='none', help='folder path to the lora model')
    parser.add_argument('--llm_device', type=str, default='cpu', help='device to run the llm model')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature for the llm model')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='max number of new tokens to generate')
    parser.add_argument('--top_p', type=float, default=0.75, help='top_p for the llm model')
    parser.add_argument('--top_k', type=int, default=40, help='top_k for the llm model')
    parser.add_argument('--num_beams', type=int, default=1, help='num_beams for the llm model')
    parser.add_argument('--keep_last_n_paragraphs', type=int, default=1, help='keep last n paragraphs in the memory')
    parser.add_argument('--distributed', action='store_true', help='enable distribution')
    parser.add_argument('--ann_path', required=True, help='annotation file')
    parser.add_argument('--save_path', default="output/eval_result", help='inference result')
    parser.add_argument('--save_name', default='vicuna_13b', required=True, help='save_name')
    parser.add_argument('--num-chunks', default=1, type=int, help="dataset chunks")
    parser.add_argument('--chunk-idx', type=int, default=0, help="chunk idx when chunks")
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


class PseudoDataset(Dataset):
    def __init__(self, ann_path):
        self.ann_path = ann_path
        self.ann_data = json.load(open(ann_path, 'r'))
        
    def __len__(self):
        return len(self.ann_data)
    
    def __getitem__(self, idx):
        return self.ann_data[idx]
    

def load_model(base_model, lora_model, cache_dir=None):
    if cache_dir is None:
        cache_dir = os.getenv('TRANSFORMERS_CACHE')
    
    if 'llama' in base_model.lower() or 'vicuna' in base_model.lower() or 'alpaca' in base_model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(
            base_model, use_fast=False, cache_dir=cache_dir)
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            cache_dir=cache_dir)
        tokenizer.pad_token_id = 0
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # load_in_8bit=True, # NOTE: ???
            torch_dtype=torch.float16,
            # device_map='auto',
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
    if lora_model != 'none':
        model = PeftModel.from_pretrained(
            model,
            lora_model,
            torch_dtype=torch.float16,)
        # set peft to inference mode
        model.config.inference_mode = True

    return tokenizer, model


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval(args):
    # param
    base_model = args.base_model
    lora_model = args.lora_model
    distributed = args.distributed
    ann_path = args.ann_path
    save_path = args.save_path
    save_name = args.save_name
    save_path = os.path.join(save_path, save_name)
    batch_size = 1
    num_chunks = args.num_chunks
    chunk_idx = args.chunk_idx
    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # mkdir
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # load model
    print("Loading Large Language Model (LLM)...")
    tokenizer, model = load_model(base_model, lora_model)
    if device == "cpu":
        model = model.float()
    else:
        model = model.half()
    model = model.to(device)

    model = model.eval()

    # split dataset to each chunks TODO: use dataloader sampler
    ann_data = json.load(open(ann_path, 'r'))
    ann_data_splited = get_chunk(ann_data, num_chunks, chunk_idx)
    
    result = []
    for samples in tqdm(ann_data_splited):
        inst = samples['instruction']
        inst_id = samples['id']
        response = eval_item(inputs=inst, model=model, tokenizer=tokenizer, generation_config=generation_config, device=device)
        response['id'] = inst_id
        result.append(response)
    
    json.dump(result, open(os.path.join(save_path, save_name + "_" + str(chunk_idx) + ".json"), 'w'))
    print("Done!")
    

def eval_item(inputs, model, tokenizer, generation_config, device):
    with torch.inference_mode():
        input_ids = tokenizer(inputs, return_tensors="pt").to(device).input_ids
        generate_ids = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,)
        response = tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,)
    # postpocessing
    response = [res.replace('### ASSISTANT:\n', '') for res in response]
    assert len(response) == 1, "only support bach_size = 1"
    response = {"output": response[0]}
    return response

def main():
    args = parse_args()
    print(args)
    eval(args)


if __name__ == "__main__":
    main()
