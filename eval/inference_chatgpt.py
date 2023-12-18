import os
import json
import argparse
import openai
from tqdm import tqdm
from evaluate_result import EvalTools_v2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', required=True, help='annotation file')
    parser.add_argument('--save_path', default="output/eval_result", help='inference result')
    parser.add_argument('--save_name', default='chatgpt', required=True, help='save_name')
    parser.add_argument('--model', default='davinci', help='model type [davinci, turbo]')
    args = parser.parse_args()
    return args


def chatgpt(messages, temperature=0.7):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature
        )
        messages.append(completion.choices[0].message)
        return completion.choices[0].message.content
    except Exception as err:
        print(err)
        return chatgpt(messages, temperature)
    

def text_davicin(prompt, temperature=0.0):
    try:
        response = openai.Completion.create(
            prompt=prompt,
            temperature=temperature,
            max_tokens=400,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            # stop=stop_sequence,
            model='text-davinci-003',
        )
        return response['choices'][0]['text'].strip()
    except Exception as err:
        print(err)
        return text_davicin(prompt, temperature)


if __name__=="__main__":

    args = parse_args()
    
    openai.api_key = os.getenv('OPENAI_API_KEY')

    print(args)
    
    val_data = json.load(open(args.ann_path, 'r'))
    predictions = []
    save_dir = os.path.join(args.save_path, args.save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, args.save_name+'.json')
    
    suffix_prompt = "Thought: Do I need to use a tool? "
    
    for idx, data_item in enumerate(tqdm(val_data)):
        inst = data_item['instruction']
        inst_id = data_item['id']
        if args.model == "turbo":
            inst = inst + "\n" + suffix_prompt
            messages = [{'role': 'system', 'content': inst}]
            response = chatgpt(messages, temperature=0.0)
            # turbo will repeat the "Thought: Do I need to use a tool?"
        else:
            inst = inst + "\n" + suffix_prompt
            response = text_davicin(inst, temperature=0.0)
            response = suffix_prompt + response
        
        predictions.append({'output': response, 'id': inst_id})
        if idx % 10 == 0:
            print(f'Dumping {idx}/{len(val_data)}')
            json.dump(predictions, open(save_path, 'w'))
    json.dump(predictions, open(save_path, 'w'))
    print("Done!")   
    
    # evaluation
    Eval = EvalTools_v2(gt_path=args.ann_path, pred_path=save_path)
    Eval.evaluate_gpt() 
    