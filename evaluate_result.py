import json
import os
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
import re
from pycocoevalcap.bleu.bleu import Bleu
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', required=True, help='annotation file')
    parser.add_argument('--save_path', default="output/eval_result", help='inference result')
    parser.add_argument('--save_name', default='vicuna_13b', required=True, help='save_name')
    parser.add_argument('--num-chunks', default=1, type=int, help="dataset chunks")
    args = parser.parse_args()
    return args


def gather_prediction(args):
    # param
    save_path = os.path.join(args.save_path, args.save_name)
    save_name = args.save_name
    num_chunks = args.num_chunks
    res_path_list = glob(save_path+f"/{args.save_name}_[0-{num_chunks-1}].json")
    res_gathered = []
    for res_path in res_path_list:
        res_chunk = json.load(open(res_path, 'r'))
        res_gathered.extend(res_chunk)
    # remove duplicate
    res_cleaned = {x['id']: x for x in res_gathered}
    res_cleaned = list(res_cleaned.values())
    # save
    save_file = os.path.join(save_path, save_name + ".json")
    json.dump(res_cleaned, open(save_file, 'w'))
    print(f"Saving gathered results to {save_file}")
    return save_file


class EvalTools():
    def __init__(self, gt_path, pred_path, debug=False):
        print("Loading...")
        gt_ann = self.load_json(gt_path)
        self.id_gt_dict = {x['id']: x for x in gt_ann}
        self.save_dir = str(Path(pred_path).parent)
        self.pred_ann = self.load_json(pred_path)
        self.thought_acc = []
        self.action_acc = []
        self.action_args_acc = []
        self.acc = []
        self.iou = []
        self.bleu_scorer = Bleu(n=1) # there may have words with length of 1.
        self.debug = debug
        
    def load_json(cls, path):
        return json.load(open(path, 'r'))

    def fun_strip(cls, x):
        return [i.strip() for i in x]

    def get_acc_item(self, gt, pred):
        len_pred, len_gt = len(pred), len(gt)
        min_len = min(len_pred, len_gt)
        max_len = max(len_pred, len_gt)

        flag = np.zeros(max_len)
        flag_gt = np.zeros(max_len)
        
        pred = pred[:min_len]
        gt = gt[:min_len]
        
        flag[:min_len] = (np.array(pred) == np.array(gt)).astype(np.float64)
        flag_gt[:len_gt] = 1.0

        return flag, flag_gt

    def get_args_acc_item(self, gt, pred):
        len_pred, len_gt = len(pred), len(gt)
        min_len = min(len_pred, len_gt)
        max_len = max(len_pred, len_gt)
        
        flag = np.zeros(max_len)
        flag_gt = np.zeros(max_len)
        
        pred = pred[:min_len]
        gt = gt[:min_len]
        
        for idx, (pred_item, gt_item) in enumerate(zip(pred, gt)):
            gt_item = re.split(',', gt_item)
            pred_item = re.split(',', pred_item) # TODO: potential bug when text of arguments has ','
            
            # the number of args is not equal
            if len(gt_item) != len(pred_item):
                flag[idx] = 0.0
                continue
            
            flag_item = []
            for (k, v) in zip(gt_item, pred_item):
                if type(k) != type(v): # different type
                    flag_item.append(-1)
                elif '.png' in k and '.png' not in v:
                    flag_item.append(-1)
                elif '.png' not in k:
                    # given object, something should be similar
                    sim = self.get_word_sim(k, v)
                    flag_item.append(sim)
                else:
                    flag_item.append(1.0)
            flag[idx] = 0.0 if (-1 in flag_item) else np.mean(flag_item)
        
        flag_gt[:len_gt] = 1.0
        
        return flag, flag_gt
    
    def get_word_sim(self, tgt_words, src_words):
        gts = {1: [tgt_words]}
        res = {1: [src_words]}
        return self.bleu_scorer.compute_score(gts, res)[0][0]

    def evaluate_item(self, gt_res, pred_res, idx):
        res = []
        # thought_res, action_res, action_args_res = [], [], []
        
        if "Action:" in gt_res and "Action Input:" in gt_res:
            # pre gt output
            gt_thought = re.findall('Thought: Do I need to use a tool\? (.*)', gt_res)
            gt_thought = self.fun_strip(gt_thought)
            gt_tool_name = re.findall('Action: (.*)', gt_res)
            gt_tool_name = self.fun_strip(gt_tool_name)
            gt_args = re.findall('Action Input: (.*)', gt_res)
            gt_args = self.fun_strip(gt_args)  
            assert len(gt_tool_name) == len(gt_args), "tool_name and args should be paired in gt"
            
            if len(gt_thought) != len(gt_tool_name):
                gt_thought = gt_thought[:len(gt_tool_name)]
            
            # check if using tool
            pred_thought = re.findall('Thought: Do I need to use a tool\? (.*)', pred_res)
            pred_thought = self.fun_strip(pred_thought)

            # action
            pred_tool_name = re.findall('Action: (.*)', pred_res)
            pred_tool_name = self.fun_strip(pred_tool_name)

            action_res, action_gt = self.get_acc_item(gt_tool_name, pred_tool_name)
                
            self.action_acc.append(np.mean(action_res))

            # args
            pred_args = re.findall('Action Input: (.*)', pred_res)
            pred_args = self.fun_strip(pred_args)

            action_args_res, action_args_gt = self.get_args_acc_item(gt_args, pred_args)
                
            self.action_args_acc.append(np.mean(action_args_res))

            # thought 的 数量或许会多一个，如果它是 No，这个不会影响 action 的执行, 且不是错的
            # 例如，一个 action 结束后，会继续提问 “Do I need use a tool? No”
            if len(pred_thought) == (1+len(pred_tool_name)) and pred_thought[-1] == "No":
                pred_thought.pop(-1)
            thought_res, thought_tgt = self.get_acc_item(gt_thought, pred_thought)
                
            self.thought_acc.append(np.mean(thought_res))

            # iou
            res = np.concatenate([thought_res, action_res, action_args_res])
            gt = np.concatenate([thought_tgt, action_gt, action_args_gt])
            intersection = np.sum(res * gt)
            union = max(np.sum(np.maximum(res, gt)), 1.0)
            self.iou.append(intersection / union)
            
            # acc
            if self.thought_acc[-1] < 1.0 or self.action_acc[-1] < 1.0 or self.action_args_acc[-1] < 0.5:
                self.acc.append(0.0)
            else:
                self.acc.append(1.0)
            
        else:
            # pre thought
            pred_thought = re.findall('Thought: Do I need to use a tool\? (.*)', pred_res)
            pred_thought = self.fun_strip(pred_thought)
            gt_thought = re.findall('Thought: Do I need to use a tool\? (.*)', gt_res)
            gt_thought = self.fun_strip(gt_thought)
            
            thought_res, thought_tgt = self.get_acc_item(gt_thought, pred_thought)
            self.thought_acc.append(np.mean(thought_res))
            
            
            if len(pred_thought) == 0 or pred_thought[0].strip() != 'No': # thought error
                self.thought_acc[-1] = 0.0

            # check has AI prefix
            pred_ai_prefix = re.findall('AI: (.*)', pred_res)
            if len(pred_ai_prefix) == 0:
                self.thought_acc[-1] = 0.0
            
            # iou
            intersection = np.sum(thought_res * thought_tgt)
            union = max(np.sum(np.maximum(thought_res, thought_tgt)), 1.0)
            self.iou.append(intersection / union)
            
            self.action_acc.append(1.0) # check length
            self.action_args_acc.append(1.0) # check length
            
            # acc
            self.acc.append(1.0)
            if self.thought_acc[-1] < 1.0:
                self.acc[-1] = 0.0
                self.action_acc[-1] = 0.0
                self.action_args_acc[-1] = 0.0
        
        if self.debug and self.acc[-1] < 1.0:
            print('-------------------------')
            print(f"id: {idx}\n***gt:\n {gt_res}\n***pred:\n {pred_res}")


    def evaluate(self):
        for pred_item in tqdm(self.pred_ann):
            gt_item = self.id_gt_dict.get(pred_item['id'], None)
            if gt_item is None:
                print("Empty Gt")
                continue
            
            inst = gt_item['instruction']
            # inst = self.id_gt_dict[pred_item['id']]['instruction']
            gt_out = gt_item['output']
            if isinstance(pred_item['output'], list):
                pred_out = pred_item['output'][0][len(inst):].strip()
            else:
                pred_out = pred_item['output'][len(inst):].strip()
            self.evaluate_item(gt_out, pred_out, pred_item['id'])
    
    def evaluate_gpt(self):
        for pred_item in tqdm(self.pred_ann):

            gt_item = self.id_gt_dict.get(pred_item['id'], None)
            if gt_item is None:
                print("Empty Gt")
                continue
            inst = gt_item['instruction']
            gt_out = gt_item['output']
            if isinstance(pred_item['output'], list):
                pred_out = pred_item['output'][0].strip()
            else:
                pred_out = pred_item['output'].strip()
            self.evaluate_item(gt_out, pred_out, pred_item['id'])
        
        record = f"""
        Thought Acc: {np.mean(self.thought_acc)}
        Action Acc: {np.mean(self.action_acc)}
        Action Args Acc: {np.mean(self.action_args_acc)}
        Acc: {np.mean(self.acc)}
        IoU: {np.mean(self.iou)}
        """
        print(record)
    
    def remove_item(cls, l):
        while -1 in l:
            l.remove(-1)
    
    def summary(self):
        self.evaluate()
        assert len(self.thought_acc) == len(self.action_acc) == len(self.action_args_acc) == len(self.iou) == len(self.acc)

        record = f"""
        Thought Acc: {np.mean(self.thought_acc)}
        Action Acc: {np.mean(self.action_acc)}
        Action Args Acc: {np.mean(self.action_args_acc)}
        Acc: {np.mean(self.acc)}
        IoU: {np.mean(self.iou)}
        """
        
        # save
        save_path = os.path.join(self.save_dir, 'eval.txt')
        with open(save_path, 'w') as w:
            w.write(record)
        
        print(record)


if __name__=="__main__":
    args = parse_args()
    print("Gathering...")
    save_file = gather_prediction(args)
    print("Evaluating...")
    Eval = EvalTools(gt_path=args.ann_path, pred_path=save_file)
    Eval.summary()
    