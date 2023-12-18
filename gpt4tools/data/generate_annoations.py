import os
import re
import json
import argparse
import openai
import random
import string
import numpy as np
from tqdm import tqdm

TOOLS = {
  'Detect the Give Object': 'useful when you only want to detect or find out given objects in the pictureThe input to this tool should be a comma separated string of two, representing the image_path, the text description of the object to be found',
  'Segment the Image': 'useful when you want to segment all the part of the image, but not segment a certain object.like: segment all the object in this image, or generate segmentations on this image, or segment the image,or perform segmentation on this image, or segment all the object in this image.The input to this tool should be a string, representing the image_path',
  'Get Photo Description': 'useful when you want to know what is inside the photo. receives image_path as input. The input to this tool should be a string, representing the image_path.',
  'Generate Image From User Input Text': 'useful when you want to generate an image from a user input text and save it to a file. like: generate an image of an object or something, or generate an image that includes some objects. The input to this tool should be a string, representing the text used to generate image.', 
  'Edge Detection On Image': 'useful when you want to detect the edge of the image. like: detect the edges of this image, or canny detection on image, or perform edge detection on this image, or detect the canny image of this image. The input to this tool should be a string, representing the image_path',
  'Generate Image Condition On Canny Image': 'useful when you want to generate a new real image from both the user description and a canny image. like: generate a real image of a object or something from this canny image, or generate a new real image of a object or something from this edge image. The input to this tool should be a comma separated string of two, representing the image_path and the user description.', 
  'Predict Depth On Image': 'useful when you want to detect depth of the image. like: generate the depth from this image, or detect the depth map on this image, or predict the depth for this image. The input to this tool should be a string, representing the image_path',
  'Generate Image Condition On Depth': 'useful when you want to generate a new real image from both the user description and depth image. like: generate a real image of a object or something from this depth image, or generate a new real image of a object or something from the depth map. The input to this tool should be a comma separated string of two, representing the image_path and the user description',
  'Answer Question About The Image': 'useful when you need an answer for a question based on an image. like: what is the background color of the last image, how many cats in this figure, what is in this figure. The input to this tool should be a comma separated string of two, representing the image_path and the question',
  'Instruct Image Using Text': 'useful when you want to the style of the image to be like the text. like: make it look like a painting. or make it like a robot. The input to this tool should be a comma separated string of two, representing the image_path and the text.', 
  'Sketch Detection On Image': 'useful when you want to generate a scribble of the image. like: generate a scribble of this image, or generate a sketch from this image, detect the sketch from this image. The input to this tool should be a string, representing the image_path',
  'Generate Image Condition On Sketch Image': 'useful when you want to generate a new real image from both the user description and a scribble image or a sketch image. The input to this tool should be a comma separated string of two, representing the image_path and the user description',
  'Generate Image Condition On Segmentations': 'useful when you want to generate a new real image from both the user description and segmentations. like: generate a real image of a object or something from this segmentation image, or generate a new real image of a object or something from these segmentations. The input to this tool should be a comma separated string of two, representing the image_path and the user description',
  'Pose Detection On Image': 'useful when you want to detect the human pose of the image. like: generate human poses of this image, or generate a pose image from this image. The input to this tool should be a string, representing the image_path',
  'Generate Image Condition On Pose Image': 'useful when you want to generate a new real image from both the user description and a human pose image. like: generate a real image of a human from this human pose image, or generate a new real image of a human from this pose. The input to this tool should be a comma separated string of two, representing the image_path and the user description',
  'Hed Detection On Image': 'useful when you want to detect the soft hed boundary of the image. like: detect the soft hed boundary of this image, or hed boundary detection on image, or perform hed boundary detection on this image, or detect soft hed boundary image of this image. The input to this tool should be a string, representing the image_path',
  'Generate Image Condition On Soft Hed Boundary Image': 'useful when you want to generate a new real image from both the user description and a soft hed boundary image. like: generate a real image of a object or something from this soft hed boundary image, or generate a new real image of a object or something from this hed boundary. The input to this tool should be a comma separated string of two, representing the image_path and the user description',
  'Predict Normal Map On Image': 'useful when you want to detect norm map of the image. like: generate normal map from this image, or predict normal map of this image. The input to this tool should be a string, representing the image_path',
  'Generate Image Condition On Normal Map': 'useful when you want to generate a new real image from both the user description and normal map. like: generate a real image of a object or something from this normal map, or generate a new real image of a object or something from the normal map. The input to this tool should be a comma separated string of two, representing the image_path and the user description',
  'Line Detection On Image': 'useful when you want to detect the straight line of the image. like: detect the straight lines of this image, or straight line detection on image, or perform straight line detection on this image, or detect the straight line image of this image. The input to this tool should be a string, representing the image_path',
  'Segment the given object': 'useful when you only want to segment the certain objects in the pictureaccording to the given textlike: segment the cat,or can you segment an obeject for meThe input to this tool should be a comma separated string of two, representing the image_path, the text description of the object to be found',
  'Remove Something From The Photo': 'useful when you want to remove and object or something from the photo from its description or location. The input to this tool should be a comma separated string of two, representing the image_path and the object need to be removed.',
  'Replace Something From The Photo': 'useful when you want to replace an object from the object description or location with another object from its description. The input to this tool should be a comma separated string of three, representing the image_path, the object to be replaced, the object to be replaced with'
}

TOOLS_OUTPUT_FORMAT = {
  'Detect the Give Object': '[output_from_detection_tool]',
  'Segment the Image': 'image_path', 
  'Get Photo Description': '[output_from_caption_tool]', 
  'Generate Image From User Input Text': 'image_path', 
  'Edge Detection On Image': 'image_path',
  'Generate Image Condition On Canny Image': 'image_path', 
  'Predict Depth On Image': 'image_path', 
  'Generate Image Condition On Depth': 'image_path',
  'Answer Question About The Image': '[output_from_vqa_tool]',
  'Instruct Image Using Text': 'image_path', 
  'Sketch Detection On Image': 'image_path',
  'Generate Image Condition On Sketch Image': 'image_path',
  'Generate Image Condition On Segmentations': 'image_path',
  'Pose Detection On Image': 'image_path',
  'Generate Image Condition On Pose Image': 'image_path',
  'Hed Detection On Image': 'image_path',
  'Generate Image Condition On Soft Hed Boundary Image': 'image_path',
  'Predict Normal Map On Image': 'image_path',
  'Generate Image Condition On Normal Map': 'image_path',
  'Line Detection On Image': 'image_path',
  'Segment the given object': 'image_path',
  'Remove Something From The Photo': 'image_path',
  'Replace Something From The Photo': 'image_path' 
}

TOOLS_NEED_IMAGE = {
  'Detect the Give Object': True, 
  'Segment the Image': True, 
  'Get Photo Description': True, 
  'Generate Image From User Input Text': False, 
  'Edge Detection On Image': True,
  'Generate Image Condition On Canny Image': True, 
  'Predict Depth On Image': True, 
  'Generate Image Condition On Depth': True,
  'Answer Question About The Image': True,
  'Instruct Image Using Text': True, 
  'Sketch Detection On Image': True, 
  'Generate Image Condition On Sketch Image': True, 
  'Generate Image Condition On Segmentations': True,
  'Pose Detection On Image': True,
  'Generate Image Condition On Pose Image': True,
  'Hed Detection On Image': True,
  'Generate Image Condition On Soft Hed Boundary Image': True,
  'Predict Normal Map On Image': True,
  'Generate Image Condition On Normal Map': True,
  'Line Detection On Image': True,
  'Segment the given object': True,
  'Remove Something From The Photo': True,
  'Replace Something From The Photo': True 
}

TOOLS_CONDITION = {
  'Detect the Give Object': None, 
  'Segment the Image': None, 
  'Get Photo Description': None, 
  'Generate Image From User Input Text': None, 
  'Edge Detection On Image': None,
  'Generate Image Condition On Canny Image': 'Edge Detection On Image', 
  'Predict Depth On Image': None, 
  'Generate Image Condition On Depth': 'Predict Depth On Image',
  'Answer Question About The Image': None,
  'Instruct Image Using Text': None, 
  'Sketch Detection On Image': None, 
  'Generate Image Condition On Sketch Image': 'Sketch Detection On Image', 
  'Generate Image Condition On Segmentations': 'Segment the Image',
  'Pose Detection On Image': None,
  'Generate Image Condition On Pose Image': 'Pose Detection On Image',
  'Hed Detection On Image': None,
  'Generate Image Condition On Soft Hed Boundary Image': 'Hed Detection On Image',
  'Predict Normal Map On Image': None,
  'Generate Image Condition On Normal Map': 'Predict Normal Map On Image',
  'Line Detection On Image': None,
  'Segment the given object': None,
  'Remove Something From The Photo': None,
  'Replace Something From The Photo': None 
}

TOOL_NUM_ARGS = {
  'Detect the Give Object': 2, 
  'Segment the Image': 1, 
  'Get Photo Description': 1, 
  'Generate Image From User Input Text': 1, 
  'Edge Detection On Image': 1,
  'Generate Image Condition On Canny Image': 2, 
  'Predict Depth On Image': 1, 
  'Generate Image Condition On Depth': 2,
  'Answer Question About The Image': 2,
  'Instruct Image Using Text': 2, 
  'Sketch Detection On Image': 1, 
  'Generate Image Condition On Sketch Image': 2, 
  'Generate Image Condition On Segmentations': 2,
  'Pose Detection On Image': 1,
  'Generate Image Condition On Pose Image': 2,
  'Hed Detection On Image': 1,
  'Generate Image Condition On Soft Hed Boundary Image': 2,
  'Predict Normal Map On Image': 1,
  'Generate Image Condition On Normal Map': 2,
  'Line Detection On Image': 1,
  'Segment the given object': 2,
  'Remove Something From The Photo': 2,
  'Replace Something From The Photo': 3
}

PROMPT_FORMAT = """GPT4Tools can handle various text and visual tasks, such as answering questions and providing in-depth explanations and discussions. It generates human-like text and uses tools to indirectly understand images. When referring to images, GPT4Tools follows strict file name rules. To complete visual tasks, GPT4Tools uses tools and stays loyal to observation outputs. Users can provide new images to GPT4Tools with a description, but tools must be used for subsequent tasks.
TOOLS:
------

GPT4Tools has access to the following tools:

{tool_descriptions}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
AI: [your response here]
```

Follow file name rules and do not fake non-existent file names. Remember to provide the image file name loyally from the last tool observation.

Previous conversation:

{chat_history}

New input: {input}
GPT4Tools needs to use tools to observe images, not directly imagine them. Thoughts and observations in the conversation are only visible to GPT4Tools. When answering human questions, repeat important information. Let's think step by step.
"""

LLM_IMAGE_FORMAT = """Human: Provide an image named {image_path}. Description: {image_caption} Understand the image using tools.
AI: Received."""

LLM_RESPONSE_TOOL_FORMAT = """Thought: Do I need to use a tool? Yes
Action: {tool_name}
Action Input: {args}
Observation:
"""

LLM_RESPONSE_HUMAN_FORMAT = """Thought: Do I need to use a tool? No
AI: {response}
"""

AGENT_RESPONSE_FORMAT = """{response}
"""

INSTRUCTION_ROLE_FORMAT = {'system': '{content}',
                           'user': 'Observation: {content}',
                           'assistant': '{content}'}


def random_image_name(num_chars=8):
    folder_name = random.choices(['image', 'images', 'examples', 'cache', None], k=1)[0]
    image_name = ''.join(random.choices(string.ascii_lowercase, k=num_chars))
    image_name = f'{image_name}.png'
    if folder_name is not None:
        image_name = f'{folder_name}/{image_name}'
    return image_name


def parse_instuct(inst):
    inst = (inst[3:].strip())
    inst = re.split('\[|\]', inst)
    instruction = inst[0].replace('"', '').strip()
    tool_name = inst[1].split(',')[0].strip()
    tool_prefix_len = len(tool_name)
    #####################
    tool_name = tool_name.replace('Detect the Given Object', 'Detect the Give Object')
    tool_name = tool_name.replace('Detect the given object', 'Detect the Give Object')
    tool_name = tool_name.replace('Detect Given Object', 'Detect the Give Object')
    tool_name = tool_name.replace('Detect Object', 'Detect the Give Object')
    tool_name = tool_name.replace('Detect Give Object', 'Detect the Give Object')
    tool_name = tool_name.replace('Segment Given Object', 'Segment the Given Object')
    tool_name = tool_name.replace('Segment Image', 'Segment the Image')
    for ori_name in TOOLS.keys():
        if tool_name.lower() == ori_name.lower():
            tool_name = ori_name
            break 
    #####################
    assert tool_name in TOOLS, f'invalid tool name: {tool_name}'
    args = inst[1][tool_prefix_len + 1:].split(',')
    for idx, arg in enumerate(args):
        args[idx] = arg.replace('"', '').strip()
    assert len(args) >= TOOL_NUM_ARGS[tool_name], f'invalid number of arguments for tool {tool_name}: {args}'
    if len(args) > TOOL_NUM_ARGS[tool_name]:
        if TOOLS_NEED_IMAGE[tool_name]:
            assert TOOL_NUM_ARGS[tool_name] == 2, f'invalid number of arguments for tool {tool_name}: {args}'
            args = [args[0], ','.join(args[1:])]
        else:
            args = [','.join(args)]
    if TOOLS_NEED_IMAGE[tool_name]:
        assert args[0] == 'example.jpg', f'invalid image name: {args[0]}'
    return instruction, tool_name, args


def generate_prompt(instruction, tool_name, caption=None, image_path=None, max_num_tools=5):
    valid_tools = []
    tool_names = [tool_name]
    if TOOLS_CONDITION[tool_name] is not None:
        tool_names.append(TOOLS_CONDITION[tool_name])
    for name in TOOLS.keys():
        if name not in tool_names:
            valid_tools.append(name)
    num_tools = random.randint(2, max_num_tools)
    valid_tools = tool_names + [str(x) for x in np.random.permutation(valid_tools)[:num_tools - len(tool_names)]]
    np.random.shuffle(valid_tools)
    tool_descriptions = '\n'.join([f'> {k}: {TOOLS[k]}'for k in valid_tools])
    tool_names = ', '.join(valid_tools)
    if caption is None:
        chat_history = ''
    else:
        chat_history = LLM_IMAGE_FORMAT.format(image_path=image_path, image_caption=caption)
    prompt = PROMPT_FORMAT.format(tool_descriptions=tool_descriptions,
                                  tool_names=tool_names,
                                  chat_history=chat_history,
                                  input=instruction) 
    return prompt


def generate_human_response(result):
    response = LLM_RESPONSE_HUMAN_FORMAT.format(response=result) 
    return response


def generate_tool_response(tool_name, args):
    args = ', '.join(args)
    response = LLM_RESPONSE_TOOL_FORMAT.format(
        tool_name=tool_name, args=args)
    return response


def generate_agent_response(tool_name, output_image_name):
    output = TOOLS_OUTPUT_FORMAT[tool_name]
    if output == 'image_path':
        output = random.choices(
            [f'Result saved as {output_image_name}',
             f'{output_image_name}'],
            k=1)[0]
    response = AGENT_RESPONSE_FORMAT.format(response=output)
    return response


def generate_annotations(data):
    annotations = []
    meta = []
    for item in tqdm(data):
        insts = item['instructions'].split('\n')
        for inst in insts:
            caption = item['caption']
            try:
                instruction, tool_name, args = parse_instuct(inst)
            except Exception as err:
                print(f'skip instruction: {inst} due to error: {err}')
                continue
            image_path = random_image_name()
            for idx, arg in enumerate(args):
                args[idx] = arg.replace('example.jpg', image_path)
            annot = []
            if not TOOLS_NEED_IMAGE[tool_name]:
                if np.random.rand() < 0.5:
                    caption = None
            annot += [{'role': 'system',
                       'content': generate_prompt(instruction, tool_name, caption, image_path)}]
            if TOOLS_CONDITION[tool_name] is not None:
                cond_tool_name = TOOLS_CONDITION[tool_name]
                output_image_name = random_image_name() 
                annot += [{'role': 'assistant',
                           'content': generate_tool_response(cond_tool_name, [image_path])}]
                annot += [{'role': 'user',
                           'content': generate_agent_response(cond_tool_name, output_image_name)}]
                for idx, arg in enumerate(args):
                    args[idx] = arg.replace(image_path, output_image_name)
            annot += [{'role': 'assistant',
                       'content': generate_tool_response(tool_name, args)}]
            annot += [{'role': 'user',
                       'content': generate_agent_response(tool_name, random_image_name())}]
            annot += [{'role': 'assistant',
                       'content': generate_human_response(annot[-1]['content'])}]
            annotations.append(annot)
            meta.append({'instruction': inst, 'tool': tool_name})
    return annotations, meta


def insert_alpaca_annotations(annotations, alpaca_data, instruct_data, ratio=1.0):
    np.random.shuffle(alpaca_data)
    alpaca_data = alpaca_data[:min(int(len(annotations) * ratio), len(alpaca_data))]
    tool_names_list = sorted(TOOLS.keys())
    for item in tqdm(alpaca_data):
        instruction = ' '.join([item['instruction'], item['input']])
        image_path = random_image_name()
        caption = None
        if np.random.rand() < 0.5:
            inst_idx = random.randrange(len(instruct_data))
            caption = instruct_data[inst_idx]['caption']
        tool_name = tool_names_list[random.randrange(len(tool_names_list))]
        annot = []
        annot += [{'role': 'system',
                   'content': generate_prompt(instruction, tool_name, caption, image_path)}]
        annot += [{'role': 'assistant',
                   'content': generate_human_response(item['output'])}]
        annotations.append(annot)
    return annotations


def filter_annotations(annotations, meta):
    filtered_annotations = []
    inst_set = set()
    for annot, meta_i in tqdm(zip(annotations, meta)):
        if meta_i['instruction'] not in inst_set:
            inst_set.add(meta_i['instruction'])
            filtered_annotations.append(annot)
    return filtered_annotations


def conversation_to_complement(annotations):
    converted_annotations = []
    for annot in tqdm(annotations):
        instruction = []
        assert len(annot) % 2 == 0, 'annotation should be even length'
        end_idx = random.randrange(0, len(annot) // 2) * 2 + 1
        for idx, msg in enumerate(annot[:end_idx]):
            inst_format = INSTRUCTION_ROLE_FORMAT[msg['role']]
            inst = inst_format.format(content=msg['content'])
            if idx != end_idx - 1:
                if inst.endswith('Observation:\n'):
                    inst = inst[:-len('Observation:\n')]
            instruction.append(inst)
        instruction = ''.join(instruction)
        output = annot[end_idx]['content']
        converted_annot = {'instruction': instruction, 'input': '', 'output': output}
        converted_annotations.append(converted_annot)
    return converted_annotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True, help='path to instruction file.')
    parser.add_argument('--output-path', required=True, help='path to annotation file.')
    parser.add_argument('--caption-path', default=None, help='path to caption file.')
    parser.add_argument('--alpaca-path', default='', help='path to alpaca data file.')
    parser.add_argument('--filter', action='store_true', help='filter duplicated instructions.')
    parser.add_argument('--complement', action='store_true', help='conversation to complement.')
    parser.add_argument('--insert-alpaca', action='store_true', help='insert alpaca data.')
    parser.add_argument('--alpaca-ratio', default=1.0, help='alpaca data ratio to input data.')
    parser.add_argument('--max_num_samples', default=100000, help='max number of samples.')
    parser.add_argument('--max_num_images', default=3000, help='max number of samples.')
    args = parser.parse_args()

    data = json.load(open(args.input_path, 'r'))
    np.random.shuffle(data)
    data = data[:min(args.max_num_images, len(data))]
    if args.caption_path is not None:
        captions = json.load(open(args.caption_path, 'r'))
        captions = {item['image_id']: item['caption'] for item in captions['annotations']}
        for item in data:
            idx = int(item['id'])
            caption = captions[idx]
            if caption is not None:
                item['caption'] = caption
            else:
                print(f'skip null caption of {idx}')

    annotations, meta = generate_annotations(data)

    if args.filter:
        annotations = filter_annotations(annotations, meta)

    if args.insert_alpaca:
        alpaca_data = json.load(open(args.alpaca_path, 'r'))
        annotations = insert_alpaca_annotations(annotations, alpaca_data, data, ratio=args.alpaca_ratio)

    if args.complement:
        annotations = conversation_to_complement(annotations)

    np.random.shuffle(annotations)
    if len(annotations) > args.max_num_samples:
        annotations = annotations[:args.max_num_samples]

    print(f'generated {len(annotations)} annotations.')
    json.dump(annotations, open(args.output_path, 'w'))
