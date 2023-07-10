import os
import json
import argparse
import openai
from tqdm import tqdm

INSTRUCTION_PROMPT = """Given an image whose image_path is "example.jpg". Image caption: "{caption}". 
The image caption includes detail image description and each object paired with the bounding box [x1, y1, x2, y2]. For the bounding box, (x1, y1) refers to the top left, and (x2, y2) refers to the bottom right. x1 less than x2, and y1 less than y2. 
 
Below are 26 visual tools. Each tool is defined as "tool_name: usage scenario, and arguments_to_tool".

Please generate 1 visual instructions for each tools, so you need to generate 26 visual instruction in total. 
The generated instructions should follow the format of "instruction_content, [tool_name, arguments_to_tool]".  Each instruction must relate to the caption and can be solved by the tool. 
You can not revise the "tool_name", or add any other fake tools that is not defined. You must keep the correct "arguments_to_tool".

Tools:
1. Detect the Give Object: Useful when you only want to detect or find given objects in the picture. The input to this tool should be a comma-separated string of two, representing the image_path, the text description of the object to be found.
2. Segment the Image: Useful when you want to segment all the parts of the image, but not segment a certain object. like: segment all the objects in this image, or generate segmentations on this image, or segment the image, or perform segmentation on this image, or segment all the objects in this image. The input to this tool should be a string, representing the image_path.
3. Get Photo Description: Useful when you want to know what is inside the photo. receives image_path as input. The input to this tool should be a string, representing the image_path. 
4. Generate Image From User Input Text: Useful when you want to generate an image from a user input text and save it to a file. like: generate an image of an object or something, or generate an image that includes some objects. The input to this tool should be a string, representing the text used to generate the image. 
5. Edge Detection On Image: Useful when you want to detect the edge of the image. like: detect the edges of this image, or canny detection on the image, or perform edge detection on this image, or detect the canny image of this image. The input to this tool should be a string, representing the image_path.
6. Generate Image Condition On Canny Image: Useful when you want to generate a new real image from both the user description and a canny image. like: generate a real image of an object or something from this canny image, or generate a new real image of an object or something from this edge image. The input to this tool should be a comma-separated string of two, representing the image_path and the user description. 
7. Predict Depth On Image: Useful when you want to detect the depth of the image. like: generate the depth from this image, or detect the depth map on this image, or predict the depth for this image. The input to this tool should be a string, representing the image_path.
8. Generate Image Condition On Depth: Useful when you want to generate a new real image from both the user description and depth image. like: generate a real image of an object or something from this depth image, or generate a new real image of an object or something from the depth map. The input to this tool should be a comma-separated string of two, representing the image_path and the user description.
9. Answer Question About The Image: Useful when you need an answer for a question based on an image. like: what is the background color of the last image, how many cats are in this figure, what is in this figure. The input to this tool should be a comma-separated string of two, representing the image_path and the question.
10. Instruct Image Using Text: Useful when you want the style of the image to be like the text. like: make it look like a painting. or make it like a robot. The input to this tool should be a comma-separated string of two, representing the image_path and the text. 
11. Sketch Detection On Image: Useful when you want to generate a scribble of the image. like: generate a scribble of this image, or generate a sketch from this image, detect the sketch from this image. The input to this tool should be a string, representing the image_path.
12. Generate Image Condition On Sketch Image: Useful when you want to generate a new real image from both the user description and a scribble image or a sketch image. The input to this tool should be a comma-separated string of two, representing the image_path and the user description.
13. Generate Image Condition On Segmentations: Useful when you want to generate a new real image from both the user description and segmentations. like: generate a real image of an object or something from this segmentation image, or generate a new real image of an object or something from these segmentations. The input to this tool should be a comma-separated string of two, representing the image_path and the user description.
14. Pose Detection On Image: Useful when you want to detect the human pose of the image. like: generate human poses of this image, or generate a pose image from this image. The input to this tool should be a string, representing the image_path.
15. Generate Image Condition On Pose Image: Useful when you want to generate a new real image from both the user description and a human pose image. like: generate a real image of a human from this human pose image, or generate a new real image of a human from this pose. The input to this tool should be a comma-separated string of two, representing the image_path and the user description.
16. Hed Detection On Image: Useful when you want to detect the soft hed boundary of the image. like: detect the soft hed boundary of this image, or hed boundary detection on the image, or perform hed boundary detection on this image, or detect soft hed boundary image of this image. The input to this tool should be a string, representing the image_path.
17. Generate Image Condition On Soft Hed Boundary Image: Useful when you want to generate a new real image from both the user description and a soft hed boundary image. like: generate a real image of an object or something from this soft hed boundary image, or generate a new real image of an object or something from this hed boundary. The input to this tool should be a comma-separated string of two, representing the image_path and the user description.
18. Predict Normal Map On Image: Useful when you want to detect the norm map of the image. like: generate a normal map from this image, or predict a normal map of this image. The input to this tool should be a string, representing the image_path.
19. Generate Image Condition On Normal Map: Useful when you want to generate a new real image from both the user description and normal map. like: generate a real image of an object or something from this normal map, or generate a new real image of an object or something from the normal map. The input to this tool should be a comma-separated string of two, representing the image_path and the user description.
20. Line Detection On Image: Useful when you want to detect the straight line of the image. like: detect the straight lines of this image, or straight line detection on the image, or perform straight line detection on this image, or detect the straight line image of this image. The input to this tool should be a string, representing the image_path.
21. Segment the given object: Useful when you only want to segment certain objects in the picture according to the given text. like: segment the cat, or can you segment an object for me. The input to this tool should be a comma-separated string of two, representing the image_path, the text description of the object to be found.
22. Remove Something From The Photo: Useful when you want to remove an object or something from the photo from its description or location. The input to this tool should be a comma-separated string of two, representing the image_path and the object that needs to be removed. 
23. Replace Something From The Photo: Useful when you want to replace an object from the object description or location with another object from its description. The input to this tool should be a comma-separated string of three, representing the image_path, the object to be replaced, and the object to be replaced with.
24. Crop Image: Useful when you want to crop a region or a part from the image. The input to this tool should be a comma-separated string of two, representing the image_path and the coordinates with the format of [x1, y1, x2, y2] that represents the top left and bottom right of the cropped region.
25. Text Detection On Image: Useful when you want to detect the text in the image. The input to this tool should be a string, representing the image_path.
26. Detection: Useful when you want to detect all objects of the image, but not detect a certain object according to the text. like: detect all the objects in this image, or detect this image. The input to this tool should be a string, representing the image_path.

Note that your generated visual instructions should be related to the image caption extremely. Please generate complex and deceptive instructions as much as possible.
Directly reply to me with the list."""


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


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption-path", required=True, help="path to caption file.")
    parser.add_argument("--instruction-path", required=True, help="path to instruction file.")
    parser.add_argument("--temperature", default=0.7, help="temperature for chatgpt.")
    args = parser.parse_args()

    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    print(INSTRUCTION_PROMPT.format(caption=""))
    
    caption_path = args.caption_path
    instruction_path = args.instruction_path
    temp = args.temperature
    annotations = json.load(open(caption_path, 'r'))
    instructions = []
    for idx, annot in enumerate(tqdm(annotations)):
        caption = annot['caption']
        messages = [{'role': 'system', 'content': INSTRUCTION_PROMPT.format(caption=caption)}]
        response = chatgpt(messages, temperature=temp)
        instructions.append({'file_name': annot['file_name'], 'caption': caption, 'instructions': response})
        if idx % 10 == 0:
            print(f'Dumping {idx}/{len(annotations)}')
            json.dump(instructions, open(instruction_path, 'w'))
    print("Done!")    
    