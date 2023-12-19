GPT4Tools can support 22 tools, including:

| No. |        Tools Name       |                                    Function                                    |
|-----|:-----------------------:|:------------------------------------------------------------------------------:|
|  1  |     InstructPix2Pix     | Style the image to be like the text.                                           |
|  2  |        Text2Image       | Generate an image from an input text.                                          |
|  3  |     ImageCaptioning     | Describe the input image.                                                      |
|  4  |       Image2Canny       | Detect the edge of the image                                                   |
|  5  |     CannyText2Image     | Generate a new real image from both the user description and a canny image.    |
|  6  |        Image2Line       | Detect the straight line of the image.                                         |
|  7  |        Image2Hed        | Detect the soft hed boundary of the image.                                     |
|  8  |      HedText2Image      | Generate a new real image from both the user description.                      |
|  9  |      Image2Scribble     | Generate a scribble of the image.                                              |
|  10 |    ScribbleText2Image   | Generate a new real image from both the user description and a scribble image. |
|  11 |        Image2Pose       | Detect the human pose of the image.                                            |
|  12 |      PoseText2Image     | Generate a new real image from both the user description.                      |
|  13 |      SegText2Image      | Generate a new real image from both the user description and segmentations.    |
|  14 |       Image2Depth       | Detect depth of the image.                                                     |
|  15 |     DepthText2Image     | Generate a new real image from both the user description and depth image.      |
|  16 |       Image2Normal      | Detect norm map of the image.                                                  |
|  17 |     NormalText2Image    | Generate a new real image from both the user description and normal map.       |
|  18 | VisualQuestionAnswering | Answer for a question based on an image.                                       |
|  19 |        Segmenting       | Segment all the part of the image.                                             |
|  20 |         Text2Box        | Detect or find out given objects in the picture.                               |
|  21 |     ObjectSegmenting    | Segment the certain objects in the picture.                                    |
|  22 |       ImageEditing      | Remove and object or something from the photo.                                 |

You can customize the used tools by specifying ```{tools_name}_{devices}``` after args ```--load``` of ```gpt4tools.py```. For example, enabling ```Text2Box```, ```Segmenting```, and ```ImageCaptioning```:
```
python gpt4tools_demo.py \
	--base_model $path_to_vicuna_with_tokenizer \
	--lora_model $path_to_lora_weights \
	--llm_device "cpu" \
	--load "Text2Box_cuda:0,Segmenting_cuda:0,ImageCaptioning_cuda:0" \
	--cache-dir $your_cache_dir \
	--share
```
More tools will be supported in the future!