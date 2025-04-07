from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from PIL.Image import Image as ImageObject
from io import BytesIO
import math
import requests

# model_path = "/home/huzhe/workspace/model_card/Qwen2.5-VL-3B-Instruct"
model_path = "./checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo_mcq/global_step_15_hf"
# model_path = "./checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo/global_step_55_hf"

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
model.eval()

# default processer
processor = AutoProcessor.from_pretrained(model_path)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {"type": "text", "text": "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}. Describe this image."},
#         ],
#     }
# ]

class ImageProcessMixin:
    max_pixels: int= 1280 * 28 * 28
    min_pixels: int= 256 * 28 * 28

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image
    

SYSTEM_PROMPT="""You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST BE enclosed within <think> and </think> tags, and the final answer MUST BE enclosed within <answer> and </answer> tags."""

instruction = """
You are gievn a situation and a question. \nBased on the situation provided, select the most appropriate option to answer the question:\n\n## Situation: \nDepicted in the image.\n\n## Question:\nA. Pick up the items and throw them back into the car.\nB. Take note of the location and pick up the trash when it is safe to do so.\nC. Call emergency serviced to promply handle the situation.\nD. Politedly remind them not to do so.\nE. No action is necessary given the situation depicted in the image.\n\nNow answer the question. Just output the choice:
""".strip()

prompt_str = " ".join((SYSTEM_PROMPT.strip(), instruction))
# prompt_str = instruction

messages = [
    # {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_str},
            {'type': 'image'},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(text)

image_processor = ImageProcessMixin()

url = "https://i.pinimg.com/564x/e3/ee/45/e3ee453fd2d9311c0fba6b6dd228bc7c.jpg"
image = Image.open(requests.get(url, stream=True).raw)

images = [image_processor.process_image(image)]
# images = None

inputs = processor(images, [text], add_special_tokens=False, return_tensors="pt")

# image_inputs, video_inputs = process_vision_info(messages)
image_inputs, video_inputs = images, None
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
# print(inputs)
inputs = inputs.to("cuda")
# print(processor.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False))
# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=4096)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

