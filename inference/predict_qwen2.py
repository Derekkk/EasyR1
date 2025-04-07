import json
import re
from PIL import Image
import torch
from transformers import BitsAndBytesConfig
import requests
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from PIL import Image
import requests
import copy
import torch

import sys
import warnings
from instruction_generation import *
import re
import tqdm


device = 'cuda'

model_path = "/home/huzhe/workspace/model_card/Qwen2.5-VL-3B-Instruct"
# model_path = "../checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo_mcq/global_step_15_hf"

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

model.eval()
print(model)


# default processer
processor = AutoProcessor.from_pretrained(model_path)


def extract_reason_and_answer(text):
    reason_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    answer = answer.replace("<answer>", "").replace("</answer>", "").strip().strip("*")
    return {"reason": reason, "answer": answer}


def qwen2_inference(instruction, image_path):
    # SYSTEM_PROMPT="""You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST BE enclosed within <think> and </think> tags, and the final answer MUST BE enclosed within <answer> and </answer> tags."""
    # instruction = " ".join((SYSTEM_PROMPT.strip(), instruction))
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": instruction},
    #             {"type": "image", "image": image_path},
    #         ],
    #     }
    # ]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": image_path},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    print(inputs.keys())
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, required=False)
    parser.add_argument('--write_path', type=str, required=False)

    args = parser.parse_args()
    read_path = args.read_path
    write_path = args.write_path

    task = "mcq" # feedback, mcq_withnorm, trajectory, mcq_withtrajectory, mcq_oracle_norm, norm_entailment
    read_path = "../../data/viva/data_annotation_v2_proc.json"
    image_folder = "../../data/viva/images_v2_all/"
    write_path = f"./result_Qwen2-VL-7B-Instruct_{task}-imagebehind.json"

    print("write_path: ", write_path)

    data = json.load(open(read_path))

    data_pred = []
    for sample in tqdm.tqdm(data):
        instruction = formulate_instruction(sample, None, task)[0]
        image_path = image_folder + sample["image_file"]
        print(f"- prompt:\n{[instruction]}")
        output = qwen2_inference(instruction, image_path)
        print(f"- original output:\n{[output]}\n")
        if "</think>" in output and "<think>" in output:
            output = extract_reason_and_answer(output)
            print(f"- parsed output:\n{output}\n")
            sample["model_output"] = output["answer"]
            sample["reason"] = output["reason"]
        else:
            print(f"- output:\n{output}\n")
            sample["model_output"] = output
        cur_preds = [{"instruction": instruction, "prediction": output}]
        sample["result"] = cur_preds
        data_pred.append(sample)
    
    with open(write_path, "w") as f_w:
        json.dump(data_pred, f_w, indent=2)



if __name__ == "__main__":

    main()
    
