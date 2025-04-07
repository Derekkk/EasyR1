from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


MODEL_PATH = "./checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo_mcq/global_step_15_hf"

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10, "video": 10},
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=256,
    stop_token_ids=[],
)

SYSTEM_PROMPT="""You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST BE enclosed within <think> and </think> tags, and the final answer MUST BE enclosed within <answer> and </answer> tags."""

instruction = """
You are gievn a situation and a question. \nBased on the situation provided, select the most appropriate option to answer the question:\n\n## Situation: \nDepicted in the image.\n\n## Question:\nA. Pick up the items and throw them back into the car.\nB. Take note of the location and pick up the trash when it is safe to do so.\nC. Call emergency serviced to promply handle the situation.\nD. Politedly remind them not to do so.\nE. No action is necessary given the situation depicted in the image.\n\nNow answer the question. Just output the choice:
""".strip()
instruction = " ".join((SYSTEM_PROMPT.strip(), instruction))
image_messages = [
    # {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                # "image": None
                # "image":"https://i.pinimg.com/564x/e3/ee/45/e3ee453fd2d9311c0fba6b6dd228bc7c.jpg",
                # "min_pixels": 224 * 224,
                # "max_pixels": 1280 * 28 * 28,
            },
            {"type": "text", "text": instruction},
        ],
    },
]

# Here we use video messages as a demonstration
messages = image_messages

processor = AutoProcessor.from_pretrained(MODEL_PATH)
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print(prompt)
# image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
image_inputs = None
mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs


llm_inputs = {
    "prompt": prompt,
    "multi_modal_data": mm_data,
}
print(f"llm_inputs: {llm_inputs}")
outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text

print(generated_text)