set -x

# MODEL_PATH=/home/huzhe/workspace/model_card/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
MODEL_PATH=./checkpoints/easy_r1/qwen2_5_vl_3b_GEOQA_8K_R1V/qwen2_5_vl_3b_GEOQA_8K_step70_hf

# SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
#  The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

SYSTEM_PROMPT="""You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST BE enclosed within <think> and </think> tags, and the final answer MUST BE enclosed within <answer> and </answer> tags."""

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/home/huzhe/workspace/data/comparative_reasoning/humor_comparative_combined@train \
    data.val_files=/home/huzhe/workspace/data/comparative_reasoning/humor_comparative_combined@test \
    data.image_key=images \
    ata.prompt_key=question \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_3b_GEOQA_8K_humor_comparative \
    worker.reward.compute_score=mcq \
    trainer.n_gpus_per_node=1
