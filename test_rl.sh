python -m src.train_rl \
    --model_name_or_path "unsloth/Llama-3.2-3B-Instruct" \
    --sampler_model_name "SamplingNetwork" \
    --sampler_config_path "configs/sampler_models/model_config_scale.yaml" \
    --sampler_checkpoint_path "path/to/sampler/checkpoint" \
    --prompt_dataset_path "data/test_prompts.jsonl" \
    --output_dir "./rl_checkpoints_8xH100" \
    --rl_config_path "configs/rl_training/rl_config.yaml" \
    --reward_model_path "Skywork/Skywork-Reward-V2-Llama-3.1-8B" \
    --sglang_host "localhost" \
    --sglang_port 30000 \
    --reward_model_host "localhost" \
    --reward_model_port 8000 \
    --reward_model_gpus 5 \
    --device_gen "cuda:7" \
    --device_sampler "cuda:0" \
    --device_embedding "cuda:0"
# 1-6 is used for loading the generation model (for rollouts) and 5 instances of the reward model
