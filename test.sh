python -m src.evaluate_with_sglang \
  --prompt "Write a short story about a programmer who discovers their cat can write Python code." \
  --model_name_or_path /mnt/data/llms/Llama-3.2-3B-Instruct \
  --sampler_model_name SamplingNetwork \
  --sampler_config_path configs/sampler_models/model_config_scale.yaml \
  --sampler_checkpoint_path ../outputs/output_sn_post_scale_gamma1/final_model/sampler_model.bin
