python src/train_supervised.py \
    --model_name_or_path <path-to-model> \
    --sampler_model_name SamplingNetwork \
    --sampler_config_path config/sn_config.yaml \
    --data_path <path-to-data.jsonl> \
    --output_dir <output-directory> \
    --max_seq_length 1000 \
    --token_batch_size 12 \
    --learning_rate 1e-2 \
    --num_epochs 1 \
    --save_steps 50 \
    --loss_gamma 1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 100 \
    --gradient_accumulation_steps 16 \
    > training.out
