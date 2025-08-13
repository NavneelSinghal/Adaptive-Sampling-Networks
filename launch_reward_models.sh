#!/bin/bash

REWARD_MODEL_PATH="Skywork/Skywork-Reward-V2-Llama-3.1-8B"
START_PORT=8000
NUM_GPUS=5

for i in $(seq 0 $((NUM_GPUS - 1)))
do
  GPU_ID=$((i + 2))
  PORT=$((START_PORT + i))
  
  echo "Launching Reward Model on GPU ${GPU_ID} at port ${PORT}"
  
  CUDA_VISIBLE_DEVICES=${GPU_ID} python -m sglang.launch_server \
    --model-path "${REWARD_MODEL_PATH}" \
    --host "0.0.0.0" \
    --port ${PORT} \
    --mem-fraction-static 0.8 \
    --is-embedding &
done

wait
