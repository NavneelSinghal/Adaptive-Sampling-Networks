#!/bin/bash
for i in $(seq 0 7)
do
  export CUDA_VISIBLE_DEVICES=$i
  PORT=$((30000 + i))
  
  python -m sglang.launch_server \
    --model-path /path/to/model \
    --host "0.0.0.0" \
    --port $PORT \
    --mem-fraction-static 0.9 \
    --dtype bfloat16 \
    --enable-custom-logit-processor \
    --random-seed 0 &
done
