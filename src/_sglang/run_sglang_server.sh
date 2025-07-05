#!/bin/bash

# This script launches the sglang server with the necessary configuration
# for generating rollouts with custom logit processors.

python -m sglang.launch_server \
  --model-path /mnt/data/llms/Gemma3-4B-it \
  --host "localhost" \
  --port 30000 \
  --tp-size 2 \
  --enable-custom-logit-processor \
  --disable-custom-all-reduce \
  --mem-fraction-static 0.3 \
  --random-seed 0
