#!/bin/bash

# This script launches the sglang server with the necessary configuration
# for generating rollouts with custom logit processors.

python -m sglang.launch_server \
  --model-path google/gemma-3-1b-it \
  --host "localhost" \
  --port 30000 \
  --tp-size 2 \
  --enable-custom-logit-processor
