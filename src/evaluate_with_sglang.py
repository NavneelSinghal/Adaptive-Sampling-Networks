import argparse
import json
import dill
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from transformers import AutoTokenizer
from src._sglang.sglang_pipeline_processor import PipelineLogitsProcessor

def send_request(session, endpoint, payload, _):
    try:
        response = session.post(endpoint, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()['text']
    except requests.exceptions.RequestException as e:
        print(f"A request failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a prompt using a trained Adaptive Sampler via SGLang.")
    parser.add_argument("--num-requests", type=int, default=8, help="Total number of requests to send.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers to send requests.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to generate text from.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the base model for tokenizer.")
    parser.add_argument("--sampler_model_name", type=str, required=True, choices=["SamplingNetwork", "LocalProbabilityTransform", "SimpleDistributionAwareTransform"])
    parser.add_argument("--sampler_config_path", type=str, required=True, help="Path to the sampler model's config YAML.")
    parser.add_argument("--sampler_checkpoint_path", type=str, required=True, help="Path to the trained sampler model's .bin file.")
    parser.add_argument("--max_new_tokens", type=int, default=250)
    args = parser.parse_args()

    pipeline_config = [
        {
            "name": "adaptive_sampler",
            "params": {
                "sampler_model_name": args.sampler_model_name,
                "sampler_config_path": args.sampler_config_path,
                "sampler_checkpoint_path": args.sampler_checkpoint_path,
            }
        }
    ]

    processor = PipelineLogitsProcessor(pipeline_config=pipeline_config)
    dill_hex_string = dill.dumps(processor).hex()
    processor_data = { "callable": dill_hex_string }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    messages = [{"role": "user", "content": args.prompt}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    payload = {
        "text": full_prompt,
        "sampling_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": 1.0,
        },
        "custom_logit_processor": json.dumps(processor_data)
    }

    endpoint = f"http://{args.host}:{args.port}/generate"
    session = requests.Session()
    worker_fn = partial(send_request, session, endpoint, payload)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        output = list(executor.map(worker_fn, range(args.num_requests)))
    for x in output:
        print('=' * 100)
        print(x)

if __name__ == "__main__":
    main()
