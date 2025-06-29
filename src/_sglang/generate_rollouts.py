import argparse
import json
import yaml
import requests
from tqdm import tqdm
from collections import defaultdict

from sglang_custom_processors import (
    TypicalLogitsProcessor,
    MinPLogitsProcessor,
    EpsilonLogitsProcessor,
    EtaLogitsProcessor,
)

CUSTOM_PROCESSOR_MAP = {
    "typical": TypicalLogitsProcessor,
    "min_p": MinPLogitsProcessor, ##this might not be necessary as sglang has a native implementation of minp already
    "epsilon": EpsilonLogitsProcessor,
    "eta": EtaLogitsProcessor,
}

def main():
    """
    Client script to generate rollouts by sending requests to a running sglang server's
    /generate endpoint.
    """
    parser = argparse.ArgumentParser(description="Generate rollouts via a running sglang server.")
    parser.add_argument("--host", type=str, default="localhost", help="Host of the sglang server.")
    parser.add_argument("--port", type=int, default=30000, help="Port of the sglang server.")
    parser.add_argument("--heuristics_config_path", type=str, required=True, help="Path to the YAML file containing sampling heuristics.")
    parser.add_argument("--source_data_path", type=str, required=True, help="Path to the source .jsonl file containing prompts.")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="The key in the source JSONL that contains the prompt text.")
    parser.add_argument("--output_path", type=str, default="./rollouts.jsonl", help="Path to save the output JSONL file.")
    parser.add_argument("--num_samples", type=int, help="Optional: Number of prompts to process.")
    parser.add_argument("--min_new_tokens", type=int, default=50, help="Minimum number of new tokens to generate.")
    parser.add_argument("--max_new_tokens", type=int, default=250, help="Maximum number of new tokens to generate.")

    args = parser.parse_args()
    server_url = f"http://{args.host}:{args.port}"
    endpoint = f"{server_url}/generate" 
    prompts = []
    with open(args.source_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                prompts.append(json.loads(line)[args.prompt_key])
            except (json.JSONDecodeError, KeyError):
                continue
    if args.num_samples:
        prompts = prompts[:args.num_samples]

    print(f"Loaded {len(prompts)} prompts. Will send requests to {endpoint}")

    with open(args.heuristics_config_path, 'r') as f:
        heuristics_config = yaml.safe_load(f)
    sampling_pipelines = heuristics_config.get("sampling_pipelines", [])
    
    all_generations = defaultdict(list)
    session = requests.Session()

    with tqdm(total=len(prompts) * len(sampling_pipelines), desc="Generating Rollouts") as pbar:
        for prompt in prompts:
            for pipeline in sampling_pipelines:
                processors = pipeline.get("processors", [])
                
                payload = {
                    "text": f"You are a helpful assistant.\nUSER: {prompt}\nASSISTANT:",
                    "sampling_params": {
                        "max_new_tokens": args.max_new_tokens,
                        "min_new_tokens": args.min_new_tokens,
                    }
                }
                
                for p in processors:
                    if p['name'] in CUSTOM_PROCESSOR_MAP:
                        processor_class = CUSTOM_PROCESSOR_MAP[p['name']]
                        payload["custom_logit_processor"] = processor_class().to_str()
                        payload["sampling_params"]["custom_params"] = p.get("params", {})
                    else:
                        payload["sampling_params"].update(p.get("params", {}))

                if 'temperature' not in payload["sampling_params"]:
                    payload["sampling_params"]['temperature'] = 1.0

                try:
                    response = session.post(endpoint, json=payload)
                    response.raise_for_status()
                    generation = response.json()["text"]
                    
                    generation_data = {
                        "generation": generation,
                        "sampler_info": { "name": pipeline.get("name"), "config": processors }
                    }
                    all_generations[prompt].append(generation_data)
                except requests.exceptions.RequestException as e:
                    print(f"\nError processing prompt: '{prompt[:50]}...'. Error: {e}")
                
                pbar.update(1)

    with open(args.output_path, 'w', encoding='utf-8') as f_out:
        for prompt, generations in all_generations.items():
            output_record = {"prompt": prompt, "generations": generations}
            f_out.write(json.dumps(output_record) + '\n')

    print(f"\nGeneration complete. Saved rollouts to '{args.output_path}'.")

if __name__ == "__main__":
    main()
