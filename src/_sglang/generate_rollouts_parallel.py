import argparse
import json
import yaml
import requests
import dill
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
from typing import List, Tuple, Dict, Any

from src._sglang.sglang_pipeline_processor import PipelineLogitsProcessor

from transformers import AutoTokenizer


# NOTE: seed can't be set here apparently - need to figure out something different for batched responses.
def prepare_and_send_request(session: requests.Session, endpoint: str, task: Tuple, tokenizer: AutoTokenizer, args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    prompt, system_prompt, pipeline_config = task
    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
    }
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    payload = { "text": full_prompt, "sampling_params": sampling_params }
    all_processor_configs = pipeline_config.get("processors", [])
    if all_processor_configs:
        pipeline_processor = PipelineLogitsProcessor(pipeline_config=all_processor_configs)
        dill_hex_string = dill.dumps(pipeline_processor).hex()
        processor_data = {
            "callable": dill_hex_string
        }
        payload["custom_logit_processor"] = json.dumps(processor_data)
    try:
        response = session.post(endpoint, json=payload, timeout=300)
        response.raise_for_status()
        response_json = response.json()
        generation_data = {
            "generation": response_json.get("text", ""),
            "sampler_info": { "name": pipeline_config.get("name"), "config": all_processor_configs }
        }
        return prompt, generation_data
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Request failed for prompt '{prompt[:50]}...'. Details: {e}")
        return prompt, None
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred for prompt '{prompt[:50]}...'. Details: {e}")
        return prompt, None

def generate_sglang_requests(tasks: List[Tuple], session: requests.Session, tokenizer: AutoTokenizer, args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    endpoint = f"http://{args.host}:{args.port}/generate"
    all_generations = defaultdict(list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = { executor.submit(prepare_and_send_request, session, endpoint, task, tokenizer, args): task for task in tasks }
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="Generating Candidates"):
            prompt, generation_data = future.result()
            if generation_data:
                all_generations[prompt].append(generation_data)
    return dict(all_generations)

def main():
    parser = argparse.ArgumentParser(
        description="Generate rollouts via a running SGLang server using a fully customizable sampling pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_path", type=str, default="google/gemma-3-1b-it", help="Model path to load the correct tokenizer for chat templating.")
    parser.add_argument("--max_workers", type=int, default=16, help="Number of parallel requests to send.")
    parser.add_argument("--host", type=str, default="localhost", help="Host of the SGLang server.")
    parser.add_argument("--port", type=int, default=30000, help="Port of the SGLang server.")
    parser.add_argument("--heuristics_config_path", type=str, required=True, help="Path to the YAML file containing sampling heuristics.")
    parser.add_argument("--source_data_path", type=str, required=True, help="Path to the source .jsonl file containing prompts.")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="The key in the source JSONL that contains the prompt text.")
    parser.add_argument("--output_path", type=str, default="./rollouts.jsonl", help="Path to save the output JSONL file.")
    parser.add_argument("--num_samples", type=int, default=None, help="Optional: Number of prompts to process from the source file.")
    parser.add_argument("--min_new_tokens", type=int, default=50, help="Minimum number of new tokens to generate.")
    parser.add_argument("--max_new_tokens", type=int, default=250, help="Maximum number of new tokens to generate.")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="The system prompt to use for the chat format.")
    args = parser.parse_args()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception as e:
        print(f"[ERROR] Could not load tokenizer for '{args.model_path}'. Details: {e}")
        return

    try:
        with open(args.source_data_path, 'r', encoding='utf-8') as f:
            prompts = [json.loads(line)[args.prompt_key] for line in f]
        if args.num_samples: prompts = prompts[:args.num_samples]
    except Exception as e:
        print(f"[ERROR] Failed to load prompts from '{args.source_data_path}': {e}")
        return

    try:
        with open(args.heuristics_config_path, 'r') as f:
            sampling_pipelines = yaml.safe_load(f).get("sampling_pipelines", [])
    except Exception as e:
        print(f"[ERROR] Failed to load heuristics from '{args.heuristics_config_path}': {e}")
        return

    tasks = [(prompt, args.system_prompt, pipeline) for prompt in prompts for i, pipeline in enumerate(sampling_pipelines)]

    session = requests.Session()
    all_generations = generate_sglang_requests(tasks, session, tokenizer, args)

    try:
        with open(args.output_path, 'w', encoding='utf-8') as f_out:
            for prompt, generations in all_generations.items():
                f_out.write(json.dumps({"prompt": prompt, "generations": generations}) + '\n')
    except IOError as e:
        print(f"[ERROR] Failed to write to output file '{args.output_path}': {e}")

    print(f"\nGeneration complete. Saved {len(all_generations)} prompts with rollouts to '{args.output_path}'.")

if __name__ == "__main__":
    main()
