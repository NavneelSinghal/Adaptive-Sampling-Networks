import argparse
import json
import yaml
import requests
from tqdm import tqdm
import concurrent.futures
from typing import Tuple, Dict, Any, Optional
from functools import partial

from transformers import AutoTokenizer

from src._sglang.generate_rollouts_parallel import prepare_and_send_request as send_sglang_request

def candidate_request_wrapper(
    task: Tuple, 
    session: requests.Session, 
    endpoint: str, 
    tokenizer: AutoTokenizer, 
    args: argparse.Namespace
) -> Optional[Dict[str, Any]]:
    """
    A wrapper function that:
    1. Takes the detailed task format for candidate generation.
    2. Translates it into the simpler format expected by the reused `send_sglang_request`.
    3. Calls the reused function to interact with the SGLang server.
    4. Reformats the output into the final, detailed record for our dataset.
    """
    source_record, prompt, system_prompt, pipeline_config, incidence_vector, seed = task
    
    sglang_task = (prompt, system_prompt, pipeline_config)
    
    original_prompt, generation_data = send_sglang_request(session, endpoint, sglang_task, tokenizer, args)

    if not generation_data:
        return None

    output_record = {
        **source_record,
        "system_prompt": system_prompt,
        "sampler_info": generation_data["sampler_info"],
        "generation": generation_data["generation"],
        "seed": seed,
        "incidence_vector": incidence_vector
    }
    return output_record


def main():
    parser = argparse.ArgumentParser(
        description="Generate candidate responses from multiple data sources via a running SGLang server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_path", type=str, default="google/gemma-3-1b-it", help="Model path for the tokenizer.")
    parser.add_argument("--dataset_sources_config_path", type=str, required=True, help="Path to the YAML config file defining dataset sources.")
    parser.add_argument("--heuristics_config_path", type=str, required=True, help="Path to the YAML file with sampling heuristics.")
    parser.add_argument("--output_path", type=str, default="./candidate_generations.jsonl", help="Path to save the output JSONL file.")
    parser.add_argument("--seed", type=int, default=42, help="A seed value to be stored with the generation record (for manual tracking).")
    parser.add_argument("--max_workers", type=int, default=16, help="Number of parallel requests to send.")
    parser.add_argument("--host", type=str, default="localhost", help="Host of the SGLang server.")
    parser.add_argument("--port", type=int, default=30000, help="Port of the SGLang server.")
    parser.add_argument("--min_new_tokens", type=int, default=50, help="Minimum number of new tokens to generate.")
    parser.add_argument("--max_new_tokens", type=int, default=250, help="Maximum number of new tokens to generate.")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="The system prompt.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    tasks = []
    print(f"Loading datasets from '{args.dataset_sources_config_path}'...")
    with open(args.dataset_sources_config_path, 'r') as f:
        dataset_sources = yaml.safe_load(f).get("datasets", [])

    for source in dataset_sources:
        try:
            with open(source['path'], 'r', encoding='utf-8') as f:
                for line in f:
                    source_record = json.loads(line)
                    prompt = source_record[source['prompt_key']]
                    tasks.append({
                        "source_record": source_record,
                        "prompt": prompt,
                        "incidence_vector": source['incidence_vector']
                    })
        except Exception as e:
            print(f"[ERROR] Failed to load prompts from '{source['path']}': {e}")
            continue
    
    print(f"Loading sampling heuristics from '{args.heuristics_config_path}'...")
    with open(args.heuristics_config_path, 'r') as f:
        sampling_pipelines = yaml.safe_load(f).get("sampling_pipelines", [])

    all_tasks_to_run = [
        (
            task['source_record'],
            task['prompt'], 
            args.system_prompt, 
            pipeline, 
            task['incidence_vector'], 
            args.seed
        )
        for pipeline in sampling_pipelines
        for task in tasks
    ]
    print(f"Created a total of {len(all_tasks_to_run)} generation tasks.")
    
    endpoint = f"http://{args.host}:{args.port}/generate"
    session = requests.Session()
    
    worker_fn = partial(
        candidate_request_wrapper, 
        session=session, 
        endpoint=endpoint, 
        tokenizer=tokenizer, 
        args=args
    )

    with open(args.output_path, 'w', encoding='utf-8') as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {executor.submit(worker_fn, task): task for task in all_tasks_to_run}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(all_tasks_to_run), desc="Generating Candidates"):
                result = future.result()
                if result:
                    f_out.write(json.dumps(result) + '\n')

    print(f"\nGeneration complete. Saved results to '{args.output_path}'.")

if __name__ == "__main__":
    main()
