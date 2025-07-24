import argparse
import json
import requests
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Dict, Generator, Any

def chunked_iterable(iterable: List[Any], size: int) -> Generator[List[Any], None, None]:
    """Yield successive chunks from a list."""
    it = iter(iterable)
    for first in it:
        yield [first] + list(islice(it, size - 1))

def score_and_update_batch(args: tuple) -> List[Dict]:
    """
    Worker function for the thread pool. It takes a batch of data, sends it to an
    SGLang server for scoring, and returns the updated batch with scores.
    """
    batch, model_path, urls, tokenizer, worker_id = args

    if not batch:
        return []

    url = urls[worker_id % len(urls)]

    try:
        conversations = [
            [
                {"role": "user", "content": item['prompt']},
                {"role": "assistant", "content": item['generation']}
            ]
            for item in batch
        ]
        
        formatted_text = [
            tokenizer.apply_chat_template(conv, tokenize=False)
            for conv in conversations
        ]
    except KeyError as e:
        print(f"\n[ERROR] Input data is missing expected key: {e}. Ensure '--prompt-key' and '--generation-key' are correct.", file=sys.stderr)
        return batch


    payload = {"model": model_path, "text": formatted_text}
    scores = [-999.0] * len(batch) # Default score in case of failure

    try:
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        results = response.json()
        
        scores = [res["embedding"][0] for res in results]
        
        if len(scores) != len(batch):
            print(f"\n[WARNING] Mismatch in response length. Expected {len(batch)}, got {len(scores)}. Some items may not be scored.", file=sys.stderr)
            scores.extend([-999.0] * (len(batch) - len(scores)))

    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Request to {url} failed. Details: {e}", file=sys.stderr)
    
    except (KeyError, IndexError) as e:
        print(f"\n[ERROR] Could not parse response from server. Error: {e}. Response: {response.text}", file=sys.stderr)

    for item, score in zip(batch, scores):
        item['bradley_terry_rating'] = score
        
    return batch

def main():
    """Main function to load, score, and save the data."""
    parser = argparse.ArgumentParser(
        description="Score generations using Skywork-Reward-V2 with an SGLang backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to write the output JSONL file.")
    parser.add_argument("--model-path", type=str, default="Skywork/Skywork-Reward-V2-Llama-3.1-8B", help="Name or path of the reward model.")
    parser.add_argument("--prompt-key", type=str, default="prompt", help="The key in the JSONL file for the prompt text.")
    parser.add_argument("--generation-key", type=str, default="generation", help="The key in the JSONL file for the generated text.")
    parser.add_argument("--sglang-base-url", type=str, default="http://127.0.0.1", help="Base URL for the SGLang servers.")
    parser.add_argument("--sglang-start-port", type=int, default=8000, help="Starting port number for SGLang servers.")
    parser.add_argument("--num-gpus", type=int, required=True, help="Number of SGLang server instances (and GPUs).")
    parser.add_argument("--num-workers", type=int, default=32, help="Number of parallel request threads.")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of items to process in a single API request.")
    
    args = parser.parse_args()

#    print("Loading dataset...")
    try:
        with open(args.input_path, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"[ERROR] Input file not found at: {args.input_path}", file=sys.stderr)
        sys.exit(1)
        
    total_items = len(all_data)
    print(f"   Found {total_items} items to score.")

#    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    urls = [f"{args.sglang_base_url}:{args.sglang_start_port + i}/classify" for i in range(args.num_gpus)]
#    print(f"   Targeting {len(urls)} SGLang server endpoints.")

    chunks = list(chunked_iterable(all_data, args.batch_size))
    worker_args = [
        (chunk, args.model_path, urls, tokenizer, i)
        for i, chunk in enumerate(chunks)
    ]
    
    all_scored_data = []
    
    print(f"Processing {total_items} items in {len(chunks)} batches using {args.num_workers} parallel threads...")
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        pbar = tqdm(executor.map(score_and_update_batch, worker_args), total=len(chunks), desc="Scoring batches")
        for scored_chunk in pbar:
            all_scored_data.extend(scored_chunk)

    print(f"Writing {len(all_scored_data)} scored datapoints to {args.output_path}...")
    with open(args.output_path, 'w', encoding='utf-8') as f_out:
        for item in all_scored_data:
            f_out.write(json.dumps(item) + '\n')

    print("Scoring complete.")

if __name__ == "__main__":
    main()
