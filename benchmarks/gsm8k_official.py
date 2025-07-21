import argparse
import ast
import json
import os
import re
import time
import dill
import requests
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from tqdm import tqdm

from sglang.utils import download_and_cache_file, read_jsonl

try:
    from src._sglang.sglang_pipeline_processor import PipelineLogitsProcessor
except ImportError:
    print("="*80)
    print("[ERROR] Could not import 'PipelineLogitsProcessor'.")
    print("Please ensure that the class is available in your Python path, for example, at")
    print("'src/_sglang/sglang_pipeline_processor.py' and that the directory is importable.")
    print("="*80)
    exit(1)


INVALID = -9999999


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def send_request(prompt, session, endpoint, payload_template):
    """Sends a single generation request to the SGLang server."""
    payload = payload_template.copy()
    payload["text"] = prompt
    try:
        response = session.post(endpoint, json=payload, timeout=300)
        response.raise_for_status()
        # The generated text is in the 'text' field of the response
        return response.json().get("text", "")
    except requests.exceptions.RequestException as e:
        print(f"A request failed: {e}")
        return "" # Return empty string on failure


def main(args):
    data_path = args.data_path
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    if not os.path.isfile(data_path):
        print(f"Data file not found. Downloading from {url}...")
        data_path = download_and_cache_file(url)
    lines = list(read_jsonl(data_path))
    num_questions = args.num_questions if args.num_questions > 0 else len(lines)
    num_shots = args.num_shots
    few_shot_prefix = get_few_shot_examples(lines, num_shots)

    prompts = []
    labels = []
    for i in range(num_questions):
        prompts.append(few_shot_prefix + get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    pipeline_config = [
        {
            "name": "temperature",
            "params": {
                "temperature": 0.6
            }
        },
        {
            "name": "top_p",
            "params": {
                "top_p": 0.9
            }
        }
    ]

    processor = PipelineLogitsProcessor(pipeline_config=pipeline_config)
    dill_hex_string = dill.dumps(processor).hex()
    processor_data = {"callable": dill_hex_string}
    payload_template = {
        "sampling_params": {
            "max_new_tokens": 2048,
            "temperature": 1.0,
            "stop": ["Question:", "Assistant:", "<|separator|>"],
        },
        "custom_logit_processor": json.dumps(processor_data)
    }
    endpoint = f"http://{args.host}:{args.port}/generate"
    session = requests.Session()
    worker_fn = partial(send_request, session=session, endpoint=endpoint, payload_template=payload_template)

    tic = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        results = list(tqdm(executor.map(worker_fn, prompts), total=len(prompts), desc="Running GSM8K Eval"))
    latency = time.perf_counter() - tic
    preds = [get_answer_value(res) for res in results]
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Invalid generations (could not parse answer): {invalid:.4f}")
    print(f"Total time for {len(prompts)} requests: {latency:.3f} s")
    print(f"Throughput: {len(prompts) / latency:.3f} requests/s")
    if args.result_file:
        with open(args.result_file, "a") as fout:
            value = {
                "task": "gsm8k_custom_sampler",
                "model": "",
                "num_gpus": 1,
                "latency": round(latency, 3),
                "accuracy": round(acc, 4),
                "invalid": round(invalid, 4),
                "num_requests": num_questions,
                "other": {
                    "parallel": args.parallel,
                    "num_shots": num_shots,
                },
            }
            fout.write(json.dumps(value) + "\n")
        print(f"Results appended to {args.result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-shots", type=int, default=5, help="Number of few-shot examples to include in the prompt.")
    parser.add_argument("--data-path", type=str, default="test.jsonl", help="Path to GSM8K test.jsonl file.")
    parser.add_argument("--num-questions", type=int, default=200, help="Number of questions to evaluate. Set to -1 to use all questions.")
    parser.add_argument("--parallel", type=int, default=16, help="Number of parallel requests to send.")
    parser.add_argument("--result-file", type=str, default="gsm8k_results.jsonl", help="File to append results.")
    
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    
    args = parser.parse_args()
    main(args)
