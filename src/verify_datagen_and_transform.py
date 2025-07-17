import argparse
import json
import yaml
from collections import defaultdict, Counter
from tqdm import tqdm

def verify_and_filter_dataset(input_path: str, output_path: str, heuristics_config_path: str, expected_seeds: int):
    """
    Reads a JSONL dataset, verifies data integrity based on uniqueness and expected counts,
    and filters out any prompts that do not meet the criteria.

    A prompt is discarded if:
    1. Any (seed, sampler_info) combination appears more than once.
    2. The number of unique seeds does not match `expected_seeds`.
    3. The number of unique samplers does not match the count in the heuristics config.

    Args:
        input_path (str): Path to the source .jsonl file.
        output_path (str): Path to write the cleaned .jsonl file.
        heuristics_config_path (str): Path to the heuristics YAML file used for generation.
        expected_seeds (int): The exact number of unique seeds expected for each prompt.
    """
    try:
        with open(heuristics_config_path, 'r', encoding='utf-8') as f:
            heuristics_config = yaml.safe_load(f)
            expected_samplers = len(heuristics_config.get("sampling_pipelines", []))
            print(f"Loaded heuristics from '{heuristics_config_path}'. Expecting {expected_samplers} unique samplers per prompt.")
    except (IOError, yaml.YAMLError) as e:
        print(f"Error: Could not read or parse heuristics config file: {e}")
        return

    print(f"Reading and grouping data from '{input_path}'...")
    data_by_prompt = defaultdict(list)
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                data_by_prompt[data['prompt']].append(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping malformed line or missing key: {e}")
                continue
    
    print(f"Found {len(data_by_prompt)} unique prompts.")
    
    prompts_to_discard = {}
    
    print("Verifying data integrity for each prompt...")
    for prompt, generations in tqdm(data_by_prompt.items(), desc="Verifying Prompts"):
        key_counts = Counter()
        unique_seeds = set()
        unique_samplers = set()
        for gen in generations:
            sampler_key = json.dumps(gen.get('sampler_info', {}), sort_keys=True)
            seed = gen.get('seed')
            
            if seed is None:
                prompts_to_discard[prompt] = "Entry with missing 'seed' found."
                break
            
            unique_seeds.add(seed)
            unique_samplers.add(sampler_key)
            unique_tuple = (seed, sampler_key)
            key_counts[unique_tuple] += 1
        if prompt in prompts_to_discard:
            continue

        for key, count in key_counts.items():
            if count > 1:
                prompts_to_discard[prompt] = f"Duplicate entry found: {count} entries for seed={key[0]} and sampler={key[1]}"
                break
        if prompt in prompts_to_discard:
            continue
        if len(unique_seeds) != expected_seeds:
            prompts_to_discard[prompt] = f"Seed count mismatch. Expected {expected_seeds}, found {len(unique_seeds)}."
            continue
        if len(unique_samplers) != expected_samplers:
            prompts_to_discard[prompt] = f"Sampler count mismatch. Expected {expected_samplers}, found {len(unique_samplers)}."
            continue

    if prompts_to_discard:
        print(f"\n[FLAG] Found {len(prompts_to_discard)} prompts to be discarded for the following reasons:")
        MAX_PRINTS
        for i, (prompt, reason) in enumerate(prompts_to_discard.items()):
            if i < MAX_PRINTS:
                 print(f"  - Prompt '{prompt[:80]}...': {reason}")
            elif i == MAX_PRINTS:
                 print(f"  ... and {len(prompts_to_discard) - 5} more.")
    else:
        print("\nData integrity check passed. All prompts are clean.")

    print(f"\nWriting clean data to '{output_path}'...")
    
    final_prompts_written = 0
    final_lines_written = 0
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for prompt, generations in data_by_prompt.items():
            if prompt not in prompts_to_discard:
                final_prompts_written += 1
                for gen in generations:
                    f_out.write(json.dumps(gen) + '\n')
                    final_lines_written += 1
                    
    print(f"Total prompts read: {len(data_by_prompt)}")
    print(f"Prompts discarded due to integrity issues: {len(prompts_to_discard)}")
    print(f"Clean prompts written: {final_prompts_written}")
    print(f"Total lines written: {final_lines_written}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify and filter a generated dataset to ensure data integrity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the input .jsonl file generated by the pipeline."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the cleaned and verified .jsonl file."
    )
    parser.add_argument(
        "--heuristics-config-path",
        type=str,
        required=True,
        help="Path to the heuristics YAML file to determine the expected number of samplers."
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        required=True,
        help="The exact number of unique seeds that is expected for each prompt."
    )
    args = parser.parse_args()
    verify_and_filter_dataset(args.input_path, args.output_path, args.heuristics_config_path, args.expected_seeds)
