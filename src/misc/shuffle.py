import json
import collections
import random
import math
import argparse
import os
def group_shuffle_split(data_list: list[dict], num_splits: int) -> list[list[list[dict]]]:
    if not data_list:
        return []
    prompt_map = collections.defaultdict(list)
    for item in data_list:
        try:
            prompt_map[item['prompt']].append(item)
        except KeyError:
            print("Warning: Skipping an item because it lacks a 'prompt' key.")
            continue
    prompt_groups = list(prompt_map.values())
    random.shuffle(prompt_groups)
    num_groups = len(prompt_groups)
    if num_groups == 0 or num_splits <= 0:
        return []
    split_size = math.ceil(num_groups / num_splits)
    
    final_splits = [
        prompt_groups[i:i + split_size]
        for i in range(0, num_groups, split_size)
    ]
    return final_splits

def main():
    parser = argparse.ArgumentParser(
        description="Group data from a JSONL file by prompt, shuffle the groups, "
                    "and write them into a specified number of split files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input .jsonl file."
    )
    parser.add_argument(
        "output_prefix",
        type=str,
        help="Path and prefix for the output files (e.g., 'output/split').\n"
             "Files will be saved as <prefix>_1.jsonl, <prefix>_2.jsonl, etc."
    )
    parser.add_argument(
        "-n", "--num_splits",
        type=int,
        default=10,
        help="The number of output files to split the data into (default: 10)."
    )
    args = parser.parse_args()
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found at {args.input_file}")
        return
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line) for line in f]
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading or parsing {args.input_file}: {e}")
        return
    splits = group_shuffle_split(all_data, args.num_splits)
    if not splits:
        print("No data was processed, either due to empty input or missing 'prompt' keys.")
        return
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    print(f"Processing complete. Writing {len(splits)} split(s) to files...\n")
    for i, split_data in enumerate(splits):
        output_path = f"{args.output_prefix}_{i+1}.jsonl"
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                items_written = 0
                flattened_split = [item for group in split_data for item in group]
                for item in flattened_split:
                    f.write(json.dumps(item) + '\n')
                    items_written += 1
                print(f"Wrote {items_written} flattened items to {output_path}")
        except IOError as e:
            print(f"Error writing to file {output_path}: {e}")
if __name__ == "__main__":
    main()
