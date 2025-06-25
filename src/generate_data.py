import argparse
import yaml
import json
import random
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sampling_heuristics import get_random_pipeline_for_generation

def main():
    parser = argparse.ArgumentParser(description="Generate data using an LLM from a source JSONL file.")
    parser.add_argument("--model_name_or_path", type=str, default="model", help="Path to the frozen base LLM.")
    parser.add_argument("--heuristics_config_path", type=str, required=True, help="Path to the heuristics config YAML file.")
    parser.add_argument("--source_data_path", type=str, required=True, help="Path to the source .jsonl file containing prompts.")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="The key in the source JSONL that contains the prompt text.")
    parser.add_argument("--output_path", type=str, default="./generated_data.jsonl", help="Path to save the output JSONL file.")
    parser.add_argument("--min_new_tokens", type=int, default=50, help="Minimum number of new tokens to generate for a response.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum number of new tokens to generate for a response.")
    parser.add_argument("--num_samples", type=int, help="Optional: Number of samples to generate. If not set, processes all prompts in the source file.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of prompts to process in a single batch.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.heuristics_config_path, 'r') as f:
        heuristics_config = yaml.safe_load(f)

    prompts = []
    try:
        with open(args.source_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if args.prompt_key in data and data[args.prompt_key]:
                        prompts.append(data[args.prompt_key])
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Source data file not found at '{args.source_data_path}'")
        exit(1)

    if not prompts:
        print(f"Error: No valid prompts found in '{args.source_data_path}' with key '{args.prompt_key}'.")
        exit(1)
        
    if args.num_samples and args.num_samples < len(prompts):
        prompts_to_process = random.sample(prompts, args.num_samples)
    else:
        prompts_to_process = prompts
        
    print(f"Loading model and tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype="auto", device_map="auto")
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Found {len(prompts)} prompts. Will process {len(prompts_to_process)} samples in batches of {args.batch_size}...")

    with open(args.output_path, 'w', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(prompts_to_process), args.batch_size), desc="Generating Batches"):
            batch_prompts = prompts_to_process[i:i + args.batch_size]
            logits_processors, sampler_info = get_random_pipeline_for_generation(heuristics_config)
            chats = [[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': p}] for p in batch_prompts]
            inputs = tokenizer.apply_chat_template(chats, add_generation_prompt=True, return_tensors="pt", padding=True, return_dict=True).to(model.device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    min_length=inputs['input_ids'].shape[1] + args.min_new_tokens,
                    max_length=inputs['input_ids'].shape[1] + args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=logits_processors
                )

            input_ids_lengths = [len(x) for x in inputs['input_ids']]
            for j, generated_sequence in enumerate(generated_ids):
                prompt_length = input_ids_lengths[j]
                generation_only_ids = generated_sequence[prompt_length:]
                generated_text = tokenizer.decode(generation_only_ids, skip_special_tokens=True)
                output_record = {
                    "prompt": batch_prompts[j],
                    "generation": generated_text,
                    "sampler_info": sampler_info
                }
                f_out.write(json.dumps(output_record) + '\n')
            del inputs
            del input_ids
            del attention_mask
            del generated_ids
            del input_ids_lengths
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nGeneration complete. Saved {len(prompts_to_process)} samples to '{args.output_path}'.")

if __name__ == "__main__":
    main()
