import argparse
import yaml
import os
import json
from tqdm import tqdm
import torch
torch.set_float32_matmul_precision('high')
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.optim import AdamW
import random
import gc

from models import SamplingNetwork, LocalProbabilityTransform, SimpleDistributionAwareTransform
from sampling_heuristics import apply_pipeline_to_logits
from loss import TruncatedLogitsLoss

SAMPLER_MODELS = {
    "SamplingNetwork": SamplingNetwork,
    "LocalProbabilityTransform": LocalProbabilityTransform,
    "SimpleDistributionAwareTransform": SimpleDistributionAwareTransform,
}

random.seed(42)

class JsonlDataset(Dataset):
    def __init__(self, file_path):
        self.samples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.samples.append(json.loads(line))
            random.shuffle(self.samples)
        except FileNotFoundError:
            print(f"Error: Data file not found at '{file_path}'.")
            exit(1)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def setup_models_and_tokenizer(args, device, device2):
    print("--- Setting up models and tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading frozen generation model: {args.model_name_or_path}")
    generation_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    generation_model.eval()
    for param in generation_model.parameters():
        param.requires_grad = False
    with open(args.sampler_config_path, 'r') as f:
        sampler_config = yaml.safe_load(f)
    sampler_model_class = SAMPLER_MODELS[args.sampler_model_name]
    sampler_model = sampler_model_class(dtype=torch.bfloat16, **sampler_config).to(device2)
    sampler_model = torch.compile(sampler_model)
    print(f"Loaded sampler model: {args.sampler_model_name}")
    print(f"Total trainable parameters in sampler: {sum(p.numel() for p in sampler_model.parameters() if p.requires_grad):,}")
    return tokenizer, generation_model, sampler_model

def train(args, generation_model, sampler_model, tokenizer, device, device2):
    optimizer = AdamW(sampler_model.parameters(), lr=args.learning_rate)
    loss_fn = TruncatedLogitsLoss(gamma=args.loss_gamma, reduction='mean')

    dataset = JsonlDataset(args.data_path)
    print(f"Loaded {len(dataset)} generation samples from {args.data_path}.")

    total_training_tokens = 0
    for sample in tqdm(dataset, desc="Preprocessing for scheduler"):
        prompt_messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': sample['prompt']}]
        assistant_messages = [{'role': 'assistant', 'content': sample['generation']}]
        prompt_inputs = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, return_tensors='pt', truncation=True, max_length=args.max_seq_length)
        all_inputs = tokenizer.apply_chat_template(prompt_messages + assistant_messages, add_generation_prompt=False, return_tensors='pt', truncation=True, max_length=args.max_seq_length)
        start_index = prompt_inputs.shape[1]
        num_target_tokens = all_inputs.shape[1] - start_index - 1
        if num_target_tokens > 0:
            total_training_tokens += num_target_tokens

    num_update_steps_per_epoch = (total_training_tokens // args.token_batch_size)
    num_training_steps = (num_update_steps_per_epoch // args.gradient_accumulation_steps) * args.num_epochs
    
    if num_training_steps == 0:
        print("\nWarning: Calculated training steps = 0. The LR scheduler will not step.")
        num_training_steps = 1

    print(f"\nTotal tokens per epoch: {total_training_tokens}")
    print(f"Token batch size: {args.token_batch_size}")
    print(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
    print(f"Effective token batch size: {args.token_batch_size * args.gradient_accumulation_steps}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Calculated total optimizer steps for scheduler: {num_training_steps}")

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps
    )
    print(f"Initialized '{args.lr_scheduler_type}' LR scheduler with {args.num_warmup_steps} warmup steps.")

    token_buffer = []
    global_step = 0
    effective_step = 0
    accumulated_loss = 0.0
    accumulated_kl_loss = 0.0
    accumulated_truncation_penalty = 0.0
    sampler_model.train()
    optimizer.zero_grad()

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}\n")
        pbar = tqdm(dataset, desc=f"Epoch {epoch+1} Processing")
        for sample in pbar:
            prompt_messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': sample['prompt']}]
            assistant_messages = [{'role': 'assistant', 'content': sample['generation']}]
            prompt_inputs = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, return_tensors='pt', truncation=True, max_length=args.max_seq_length).to(device)
            all_inputs = tokenizer.apply_chat_template(prompt_messages + assistant_messages, add_generation_prompt=False, return_tensors='pt', truncation=True, max_length=args.max_seq_length).to(device)
            start_index = prompt_inputs.shape[1]

            with torch.no_grad():
                outputs = generation_model(all_inputs)
                raw_logits_sequence = outputs.logits[:, start_index:-1, :].clone()
                del outputs

            seq_len = raw_logits_sequence.shape[1]
            if seq_len == 0:
                continue
            raw_logits_sequence = raw_logits_sequence.squeeze(0)

            pipeline_config = sample['sampler_info']['config']

            target_logits_list = []
            with torch.no_grad():
                for i in range(seq_len):
                    current_raw_logits = raw_logits_sequence[i].unsqueeze(0)
                    target_logits = apply_pipeline_to_logits(
                        input_ids=None,
                        logits=current_raw_logits,
                        pipeline_config=pipeline_config
                    )
                    target_logits_list.append(target_logits)

            for i in range(seq_len):
                token_buffer.append((raw_logits_sequence[i], target_logits_list[i]))

            while len(token_buffer) >= args.token_batch_size:
                batch_data = token_buffer[:args.token_batch_size]
                token_buffer = token_buffer[args.token_batch_size:]

                raw_logits_batch, target_logits_batch = zip(*batch_data)
                raw_logits_tensor = torch.stack(raw_logits_batch).to(device2)
                target_logits_tensor = torch.stack(target_logits_batch).squeeze(1).to(device2)

                predicted_logits = sampler_model(raw_logits_tensor)
                loss, kl_loss, truncation_penalty = loss_fn(predicted_logits, target_logits_tensor)
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    kl_loss = kl_loss / args.gradient_accumulation_steps
                    truncation_penalty = truncation_penalty / args.gradient_accumulation_steps

                accumulated_loss += loss.item()
                accumulated_kl_loss += kl_loss.item()
                accumulated_truncation_penalty += truncation_penalty.item()
                
                loss.backward()
                global_step += 1

                if global_step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    effective_step += 1

                    if effective_step % 50 == 0 or effective_step in [1, 2, 4, 8, 16, 32]:
                        denom = 50
                        if effective_step == 50:
                            denom = 50 - 32
                        elif effective_step == 1:
                            denom = 1
                        elif effective_step < 50:
                            denom = effective_step // 2
                        else:
                            denom = 50
                        avg_loss = accumulated_loss / denom
                        avg_kl = accumulated_kl_loss / denom
                        avg_penalty = accumulated_truncation_penalty / denom
                        current_lr = lr_scheduler.get_last_lr()[0]
                        pbar.set_postfix(last_effective_step_logged=f"{effective_step}", avg_loss=f"{avg_loss:.4f}", avg_kl_loss=f"{avg_kl:.4f}", avg_penalty=f"{avg_penalty:.4f}", lr=f"{current_lr:.2e}")
                        accumulated_loss = 0.0
                        accumulated_kl_loss = 0.0
                        accumulated_truncation_penalty = 0.0

                    if effective_step > 0 and effective_step % args.save_steps == 0:
                        output_path = os.path.join(args.output_dir, f"checkpoint-{effective_step}")
                        os.makedirs(output_path, exist_ok=True)
                        torch.save(sampler_model.state_dict(), os.path.join(output_path, "sampler_model.bin"))
                        pbar.write(f"\nSaved checkpoint to {output_path}\n")
                        pbar.refresh()

            try:
                del batch_data, raw_logits_batch, target_logits_batch, raw_logits_tensor, target_logits_tensor, predicted_logits, loss
            except Exception:
                pass
            try:
                del current_raw_logits, target_logits, target_logits_list
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()

    if global_step % args.gradient_accumulation_steps != 0:
        print("\nPerforming final optimizer step for remaining gradients...")
        optimizer.step()
        optimizer.zero_grad()

    print("\n--- Training Complete ---")
    final_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    torch.save(sampler_model.state_dict(), os.path.join(final_path, "sampler_model.bin"))
    print(f"Saved final model to {final_path}")

def main():
    parser = argparse.ArgumentParser(description="Supervised Pre-training for Adaptive Samplers.")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2", help="Path to the frozen base LLM.")
    parser.add_argument("--sampler_model_name", type=str, required=True, choices=SAMPLER_MODELS.keys())
    parser.add_argument("--sampler_config_path", type=str, required=True, help="Path to the sampler model's config YAML.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the pre-generated .jsonl training data.")
    parser.add_argument("--output_dir", type=str, default="./sampler_checkpoints")
    parser.add_argument("--max_seq_length", type=int, default=1000, help="Max sequence length to process from generations.")
    parser.add_argument("--token_batch_size", type=int, default=8, help="Batch size in terms of individual tokens.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of passes over the entire generation dataset.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save a checkpoint every N optimizer steps.")
    parser.add_argument("--loss_gamma", type=float, default=5.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=100, help="Number of steps for the linear warmup phase.")

    args = parser.parse_args()
    device = torch.device("cuda:1")
    device2 = torch.device("cuda:0")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer, generation_model, sampler_model = setup_models_and_tokenizer(args, device, device2)
    train(args, generation_model, sampler_model, tokenizer, device, device2)

if __name__ == "__main__":
    main()
