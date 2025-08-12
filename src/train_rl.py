import argparse
import yaml
import os
import json
import random
import gc
import copy
import requests
import dill
import time
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.optim import AdamW

from src.models import SamplingNetwork, LocalProbabilityTransform, SimpleDistributionAwareTransform
from src.rl_env.direct_ratings import compute_unnormalized_direct_ratings
from src.rl_env.diversity import compute_unnormalized_diversity_reward_embedding_entropy
from src.rl_env.verifiable_rewards import compute_unnormalized_verifiable_reward
from src._sglang.sglang_pipeline_processor import PipelineLogitsProcessor
from src.data_labelling.efficient_diversity import init_worker_list_gpus

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.set_float32_matmul_precision('high')

SAMPLER_MODELS = {
    "SamplingNetwork": SamplingNetwork,
    "LocalProbabilityTransform": LocalProbabilityTransform,
    "SimpleDistributionAwareTransform": SimpleDistributionAwareTransform,
}

class PromptDataset(Dataset):
    def __init__(self, file_path: str, prompt_key: str = "prompt"):
        self.prompts = []
        self.all = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    x = json.loads(line)
                    self.prompts.append(x[prompt_key])
                    self.all.append(x)
                except (json.JSONDecodeError, KeyError):
                    print(f"Warning: Skipping malformed line or missing key in {file_path}")
        assert len(self.prompts) == len(self.all)
    
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.all[idx]

def setup_models_and_tokenizer(args, device_gen, device_sampler):
    print("Setting up models and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"Loading frozen generation model: {args.model_name_or_path}")
    generation_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device_gen)
    generation_model.eval()
    for param in generation_model.parameters():
        param.requires_grad = False

    print(f"Loading sampler model: {args.sampler_model_name}")
    with open(args.sampler_config_path, 'r') as f:
        sampler_config = yaml.safe_load(f)

    sampler_model_class = SAMPLER_MODELS[args.sampler_model_name]
    
    sampler_model = sampler_model_class(dtype=torch.bfloat16, **sampler_config).to(device_sampler)
    if args.sampler_checkpoint_path:
        print(f"Loading sampler checkpoint from: {args.sampler_checkpoint_path}")
        sampler_checkpoint = torch.load(args.sampler_checkpoint_path, map_location=device_sampler)
        sampler_model.load_state_dict(sampler_checkpoint)
    sampler_model = torch.compile(sampler_model)
    
    ref_sampler_model = copy.deepcopy(sampler_model)
    ref_sampler_model.eval()
    for param in ref_sampler_model.parameters():
        param.requires_grad = False

    print(f"Total trainable parameters in sampler: {sum(p.numel() for p in sampler_model.parameters() if p.requires_grad):,}")
    return tokenizer, generation_model, sampler_model, ref_sampler_model

def send_sglang_request(session, endpoint, payload):
    try:
        response = session.post(endpoint, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()['text']
    except requests.exceptions.RequestException as e:
        print(f"A request failed: {e}")
        return None

def generate_rollouts(prompts_text_batch, source_data_batch, tokenizer, policy_sampler_path, rl_config, args):
    G = rl_config['G']
    session = requests.Session()
    endpoint = f"http://{args.sglang_host}:{args.sglang_port}/generate"
    
    tasks_text = []
    tasks_source_data = []
    for i, prompt_text in enumerate(prompts_text_batch):
        tasks_text.extend([copy.deepcopy(prompt_text) for _ in range(G)])
        tasks_source_data.extend([copy.deepcopy(source_data_batch[i]) for _ in range(G)])

    pipeline_config = [{
        "name": "adaptive_sampler",
        "params": {
            "sampler_model_name": args.sampler_model_name,
            "sampler_config_path": args.sampler_config_path,
            "sampler_checkpoint_path": policy_sampler_path,
        }
    }]
    processor = PipelineLogitsProcessor(pipeline_config=pipeline_config)
    dill_hex_string = dill.dumps(processor).hex()
    processor_data = {"callable": dill_hex_string}
    
    payloads = []
    for prompt in tasks_text:
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        payloads.append({
            "text": full_prompt,
            "sampling_params": {
                "max_new_tokens": rl_config['max_new_tokens'],
                "temperature": 1.0,
            },
            "custom_logit_processor": json.dumps(processor_data)
        })

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(tqdm(executor.map(lambda p: send_sglang_request(session, endpoint, p), payloads), total=len(payloads), desc="Generating Rollouts"))
        completions = [res if res is not None else "" for res in results]

    completion_tokens = tokenizer(completions, return_tensors="pt", padding=True, truncation=True, max_length=rl_config['max_new_tokens'])
    attention_mask = completion_tokens.attention_mask

    return tasks_text, tasks_source_data, completions

def compute_rewards(source_data, completions, rl_config, tokenizer, args):
    device = torch.device(args.device_sampler)
    batch_size = len(completions)
    G = rl_config['G']
    G_div = rl_config['G_div']
    assert G % G_div == 0
    
    rewards = {"quality": torch.zeros(batch_size, device=device), "diversity": torch.zeros(batch_size, device=device), "verifiable": torch.zeros(batch_size, device=device)}
    is_applicable = {"quality": torch.zeros(batch_size, device=device), "diversity": torch.zeros(batch_size, device=device), "verifiable": torch.zeros(batch_size, device=device)}
    
    reward_data = [{'prompt': copy.deepcopy(src['prompt']), 'generation': copy.deepcopy(gen), **copy.deepcopy(src)} for src, gen in zip(source_data, completions)]
    
    verifiable_rewards = compute_unnormalized_verifiable_reward(reward_data)
    for i, item in enumerate(verifiable_rewards):
        if 'unnormalized_verifiable_reward' in item:
            rewards['verifiable'][i] = item['unnormalized_verifiable_reward']
            is_applicable['verifiable'][i] = 1.0

    compute_unnormalized_diversity_reward_embedding_entropy([x['generation'] for x in reward_data], group_size=G_div)
    for i, item in enumerate(reward_data):
        if 'unnormalized_diversity_reward_embedding_entropy' in item:
            rewards['diversity'][i] = item['unnormalized_diversity_reward_embedding_entropy']
            is_applicable['diversity'][i] = 1.0

    reward_model_urls = [f"http://{args.reward_model_host}:{args.reward_model_port + i}/classify" for i in range(args.reward_model_gpus)]
    quality_rewards = compute_unnormalized_direct_ratings(reward_data, args.reward_model_path, reward_model_urls, tokenizer, 0)
    for i, item in enumerate(quality_rewards):
         if 'bradley_terry_rating' in item:
            rewards['quality'][i] = item['bradley_terry_rating']
            is_applicable['quality'][i] = 1.0

    combined_rewards = torch.zeros(batch_size, device=device)
    total_weights = torch.zeros(batch_size, device=device)
    
    component_means = {}
    norm_params = rl_config.get('reward_normalization', {})

    for name in rewards:
        if is_applicable[name].sum() > 0 and name in norm_params:
            mean = norm_params[name].get('mean', 0.0)
            std = norm_params[name].get('std', 1.0)
            component_means[f'reward_{name}_mean_unnormalized'] = rewards[name][is_applicable[name] == 1.0].mean().item()
            rewards[name] = (rewards[name] - mean) / (std + 1e-8)
        weight = rl_config['reward_weights'].get(name, 0.0)
        combined_rewards += is_applicable[name] * weight * rewards[name]
        total_weights += is_applicable[name] * weight
        
    final_rewards = torch.where(total_weights > 0, combined_rewards / total_weights, 0.0)
    
    if final_rewards.numel() > 0:
      component_means['reward_mean_final'] = final_rewards.mean().item()

    return final_rewards, component_means


def get_log_probs(prompts, completions, gen_model, policy_sampler, ref_sampler, tokenizer, device_gen, device_sampler):
    gen_model.eval()
    policy_sampler.eval()
    ref_sampler.eval()

    full_chats = []
    for p, c in zip(prompts, completions):
        if c:
            full_chats.append([{"role": "user", "content": p}, {"role": "assistant", "content": c}])
        else:
            full_chats.append([{"role": "user", "content": p}])

    inputs = tokenizer.apply_chat_template(
        full_chats,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096
    ).to(device_gen)

    with torch.no_grad():
        base_outputs = gen_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        base_logits = base_outputs.logits.to(device_sampler)

    base_logits = base_logits[:, :-1, :]

    flat_base_logits = base_logits.reshape(-1, base_logits.size(-1))
    flat_policy_logits = policy_sampler(flat_base_logits)
    with torch.no_grad():
        flat_ref_logits = ref_sampler(flat_base_logits)

    flat_policy_lps = F.log_softmax(flat_policy_logits, dim=-1)
    flat_ref_lps = F.log_softmax(flat_ref_logits, dim=-1)

    policy_lps_full = flat_policy_lps.view_as(base_logits)
    ref_lps_full = flat_ref_lps.view_as(base_logits)

    target_ids = inputs.input_ids[:, 1:].to(device_sampler)
    gathered_policy_lps = torch.gather(policy_lps_full, 2, target_ids.unsqueeze(-1)).squeeze(-1)
    gathered_ref_lps = torch.gather(ref_lps_full, 2, target_ids.unsqueeze(-1)).squeeze(-1)

    loss_mask = torch.zeros_like(target_ids, dtype=torch.float)

    prompts_with_template = tokenizer.apply_chat_template(
        [[{"role": "user", "content": p}] for p in prompts],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        padding=True
    )
    prompt_lengths = prompts_with_template.attention_mask.sum(dim=1)

    for i in range(len(prompts)):
        seq_len = inputs.attention_mask[i, 1:].sum()
        prompt_len = prompt_lengths[i] - 1
        loss_mask[i, prompt_len:seq_len] = 1.0

    return gathered_policy_lps, gathered_ref_lps, loss_mask

def grpo_loss_fn(rewards, policy_lps, ref_lps, mask, G, beta, eps_denom=1e-8):
    num_groups_per_batch = rewards.shape[0] // G
    
    rewards_grouped = rewards.view(num_groups_per_batch, G)
    mean_rewards = rewards_grouped.mean(dim=-1, keepdim=True)
    std_rewards = rewards_grouped.std(dim=-1, keepdim=True)
    advantages = (rewards_grouped - mean_rewards) / (std_rewards + eps_denom)
    advantages = advantages.flatten().detach()

    advantages = advantages.unsqueeze(1).expand_as(policy_lps)
    
    pg_term = (advantages * policy_lps * mask).sum() / mask.sum()
    
    log_ratios = policy_lps - ref_lps
    kl_div = (torch.exp(log_ratios) - 1 - log_ratios) * mask
    kl_term = beta * kl_div.sum() / mask.sum()

    loss = -(pg_term - kl_term)
    
    return loss, pg_term, kl_term

def main():
    parser = argparse.ArgumentParser(description="RL for Adaptive Sampling Networks using GRPO.")

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--sampler_model_name", type=str, required=True, choices=SAMPLER_MODELS.keys())
    parser.add_argument("--sampler_config_path", type=str, required=True)
    parser.add_argument("--sampler_checkpoint_path", type=str)
    
    parser.add_argument("--prompt_dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./rl_checkpoints")
    
    parser.add_argument("--rl_config_path", type=str, required=True)
    
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_rl_steps", type=int, default=1000)
    parser.add_argument("--num_inner_updates", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=50)
    
    parser.add_argument("--sglang_host", type=str, default="localhost")
    parser.add_argument("--sglang_port", type=int, default=30000)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--device_gen", type=str, default="cuda:1")
    parser.add_argument("--device_sampler", type=str, default="cuda:0")
    parser.add_argument("--device_embedding", type=str, default="cuda:1")

    parser.add_argument("--reward_model_path", type=str, help="Path for the quality reward model tokenizer.")
    parser.add_argument("--reward_model_host", type=str, default="localhost")
    parser.add_argument("--reward_model_port", type=int, default=8000)
    parser.add_argument("--reward_model_gpus", type=int, default=1)
    
    args = parser.parse_args()
    
    device_gen = torch.device(args.device_gen)
    device_sampler = torch.device(args.device_sampler)
    device_embedding = torch.device(args.device_embedding)
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = os.path.join(args.output_dir, "rl_log.jsonl")

    init_worker_list_gpus([args.device_embedding])

    with open(args.rl_config_path, 'r') as f:
        rl_config = yaml.safe_load(f)

    tokenizer, gen_model, sampler_model, ref_sampler_model = setup_models_and_tokenizer(args, device_gen, device_sampler)
    prompt_dataset = PromptDataset(args.prompt_dataset_path)
    prompt_loader = DataLoader(prompt_dataset, batch_size=rl_config['prompts_per_batch'], shuffle=True)
    
    optimizer = AdamW(sampler_model.parameters(), lr=rl_config['lr'])
    total_training_steps = min(args.num_rl_steps, len(prompt_loader) * args.num_epochs) * args.num_inner_updates
    lr_scheduler = get_scheduler(
        name=rl_config.get("lr_scheduler_type", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=rl_config.get("num_warmup_steps", 100),
        num_training_steps=total_training_steps
    )
    
    current_step = 0
    temp_file_idx = 0

    print("Starting Training Loop")
    for epoch in range(args.num_epochs):
        pbar = tqdm(prompt_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        for prompts_text_batch, source_data_batch in pbar:
            if current_step >= args.num_rl_steps:
                break
            
            sampler_model.eval()
            temp_policy_path = os.path.join(args.output_dir, f"_current_policy_{temp_file_idx}.bin")
            temp_file_idx += 1
            torch.save(sampler_model.state_dict(), temp_policy_path)
            time.sleep(1)

            tasks_text, tasks_source, completions = generate_rollouts(prompts_text_batch, source_data_batch, tokenizer, temp_policy_path, rl_config, args)

            rewards, reward_means = compute_rewards(tasks_source, completions, rl_config, tokenizer, args)

            sampler_model.train()
            for _ in range(args.num_inner_updates):
                optimizer.zero_grad()
                policy_lps, ref_lps, loss_mask = get_log_probs(tasks_text, completions, gen_model, sampler_model, ref_sampler_model, tokenizer, device_gen, device_sampler)
                loss, pg_term, kl_term = grpo_loss_fn(rewards, policy_lps, ref_lps, loss_mask, rl_config['G'], rl_config['beta'])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(sampler_model.parameters(), rl_config['max_grad_norm'])
                optimizer.step()
                lr_scheduler.step()
            
            kl_val = kl_term.item() / rl_config['beta'] if rl_config['beta'] > 0 else 0
            log_data = {
                "step": current_step,
                "loss": loss.item(),
                "reward_mean": rewards.mean().item(),
                "kl": kl_val,
                **reward_means
            }
            
            with open(log_file_path, 'a') as f:
                f.write(json.dumps(log_data) + '\n')
            pbar.set_postfix(log_data)
            current_step += 1

            if current_step > 0 and current_step % args.save_every == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{current_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(sampler_model.state_dict(), os.path.join(checkpoint_dir, "sampler_model.bin"))
                print(f"\nSaved checkpoint to {checkpoint_dir}")

            gc.collect()
            torch.cuda.empty_cache()
        
        if current_step >= args.num_rl_steps:
            break

    final_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    torch.save(sampler_model.state_dict(), os.path.join(final_path, "sampler_model.bin"))
    print(f"\nTraining Complete")
    print(f"Saved final model to {final_path}")
    print(f"Training logs saved to {log_file_path}")

if __name__ == "__main__":
    main()
