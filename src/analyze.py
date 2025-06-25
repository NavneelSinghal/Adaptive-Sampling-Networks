import argparse
import yaml
import os
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
torch._dynamo.config.disable = True

torch.set_float32_matmul_precision('high')
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from models import SamplingNetwork, LocalProbabilityTransform, SimpleDistributionAwareTransform
from sampling_heuristics import apply_pipeline_to_logits
from loss import TruncatedLogitsLoss

SAMPLER_MODELS = {
    "SamplingNetwork": SamplingNetwork,
    "LocalProbabilityTransform": LocalProbabilityTransform,
    "SimpleDistributionAwareTransform": SimpleDistributionAwareTransform,
}

def setup_models(args, device_gen, device_sampler):
    print("--- Setting up models and tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading frozen generation model: {args.model_name_or_path}")
    gen_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device_gen)
    gen_model.eval()
    print(f"Loading sampler model: {args.sampler_model_name}")
    with open(args.sampler_config_path, 'r') as f:
        sampler_config = yaml.safe_load(f)
    sampler_model_class = SAMPLER_MODELS[args.sampler_model_name]
    sampler_model = sampler_model_class(dtype=torch.float32, **sampler_config).to(torch.float32)
    sampler_checkpoint = torch.load(args.sampler_checkpoint_path, map_location=device_sampler)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in sampler_checkpoint.items():
        if k.startswith('_orig_mod.'):
            name = k[10:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    sampler_model.load_state_dict(new_state_dict)
    sampler_model.to(device_sampler)
    sampler_model.eval()
    print("--- Models loaded successfully ---")
    return tokenizer, gen_model, sampler_model

def calculate_analysis_metrics(pred_logits, target_logits):
    with torch.no_grad():
        pred_logits = pred_logits.float()
        target_logits = target_logits.float()
        truncated_mask = torch.isneginf(target_logits)
        log_p_pred = F.log_softmax(pred_logits, dim=-1)
        target_logits_masked = target_logits.masked_fill(truncated_mask, -torch.inf)
        p_target = F.softmax(target_logits_masked, dim=-1)
        log_p_target = F.log_softmax(target_logits_masked, dim=-1)
        kl_div_elements = p_target * (log_p_target - log_p_pred)
        kl_div_elements[truncated_mask] = torch.tensor(0.0, dtype=torch.float32)
        kl_divergence = kl_div_elements.sum()
        p_pred = torch.exp(log_p_pred)
        truncated_mass = (p_pred * truncated_mask.float()).sum()
        return kl_divergence.item(), truncated_mass.item()

def analyze_and_plot(args, tokenizer, gen_model, sampler_model, device_gen, device_sampler):
    print(f"\n--- Generating continuation for prompt ---\nPrompt: '{args.prompt}'")

    chat = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': args.prompt}]
    inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", padding=True, return_dict=True).to(device_gen)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    generated_output = gen_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.num_tokens_to_analyze + 5,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    
    full_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    print(f"Generated Text: {full_text}\n")
    
    with torch.no_grad():
        full_sequence_inputs = generated_output.to(device_gen)
        outputs = gen_model(full_sequence_inputs)
        raw_logits_sequence = outputs.logits[0, input_ids.shape[1]-1:-1, :].clone()

    with open(args.pipeline_config_path, 'r') as f:
        pipeline_config = yaml.safe_load(f)['processors']
    print(f"Loaded analysis pipeline config from '{args.pipeline_config_path}'")
    print(pipeline_config)

    print("\n--- Analysis Results (Lower is Better) ---")
    print(f"{'Token':<15} | {'Metric':<25} | {'Identity (Raw Logits)':<25} | {'Trained Sampler':<20}")
    print("-" * 90)

    for i in range(min(args.num_tokens_to_analyze, raw_logits_sequence.shape[0])):
        token_id = generated_output[0, input_ids.shape[1] + i].item()
        token_str = tokenizer.decode(token_id)

        raw_logits = raw_logits_sequence[i].unsqueeze(0)
        
        with torch.no_grad():
            target_logits = apply_pipeline_to_logits(
                input_ids=None,
                logits=raw_logits.clone(),
                pipeline_config=pipeline_config
            )

        with torch.no_grad():
            raw_logits_for_sampler = raw_logits.to(device_sampler).to(torch.float32)
            predicted_logits = sampler_model(raw_logits_for_sampler).to(raw_logits.device)

        identity_logits = raw_logits

        kl_identity, mass_identity = calculate_analysis_metrics(identity_logits, target_logits)
        kl_sampler, mass_sampler = calculate_analysis_metrics(predicted_logits, target_logits)

        print(f"{token_str:<15} | {'KL Divergence':<25} | {kl_identity:<25.4f} | {kl_sampler:<20.4f}")
        print(f"{'':<15} | {'Truncated Prob. Mass':<25} | {mass_identity:<25.4f} | {mass_sampler:<20.4f}")
        print("-" * 90)

        if i < 20:
            print("\nGenerating probability distribution plot for the first analyzed token...")
            plot_distributions(
                raw_logits,
                target_logits,
                predicted_logits,
                tokenizer,
                args.top_k_plot,
                f"Token: '{token_str}' (ID: {token_id})",
                i
            )

def plot_distributions(raw_logits, target_logits, predicted_logits, tokenizer, top_k, title_str, suffix):
    with torch.no_grad():
        raw_logits = raw_logits.cpu().float()
        target_logits = target_logits.cpu().float()
        predicted_logits = predicted_logits.cpu().float()

        p_raw = F.softmax(raw_logits, dim=-1)
        p_target = F.softmax(target_logits, dim=-1)
        p_predicted = F.softmax(predicted_logits, dim=-1)
        
        top_probs_raw, top_indices_raw = torch.topk(p_raw.squeeze(), k=top_k)
        
        p_target_slice = p_target.squeeze()[top_indices_raw]
        p_predicted_slice = p_predicted.squeeze()[top_indices_raw]
        
        token_labels = [tokenizer.decode(idx.item()) for idx in top_indices_raw]
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(18, 10))
        
        x = np.arange(len(token_labels))
        width = 0.2
        ax.bar(x - width*1.5, top_probs_raw.numpy(), width, label='Original (Raw Logits)', color='skyblue')
        ax.bar(x - width*0.5, p_target_slice.numpy(), width, label='Target (Algorithm)', color='salmon')
        ax.bar(x + width*0.5, p_predicted_slice.numpy(), width, label='Predicted (Learned Sampler)', color='mediumseagreen')
        
        ax.set_ylabel('Probability', fontsize=14)
        ax.set_title(f'Probability Distributions for Top {top_k} Original Tokens\n{title_str}', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(token_labels, rotation=75, ha='right', fontsize=10)
        ax.legend(fontsize=12)
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-6)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        fig.tight_layout()
        
        output_filename = f'distribution_analysis_{suffix}.png'
        plt.savefig(output_filename)
        print(f"Plot saved to '{output_filename}'")


def main():
    parser = argparse.ArgumentParser(description="Inference and analysis for adaptive samplers.")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2", help="Path to the frozen LLM.")
    parser.add_argument("--sampler_model_name", type=str, required=True, choices=SAMPLER_MODELS.keys(), help="The type of sampler model to load.")
    parser.add_argument("--sampler_config_path", type=str, required=True, help="Path to the sampler model's config YAML.")
    parser.add_argument("--sampler_checkpoint_path", type=str, required=True, help="Path to the trained sampler model's .bin file.")
    parser.add_argument("--pipeline_config_path", type=str, required=True, help="Path to a YAML file defining the sampling pipeline for generating target logits.")
    parser.add_argument("--prompt", type=str, default="Please write me a random story.", help="The user prompt.")
    parser.add_argument("--num_tokens_to_analyze", type=int, default=5, help="How many generated tokens to analyze in detail.")
    parser.add_argument("--top_k_plot", type=int, default=50, help="How many of the top original tokens to include in the plot.")
    args = parser.parse_args()
    device_gen = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device_sampler = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer, gen_model, sampler_model = setup_models(args, device_gen, device_sampler)
    analyze_and_plot(args, tokenizer, gen_model, sampler_model, device_gen, device_sampler)

if __name__ == "__main__":
    main()
