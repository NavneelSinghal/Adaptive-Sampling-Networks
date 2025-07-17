import argparse
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import itertools

def process_and_score_data(
    input_path: str,
    output_path: str,
    a_coeffs: list[float],
    b_coeffs: list[float],
    c_coeffs: list[float],
    top_k: int,
):
    print("Reading data and grouping by prompt")
    data_by_prompt_seed = defaultdict(list)
    all_quality_scores = []
    diversity_scores_for_norm = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading and Grouping"):
            datapoint = json.loads(line)
            group_key = (datapoint['prompt'], datapoint['seed'])
            data_by_prompt_seed[group_key].append(datapoint)
            if 'bradley_terry_rating' in datapoint:
                all_quality_scores.append(datapoint['bradley_terry_rating'])
            if datapoint.get('incidence_vector', {}).get('check_diversity'):
                if 'diversity' in datapoint and 'embedding_entropy' in datapoint['diversity']:
                    diversity_scores_for_norm.append(datapoint['diversity']['embedding_entropy'])

    print(f"Found {len(data_by_prompt_seed)} unique (prompt, seed) groups.")

    print("\nCalculating normalization stats")
    quality_mean = np.mean(all_quality_scores)
    quality_std = np.std(all_quality_scores)
    print(f"Quality score stats: mean={quality_mean:.4f}, std={quality_std:.4f}")
    
    diversity_mean = np.mean(diversity_scores_for_norm)
    diversity_std = np.std(diversity_scores_for_norm)
    print(f"Diversity score stats: mean={diversity_mean:.4f}, std={diversity_std:.4f}")
    
    quality_std = quality_std if quality_std > 1e-6 else 1.0
    diversity_std = diversity_std if diversity_std > 1e-6 else 1.0

    print("\nNormalization, scoring and filtering")
    final_data_to_write = []
    for group_key, generations in tqdm(data_by_prompt_seed.items(), desc="Processing Groups"):
        processed_generations = []
        is_verifiable_task = any(dp.get('verifiable_reward') is not None for dp in generations)
        if is_verifiable_task:
            generations = [dp for dp in generations if dp.get('verifiable_reward') == 1.0]
            if not generations:
                print(f'WARNING: All generations failed for prompt {group_key[0][:50]}... skipping.')
                continue
        for dp in generations:
            norm_quality = (dp.get('bradley_terry_rating', quality_mean) - quality_mean) / quality_std
            dp['normalized_quality'] = norm_quality
            norm_diversity = None
            if dp.get('incidence_vector', {}).get('check_diversity'):
                diversity_score = dp.get('diversity', {}).get('embedding_entropy', diversity_mean)
                norm_diversity = (diversity_score - diversity_mean) / diversity_std
                dp['normalized_diversity'] = norm_diversity
            dp['combined_scores'] = {}
            for b, a, c in itertools.product(b_coeffs, a_coeffs, c_coeffs):
                score_key = f"a{a}_b{b}_c{c}"
                total_score = 0
                divisor = 0
                total_score += b * norm_quality
                divisor += b
                if norm_diversity is not None:
                    total_score += a * norm_diversity
                    divisor += a
                if is_verifiable_task:
                    total_score += c * 1.0
                    divisor += c
                final_score = total_score / divisor if divisor > 0 else 0
                dp['combined_scores'][score_key] = final_score
            processed_generations.append(dp)
        if not processed_generations:
            continue
        default_sort_key = f"a0.1_b0.9_c0.9"
        processed_generations.sort(
            key=lambda x: x['combined_scores'].get(default_sort_key, -999),
            reverse=True
        )
        top_generations = processed_generations[:top_k]
        final_data_to_write.extend(top_generations)
    print(f"\nWriting {len(final_data_to_write)} filtered data points to output")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for datapoint in final_data_to_write:
            f_out.write(json.dumps(datapoint) + '\n')
    print(f"\nOutput saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Normalize scores, apply combined scoring, and filter the dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input .jsonl file with rewards.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the final processed and filtered .jsonl file.")
    parser.add_argument("--top-k", type=int, default=5, help="Keep the top K best-scoring generations for each prompt.")
    parser.add_argument("--sort-key", type=str, default='a0.1_b0.9_c0.9', help="Key for ranking: ax_by_cz where x = weight for diversity, y = weight for quality, z = weight for correctness")
    
    args = parser.parse_args()
    
    A_COEFFS = [0.1, 0.3, 0.5, 0.7, 0.9]
    B_COEFFS = [0.5, 0.6, 0.7, 0.8, 0.9]
    C_COEFFS = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    process_and_score_data(
        input_path=args.input_path,
        output_path=args.output_path,
        a_coeffs=A_COEFFS,
        b_coeffs=B_COEFFS,
        c_coeffs=C_COEFFS,
        top_k=args.top_k,
    )

if __name__ == "__main__":
    main()
