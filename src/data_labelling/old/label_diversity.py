import argparse
import json
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from diversity_metrics import calculate_self_bleu, calculate_embedding_entropy

def label_diversity(data_path: str, output_path: str, reference_seed: int):
    """
    Labels a multi-seed generation dataset for diversity and appends scores
    to the reference seed datapoint.
    """
    print("Loading SentenceTransformer model for embedding calculations...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    print(f"Reading and grouping data from {data_path}...")
    data_groups = defaultdict(lambda: defaultdict(list))
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            datapoint = json.loads(line)
            sampler_key = json.dumps(datapoint['sampler_info'], sort_keys=True)
            data_groups[datapoint['prompt']][sampler_key].append(datapoint)

    print("Calculating diversity scores...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for prompt, pipelines in tqdm(data_groups.items(), desc="Processing Prompts"):
            for sampler_key, datapoints in pipelines.items():
                if len(datapoints) < 2:
                    continue
                reference_datapoint = next((dp for dp in datapoints if dp['seed'] == reference_seed), None)
                if not reference_datapoint:
                    continue
                generations = [dp['generation'] for dp in datapoints]
                self_bleu = calculate_self_bleu(generations)
                embedding_entropy = calculate_embedding_entropy(generations, embedding_model)
                reference_datapoint['diversity'] = {
                    'self_bleu': float(self_bleu),
                    'embedding_entropy': float(embedding_entropy),
                    'sample_count': len(generations)
                }
                f_out.write(json.dumps(reference_datapoint) + '\n')

    print(f"\nDiversity labelling complete. Annotated data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Labelling generation diversity from a multi-seed dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the multi-seed .jsonl dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the annotated output .jsonl file.")
    parser.add_argument("--reference_seed", type=int, default=0, help="The seed of the datapoint to append scores to.")
    args = parser.parse_args()
    
    label_diversity(args.data_path, args.output_path, args.reference_seed)
