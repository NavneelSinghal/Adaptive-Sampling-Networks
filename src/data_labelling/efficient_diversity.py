import argparse
import json
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from functools import partial
import multiprocessing
from typing import List, Dict, Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def calculate_self_bleu(generations: List[str]) -> float:
    from sacrebleu.metrics import BLEU
    if len(generations) < 2:
        return 0.0
    total_bleu_score = 0.0
    bleu = BLEU(effective_order=True)
    for i in range(len(generations)):
        hypothesis = generations[i]
        references = generations[:i] + generations[i+1:]
        score = bleu.sentence_score(hypothesis, references)
        total_bleu_score += score.score
    return total_bleu_score / len(generations)

def calculate_embedding_entropy(generations: List[str], model) -> float:
    from sklearn.metrics.pairwise import cosine_similarity
    if not generations:
        return 0.0
    embeddings = model.encode(generations, convert_to_numpy=True)
    similarity_matrix = cosine_similarity(embeddings)
    eigenvalues = np.linalg.eigvalsh(similarity_matrix)
    non_zero_eigenvalues = eigenvalues[eigenvalues > 1e-9]
    if non_zero_eigenvalues.size == 0:
        return 0.0
    normalized_eigenvalues = non_zero_eigenvalues / np.sum(non_zero_eigenvalues)
    normalized_eigenvalues = normalized_eigenvalues[normalized_eigenvalues > 1e-9]
    if normalized_eigenvalues.size == 0:
        return 0.0
    entropy = -np.sum(normalized_eigenvalues * np.log2(normalized_eigenvalues))
    return entropy

embedding_model = None

def init_worker(num_gpus_available: int):
    """
    Initializer for each worker. Uses the worker's unique ID to select a GPU.
    This ensures the model is loaded only once per process on the correct device.
    """
    global embedding_model
    from sentence_transformers import SentenceTransformer
    
    worker_id = multiprocessing.current_process()._identity[0] - 1
    gpu_id = worker_id % num_gpus_available
    process_name = multiprocessing.current_process().name
    print(f"{process_name}: Initializing model on cuda:{gpu_id}...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=f'cuda:{gpu_id}')
    print(f"{process_name}: Model loaded on cuda:{gpu_id}.")


def process_group(datapoints: List[Dict], reference_seed: int) -> str | None:
    global embedding_model
    if len(datapoints) < 2:
        return None
    generations = [dp['generation'] for dp in datapoints]
    self_bleu = calculate_self_bleu(generations)
    embedding_entropy = calculate_embedding_entropy(generations, embedding_model)
    for dp in datapoints:
        dp['diversity'] = {
            'self_bleu': float(self_bleu),
            'embedding_entropy': float(embedding_entropy),
            'sample_count': len(generations)
        }
    return '\n'.join([json.dumps(dp) for dp in datapoints])

def label_diversity_parallel(data_path: str, output_path: str, reference_seed: int, num_gpus: int):
    print(f"Reading and grouping data from {data_path}...")
    data_groups = defaultdict(lambda: defaultdict(list))
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            datapoint = json.loads(line)
            sampler_key = json.dumps(datapoint['sampler_info'], sort_keys=True)
            data_groups[datapoint['prompt']][sampler_key].append(datapoint)

    jobs = [datapoints for pipelines in data_groups.values() for datapoints in pipelines.values()]
    num_processes = os.cpu_count() - 1
    print(f"\nFound {len(jobs)} groups to process.")
    print(f"Starting parallel processing with {num_processes} workers across {num_gpus} GPUs.")
    
    worker_func = partial(process_group, reference_seed=reference_seed)

    with open(output_path, 'w', encoding='utf-8') as f_out:
        with multiprocessing.Pool(
            processes=num_processes, 
            initializer=init_worker, 
            initargs=(num_gpus,)
        ) as pool:
            progress_bar = tqdm(pool.imap_unordered(worker_func, jobs), total=len(jobs), desc="Processing Groups")
            for result_json in progress_bar:
                if result_json:
                    f_out.write(result_json + '\n')

    print(f"\nDiversity labelling complete. Annotated data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Efficiently label generation diversity from a multi-seed dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the multi-seed .jsonl dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the annotated output .jsonl file.")
    parser.add_argument("--reference_seed", type=int, default=0, help="The seed of the datapoint to append scores to.")
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs to distribute work across.")
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)
    label_diversity_parallel(args.data_path, args.output_path, args.reference_seed, args.num_gpus)
