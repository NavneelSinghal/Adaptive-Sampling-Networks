import argparse
import json
from tqdm import tqdm

from src.data_labelling.infinigram_metrics import InfiniGramApiLabeller

def label_infinigram(data_path: str, output_path: str, index_name: str):
    """
    Reads a dataset, applies Infini-gram labelling to each generation,
    and saves the annotated data.
    """
    print(f"Initializing Infini-gram labeller with index: {index_name}")
    labeller = InfiniGramApiLabeller(index_name=index_name)
    print(f"Reading data from {data_path} and writing annotated results to {output_path}...")
    with open(data_path, 'r', encoding='utf-8') as f_in, \
        open(output_path, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc="Labelling Generations"):
            datapoint = json.loads(line)
            generation_text = datapoint.get('generation')
            if not generation_text:
                continue
            scores = labeller.label(generation_text)
            datapoint['infinigram_scores'] = scores
            f_out.write(json.dumps(datapoint) + '\n')
    print("\nInfini-gram labelling complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label generation quality using the Infini-gram API.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input .jsonl dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the annotated output .jsonl file.")
    parser.add_argument("--infogram_index", type=str, required=True, help="The Infini-gram API index to use (e.g., 'v4_rpj_llama_s4').")
    args = parser.parse_args()
    label_infinigram(args.data_path, args.output_path, args.infogram_index)
