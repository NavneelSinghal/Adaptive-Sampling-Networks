import argparse
import json
from tqdm import tqdm
import logging

from src.data_labelling.verifiers import GSM8KVerifier, MathVerifier, IFEvalVerifierOld

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_relevant_verifiers() -> dict:
    verifiers = {
        "gsm8k": GSM8KVerifier(),
        "MATH": MathVerifier(),
        "ifeval": IFEvalVerifierOld(),
    }
    logging.info(f"Initialized verifiers for: {list(verifiers.keys())}")
    return verifiers

def process_file(input_path: str, output_path: str, verifiers: dict):
    logging.info(f"Starting processing for '{input_path}'")
    try:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:

            for line in tqdm(f_in, desc="Labeling Verifiable Rewards"):
                try:
                    datapoint = json.loads(line)
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed JSON line: {line.strip()}")
                    continue
                
                verifiable_reward = None
                dataset_type = datapoint.get('dataset')

                if dataset_type in verifiers:
                    verifier = verifiers[dataset_type]
                    
                    ground_truth = datapoint.get('ground_truth')
                    if ground_truth is not None:
                        try:
                            result = verifier(
                                tokenized_prediction=None,
                                prediction=datapoint.get('generation', ''),
                                label=ground_truth,
                                query=datapoint.get('prompt')
                            )
                            verifiable_reward = result.score
                        except Exception as e:
                            logging.error(f"Error verifying datapoint with prompt '{datapoint.get('prompt', 'N/A')[:50]}...': {e}")
                            verifiable_reward = 0.0
                    else:
                        logging.warning(f"No 'ground_truth' key found for verifiable datapoint. Prompt: {datapoint.get('prompt', 'N/A')[:50]}...")
                
                datapoint['verifiable_reward'] = verifiable_reward
                f_out.write(json.dumps(datapoint) + '\n')

    except FileNotFoundError:
        logging.error(f"Input file not found: {input_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    logging.info(f"Processing complete. Annotated data saved to '{output_path}'")

def main():
    parser = argparse.ArgumentParser(description="Label a dataset with verifiable rewards using custom verifier functions.")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the input .jsonl file containing generations."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the output .jsonl file with verifiable rewards."
    )
    args = parser.parse_args()
    verifiers = build_relevant_verifiers()
    process_file(args.data_path, args.output_path, verifiers)

if __name__ == "__main__":
    main()

