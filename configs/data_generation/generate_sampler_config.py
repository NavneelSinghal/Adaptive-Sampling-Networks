import yaml
import argparse
import os
import re
from typing import List, Dict, Any

def generate_adaptive_sampler_pipelines(model_dir: str) -> List[Dict[str, Any]]:
    """
    Scans a directory for trained models and generates adaptive sampler pipelines.
    
    Args:
        model_dir (str): The path to the directory containing model output folders.

    Returns:
        A list of dictionaries, each representing a sampling pipeline for a found model.
    """
    if not os.path.isdir(model_dir):
        print(f"Error: Model directory not found at '{model_dir}'. Cannot generate adaptive samplers.")
        return []
    
    adaptive_pipelines = []
    print(f"Scanning '{model_dir}' for trained adaptive samplers...")

    for dirname in sorted(os.listdir(model_dir)):
        match = re.match(r'output_final_training_data_(.*)', dirname)
        if not match:
            continue
        
        model_identifier = match.group(1)
        checkpoint_path = os.path.join(model_dir, dirname, "final_model", "sampler_model.bin")

        if os.path.exists(checkpoint_path):
            pipeline_name = f"adaptive_sampler_{model_identifier}"
            
            pipeline = {
                "name": pipeline_name,
                "weight": 1.0,
                "processors": [
                    {
                        "name": "adaptive_sampler",
                        "params": {
                            "sampler_model_name": "SamplingNetwork",
                            "sampler_config_path": "configs/sampler_models/model_config_scale.yaml",
                            "sampler_checkpoint_path": checkpoint_path
                        }
                    }
                ]
            }
            adaptive_pipelines.append(pipeline)
            print(f"  > Found and added pipeline: {pipeline_name}")
        else:
            print(f"  > Warning: Directory '{dirname}' found, but missing model bin file at '{checkpoint_path}'. Skipping.")
            
    return adaptive_pipelines

def main():
    parser = argparse.ArgumentParser(
        description="Scans a directory of trained models and generates a YAML file with their sampler configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the trained adaptive sampler models to scan."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the YAML file containing the discovered adaptive sampler pipelines."
    )
    args = parser.parse_args()

    adaptive_pipelines = generate_adaptive_sampler_pipelines(args.model_dir)

    if adaptive_pipelines:
        try:
            with open(args.output_file, "w") as f:
                yaml.dump({"sampling_pipelines": adaptive_pipelines}, f, indent=2, sort_keys=False)
            print(f"\nSuccessfully saved {len(adaptive_pipelines)} discovered sampler pipelines to '{args.output_file}'.")
        except Exception as e:
            print(f"Error writing to output file '{args.output_file}': {e}")
    else:
        print("\nNo adaptive samplers were found. Output file was not created.")

if __name__ == "__main__":
    main()
