import yaml
import argparse
from copy import deepcopy
from typing import List, Dict, Any

def get_base_parameters() -> Dict[str, List[float]]:
    return {
        "temperatures": [round(1.2 ** x * 10) / 10 for x in range(-6, 6)],
        "top_p_values": [0.5, 0.8, 0.9, 0.95, 0.98],
        "top_k_values": [8, 16, 32, 64, 128],
        "min_p_values": [0.01, 0.02, 0.05, 0.1],
        "eta_values": [1e-3, 8e-4, 5e-4, 2e-4, 1e-4],
        "epsilon_values": [4e-3, 2e-3, 1e-3, 8e-4, 5e-4, 2e-4],
        "typical_values": [0.2, 0.5, 0.8, 0.9, 0.92, 0.95],
    }

def get_model_specific_defaults(model_type: str) -> List[Dict[str, Any]]:
    if model_type == 'llama':
        return [{"name": "top_p", "params": {"top_p": 0.9}}]
    elif model_type == 'gemma':
        return [
            {"name": "top_p", "params": {"top_p": 0.95}},
            {"name": "top_k", "params": {"top_k": 64}},
        ]
    elif model_type == 'qwen':
        return [
            {"name": "top_p", "params": {"top_p": 0.8}},
            {"name": "top_k", "params": {"top_k": 20}},
        ]
    else:
        print(f"Warning: Unknown model_type '{model_type}'. No model-specific defaults will be added.")
        return []

def generate_sampling_pipelines(model_type: str) -> Dict[str, List[Dict[str, Any]]]:
    params = get_base_parameters()
    all_pipelines = []

    def create_pipeline(name: str, processors: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"name": name, "weight": 1.0, "processors": processors}

    for temp in params["temperatures"]:
        temp_processor = {"name": "temperature", "params": {"temperature": temp}}
        
        all_pipelines.append(create_pipeline(f"temp_{temp}", [deepcopy(temp_processor)]))

        model_defaults = get_model_specific_defaults(model_type)
        if model_defaults:
            all_pipelines.append(
                create_pipeline(f"official_temp_{temp}", [deepcopy(temp_processor)] + model_defaults)
            )

        for p in params["top_p_values"]:
            all_pipelines.append(create_pipeline(f"top_p_{p}_temp_{temp}", [deepcopy(temp_processor), {"name": "top_p", "params": {"top_p": p}}]))
        for k in params["top_k_values"]:
            all_pipelines.append(create_pipeline(f"top_k_{k}_temp_{temp}", [deepcopy(temp_processor), {"name": "top_k", "params": {"top_k": k}}]))
        for p in params["min_p_values"]:
            all_pipelines.append(create_pipeline(f"min_p_{p}_temp_{temp}", [deepcopy(temp_processor), {"name": "min_p", "params": {"min_p": p}}]))
        for e in params["eta_values"]:
            all_pipelines.append(create_pipeline(f"eta_{e:.1e}_temp_{temp}", [deepcopy(temp_processor), {"name": "eta", "params": {"epsilon": e}}]))
        for e in params["epsilon_values"]:
            all_pipelines.append(create_pipeline(f"epsilon_{e:.1e}_temp_{temp}", [deepcopy(temp_processor), {"name": "epsilon", "params": {"epsilon": e}}]))
        for t in params["typical_values"]:
            all_pipelines.append(create_pipeline(f"typical_{t}_temp_{temp}", [deepcopy(temp_processor), {"name": "typical", "params": {"mass": t}}]))

    return {"sampling_pipelines": all_pipelines}

def main():
    parser = argparse.ArgumentParser(
        description="Generate a YAML configuration file with a wide range of sampling heuristics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=['llama', 'gemma', 'qwen', 'none'],
        help="The base model type to generate specific default configurations for."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output YAML file."
    )
    parser.add_argument(
        "--append_from_file",
        type=str,
        default=None,
        help="Optional path to a YAML file containing additional 'sampling_pipelines' to append to the generated config."
    )
    args = parser.parse_args()

    config_data = generate_sampling_pipelines(args.model_type)

    if args.append_from_file:
        print(f"Appending additional pipelines from '{args.append_from_file}'...")
        try:
            with open(args.append_from_file, 'r') as f:
                extra_config = yaml.safe_load(f)
            
            if 'sampling_pipelines' in extra_config and isinstance(extra_config['sampling_pipelines'], list):
                existing_names = {p['name'] for p in config_data['sampling_pipelines']}
                new_pipelines = extra_config['sampling_pipelines']
                
                appended_count = 0
                for pipeline in new_pipelines:
                    if pipeline.get('name') in existing_names:
                        print(f"Warning: Pipeline '{pipeline.get('name')}' from appended file already exists. Skipping.")
                    else:
                        config_data['sampling_pipelines'].append(pipeline)
                        appended_count += 1
                print(f"Successfully appended {appended_count} new unique pipelines.")
            else:
                print("Warning: '--append_from_file' did not contain a valid 'sampling_pipelines' list. Skipping.")
        except Exception as e:
            print(f"Error loading or processing append file: {e}")

    with open(args.output_file, "w") as f:
        yaml.dump(config_data, f, indent=2, sort_keys=False)
        
    print(f"\nSuccessfully generated {len(config_data['sampling_pipelines'])} total sampling pipelines.")
    print(f"Configuration saved to '{args.output_file}'")

if __name__ == "__main__":
    main()
