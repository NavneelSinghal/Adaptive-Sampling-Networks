import yaml
from copy import deepcopy

def generate_sampling_pipelines():
    temperatures = [round(1.2 ** x * 10) / 10 for x in range(-6, 6)]
    top_p_values = [0.5, 0.8, 0.9, 0.95, 0.98]
    top_k_values = [8, 16, 32, 64, 128]
    min_p_values = [0.01, 0.02, 0.05, 0.1]
    eta_values = [1e-3, 8e-4, 5e-4, 2e-4, 1e-4]
    epsilon_values = [4e-3, 2e-3, 1e-3, 8e-4, 5e-4, 2e-4]
    typical_values = [0.2, 0.5, 0.8, 0.9, 0.92, 0.95]

    all_pipelines = []

    for temp in temperatures:
        temp_processor = {
            "name": "temperature",
            "params": {"temperature": temp}
        }
        
        def create_pipeline(name, processors):
            return {"name": name, "weight": 1, "processors": processors}

        all_pipelines.append(
            create_pipeline(f"temp {temp}", [deepcopy(temp_processor)])
        )

        all_pipelines.append(
            create_pipeline(
                f"official temp {temp}",
                [
                    deepcopy(temp_processor),
                    {"name": "top_p", "params": {"top_p": 0.8}},
                    {"name": "top_k", "params": {"top_k": 20}},
                ],
            )
        )

        for p in top_p_values:
            all_pipelines.append(
                create_pipeline(
                    f"top_p {p} temp {temp}",
                    [deepcopy(temp_processor), {"name": "top_p", "params": {"top_p": p}}],
                )
            )

        for k in top_k_values:
            all_pipelines.append(
                create_pipeline(
                    f"top_k {k} temp {temp}",
                    [deepcopy(temp_processor), {"name": "top_k", "params": {"top_k": k}}],
                )
            )

        for p in min_p_values:
            all_pipelines.append(
                create_pipeline(
                    f"min_p {p} temp {temp}",
                    [deepcopy(temp_processor), {"name": "min_p", "params": {"min_p": p}}],
                )
            )
            
        for e in eta_values:
            all_pipelines.append(
                create_pipeline(
                    f"eta {e:.1e} temp {temp}",
                    [deepcopy(temp_processor), {"name": "eta", "params": {"epsilon": e}}],
                )
            )

        for e in epsilon_values:
            all_pipelines.append(
                create_pipeline(
                    f"epsilon {e:.1e} temp {temp}",
                    [deepcopy(temp_processor), {"name": "epsilon", "params": {"epsilon": e}}],
                )
            )

        for t in typical_values:
            all_pipelines.append(
                create_pipeline(
                    f"typical {t} temp {temp}",
                    [deepcopy(temp_processor), {"name": "typical", "params": {"mass": t}}],
                )
            )

    final_config = {"sampling_pipelines": all_pipelines}
    
    return final_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a YAML configuration for sampling pipelines.")
    parser.add_argument(
        "--add_from_file",
        type=str,
        default=None,
        help="Optional: Path to a YAML file with additional sampling pipelines to include."
    )
    args = parser.parse_args()

    config_data = generate_sampling_pipelines()
    
    if args.add_from_file:
        print(f"INFO: Attempting to add pipelines from '{args.add_from_file}'...")
        try:
            with open(args.add_from_file, 'r') as f:
                additional_config = yaml.safe_load(f)
                additional_pipelines = additional_config.get("sampling_pipelines", [])
                if additional_pipelines:
                    config_data["sampling_pipelines"].extend(additional_pipelines)
                    print(f"Successfully added {len(additional_pipelines)} pipelines from file.")
                else:
                    print(f"WARNING: No 'sampling_pipelines' key found in '{args.add_from_file}'.")
        except FileNotFoundError:
            print(f"ERROR: File not found: '{args.add_from_file}'. Skipping.")
        except yaml.YAMLError as e:
            print(f"ERROR: Could not parse YAML from '{args.add_from_file}': {e}. Skipping.")

    output_filename = "generated_config_qwen2.5_3b.yaml"
    
    with open(output_filename, "w") as f:
        yaml.dump(config_data, f, indent=2, sort_keys=False)
        
    print(f"Successfully generated {len(config_data['sampling_pipelines'])} sampling pipelines.")
    print(f"Configuration saved to '{output_filename}'")
