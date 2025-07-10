import yaml
from copy import deepcopy

def generate_sampling_pipelines():
    temperatures = [round(1.2 ** x * 10) / 10 for x in range(-6, 3)]
    temperatures_min_p = [round(1.2 ** x * 10) / 10 for x in range(-6, 5)]
    top_p_values = [0.8, 0.9, 0.95]
    top_k_values = [16, 64, 256]
    min_p_values = [0.01, 0.02, 0.05, 0.1]
    epsilon_values = [4e-3, 2e-3, 1e-3]

    all_pipelines = []

    for temp in temperatures_min_p:

        temp_processor = {
            "name": "temperature",
            "params": {"temperature": temp}
        }

        def create_pipeline(name, processors):
            return {"name": name, "weight": 1, "processors": processors}

        for p in min_p_values:
            all_pipelines.append(
                create_pipeline(
                    f"min_p {p} temp {temp}",
                    [deepcopy(temp_processor), {"name": "min_p", "params": {"min_p": p}}],
                )
            )

        if temp not in temperatures:
            continue

        all_pipelines.append(
            create_pipeline(f"temp {temp}", [deepcopy(temp_processor)])
        )

        all_pipelines.append(
            create_pipeline(
                f"official temp {temp}",
                [
                    deepcopy(temp_processor),
                    {"name": "top_p", "params": {"top_p": 0.9}},
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

        for e in epsilon_values:
            all_pipelines.append(
                create_pipeline(
                    f"epsilon {e:.1e} temp {temp}",
                    [deepcopy(temp_processor), {"name": "epsilon", "params": {"epsilon": e}}],
                )
            )

    final_config = {"sampling_pipelines": all_pipelines}
    return final_config

if __name__ == "__main__":
    config_data = generate_sampling_pipelines()
    
    output_filename = "generated_config_llama3.2_3b.yaml"
    
    with open(output_filename, "w") as f:
        yaml.dump(config_data, f, indent=2, sort_keys=False)
        
    print(f"Successfully generated {len(config_data['sampling_pipelines'])} sampling pipelines.")
    print(f"Configuration saved to '{output_filename}'")


