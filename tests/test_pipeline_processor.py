import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src._sglang.sglang_pipeline_processor import PipelineLogitsProcessor

def run_test(test_name, pipeline_config_dict, vocab_size=50):
    """Helper function to run a single test case."""
    print(f"\nRunning Test: {test_name}\n")
    logits = torch.randn(2, vocab_size)
    print(f"Original logits shape: {logits.shape}")
    try:
        processors_list = pipeline_config_dict.get("processors", [])
        pipeline_processor = PipelineLogitsProcessor(pipeline_config=processors_list)
        print("Pipeline Processor initialized successfully with config:")
        for processor in pipeline_processor.pipeline:
            print(f"  - {processor.__class__.__name__} with params: {processor.__dict__}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize processor: {e}")
        raise
    try:
        processed_logits = pipeline_processor(logits.clone(), custom_param_list=None)
        print(f"Processed logits shape: {processed_logits.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to execute processor pipeline: {e}")
        raise
    print("\nVerifying output...")
    last_top_k_config = None
    for conf in reversed(processors_list):
        if conf["name"] == "top_k":
            last_top_k_config = conf
            break
    if last_top_k_config:
        k = last_top_k_config["params"]["top_k"]
        for j in range(processed_logits.shape[0]):
            num_valid_logits = torch.isfinite(processed_logits[j]).sum().item()
            if num_valid_logits == k:
                print(f"  OK [top_k Verification PASSED for batch item {j}]: Found exactly {k} valid logits as expected.")
            else:
                print(f"  ERROR [top_k Verification FAILED for batch item {j}]: Expected {k} valid logits, but found {num_valid_logits}.")
    
    if any(p.get("name") == "temperature" for p in processors_list):
        original_max = logits.max()
        processed_max = processed_logits[torch.isfinite(processed_logits)].max()
        if not torch.isclose(original_max, processed_max):
             print(f"  OK [Temperature Verification PASSED]: Logits were successfully scaled (original max: {original_max:.2f}, new max: {processed_max:.2f}).")
        else:
             print("  ERROR [Temperature Verification FAILED]: Logits were not scaled by temperature.")
    
    print("-" * 30 + "\n")


if __name__ == "__main__":
    top_k_config = {
        "processors": [
            {"name": "top_k", "params": {"top_k": 5}}
        ]
    }
    run_test("Simple Top-K", top_k_config)
    chain_config = {
        "processors": [
            {"name": "temperature", "params": {"temperature": 0.5}},
            {"name": "top_k", "params": {"top_k": 10}}
        ]
    }
    run_test("Chained Temperature and Top-K", chain_config, vocab_size=100)
