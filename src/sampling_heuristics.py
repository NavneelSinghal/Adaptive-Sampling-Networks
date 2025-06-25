import yaml
import random
import torch
from typing import List, Dict, Any, Tuple
from transformers.generation.logits_process import (
    LogitsProcessor,
    LogitsProcessorList,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    MinPLogitsWarper,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    TemperatureLogitsWarper,
)

def _create_processor_list_from_config(
    pipeline_config: List[Dict[str, Any]]
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    for config in pipeline_config:
        name = config.get("name")
        params = config.get("params", {})
        if name == "temperature":
            processor = TemperatureLogitsWarper(**params)
        elif name == "top_k":
            processor = TopKLogitsWarper(**params)
        elif name == "top_p":
            processor = TopPLogitsWarper(**params)
        elif name == "typical":
            processor = TypicalLogitsWarper(**params)
        elif name == "min_p":
            processor = MinPLogitsWarper(**params)
        elif name == "epsilon":
            processor = EpsilonLogitsWarper(**params)
        elif name == "eta":
            processor = EtaLogitsWarper(**params)
        else:
            raise ValueError(f"Unknown heuristic name in pipeline: {name}")
        processor_list.append(processor)
    return processor_list


def apply_pipeline_to_logits(
    input_ids: torch.LongTensor,
    logits: torch.Tensor,
    pipeline_config: List[Dict[str, Any]]
) -> torch.Tensor:
    processor_list = _create_processor_list_from_config(pipeline_config)
    return processor_list(input_ids, logits)


def get_random_pipeline_for_generation(
    config
) -> Tuple[LogitsProcessorList, Dict[str, Any]]:

    pipelines = config.get("sampling_pipelines", [])
    if not pipelines:
        raise ValueError("Config must have a 'sampling_pipelines' key.")

    weights = [p.get('weight', 0) for p in pipelines]
    chosen_pipeline = random.choices(pipelines, weights=weights, k=1)[0]
    
    pipeline_config = chosen_pipeline.get("processors", [])
    processor_list = _create_processor_list_from_config(pipeline_config)

    sampler_info = {
        "name": chosen_pipeline.get("name"),
        "config": pipeline_config
    }
    
    return processor_list, sampler_info
