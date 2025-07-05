import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from transformers.generation.logits_process import (
    TopKLogitsWarper,
    TopPLogitsWarper,
    TemperatureLogitsWarper,
    MinPLogitsWarper,
    TypicalLogitsWarper,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
)

PROCESSOR_MAP = {
    "top_k": TopKLogitsWarper,
    "top_p": TopPLogitsWarper,
    "temperature": TemperatureLogitsWarper,
    "min_p": MinPLogitsWarper,
    "typical": TypicalLogitsWarper,
    "epsilon": EpsilonLogitsWarper,
    "eta": EtaLogitsWarper,
}

class PipelineLogitsProcessor(CustomLogitProcessor):
    def __init__(self, pipeline_config: list[dict], **kwargs):
        super().__init__(**kwargs)
        self.pipeline = []
        for conf in pipeline_config:
            name = conf.get("name")
            params = conf.get("params", {})
            if name in PROCESSOR_MAP:
                self.pipeline.append(PROCESSOR_MAP[name](**params))

    def __call__(self, logits=None, custom_param_list=None):
        if logits is None and custom_param_list is None:
            return self
        dummy_input_ids = None
        for processor in self.pipeline:
            logits = processor(input_ids=dummy_input_ids, scores=logits)
        return logits
