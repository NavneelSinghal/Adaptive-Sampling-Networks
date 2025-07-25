import torch
import yaml
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from src.models import SamplingNetwork, LocalProbabilityTransform, SimpleDistributionAwareTransform
from cachetools import LRUCache
from collections import OrderedDict
import gc
from transformers.generation.logits_process import (
    TopKLogitsWarper,
    TopPLogitsWarper,
    TemperatureLogitsWarper,
    MinPLogitsWarper,
    TypicalLogitsWarper,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
)

class ModelLRUCache(LRUCache):
    def popitem(self):
        key, value = super().popitem()
        del value
        gc.collect()
        torch.cuda.empty_cache()
        return key, value

_SAMPLER_CACHE_SIZE = 16
_GLOBAL_SAMPLER_CACHE = ModelLRUCache(maxsize=_SAMPLER_CACHE_SIZE)

PROCESSOR_MAP = {
    "top_k": TopKLogitsWarper,
    "top_p": TopPLogitsWarper,
    "temperature": TemperatureLogitsWarper,
    "min_p": MinPLogitsWarper,
    "typical": TypicalLogitsWarper,
    "epsilon": EpsilonLogitsWarper,
    "eta": EtaLogitsWarper,
}

SAMPLER_MODELS = {
    "SamplingNetwork": SamplingNetwork,
    "LocalProbabilityTransform": LocalProbabilityTransform,
    "SimpleDistributionAwareTransform": SimpleDistributionAwareTransform,
}

class _AdaptiveSamplerWrapper:
    def __init__(self, sampler_model_name: str, sampler_config_path: str, sampler_checkpoint_path: str, device: str = "cuda:0"):
        self.sampler_model_name = sampler_model_name
        self.sampler_config_path = sampler_config_path
        self.sampler_checkpoint_path = sampler_checkpoint_path
        self.device = device
        self.sampler_model = None

    def _load_model_if_needed(self):
        if self.sampler_checkpoint_path in _GLOBAL_SAMPLER_CACHE:
            self.sampler_model = _GLOBAL_SAMPLER_CACHE[self.sampler_checkpoint_path]
        else:
            print(f"INFO: Caching new sampler model from: {self.sampler_checkpoint_path}")
            with open(self.sampler_config_path, 'r') as f:
                config = yaml.safe_load(f)

            model_class = SAMPLER_MODELS[self.sampler_model_name]
            model = model_class(dtype=torch.bfloat16, **config)

            state_dict = torch.load(self.sampler_checkpoint_path, map_location=self.device)
            from collections import OrderedDict
            new_state_dict = OrderedDict((k[10:] if k.startswith('_orig_mod.') else k, v) for k, v in state_dict.items())
            model.load_state_dict(new_state_dict)
            model.to(self.device).eval()
            self.sampler_model = torch.compile(model)
            _GLOBAL_SAMPLER_CACHE[self.sampler_checkpoint_path] = self.sampler_model

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        if self.sampler_model is None:
            self._load_model_if_needed()
        logits_for_sampler = scores.to(self.device, dtype=torch.bfloat16)
        modified_logits = self.sampler_model(logits_for_sampler)
        return modified_logits.to(scores.device, dtype=scores.dtype)

class PipelineLogitsProcessor(CustomLogitProcessor):
    def __init__(self, pipeline_config: list[dict], **kwargs):
        super().__init__(**kwargs)
        self.pipeline = []
        for conf in pipeline_config:
            name = conf.get("name")
            params = conf.get("params", {})
            if name == "adaptive_sampler":
                self.pipeline.append(_AdaptiveSamplerWrapper(**params))
            elif name in PROCESSOR_MAP:
                self.pipeline.append(PROCESSOR_MAP[name](**params))
            else:
                print(f"WARNING: Unknown processor name '{name}' in pipeline config. Skipping.")

    def __call__(self, logits=None, custom_param_list=None):
        if logits is None and custom_param_list is None:
            return self
        dummy_input_ids = None
        for processor in self.pipeline:
            logits = processor(input_ids=dummy_input_ids, scores=logits)
        return logits
