import torch
import torch.nn.functional as F
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

class MinPLogitsProcessor(CustomLogitProcessor):
    """
    sglang-native implementation of Min-P Sampling.
    based on transformers.MinPLogitsWarper.
    """

    def __call__(self, logits, custom_param_list):
        min_p_key = "min_p"
        filter_value = -float("Inf")
        min_tokens_to_keep_key = "min_tokens_to_keep"

        assert logits.shape[0] == len(custom_param_list)

        for i, param_dict in enumerate(custom_param_list):
            min_p = param_dict.get(min_p_key, 0.1) # Default from example, adjust as needed
            min_tokens_to_keep = param_dict.get(min_tokens_to_keep_key, 1)
            probs = torch.softmax(logits[i], dim=-1)
            top_probs, _ = probs.max(dim=-1, keepdim=True)
            scaled_min_p = min_p * top_probs
            tokens_to_remove = probs < scaled_min_p
            sorted_indices = torch.argsort(logits[i], descending=True)
            sorted_indices_to_remove = torch.gather(tokens_to_remove, dim=-1, index=sorted_indices)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = False
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits[i] = logits[i].masked_fill(indices_to_remove, filter_value)

        return logits
    
class TypicalLogitsProcessor(CustomLogitProcessor):
    """
    sglang-native implementation of Typical Sampling.
    based on transformers.TypicalLogitsWarper.
    """

    def __call__(self, logits, custom_param_list):
        mass_key = "mass"
        filter_value = -float("Inf")
        min_tokens_to_keep_key = "min_tokens_to_keep"
        
        assert logits.shape[0] == len(custom_param_list)

        for i, param_dict in enumerate(custom_param_list):
            mass = param_dict.get(mass_key, 0.9)
            min_tokens_to_keep = param_dict.get(min_tokens_to_keep_key, 1)
            normalized = F.log_softmax(logits[i], dim=-1)
            p = torch.exp(normalized)
            ent = -(normalized * p).nansum(-1, keepdim=True)
            shifted_scores = torch.abs((-normalized) - ent)
            sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
            sorted_logits = logits[i].gather(-1, sorted_indices)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            last_ind = (cumulative_probs < mass).sum(dim=0)
            last_ind = torch.clamp(last_ind, max=sorted_scores.shape[-1] - 1)
            cutoff_score = sorted_scores.gather(0, last_ind.view(1))[0]
            sorted_indices_to_remove = sorted_scores > cutoff_score
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits[i] = logits[i].masked_fill(indices_to_remove, filter_value)

        return logits
    
class EpsilonLogitsProcessor(CustomLogitProcessor):
    """
    sglang-native implementation of Epsilon Sampling.
    based on transformers.EpsilonLogitsWarper
    """

    def __call__(self, logits, custom_param_list):
        epsilon_key = "epsilon"
        filter_value = -float("Inf")
        min_tokens_to_keep_key = "min_tokens_to_keep"

        assert logits.shape[0] == len(custom_param_list)

        for i, param_dict in enumerate(custom_param_list):
            epsilon = param_dict.get(epsilon_key, 0.01)
            min_tokens_to_keep = param_dict.get(min_tokens_to_keep_key, 1)
            probabilities = logits[i].softmax(dim=-1)
            indices_to_remove = probabilities < epsilon
            top_k = min(min_tokens_to_keep, logits[i].size(-1))
            top_k_values, _ = torch.topk(logits[i], top_k)
            indices_to_remove = indices_to_remove & (logits[i] < top_k_values[..., -1, None])
            logits[i] = logits[i].masked_fill(indices_to_remove, filter_value)

        return logits
    
class EtaLogitsProcessor(CustomLogitProcessor):
    """
    sglang-native implementation of Eta Sampling.
    based on transformers.EtaLogitsWarper
    """
    def __call__(self, logits, custom_param_list):
        epsilon_key = "epsilon"
        filter_value = -float("Inf")
        min_tokens_to_keep_key = "min_tokens_to_keep"
        assert logits.shape[0] == len(custom_param_list)

        for i, param_dict in enumerate(custom_param_list):
            epsilon = param_dict.get(epsilon_key, 3e-4)
            min_tokens_to_keep = param_dict.get(min_tokens_to_keep_key, 1)
            probabilities = logits[i].softmax(dim=-1)
            entropy = torch.distributions.Categorical(logits=logits[i]).entropy()
            eta = torch.min(torch.tensor(epsilon), torch.sqrt(torch.tensor(epsilon)) * torch.exp(-entropy))[..., None]
            indices_to_remove = probabilities < eta
            top_k = min(min_tokens_to_keep, logits[i].size(-1))
            top_k_values, _ = torch.topk(logits[i], top_k)
            indices_to_remove = indices_to_remove & (logits[i] < top_k_values[..., -1, None])

            logits[i] = logits[i].masked_fill(indices_to_remove, filter_value)
        return logits
