import abc
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

class BaseRewardScorer(abc.ABC):
    """Abstract base class for reward scorers."""
    
    _PROMPT_TEMPLATE = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better.
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.
Please directly output your final verdict by strictly following this format: "A" if assistant A is better, "B" if assistant B is better.
[User Question]
{input}
[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]
[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]
"""

    @abc.abstractmethod
    def compare(self, prompt: str, response_a: str, response_b: str):
        """
        Compares two responses for a given prompt and returns the scores of A and B

        Args:
            prompt: The user's input/question.
            response_a: The first assistant's response.
            response_b: The second assistant's response.

        Returns:
            'A' if response_a is preferred, 'B' if response_b is preferred.
        """
        raise NotImplementedError

class HFRewardScorer(BaseRewardScorer):
    """
    A reward scorer that uses a local Hugging Face transformers model 
    to perform pairwise comparison.
    """
    def __init__(self, model_name_or_path: str, device: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "left"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.bfloat16, 
            device_map=device
        )
        self.device = self.model.device
        target_choices = ["A", "B"]
        self._target_token_ids = torch.tensor(
            [self.tokenizer(item, add_special_tokens=False).input_ids for item in target_choices],
            device=self.device
        )

    @torch.no_grad()
    def compare(self, prompt: str, response_a: str, response_b: str):
        messages = [
            [{"role": "user", "content": self._PROMPT_TEMPLATE.format(input=prompt, response_a=response_a, response_b=response_b)}],
            [{"role": "user", "content": self._PROMPT_TEMPLATE.format(input=prompt, response_a=response_b, response_b=response_a)}],
        ]
        target_choices_token_ids = torch.cat(
            (self._target_token_ids, torch.flip(self._target_token_ids, dims=(0,))), 
            dim=1
        )
        full_prompts = [self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        inputs = self.tokenizer(full_prompts, return_tensors="pt", padding=True).to(self.device)
        output = self.model(**inputs)
        logits = torch.gather(output.logits[:, -1, :], 1, target_choices_token_ids)
        probabilities = torch.nn.Softmax(dim=0)(logits)
        score_a, score_b = torch.mean(probabilities, dim=1).tolist()
        denominator = max(1e-9, score_a + score_b)
        return score_a / denominator, score_b / denominator

class SGLangRewardScorer(BaseRewardScorer):
    """
    A reward scorer that sends requests to a running SGLang server
    to perform pairwise comparison. (Not yet implemented)
    """
    def __init__(self, url: str):
        self.url = url
        # TODO
        raise NotImplementedError("SGLangRewardScorer is not yet implemented.")

    def compare(self, prompt: str, response_a: str, response_b: str):
        # TODO: Make a request to the model loaded in SGLang and return scores based on logits.
        pass

def get_scorer(backend: str, **kwargs) -> BaseRewardScorer:
    """
    Factory function to get an instance of a reward scorer.
    
    Args:
        backend: The desired backend, either "hf" or "sglang".
        **kwargs: Arguments to be passed to the scorer's constructor.
    
    Returns:
        An instance of a BaseRewardScorer subclass.
    """
    if backend == "hf":
        return HFRewardScorer(model_name_or_path=kwargs.get("model_name_or_path"))
    elif backend == "sglang":
        return SGLangRewardScorer(url=kwargs.get("url"))
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'hf' or 'sglang'.")
