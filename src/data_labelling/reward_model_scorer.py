import abc
import requests
import torch
import math
import openai # Added import
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

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
        raise NotImplementedError
    
    @abc.abstractmethod
    def compare_batch(self, triplets: List[tuple[str, str, str]]) -> List[tuple[float, float]]:
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
            [self.tokenizer.convert_tokens_to_ids(item) for item in target_choices],
            device=self.device
        )

    @torch.no_grad()
    def compare(self, prompt: str, response_a: str, response_b: str):
        return self.compare_batch([(prompt, response_a, response_b)])[0]

    @torch.no_grad()
    def compare_batch(self, triplets: List[tuple[str, str, str]]) -> List[tuple[float, float]]:
        all_prompts = []
        for prompt, response_a, response_b in triplets:
            all_prompts.append(self._PROMPT_TEMPLATE.format(input=prompt, response_a=response_a, response_b=response_b))
            all_prompts.append(self._PROMPT_TEMPLATE.format(input=prompt, response_a=response_b, response_b=response_a))

        chat_prompts = [
            self.tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in all_prompts
        ]
        
        inputs = self.tokenizer(chat_prompts, return_tensors="pt", padding=True).to(self.device)
        logits = self.model(**inputs).logits[:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        final_scores = []
        for i in range(len(triplets)):
            log_probs1 = log_probs[2*i]
            log_probs2 = log_probs[2*i + 1]

            logprob_a1 = log_probs1[self._target_token_ids[0]].item()
            logprob_b1 = log_probs1[self._target_token_ids[1]].item()
            
            logprob_a2 = log_probs2[self._target_token_ids[1]].item()
            logprob_b2 = log_probs2[self._target_token_ids[0]].item()

            prob_a1, prob_b1 = math.exp(logprob_a1), math.exp(logprob_b1)
            prob_a2, prob_b2 = math.exp(logprob_a2), math.exp(logprob_b2)

            norm1 = prob_a1 + prob_b1
            norm2 = prob_a2 + prob_b2

            p_a_prompt1 = prob_a1 / norm1 if norm1 > 1e-9 else 0.5
            p_a_prompt2 = prob_a2 / norm2 if norm2 > 1e-9 else 0.5
            p_b_prompt1 = prob_b1 / norm1 if norm1 > 1e-9 else 0.5
            p_b_prompt2 = prob_b2 / norm2 if norm2 > 1e-9 else 0.5

            avg_prob_a = (p_a_prompt1 + p_a_prompt2) / 2
            avg_prob_b = (p_b_prompt1 + p_b_prompt2) / 2
            
            denominator = max(1e-9, avg_prob_a + avg_prob_b)
            score_a = avg_prob_a / denominator
            score_b = avg_prob_b / denominator
            
            final_scores.append((score_a, score_b))
            
        return final_scores

class SGLangRewardScorer(BaseRewardScorer):
    """
    A reward scorer that sends requests to a running SGLang server
    using the OpenAI-compatible API.
    """
    def __init__(self, url: str, model_name_or_path: str):
        self.url = url
        self.model_name = 'reward-model'

        self.client = openai.OpenAI(
            base_url=self.url,
            api_key="sglang"
        )
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.token_id_A = self.tokenizer.convert_tokens_to_ids("A")
            self.token_id_B = self.tokenizer.convert_tokens_to_ids("B")
            self.str_token_id_A = str(self.token_id_A)
            self.str_token_id_B = str(self.token_id_B)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer for SGLang client: {e}")

        print(f"SGLang scorer initialized. Endpoint: {self.url}. Token IDs: A={self.token_id_A}, B={self.token_id_B}")

    def _send_single_request(self, prompt_text: str) -> dict:
        """Sends one request to the SGLang server using the openai client."""
        error_logprobs = {self.str_token_id_A: -1e9, self.str_token_id_B: -1e9}

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=1.0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=15
            )

            logprobs_content = response.choices[0].logprobs.content
            if not logprobs_content or not logprobs_content[0].top_logprobs:
                return error_logprobs

            top_logprobs = logprobs_content[0].top_logprobs
            
            logprob_dict = {
                str(self.tokenizer.convert_tokens_to_ids(lp.token)): lp.logprob
                for lp in top_logprobs
            }
            return logprob_dict

        except Exception as e:
            print(f"\n[ERROR] Request failed. Details: {e}")
            return error_logprobs

    def compare(self, prompt: str, response_a: str, response_b: str) -> tuple[float, float]:
        """Runs a single comparison."""
        return self.compare_batch([(prompt, response_a, response_b)])[0]

    def compare_batch(self, triplets: List[tuple[str, str, str]], max_workers: int = 16) -> List[tuple[float, float]]:
        """
        Performs batch comparison using a ThreadPoolExecutor.
        """
        all_formatted_prompts = []
        for prompt, response_a, response_b in triplets:
            msg1 = [{"role": "user", "content": self._PROMPT_TEMPLATE.format(input=prompt, response_a=response_a, response_b=response_b)}]
            msg2 = [{"role": "user", "content": self._PROMPT_TEMPLATE.format(input=prompt, response_a=response_b, response_b=response_a)}]
            all_formatted_prompts.append(self.tokenizer.apply_chat_template(msg1, tokenize=False, add_generation_prompt=True))
            all_formatted_prompts.append(self.tokenizer.apply_chat_template(msg2, tokenize=False, add_generation_prompt=True))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self._send_single_request, all_formatted_prompts))

        final_scores = []
        if len(results) != len(all_formatted_prompts):
            raise RuntimeError("Mismatch between sent requests and received results.")

        for i in range(0, len(results), 2):
            logprobs_a_first = results[i]
            logprobs_b_first = results[i+1]
            
            logprob_a1 = logprobs_a_first.get(self.str_token_id_A, -1e9)
            logprob_b1 = logprobs_a_first.get(self.str_token_id_B, -1e9)

            logprob_b2 = logprobs_b_first.get(self.str_token_id_A, -1e9)
            logprob_a2 = logprobs_b_first.get(self.str_token_id_B, -1e9)

            prob_a1, prob_b1 = math.exp(logprob_a1), math.exp(logprob_b1)
            prob_a2, prob_b2 = math.exp(logprob_a2), math.exp(logprob_b2)

            norm1 = prob_a1 + prob_b1
            norm2 = prob_a2 + prob_b2

            p_a_prompt1 = prob_a1 / norm1 if norm1 > 1e-9 else 0.5
            p_a_prompt2 = prob_a2 / norm2 if norm2 > 1e-9 else 0.5
            p_b_prompt1 = prob_b1 / norm1 if norm1 > 1e-9 else 0.5
            p_b_prompt2 = prob_b2 / norm2 if norm2 > 1e-9 else 0.5

            avg_prob_a = (p_a_prompt1 + p_a_prompt2) / 2
            avg_prob_b = (p_b_prompt1 + p_b_prompt2) / 2
            
            denominator = max(1e-9, avg_prob_a + avg_prob_b)
            score_a = avg_prob_a / denominator
            score_b = avg_prob_b / denominator

            final_scores.append((score_a, score_b))

        return final_scores


def get_scorer(backend: str, **kwargs) -> BaseRewardScorer:
    """Factory function to get an instance of a reward scorer."""
    if backend == "hf":
        return HFRewardScorer(model_name_or_path=kwargs.get("model_name_or_path"), device=kwargs.get("device", "auto"))
    elif backend == "sglang":
        return SGLangRewardScorer(url=kwargs.get("url"), model_name_or_path=kwargs.get("model_name_or_path"))
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'hf' or 'sglang'.")
