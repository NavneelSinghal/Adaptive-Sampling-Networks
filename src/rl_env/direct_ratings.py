from src.data_labelling.direct_scorer import score_and_update_batch
from typing import List

def compute_unnormalized_direct_ratings(generations: List[dict], model_path: str, urls: List[str], tokenizer: List[str], worker_id: int):
    return score_and_update_batch((generations, model_path, urls, tokenizer, worker_id))
