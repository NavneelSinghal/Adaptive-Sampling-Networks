import src.data_labelling.efficient_diversity as diversity
from typing import List

def compute_unnormalized_diversity_reward_self_bleu(generations: List[dict], group_size: int, log_transform: bool = True) -> None:
    for i in range(0, len(generations), group_size):
        group = generations[i : i + group_size]
        reward = -diversity.calculate_self_bleu([x['generation'] for x in group], log_transform)
        for x in group:
            x['unnormalized_diversity_reward_self_bleu'] = reward

def compute_unnormalized_diversity_reward_embedding_entropy(generations: List[dict], group_size: int, log_transform: bool = True) -> None:
    for i in range(0, len(generations), group_size):
        group = generations[i : i + group_size]
        reward = diversity.calculate_embedding_entropy([x['generation'] for x in group], diversity.embedding_model, log_transform)
        for x in group:
            x['unnormalized_diversity_reward_embedding_entropy'] = reward
