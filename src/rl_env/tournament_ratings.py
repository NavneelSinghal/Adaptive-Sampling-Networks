import src.data_labelling.label_ratings import process_job
from typing import List

def compute_unnormalized_tournament_ratings(prompt: str, generations: List[dict], config: dict):
    temp_seed_set = False
    if generations and generations[0].get('seed') is None:
        temp_seed_set = True
        for i, x in enumerate(generations):
            x['seed'] = i
    generations = process_job(((prompt, seed, generations), config))
    if temp_seed_set:
        for i, x in enumerate(generations):
            del x['seed']
    return generations
