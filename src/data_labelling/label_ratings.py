import argparse
import yaml
import json
import random
import math
import multiprocessing
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict

from reward_model_scorer import get_scorer

class SwissTournamentRunner:
    """
    Manages and executes a Swiss-style tournament for a given set of players.
    """
    def __init__(self, player_ids: List[str]):
        if not player_ids or len(player_ids) < 2:
            raise ValueError("Player IDs must contain at least two unique items.")
        
        self.players = {pid: {"id": pid, "score": 0.0, "opponents": set()} for pid in player_ids}
        self.comparison_history = []

    def _get_pairs(self, round_num: int) -> tuple[List[tuple[str, str]], str | None]:
        """
        Generates pairs for a tournament round.
        Returns a tuple containing the list of pairs and the ID of any player receiving a bye.
        """
        bye_player = None
        if round_num == 1:
            shuffled_ids = list(self.players.keys())
            random.shuffle(shuffled_ids)
            if len(shuffled_ids) % 2 != 0:
                bye_player = shuffled_ids.pop()
            pairs = [(shuffled_ids[i], shuffled_ids[i+1]) for i in range(0, len(shuffled_ids), 2)]
            return pairs, bye_player

        sorted_players = sorted(self.players.values(), key=lambda p: p['score'], reverse=True)
        to_be_paired = [p['id'] for p in sorted_players]
        pairs = []

        while len(to_be_paired) >= 2:
            p1_id = to_be_paired.pop(0)
            best_opponent_idx = -1
            for i, p2_id in enumerate(to_be_paired):
                if p2_id not in self.players[p1_id]['opponents']:
                    best_opponent_idx = i
                    break
            
            if best_opponent_idx != -1:
                p2_id = to_be_paired.pop(best_opponent_idx)
                pairs.append((p1_id, p2_id))
            elif to_be_paired:
                p2_id = to_be_paired.pop(0)
                pairs.append((p1_id, p2_id))
        
        if to_be_paired:
            bye_player = to_be_paired[0]
        return pairs, bye_player

    def run_round(self, round_num: int, scorer, prompt: str, generation_map: Dict[str, str]):
        """
        Runs a single tournament round by preparing all pairs and sending them
        as a single batch to the scorer.
        """
        pairs, bye_player_id = self._get_pairs(round_num)
        
        if pairs:
            comparison_triplets = [
                (prompt, generation_map[p1_id], generation_map[p2_id])
                for p1_id, p2_id in pairs
            ]
            
            batch_scores = scorer.compare_batch(comparison_triplets)

            for i, (p1_id, p2_id) in enumerate(pairs):
                score_a, score_b = batch_scores[i]
                self.comparison_history.append((p1_id, p2_id, score_a))
                self.players[p1_id]['score'] += score_a
                self.players[p2_id]['score'] += score_b
                self.players[p1_id]['opponents'].add(p2_id)
                self.players[p2_id]['opponents'].add(p1_id)

        if bye_player_id:
            self.players[bye_player_id]['score'] += 0.5

class BradleyTerryRanker:
    """
    Calculates latent quality ratings from pairwise comparison data using
    gradient descent to optimize a Bradley-Terry model.
    """
    def __init__(self, comparison_history: List[tuple[str, str, float]]):
        self.history = comparison_history
        self.player_ids = list(set([p[0] for p in self.history] + [p[1] for p in self.history]))

    def calculate_ratings(self, iterations: int = 1000, learning_rate: float = 0.05) -> Dict[str, float]:
        """Calculates the final ratings."""
        ratings = {pid: 0.0 for pid in self.player_ids}
        def sigmoid(x):
            x = max(-35.0, min(35.0, x))
            return 1 / (1 + math.exp(-x))
        for _ in range(iterations):
            gradients = defaultdict(float)
            for p1_id, p2_id, observed_score in self.history:
                if p1_id not in ratings or p2_id not in ratings: continue
                predicted_score = sigmoid(ratings[p1_id] - ratings[p2_id])
                error = observed_score - predicted_score
                gradients[p1_id] += error
                gradients[p2_id] -= error
            for pid in ratings:
                ratings[pid] += learning_rate * gradients.get(pid, 0.0)
        if not ratings: return {}
        mean_rating = sum(ratings.values()) / len(ratings)
        for pid in ratings:
            ratings[pid] -= mean_rating
        return ratings

def process_job(args: tuple[tuple[str, int, List[Dict]], Dict]):
    """
    Worker function for multiprocessing.
    This function is a lightweight client that sends requests to the SGLang server.
    """
    (prompt, seed, datapoints), config = args
    
    scorer = get_scorer(
        backend=config['reward_model_backend'],
        url=config.get("sglang_url"),
        model_name_or_path=config.get("model_name_or_path")
    )

    if len(datapoints) < 2:
        return datapoints
        
    generation_map = {f"{json.dumps(dp[config['heuristic_key']], sort_keys=True)}_{dp['seed']}": dp[config['generation_key']] for dp in datapoints}
    player_ids = list(generation_map.keys())
    
    num_rounds = math.ceil(math.log2(len(player_ids))) + config.get('tournament_extra_rounds', 2)
    
    tournament = SwissTournamentRunner(player_ids)
    for i in range(num_rounds):
        tournament.run_round(i + 1, scorer, prompt, generation_map)
        
    ranker = BradleyTerryRanker(tournament.comparison_history)
    final_ratings = ranker.calculate_ratings()
    
    for dp in datapoints:
        player_id = f"{json.dumps(dp[config['heuristic_key']], sort_keys=True)}_{dp['seed']}"
        if player_id in final_ratings:
            dp['bradley_terry_rating'] = final_ratings[player_id]
            
    return datapoints

def main(config_path: str):
    """Main function to run the rating labelling in parallel using SGLang."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("Loading and grouping dataset by prompt and seed...")
    data_by_prompt_and_seed = defaultdict(lambda: defaultdict(list))
    total_size = 0
    with open(config['data_path'], 'r') as f:
        for line in f:
            item = json.loads(line)
            prompt = item[config['prompt_key']]
            seed = item['seed']
            data_by_prompt_and_seed[prompt][seed].append(item)
            total_size += 1
    jobs = []
    for prompt, seed_groups in data_by_prompt_and_seed.items():
        for seed, datapoints in seed_groups.items():
            jobs.append((prompt, seed, datapoints))
            
    worker_args = [(job, config) for job in jobs]
    num_workers = min(16, multiprocessing.cpu_count() - 1)

    print(f"Processing ~{total_size} datapoints across {len(jobs)} jobs using {num_workers} parallel workers.")
    
    all_annotated_data = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        pbar = tqdm(pool.imap_unordered(process_job, worker_args), total=len(jobs))
        for annotated_group in pbar:
            all_annotated_data.extend(annotated_group)

    print(f"Writing {len(all_annotated_data)} annotated datapoints to {config['output_path']}...")
    with open(config['output_path'], 'w') as f_out:
        for item in all_annotated_data:
            f_out.write(json.dumps(item) + '\n')

    print("Rating labelling complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank individual generations via Swiss tournament and Bradley-Terry model using an SGLang backend.")
    parser.add_argument("--config", type=str, required=True, help="Path to the rating labelling configuration YAML file.")
    args = parser.parse_args()
    multiprocessing.set_start_method('spawn', force=True)
    main(args.config)
