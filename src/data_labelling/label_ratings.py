import argparse
import yaml
import json
import random
import math
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Tuple

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

    def _get_pairs(self, round_num: int) -> List[Tuple[str, str]]:
        """Generates pairs for a tournament round."""
        if round_num == 1:
            shuffled_ids = list(self.players.keys())
            random.shuffle(shuffled_ids)
            return [(shuffled_ids[i], shuffled_ids[i+1]) for i in range(0, len(shuffled_ids) -1, 2)]

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
        
        return pairs

    def run_round(self, round_num: int, scorer, prompt: str, generation_map: Dict[str, str]):
        """Runs a single round of the tournament."""
        pairs = self._get_pairs(round_num)
        for p1_id, p2_id in pairs:
            response_a = generation_map[p1_id]
            response_b = generation_map[p2_id]
            score_a, score_b = scorer.compare(prompt, response_a, response_b)
            self.comparison_history.append((p1_id, p2_id, score_a))
            self.players[p1_id]['score'] += score_a
            self.players[p2_id]['score'] += score_b
            self.players[p1_id]['opponents'].add(p2_id)
            self.players[p2_id]['opponents'].add(p1_id)

class BradleyTerryRanker:
    """
    Calculates latent quality ratings from pairwise comparison data using
    gradient descent to optimize a Bradley-Terry model.
    """
    def __init__(self, comparison_history: List[Tuple[str, str, float]]):
        self.history = comparison_history
        self.player_ids = list(set([p[0] for p in history] + [p[1] for p in history]))

    def calculate_ratings(self, iterations: int = 1000, learning_rate: float = 0.05) -> Dict[str, float]:
        """Calculates the final ratings."""
        ratings = {pid: 0.0 for pid in self.player_ids}
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))
        for _ in range(iterations):
            gradients = defaultdict(float)
            for p1_id, p2_id, observed_score in self.history:
                predicted_score = sigmoid(ratings[p1_id] - ratings[p2_id])
                error = observed_score - predicted_score
                gradients[p1_id] += error
                gradients[p2_id] -= error
            for pid in self.player_ids:
                ratings[pid] += learning_rate * gradients[pid]
        mean_rating = sum(ratings.values()) / len(ratings)
        for pid in self.player_ids:
            ratings[pid] -= mean_rating
        return ratings

def main(config_path: str):
    """Main function to run the rating labelling."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Initializing '{config['reward_model_backend']}' reward scorer...")
    scorer = get_scorer(
        backend=config['reward_model_backend'],
        model_name_or_path=config.get("hf_model_name_or_path")
    )
    print("Loading and grouping dataset by prompt...")
    data_by_prompt = defaultdict(list)
    with open(config['data_path'], 'r') as f:
        for line in f:
            item = json.loads(line)
            data_by_prompt[item[config['prompt_key']]].append(item)
    all_annotated_data = []
    pbar = tqdm(data_by_prompt.items(), desc="Processing Prompts")
    for prompt, datapoints in pbar:
        if len(datapoints) < 2:
            all_annotated_data.extend(datapoints)
            continue
        generation_map = {f"{dp[config['heuristic_key']]}_{dp['seed']}": dp[config['generation_key']] for dp in datapoints}
        player_ids = list(generation_map.keys())
        num_rounds = math.ceil(math.log2(len(player_ids))) + config.get('tournament_extra_rounds', 2)
        tournament = SwissTournamentRunner(player_ids)
        for i in range(num_rounds):
            tournament.run_round(i + 1, scorer, prompt, generation_map)
        ranker = BradleyTerryRanker(tournament.comparison_history)
        final_ratings = ranker.calculate_ratings()
        for dp in datapoints:
            player_id = f"{dp[config['heuristic_key']]}_{dp['seed']}"
            if player_id in final_ratings:
                dp['bradley_terry_rating'] = final_ratings[player_id]
            all_annotated_data.append(dp)
    print(f"Writing {len(all_annotated_data)} annotated datapoints to {config['output_path']}...")
    with open(config['output_path'], 'w') as f_out:
        for dp in all_annotated_data:
            f_out.write(json.dumps(dp) + '\n')
    print("Rating labelling complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank individual generations via Swiss tournament and Bradley-Terry model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the rating labelling configuration YAML file.")
    args = parser.parse_args()
    main(args.config)
