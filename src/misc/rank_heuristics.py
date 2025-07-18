import json
import argparse
import os
from collections import defaultdict

def analyze_sampler_performance(data: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    accumulator = defaultdict(lambda: {"entropy_sum": 0.0, "reward_sum": 0.0, "quality_sum": 0.0, "count": 0})

    for item in data:
        sampler_info = item.get('sampler_info')
        diversity_info = item.get('diversity', {})
        reward = item.get('verifiable_reward')
        quality = item.get('bradley_terry_rating')

        if not sampler_info or not isinstance(diversity_info, dict) or reward is None or quality is None:
            continue

        entropy = diversity_info.get('embedding_entropy')
        if entropy is None:
            continue
            
        sampler_key = json.dumps(sampler_info, sort_keys=True)
        
        accumulator[sampler_key]["entropy_sum"] += entropy
        accumulator[sampler_key]["reward_sum"] += reward
        accumulator[sampler_key]["quality_sum"] += quality
        accumulator[sampler_key]["count"] += 1

    results_list = [
        {
            "sampler_info": json.loads(key),
            "total_embedding_entropy": stats["entropy_sum"],
            "total_verifiable_reward": stats["reward_sum"],
            "total_quality": stats["quality_sum"],
            "sample_count": stats["count"]
        }
        for key, stats in accumulator.items()
    ]

    entropy_ranking = sorted(
        results_list,
        key=lambda x: x["total_embedding_entropy"],
        reverse=True
    )
    
    reward_ranking = sorted(
        results_list,
        key=lambda x: x["total_verifiable_reward"],
        reverse=True
    )

    quality_ranking = sorted(
        results_list,
        key=lambda x: x["total_quality"],
        reverse=True
    )

    return entropy_ranking, reward_ranking, quality_ranking

def print_ranking(title: str, ranking_data: list[dict], score_key: str):
    print(f"## {title}")
    print("-" * 40)
    if not ranking_data:
        print("No data to rank.")
        return
    for i, stats in enumerate(ranking_data):
        sampler_name = stats["sampler_info"].get("name", "N/A")
        score = stats[score_key]
        count = stats["sample_count"]
        print(f"Rank {i+1}: {sampler_name}")
        print(f"  - Total Score: {score:.4f}")
        print(f"  - Sample Count: {count}")
    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Rank sampler configurations from a .jsonl file based on accumulated scores."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input .jsonl file."
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found at {args.input_file}")
        return

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading or parsing {args.input_file}: {e}")
        return
        
    entropy_ranking, reward_ranking, quality_ranking = analyze_sampler_performance(data)

    print_ranking(
        "Ranking by Total Embedding Entropy",
        entropy_ranking,
        "total_embedding_entropy"
    )
    
    print_ranking(
        "Ranking by Total Verifiable Reward",
        reward_ranking,
        "total_verifiable_reward"
    )

    print_ranking(
        "Ranking by Total Quality",
        quality_ranking,
        "total_quality"
    )

if __name__ == "__main__":
    main()
