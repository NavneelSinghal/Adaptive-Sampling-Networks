# Adaptive Sampling Networks

This repository contains the code and implementation for training and analyzing adapting sampling networks, a collection of lightweight, permutation-equivariant neural networks designed to learn and potentially replace traditional heuristic-based sampling methods for Large Language Models.

The core idea is to train a small, efficient "sampler" model that takes the raw output logits from a frozen, pre-trained LLM and transforms them into a new distribution. This learned transformation can emulate and potentially combine widely used sampling strategies in a single forward pass, without having to do any external scaffolding on top.

## Key Features

  * **Learned Sampling**: Moves beyond hand-tuned heuristics to a learnable algorithm for sampling.
  * **Reasonable Efficiency**: The sampler models are designed to be small and fast, adding minimal overhead (\< 1ms) to the generation process.
  * **Permutation-Equivariant by Design**: The models treat the vocabulary as an unordered set, ensuring they learn general distribution transformations rather than memorizing token-specific information.
  * **Data Curation**: Features a pipeline for generating candidate texts and labeling them for quality, diversity, and similarity to datasets to create high-quality training data.
  * **Supervised Training**: The samplers are trained via supervised learning to mimic the output distributions of complex, high-quality sampling pipelines.
  * **Analysis Tools**: Includes scripts to visualize and compare the learned sampler's output distribution against the original and target distributions.

## Planned Features

  * **Reinforcement learning**: Setting up rewards for such metrics (diversity, coherence, factuality, subjective quality and so on) can be looked into, with standard loss functions (and not direct logit-matching as done for the above methods). Penalties like KL divergence with some older checkpoint might help stabilize training.
  * **Multi-layer inputs**: This can be augmented with tuned-lens or similar decoding methods applied on hidden states from intermediate layers. There is research that suggests that this can improve robustness (for instance, decoding by contrasting layers improves factuality in language models).
  * **Context-based inputs**: By training a lightweight permutation-invariant recurrent model or another approach of this sort, we could try implementing architectures that can express (and learn) stateful algorithms like Mirostat, keeping in mind that metrics over the whole context give us more information that can be used to better pick the next token.
  * **Interpretability**: Understanding what kinds of functions the model learns, and whether we can, reliably and across models, reproduce part of the performance gains, if any, through a simpler algorithmic sampling strategy.

## How It Works

The project aims to create a self-contained, efficient, and expressive neural network which takes raw logits from a base language model and produces a modified set of logits suitable for sampling. The network internally learns to apply all transformations, including value-based modifications and dynamic soft truncation, without requiring external scaffolding like hard cutoffs.

The current workflow involves generating a diverse set of candidate responses, scoring and ranking them to identify the best-performing sampling strategies, and then training the sampler network to mimic those strategies.

### 1\. Candidate Generation

Instead of a simple data generation script, the repository uses a parallelized pipeline to generate a large set of candidate responses.

1.  A high-throughput `SGLang` server is launched to serve the base LLM. Multiple servers can be run in parallel for even greater throughput.
2.  The `src/rejection_sampling/generate_candidates_multi.py` script reads prompts from various source datasets (defined in `configs/data_generation/dataset_sources.yaml`).
3.  For each prompt, it generates multiple responses, each using a different sampling heuristic from a broad configuration file (e.g., `configs/data_generation/generated_config_llama3.2_3b.yaml`).
4.  This process creates a rich dataset where each prompt is associated with dozens of generations, each tagged with the specific heuristic used.

### 2\. Data Annotation & Labeling

Once candidates are generated, they are annotated with objective scores to identify the highest-quality outputs. This is a crucial step for creating the "target" data for the sampler model.

1.  **Quality Ranking**: The `src/data_labelling/label_ratings.py` script uses a reward model (e.g., `GRAM-LLaMA3.2-3B-RewardModel`) to conduct a Swiss-style tournament for the generations associated with each prompt. It then uses a Bradley-Terry model to calculate a latent quality rating for each generation. This effectively ranks the different sampling heuristics for a given prompt.
2.  **Diversity Scoring**: `src/data_labelling/efficient_diversity.py` calculates `self-BLEU` and `embedding_entropy` scores to measure the diversity of generations produced by each heuristic.
3.  **Data Matching**: `src/data_labelling/label_infinigram.py` uses the Infini-gram API to check for n-gram overlap with a large reference corpus, to match the quality of generations with
    that of high-quality datasets.

### 3\. Supervised Training

With the annotated dataset, the `train_supervised.py` script trains the sampler model. The data can be filtered to only include generations from the top-ranked heuristics.

1.  For each sample in the dataset, the script replays the generation to get the base model's raw logits at each step.
2.  It then applies the saved heuristic pipeline to these raw logits to compute the "target" logits. The target logits are typically sparse, with many values set to `-inf` where tokens have been filtered out.
3.  The sampler model takes the raw logits and produces its own "predicted" logits.
4.  The goal is to make the predicted logits match the target logits. This is achieved by minimizing a custom `TruncatedLogitsLoss` function, which has two components:
      * **KL Divergence**: Pushes the sampler's predicted probability distribution to match the target distribution for the tokens that were *not* filtered out. It still penalizes the tokens which were filtered out, owing to the normalization of probabilities required. In particular, the KL divergence of a truncated and the original non-truncated distribution is $- \log m$ where $m$ is the mass of the surviving tokens.
      * **Truncation Penalty**: Penalizes the sampler for assigning any probability mass to tokens that were filtered out (i.e., where the target logit is `-inf`).

The loss is defined as:
$$\mathcal{L} = D_{KL}(P_{\text{target}} || P_{\text{pred}}) + \gamma \sum_{i \in \text{truncated}} P_{\text{pred}, i}$$

## Model Architectures

There are three different sampler architectures of increasing complexity defined in `src/models.py`.

1.  **`LocalProbabilityTransform`**:

  * **Description**: The simplest model. It applies a learned transformation to each log-probability value independently using a small MLP. It also learns a soft truncation gate.
  * **Can Learn**: Simple transformations like temperature scaling, epsilon sampling, and polynomial functions of the log-probabilities.

2.  **`SimpleDistributionAwareTransform`**:

  * **Description**: This model improves the expressivity of the local transform by incorporating global statistics from the probability distribution, specifically the maximum log-probability and the distribution's entropy. Can be extended by adding more layers and/or adding more pooling functions (e.g. higher order moments of log-probabilities).
  * **Can Learn**: More complex strategies that depend on the overall "shape" of the distribution, such as min-p and eta sampling, in addition to whatever the `LocalProbabilityTransform` can learn. It can also learn to make its soft truncation decisions (how aggressively to filter) dynamic based on the distribution's entropy for instance.

3.  **`SamplingNetwork`**:

  * **Description**: The most expressive model. It uses a general architecture with linear attention to model vocabulary-wide interactions. This allows it to construct a large class of arbitrary permutation-equivariant functions on the tokens that can be used for more complex sampling.
  * **Can Learn**: Highly complex sampling strategies that can, for instance, approximate top-k and top-p (which require sorting) and typical sampling, in addition to all the strategies learnable by the simpler models above. The permutation-equivariant design ensures that there's no (or at least minimal) token-level knowledge being learnt by the sampling model.

## Structure

```
.
├── src/
│   ├── _sglang/                          # SGLang server integration and custom processors.
│   │   ├── generate_rollouts_parallel.py # Client for parallel generation.
│   │   └── run_sglang_server*.sh         # Scripts to launch SGLang servers.
│   ├── rejection_sampling/               # Advanced candidate generation pipeline.
│   │   ├── generate_candidates_multi.py  # Main script for generating candidates in parallel.
│   │   └── run_on_multiple_servers.sh    # Script to launch multiple servers across GPUs.
│   ├── data_labelling/                   # Scripts for annotating generated data.
│   │   ├── label_ratings.py              # Ranks generations using a reward model tournament.
│   │   ├── efficient_diversity.py        # Labels for self-BLEU and embedding entropy.
│   │   └── label_infinigram.py           # Labels for data-matching using Infini-gram API.
│   ├── analyze.py                        # Script to analyze and compare a trained sampler.
│   ├── generate_data.py                  # (Legacy) Simple script to generate training data.
│   ├── train_supervised.py               # Main training script for the sampler models.
│   ├── models.py                         # Definitions for the sampler model architectures.
│   ├── loss.py                           # The custom TruncatedLogitsLoss.
│   └── sampling_heuristics.py            # Utilities to build and apply sampling pipelines.
├── configs/
│   ├── data_generation/                  # Configs defining heuristics and data sources.
│   ├── reward_model/                     # Configs for the reward model-based tournament.
│   └── sampler_models/                   # Configs for the sampler models' architectures.
└── requirements.txt
```

## Usage

The process involves generating candidates, labeling them for quality, and then training a sampler model on the best-resulting data.

### Workflow 1: Advanced Data Generation & Labeling

This workflow creates the high-quality dataset needed for training.

**Step 1: Launch SGLang Server(s)**

First, launch one or more SGLang servers to serve the base LLM. You can launch one server per GPU for maximum throughput. Modify the `model-path` in the script. Note that something similar can be
achieved via `sglang_router` as well.

```bash
# To launch 8 servers, one for each GPU from 0 to 7
bash src/rejection_sampling/run_on_multiple_servers.sh
```

**Step 2: Generate Candidate Responses**

In a separate terminal, run the candidate generation script. This will connect to the running SGLang servers and generate responses for every prompt using every heuristic specified.

  * `dataset_sources_config_path`: A YAML file listing the input prompt datasets (see `configs/data_generation/dataset_sources.yaml`).
  * `heuristics_config_path`: A YAML file listing all sampling pipelines to use (see `configs/data_generation/generate_yaml_*.py` for examples).

```bash
python src/rejection_sampling/generate_candidates_multi.py \
    --model_path "path/to/your/model" \
    --dataset_sources_config_path "configs/data_generation/dataset_sources.yaml" \
    --heuristics_config_path "configs/data_generation/generated_config_llama3.2_3b.yaml" \
    --output_path "./candidate_generations.jsonl" \
    --num_servers 8 \
    --max_workers 32 \
    --max_new_tokens 250
```

**Step 3: Label Data for Quality (Tournament)**

Run the tournament script to score the generated candidates using a reward model. This adds a `bradley_terry_rating` to each datapoint.

  * `--config`: Path to the tournament configuration YAML, which specifies the data paths and reward model details (see `configs/reward_model/tournament_config.yaml`).

```bash
python src/data_labelling/label_ratings.py \
    --config configs/reward_model/tournament_config.yaml
```

### Workflow 2: Train the Sampler Model

After generating and labeling data, you can train your sampler model. You may want to first filter your labeled `.jsonl` file to only include generations with high `bradley_terry_rating`.

```bash
python src/train_supervised.py \
  --model_name_or_path "path/to/your/model" \
  --sampler_model_name "SamplingNetwork" \
  --sampler_config_path "configs/sampler_models/model_config_scale.yaml" \
  --data_path "./path/to/your/filtered_labelled_data.jsonl" \
  --output_dir "./sampler_checkpoints" \
  --max_seq_length 1000 \
  --token_batch_size 12 \
  --learning_rate 1e-2 \
  --num_epochs 1 \
  --save_steps 50 \
  --loss_gamma 5.0 \
  --lr_scheduler_type "cosine" \
  --num_warmup_steps 100 \
  --gradient_accumulation_steps 16
```

*Note: The script is configured to use two GPUs (`cuda:0` and `cuda:1`). You may need to adjust the `device` and `device2` variables in the script's `main` function for your specific hardware setup.*

### Workflow 3: Analyze the Trained Sampler

Use `analyze.py` to compare your trained sampler's output distribution against the base model's raw logits and a target heuristic pipeline.

```bash
python src/analyze.py \
  --model_name_or_path "path/to/your/model" \
  --sampler_model_name "SamplingNetwork" \
  --sampler_config_path "configs/sampler_models/model_config.yaml" \
  --sampler_checkpoint_path "./sampler_checkpoints/final_model/sampler_model.bin" \
  --pipeline_config_path "configs/analysis/heuristics_min_p_0.05.yaml" \
  --prompt "Write me a random story." \
  --num_tokens_to_analyze 20 \
  --top_k_plot 100
```
