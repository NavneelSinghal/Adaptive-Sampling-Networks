# Adaptive Sampling Networks

This repository contains the code and implementation for training and analyzing adapting sampling networks, a collection of lightweight, permutation-equivariant neural networks designed to learn and potentially replace traditional heuristic-based sampling methods for Large Language Models.

The core idea is to train a small, efficient "sampler" model that takes the raw output logits from a frozen, pre-trained LLM and transforms them into a new distribution. This learned transformation can emulate and potentially combine widely used sampling strategies in a single forward pass, without having to do any external scaffolding on top.

## Key Features

  * **Learned Sampling**: Moves beyond hand-tuned heuristics to a learnable algorithm for sampling.
  * **Reasonable Efficiency**: The sampler models are designed to be small and fast, adding minimal overhead (< 1ms) to the generation process.
  * **Permutation-Equivariant by Design**: The models treat the vocabulary as an unordered set, ensuring they learn general distribution transformations rather than memorizing token-specific information.
  * **End-to-End Training**: The samplers are (currently) trained via supervised learning to mimic the output of complex sampling pipelines. See **Planned Features** for more.
  * **Analysis Tools**: Includes scripts to visualize and compare the learned sampler's output distribution against the original and target distributions.

## Planned Features

  * **Rejection sampling to combine the best of existing sampling strategies**: Learning to mimic multiple algorithms after rejection sampling based on reward models/other metrics. Reranking steps with the sets of high-quality heuristics can be applied to improve metrics like diversity by reranking, score-mixing, or up/down-sampling different sampling algorithms.
  * **Reinforcement learning**: Setting up rewards for such metrics (diversity, coherence, factuality, subjective quality and so on) can be looked into, with standard loss functions (and not direct logit-matching as done for the above methods). Penalties like KL divergence with some older checkpoint might help stabilize training.

## How It Works

The project aims to create a self-contained, efficient, and expressive neural network which takes raw logits from a base language model and produces a modified set of logits suitable for sampling. The network internally learns to apply all transformations, including value-based modifications and dynamic soft truncation, without requiring external scaffolding like hard cutoffs.

### 1\. Data Generation

The training process starts by creating a dataset of recipes used to generate `(raw_logits, target_logits)` pairs at training time.

1.  A large set of prompts is prepared.
2.  The `generate_data.py` script feeds these prompts to a frozen base LLM.
3.  For each generation, a random sampling pipeline (e.g., a combination of top-k, top-p and temperature) is chosen from a configuration file.
4.  The script saves the prompt, the text generated using this random heuristic, and the configuration of the heuristic used.

This can be used as the first step for data generation pipelines for multiple training algorithms as mentioned earlier.

### 2\. Supervised Training

With the generated dataset (or any other dataset with the same format), the `train_supervised.py` script trains the sampler model.

1.  For each sample in the dataset, the script replays (in parallel) the generation for all tokens to get the base model's raw logits at each step.
2.  It then applies the saved heuristic pipeline to these raw logits to compute the "target" logits. The target logits are typically sparse, with many values set to `-inf` where tokens have been filtered out.
3.  The sampler model takes the raw logits and produces its own "predicted" logits.
4.  The goal is to make the predicted logits match the target logits. This is achieved by minimizing a custom `TruncatedLogitsLoss` function, which has two components:
      * **KL Divergence**: Pushes the sampler's predicted probability distribution to match the target distribution for the tokens that were *not* filtered out. It still penalizes the tokens which were filtered out, owing to the normalization of probabilities required. In particular, the KL divergence of a truncated and the original non-truncated distribution is $- \log m$ where $m$ is the mass of the surviving tokens.
      * **Truncation Penalty**: Penalizes the sampler for assigning any probability mass to tokens that were filtered out (i.e., where the target logit is `-inf`).

The loss is defined as:
$$\mathcal{L} = D_{KL}(P_{\text{target}} || P_{\text{pred}}) + \gamma \sum_{i \in \text{truncated}} P_{\text{pred}, i}$$

### 3\. Further Training

Please see `Planned Features` for details on TODOs for further training.

## Model Architectures

There are three different sampler architectures of increasing complexity.

1.  **`LocalProbabilityTransform`**:

  * **Description**: The simplest model. It applies a learned transformation to each log-probability value independently using a small MLP (`SwiGLUMLP`). It also learns a soft truncation gate.
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
├── src/                        # Directory for code
│   ├── analyze.py              # Script to analyze and compare a trained sampler with basic sampling.
│   ├── generate_data.py        # Script to generate training data using heuristics.
│   ├── train_supervised.py     # Main training script for the sampler models.
│   ├── models.py               # Contains definitions for the three sampler model architectures.
│   ├── loss.py                 # Defines the custom TruncatedLogitsLoss.
│   └── sampling_heuristics.py  # Utilities to build and apply sampling pipelines.
├── configs/                    # Directory for configuration files.
│   ├── sampler_models/         # Configs for the sampler models' architectures.
│   │   └── *.yaml
│   ├── data_generation/        # Configs defining the heuristics for data generation.
│   │   └── *.yaml
│   └── analysis/               # Configs defining the target pipeline for analysis.
│       └── *.yaml
└── README.md
```

## Usage

The process involves three main steps: data generation, model training, and analysis. For more detailed information on usage, run each script with `--help`.

### Step 1: Prepare Configuration Files

You will need to create several YAML configuration files. See the `configs/` directory for examples.

**Sampler Model Config (`configs/sampler_models/SamplingNetwork.yaml`)**:
Defines the architecture of the sampler you want to train.

```yaml
# For SamplingNetwork
d_model: 32
d_ff: 64
num_blocks: 4

# For SimpleDistributionAwareTransform
# hidden_dims: 64
# use_dynamic_truncation: true
```

**Data Generation Heuristics Config (`configs/data_generation/default_heuristics.yaml`)**:
Defines the different sampling pipelines that will be randomly chosen during data generation. Would generally be used with a single sampling pipeline, though.

```yaml
sampling_pipelines:
  - name: "Conservative Top-P"
    weight: 1.0
    processors:
      - name: "top_p"
        params:
          top_p: 0.92
      - name: "temperature"
        params:
          temperature: 0.9

  - name: "Min-P"
    weight: 0.8
    processors:
      - name: "min_p"
        params:
          min_p: 0.05

  - name: "Aggressive Top-K"
    weight: 1.0
    processors:
      - name: "top_k"
        params:
          top_k: 10
```

### Step 2: Generate Training Data

Use `generate_data.py` to create a dataset for training. You need a `.jsonl` file where each line is a JSON object with a key for the prompt text (e.g., `{"prompt": "Write a story about a dragon."}`).

This step is optional and can be skipped if it is not important that the generations are done using the sampler that is specified. In this case, ensure that your data format is the same as if it were
generated by this script, for the later steps.

```bash
python src/generate_data.py \
  --model_name_or_path "model-name" \
  --heuristics_config_path "configs/data_generation/<config>.yaml" \
  --source_data_path "path/to/your/prompts.jsonl" \
  --output_path "./generated_training_data.jsonl" \
  --num_samples 1000 \
  --batch_size 8 \
  --min_new_tokens 50 \
  --max_new_tokens 250
```

### Step 3: Train the Sampler Model

Use `train_supervised_schedule_acc.py` to train your chosen sampler model on the data you just generated.

```bash
python src/train_supervised_schedule_acc.py \
  --model_name_or_path "model-name" \
  --sampler_model_name "SamplingNetwork" \
  --sampler_config_path "configs/sampler_models/<config>.yaml" \
  --data_path "./generated_training_data.jsonl" \
  --output_dir "./sampler_checkpoints" \
  --max_seq_length 1000 \
  --token_batch_size 8 \
  --learning_rate 1e-2 \
  --num_epochs 1 \
  --save_steps 50 \
  --loss_gamma 5.0 \
  --lr_scheduler_type "cosine" \
  --num_warmup_steps 100
  --gradient_accumulation_steps 24 \
```

*Note: The script is configured to use two GPUs (`cuda:0` and `cuda:1`). You may need to adjust the `device` and `device2` variables in the script's `main` function for your specific hardware setup.*

### Step 4: Analyze the Trained Sampler

Use `analyze.py` to see how well your trained sampler performs. This script will generate a text continuation and then, for each new token, it will compare the probability distributions of:

1.  The generation model's raw logits.
2.  A target heuristic (e.g., top\_k = 50).
3.  Your trained sampler.

It will print metrics (KL Divergence, Truncated Probability Mass) and save plots for visual comparison.

```bash
python analyze.py \
  --model_name_or_path "model-name" \
  --sampler_model_name "SamplingNetwork" \
  --sampler_config_path "configs/sampler_models/<config>.yaml" \
  --sampler_checkpoint_path "./sampler_checkpoints/final_model/sampler_model.bin" \
  --pipeline_config_path "configs/analysis/<config>.yaml" \
  --prompt "Write me a random story." \
  --num_tokens_to_analyze 20 \
  --top_k_plot 100
```
