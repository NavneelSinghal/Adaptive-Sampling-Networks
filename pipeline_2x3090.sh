#!/bin/bash
set -euo pipefail

# NOTE: try with a smaller data source before using a bigger one
# PLEASE CONFIGURE THE VARIABLES IN THE SECTION BELOW BEFORE RUNNING.
# TODO: MUST BE EDITED BY USER
export BASE_MODEL_PATH="/mnt/data/llms/Llama-3.2-3B-Instruct"
export REWARD_MODEL_PATH="/mnt/data/llms/GRAM-LLaMA3.2-3B-RewardModel"
export MODEL_TYPE="llama" # 'llama', 'gemma', or 'qwen'. Used for generating heuristics.

# Parameters
export PROJECT_ROOT="$(pwd)"
export OUTPUT_DIR="${PROJECT_ROOT}/data/outputs/run_$(date +%Y-%m-%d_%H-%M)"
export NUM_SEEDS=8
export NUM_GPUS=2
export GENERATION_WORKERS=32
export DIVERSITY_GPUS=2
# Set to "" to disable, "v4_rpj_llama_s4" for an actual value for instance
export INFINIGRAM_INDEX=""
export TOP_K_FILTER=1
export LABELING_SEED=0

# File Names
export HEURISTICS_CONFIG_FILE="${OUTPUT_DIR}/generated_heuristics.yaml"
export ALL_GENERATIONS_FILE="${OUTPUT_DIR}/all_generations_raw.jsonl"
export VERIFIED_DATA_FILE="${OUTPUT_DIR}/all_generations_verified.jsonl"
export QUALITY_LABELLED_FILE="${OUTPUT_DIR}/quality_labelled.jsonl"
export DIVERSITY_LABELLED_FILE="${OUTPUT_DIR}/diversity_labelled.jsonl"
export VERIFIER_LABELLED_FILE="${OUTPUT_DIR}/verifier_labelled.jsonl"
export INFINIGRAM_LABELLED_FILE="${OUTPUT_DIR}/infinigram_labelled.jsonl"
export FINAL_TRAINING_DATA_FILE="${OUTPUT_DIR}/final_training_data.jsonl"
export TOURNAMENT_CONFIG_FILE="${OUTPUT_DIR}/tournament_config.yaml"

trap 'echo -e "\n\nINTERRUPT RECEIVED"; echo "Killing background processes..."; pkill -P $$ || true; wait || true; echo "Cleanup complete. Exiting."; exit 1' INT TERM EXIT

mkdir -p "${OUTPUT_DIR}"
echo "=============================================================================="
echo "Starting Data Pipeline for 2x RTX 3090"
echo "Project Root: ${PROJECT_ROOT}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Base Model: ${BASE_MODEL_PATH}"
echo "=============================================================================="

echo -e "\nGenerating heuristics configs"
# python configs/data_generation/generate_heuristics_config.py \
python configs/data_generation/generate_heuristics_config_filtered.py \
    --model_type "${MODEL_TYPE}" \
    --output_file "${HEURISTICS_CONFIG_FILE}"
echo "Heuristics configuration saved to ${HEURISTICS_CONFIG_FILE}"

echo -e "\nGenerating candidate responses (multi-seed: 0 to $(($NUM_SEEDS - 1)))"
SEED_FILES=""
for seed in $(seq 0 $(($NUM_SEEDS - 1)))
do
    echo "-------------------"
    echo "PROCESSING SEED ${seed}"
    echo "-------------------"
    
    echo "  > Launching SGLang Router with --dp-size ${NUM_GPUS} and --random-seed ${seed}..."
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($NUM_GPUS - 1)))
    LOG_FILE="${OUTPUT_DIR}/sglang_router_gen_seed_${seed}.log"
    python -m sglang_router.launch_server \
        --model-path "${BASE_MODEL_PATH}" --host "0.0.0.0" --port 30000 \
        --dp-size "${NUM_GPUS}" \
        --mem-fraction-static 0.9 --dtype bfloat16 --enable-custom-logit-processor \
        --random-seed "${seed}" &> "${LOG_FILE}" &
    echo "  > Waiting 60 seconds for router to initialize..."
    sleep 60
    
    SEED_OUTPUT_PATH="${OUTPUT_DIR}/candidate_generations_seed_${seed}.jsonl"
    SEED_FILES="${SEED_FILES}${SEED_OUTPUT_PATH} "
    echo "  > Running generation client with --seed ${seed}..."
    python -m src.rejection_sampling.generate_candidates_multi \
        --model_path "${BASE_MODEL_PATH}" \
        --dataset_sources_config_path "configs/data_generation/dataset_sources.yaml" \
        --heuristics_config_path "${HEURISTICS_CONFIG_FILE}" \
        --output_path "${SEED_OUTPUT_PATH}" \
        --seed "${seed}" \
        --num_servers 1 \
        --base_port 30000 \
        --max_workers "${GENERATION_WORKERS}" \
        --max_new_tokens 512
    echo "  > Shutting down SGLang Router for seed ${seed}..."
    pkill -P $$ || true
    wait || true
    echo "  > Seed ${seed} complete."
done
echo "Multi-seed generation complete."

echo -e "\nConsolidating and verifying data"
cat ${SEED_FILES} > "${ALL_GENERATIONS_FILE}"
rm ${SEED_FILES}

echo "Verifying combined file..."
python -m src.verify_datagen_and_transform \
    --input-path "${ALL_GENERATIONS_FILE}" \
    --output-path "${VERIFIED_DATA_FILE}" \
    --heuristics-config-path "${HEURISTICS_CONFIG_FILE}" \
    --expected-seeds "${NUM_SEEDS}"
echo "Verification complete. Clean data at ${VERIFIED_DATA_FILE}"

echo -e "\nLaunching reward model router for annotation"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($NUM_GPUS - 1)))
LOG_FILE="${OUTPUT_DIR}/sglang_router_reward.log"
python -m sglang_router.launch_server \
    --model-path "${REWARD_MODEL_PATH}" --host "0.0.0.0" --port 30000 \
    --dp-size $(NUM_GPUS) \
    --mem-fraction-static 0.9 --dtype bfloat16 \
    --random-seed "${LABELING_SEED}" &> "${LOG_FILE}" &
echo "Waiting 60 seconds for reward router to initialize..."
sleep 60

echo -e "\nAnnotating Data"
CURRENT_INPUT_FILE="${VERIFIED_DATA_FILE}"

echo "  > 4.1: Labeling for Quality..."
cp configs/reward_model/tournament_config.yaml "${TOURNAMENT_CONFIG_FILE}"
sed -i "s|data_path:.*|data_path: \"${CURRENT_INPUT_FILE}\"|" "${TOURNAMENT_CONFIG_FILE}"
sed -i "s|output_path:.*|output_path: \"${QUALITY_LABELLED_FILE}\"|" "${TOURNAMENT_CONFIG_FILE}"
sed -i "s|model_name_or_path:.*|model_name_or_path: \"${REWARD_MODEL_PATH}\"|" "${TOURNAMENT_CONFIG_FILE}"
sed -i "s|sglang_url:.*|sglang_url: \"http://127.0.0.1:30000/v1\"|" "${TOURNAMENT_CONFIG_FILE}" 
python -m src.data_labelling.label_ratings --config "${TOURNAMENT_CONFIG_FILE}"
CURRENT_INPUT_FILE="${QUALITY_LABELLED_FILE}"
pkill -P $$ || true
wait || true

echo -e "  > 4.2: Labeling for Diversity..."
python -m src.data_labelling.efficient_diversity \
    --data_path "${CURRENT_INPUT_FILE}" \
    --output_path "${DIVERSITY_LABELLED_FILE}" \
    --num_gpus "${DIVERSITY_GPUS}"
CURRENT_INPUT_FILE="${DIVERSITY_LABELLED_FILE}"

echo -e "  > 4.3: Labeling with verifiable rewards..."
python -m src.data_labelling.label_verifier_rewards \
    --data-path "${CURRENT_INPUT_FILE}" \
    --output-path "${VERIFIER_LABELLED_FILE}"
CURRENT_INPUT_FILE="${VERIFIER_LABELLED_FILE}"

if [[ -n "${INFINIGRAM_INDEX}" ]]; then
    echo -e "  > 4.4: Labeling with Infini-gram..."
    python -m src.data_labelling.label_infinigram \
        --data_path "${CURRENT_INPUT_FILE}" \
        --output_path "${INFINIGRAM_LABELLED_FILE}" \
        --infogram_index "${INFINIGRAM_INDEX}"
    CURRENT_INPUT_FILE="${INFINIGRAM_LABELLED_FILE}"
fi

# TODO: generate many of these
echo -e "\nFinal scoring and filtering"
python -m src.data_labelling.process_and_score \
    --input-path "${CURRENT_INPUT_FILE}" \
    --output-path "${FINAL_TRAINING_DATA_FILE}" \
    --top-k "${TOP_K_FILTER}"

echo -e "\nPipeline complete."

trap - INT TERM EXIT
echo -e "\nThe data generation and labeling pipeline has finished."
