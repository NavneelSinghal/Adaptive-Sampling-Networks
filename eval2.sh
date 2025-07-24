#!/bin/bash
set -euo pipefail

# NOTE: try with a smaller data source before using a bigger one
# PLEASE CONFIGURE THE VARIABLES IN THE SECTION BELOW BEFORE RUNNING.
# TODO: MUST BE EDITED BY USER
export BASE_MODEL_PATH="unsloth/Llama-3.2-3B-Instruct"
export REWARD_MODEL_PATH="Skywork/Skywork-Reward-V2-Llama-3.1-8B"
export MODEL_TYPE="llama" # 'llama', 'gemma', or 'qwen'. Used for generating heuristics.

# Parameters
export PROJECT_ROOT="$(pwd)"
export OUTPUT_DIR="${PROJECT_ROOT}/data/outputs/run_2025-07-23_21-08"
export NUM_SEEDS=8
export NUM_GPUS=8
export DIVERSITY_GPUS=4
export INFINIGRAM_INDEX="" # Set to "" to disable, "v4_rpj_llama_s4" for an actual value for instance
export TOP_K_FILTER=10000
export LABELING_SEED=0
export BASE_PORT=30000

# File Names
export HEURISTICS_CONFIG_FILE="${OUTPUT_DIR}/generated_heuristics.yaml"
export SAMPLING_MODELS_CONFIG_FILE="${OUTPUT_DIR}/sampling_model_configs.yaml"
export ALL_GENERATIONS_FILE="${OUTPUT_DIR}/all_sampler_model_generations_raw.jsonl"
export VERIFIED_DATA_FILE="${OUTPUT_DIR}/all_sampler_model_generations_verified.jsonl"
export QUALITY_LABELLED_FILE="${OUTPUT_DIR}/sampler_model_quality_labelled.jsonl"
export DIVERSITY_LABELLED_FILE="${OUTPUT_DIR}/sampler_model_diversity_labelled.jsonl"
export OLD_VERIFIER_LABELLED_FILE="${OUTPUT_DIR}/verifier_labelled.jsonl"
export VERIFIER_LABELLED_FILE="${OUTPUT_DIR}/sampler_model_verifier_labelled.jsonl"
export COMBINED_VERIFIER_LABELLED_FILE="${OUTPUT_DIR}/combined_verifier_labelled.jsonl"
export INFINIGRAM_LABELLED_FILE="${OUTPUT_DIR}/combined_infinigram_labelled.jsonl"
export FINAL_EVAL_DATA_FILE="${OUTPUT_DIR}/final_eval_data.jsonl"
export TOURNAMENT_CONFIG_FILE="${OUTPUT_DIR}/tournament_config.yaml"
export BASE_OUTPUT_PATH="${OUTPUT_DIR}/sampler_model_generations"
export MODEL_DIR="${OUTPUT_DIR}/model"

trap 'echo -e "\n\nINTERRUPT RECEIVED"; echo "Killing background processes..."; pkill -P $$ || true; wait || true; echo "Cleanup complete. Exiting."; exit 1' INT TERM EXIT

mkdir -p "${OUTPUT_DIR}"
echo "=============================================================================="
echo "Starting Data Pipeline for 8xH100"
echo "Project Root: ${PROJECT_ROOT}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Base Model: ${BASE_MODEL_PATH}"
echo "=============================================================================="

echo -e "\nGenerating heuristics configs"
python configs/data_generation/generate_sampler_config.py \
    --model_dir "${MODEL_DIR}" \
    --output_file "${SAMPLING_MODELS_CONFIG_FILE}"

echo "Sampling Model Configs saved to ${SAMPLING_MODELS_CONFIG_FILE}"

echo -e "\nLaunching servers in parallel for seeds 0 to $(($NUM_SEEDS - 1))"
SERVER_PIDS=()
for i in $(seq 0 7)
do
  LOG_FILE="${OUTPUT_DIR}/sglang_server_seed_${i}.log"
  echo "  > Launching SGLang Server on GPU $i at port $((30000 + i))...Logs will be streamed to: ${LOG_FILE}"
  CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
    --model-path "${BASE_MODEL_PATH}" \
    --host "0.0.0.0" \
    --port $((30000 + i)) \
    --mem-fraction-static 0.7 \
    --dtype bfloat16 \
    --enable-custom-logit-processor \
    --random-seed $i &> "${LOG_FILE}" &
  SERVER_PIDS+=($!)
done

echo "  > Waiting 120 seconds for all servers to initialize..."
sleep 120

echo -e "\nLaunching clients in parallel"
CLIENT_PIDS=()
SEED_FILES=""
for i in $(seq 0 $(($NUM_SEEDS - 1)))
do
  PORT=$(($BASE_PORT + i))
  OUTPUT_FILE="${BASE_OUTPUT_PATH}_seed_${i}.jsonl"
  SEED_FILES+="$OUTPUT_FILE "
  SEED=$i

  echo "-> Starting client for server on port $PORT with seed $SEED. Outputting to $OUTPUT_FILE"

  python -m src.rejection_sampling.generate_candidates_multi \
    --base_port $PORT \
    --num_servers 1 \
    --model_path "${BASE_MODEL_PATH}" \
    --dataset_sources_config_path "configs/data_generation/dataset_sources.yaml" \
    --heuristics_config_path "${SAMPLING_MODELS_CONFIG_FILE}" \
    --output_path "$OUTPUT_FILE" \
    --seed $SEED \
    --max_workers 128 \
    --min_new_tokens 0 \
    --max_new_tokens 2048 &
  CLIENT_PIDS+=($!)
done

echo "Waiting for all client processes to complete..."
wait "${CLIENT_PIDS[@]}"
echo "All client processes have completed."

echo "Shutting down SGLang Servers..."
kill "${SERVER_PIDS[@]}" || true
wait || true
echo "Parallel generation complete."

echo -e "\nConsolidating and verifying data"
cat ${SEED_FILES} > "${ALL_GENERATIONS_FILE}"
rm ${SEED_FILES}

echo "Verifying combined file..."
python -m src.verify_datagen_and_transform \
    --input-path "${ALL_GENERATIONS_FILE}" \
    --output-path "${VERIFIED_DATA_FILE}" \
    --heuristics-config-path "${SAMPLING_MODELS_CONFIG_FILE}" \
    --expected-seeds "${NUM_SEEDS}"
echo "Verification complete. Clean data at ${VERIFIED_DATA_FILE}"

SERVER_PIDS=()

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($NUM_GPUS - 1)))
for (( i=0; i<$NUM_GPUS; i++ ));
do
    LOG_FILE="${OUTPUT_DIR}/sglang_scorer_${i}.log"
    PORT=$(($BASE_PORT + i))
    echo "   > Starting server on port ${PORT} with GPU: $i"
    CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
        --model-path "${REWARD_MODEL_PATH}" \
        --host "0.0.0.0" \
        --port ${PORT} \
        --mem-fraction-static 0.8 \
        --tp 1 \
        --context-length 4096 \
        --is-embedding \
        &> "${LOG_FILE}" &
    SERVER_PIDS+=($!)
done

echo "Waiting 150 seconds for reward router to initialize..."
sleep 150

echo -e "\nAnnotating Data"
CURRENT_INPUT_FILE="${VERIFIED_DATA_FILE}"

echo "  > 4.1: Labeling for Quality..."

#cp configs/reward_model/tournament_config.yaml "${TOURNAMENT_CONFIG_FILE}"
#sed -i "s|data_path:.*|data_path: \"${CURRENT_INPUT_FILE}\"|" "${TOURNAMENT_CONFIG_FILE}"
#sed -i "s|output_path:.*|output_path: \"${QUALITY_LABELLED_FILE}\"|" "${TOURNAMENT_CONFIG_FILE}"
#sed -i "s|model_name_or_path:.*|model_name_or_path: \"${REWARD_MODEL_PATH}\"|" "${TOURNAMENT_CONFIG_FILE}"
#sed -i "s|sglang_url:.*|sglang_url: \"http://127.0.0.1:30000/v1\"|" "${TOURNAMENT_CONFIG_FILE}"

python -m src.data_labelling.direct_scorer \
    --input-path "${CURRENT_INPUT_FILE}" \
    --output-path "${QUALITY_LABELLED_FILE}" \
    --model-path "${REWARD_MODEL_PATH}" \
    --num-gpus ${NUM_GPUS} \
    --sglang-start-port ${BASE_PORT} \
    --prompt-key "prompt" \
    --generation-key "generation"

CURRENT_INPUT_FILE="${QUALITY_LABELLED_FILE}"

pkill -f "sglang.launch_server.*${REWARD_MODEL_PATH}" || true
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

cat ${VERIFIER_LABELLED_FILE} ${OLD_VERIFIER_LABELLED_FILE} > ${COMBINED_VERIFIER_LABELLED_FILE}
CURRENT_INPUT_FILE="${COMBINED_VERIFIER_LABELLED_FILE}"

if [[ -n "${INFINIGRAM_INDEX}" ]]; then
    echo -e "  > 4.4: Labeling with Infini-gram..."
    python -m src.data_labelling.label_infinigram \
        --data_path "${CURRENT_INPUT_FILE}" \
        --output_path "${INFINIGRAM_LABELLED_FILE}" \
        --infogram_index "${INFINIGRAM_INDEX}"
    CURRENT_INPUT_FILE="${INFINIGRAM_LABELLED_FILE}"
fi

echo -e "\nFinal scoring and filtering"
python -m src.data_labelling.process_and_score \
    --input-path "${CURRENT_INPUT_FILE}" \
    --output-path "${FINAL_EVAL_DATA_FILE}" \
    --top-k "${TOP_K_FILTER}" \
    --ignore-verifiable-filtering

echo -e "\nPipeline complete."

trap - INT TERM EXIT
echo -e "\nThe eval pipeline has finished."
