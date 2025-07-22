#!/bin/bash

set -e
DATA_PREP_SCRIPT="src/data_labelling/process_and_score.py"
TRAIN_SCRIPT="src/train_supervised.py"

declare -a KEYS_TO_PROCESS=(
    "0.1 0.9 0.9"
    "0.3 0.7 0.9"
    "0.5 0.5 0.9"
    "0.7 0.3 0.9"
    "0.9 0.1 0.9"
)

declare -a TOP_K_VALUES=(1 2 5)

MAX_PARALLEL_JOBS=8

RUN_NAME="run_2025-07-20_11-19"
DATA_BASE_PATH="data/outputs/${RUN_NAME}/training_data"
INPUT_FILE="data/outputs/${RUN_NAME}/final_training_data.jsonl"
MODEL_DIR="data/outputs/${RUN_NAME}/model"
TRAINING_LOGS_DIR="data/outputs/${RUN_NAME}/training_logs"

echo "STAGE 1: Generating Datasets"

mkdir -p ${DATA_BASE_PATH}

for key_coeffs in "${KEYS_TO_PROCESS[@]}"; do
    for k in "${TOP_K_VALUES[@]}"; do
        # Read the space-separated coefficients into individual variables
        read -r a b c <<< "$key_coeffs"

        # Format the sort key string and the output filename
        SORT_KEY="a${a}_b${b}_c${c}"
        OUTPUT_FILE="${DATA_BASE_PATH}/final_training_data_${SORT_KEY}_top${k}.jsonl"

        echo ">> Preparing data for sort key: ${SORT_KEY} and top-k: ${k}"
        echo ">> Output file: ${OUTPUT_FILE}"

        python "${DATA_PREP_SCRIPT}" \
            --input-path "${INPUT_FILE}" \
            --output-path "${OUTPUT_FILE}" \
            --top-k "${k}" \
            --sort-key "${SORT_KEY}"
    done
done

echo ">> All ${TOTAL_DATASETS} datasets generated successfully."

echo "STAGE 2: Launching Training Runs"

echo ">> Creating base directories for models and logs..."

mkdir -p ${MODEL_DIR}
mkdir -p ${TRAINING_LOGS_DIR}
echo ">> Directories created."

JOB_COUNT=0
TOTAL_JOBS_LAUNCHED=0

for key_coeffs in "${KEYS_TO_PROCESS[@]}"; do
    for k in "${TOP_K_VALUES[@]}"; do

        read -r a b c <<< "$key_coeffs"
        SORT_KEY_STR="a${a}_b${b}_c${c}"
        DATA_NAME="final_training_data_${SORT_KEY_STR}_top${k}"

        DATA_PATH="${DATA_BASE_PATH}/${DATA_NAME}.jsonl"
        OUTPUT_DIR="${MODEL_DIR}/output_${DATA_NAME}"
        LOG_FILE="${TRAINING_LOGS_DIR}/logs_${DATA_NAME}.out"

        if [ ! -f "$DATA_PATH" ]; then
            echo ">> ERROR: Data file not found, skipping: ${DATA_PATH}"
            echo ">> This should not happen. Please check Stage 1 for errors"
            continue
        fi
        GPU_ID=$((JOB_COUNT % MAX_PARALLEL_JOBS))

        echo "----------------------------------------------------"
        echo "Launching Job #$((TOTAL_JOBS_LAUNCHED + 1)) of ${TOTAL_CONFIGS}: ${DATA_NAME}"
        echo "  - GPU ID: ${GPU_ID}"
        echo "  - Output Dir: ${OUTPUT_DIR}"
        echo "  - Log File: ${LOG_FILE}"
        echo "----------------------------------------------------"

        CUDA_VISIBLE_DEVICES=${GPU_ID} python ${TRAIN_SCRIPT} \
            --model_name_or_path unsloth/Llama-3.2-3B-Instruct \
            --sampler_model_name SamplingNetwork \
            --sampler_config_path configs/sampler_models/model_config_scale.yaml \
            --data_path "${DATA_PATH}" \
            --output_dir "${OUTPUT_DIR}" \
            --max_seq_length 1000 \
            --token_batch_size 12 \
            --learning_rate 1e-3 \
            --num_epochs 1 \
            --save_steps 50 \
            --loss_gamma 1 \
            --lr_scheduler_type cosine \
            --num_warmup_steps 100 \
            --gradient_accumulation_steps 2 \
            > "${LOG_FILE}" 2>&1 &

        JOB_COUNT=$((JOB_COUNT + 1))
        TOTAL_JOBS_LAUNCHED=$((TOTAL_JOBS_LAUNCHED + 1))

        if (( JOB_COUNT % MAX_PARALLEL_JOBS == 0 )); then
            echo ""
            echo ">> Maximum parallel jobs (${MAX_PARALLEL_JOBS}) reached. Waiting for current batch to finish..."
            wait
            echo ">> Batch complete. Launching next batch of jobs."
            echo ""
            JOB_COUNT=0
        fi
    done
done

echo "All training jobs have been launched."
echo "Waiting for the final batch of jobs to complete"
wait
echo "All training runs have finished successfully."
echo "Total jobs executed: ${TOTAL_JOBS_LAUNCHED}"
