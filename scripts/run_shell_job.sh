#!/bin/bash

# Script to run a single job locally (without job scheduler)
CUDA_VISIBLE_DEVICES=2

LOCAL_HOME="~"
NUM_GPUS=2

PREVIOUS_TRIAL=${1:-1}
NEW_TRIAL=${2:-1}
PROMPT_TYPE=${3:-"kernelbench"}

ONLINE_SERVICE_URL=http://localhost:30000/v1
USE_ONLINE=${11:-0}

EXPERIMENT_NAME=${4:-""}
# MODEL_NAME=${5:-"Qwen/Qwen3-Coder-30B-A3B-Instruct"}
MODEL_NAME=${5:-"default"}


INFERENCE=${6:-1}
COMPILE=${7:-1}
EVAL=${8:-1}

LEVEL=${9:-1}

VERSION=${10:-"V8"}

NUM_ITEMS=${12:-}

# Timestamps
timestamp=$(date +%Y%m%d_%H%M%S)
timestamp2=$(date +%Y-%m-%d_%H:%M:%S)

# Key params
PREVIOUS_TRIAL=$PREVIOUS_TRIAL
TRIAL=$NEW_TRIAL

# Params
FILTER=true
NUM_SAMPLES=1

MAX_MEM_UTIL=0.90
MAX_MODEL_LEN=16000
MAX_TOKENS=10000

# Generate experiment name if not provided
SHORT_MODEL_NAME=$(echo "$MODEL_NAME" | cut -d'/' -f2 | sed 's/[^a-zA-Z0-9]/_/g' | cut -c1-32)

SIGNATURE="L${LEVEL}_${VERSION}_${SHORT_MODEL_NAME}_${PROMPT_TYPE}_API${USE_ONLINE}"

if [ -z "$EXPERIMENT_NAME" ]; then
    EXPERIMENT_NAME="exp_local_${SIGNATURE}"
fi

# Build arguments
ARGS="--experiment_name ${EXPERIMENT_NAME} \
--model_name ${MODEL_NAME} \
--max_model_len ${MAX_MODEL_LEN} \
--max_tokens ${MAX_TOKENS}"

if [ "$USE_ONLINE" -eq 1 ]; then
    ARGS="$ARGS --online_service_url ${ONLINE_SERVICE_URL}"
    source ./api_key.sh
fi

if [ -n "$TOKENIZER_NAME" ]; then
    ARGS="$ARGS --tokenizer $TOKENIZER_NAME"
fi

if [ -n "$PROMPT_TYPE" ]; then
    ARGS="$ARGS --prompt_type $PROMPT_TYPE"
fi

if [ "${FILTER:-false}" = true ]; then
    ARGS="$ARGS --filter"
fi

if [ -n "${NUM_ITEMS}" ];then
    ARGS="$ARGS --num_items ${NUM_ITEMS}"
fi

if [ -n "${NUM_SAMPLES}" ];then
    ARGS="$ARGS --num_samples ${NUM_SAMPLES}"
fi

if [ -n "${TRIAL}" ];then
    ARGS="$ARGS --trial ${TRIAL}"
fi

if [ -n "${PREVIOUS_TRIAL}" ];then
    ARGS="$ARGS --previous_trial ${PREVIOUS_TRIAL}"
fi

if [ -n "${LEVEL}" ];then
    ARGS="$ARGS --level ${LEVEL}"
fi

if [ -n "${MAX_MEM_UTIL}" ];then
    ARGS="$ARGS --max_mem_util ${MAX_MEM_UTIL}"
fi

NUM_GPUS=${NUM_GPUS:-1}

if [ -n "${NUM_GPUS}" ];then
    ARGS="$ARGS --tensor_parallel_size ${NUM_GPUS}"
fi

# Print job configuration
echo "=== Job Configuration ==="
echo "Experiment Name: $EXPERIMENT_NAME"
echo "Model Name: $MODEL_NAME"
echo "Short Model Name: ${SHORT_MODEL_NAME}"
echo "Prompt Type: ${PROMPT_TYPE}"
echo "Previous Trial: ${PREVIOUS_TRIAL}"
echo "New Trial: ${TRIAL}"
echo "USE ONLINE: $USE_ONLINE"
echo "NUM ITEMS: $NUM_ITEMS"
echo "ARGS: $ARGS"
echo "INFERENCE: $INFERENCE"
echo "COMPILE: $COMPILE"
echo "EVAL: $EVAL"
echo "Timestamp: $timestamp2"
echo "=========================="

# Setup environment
ENV="env_robust_kernelbench"

# Execute the job
echo "Welcome to the script..."
echo $EXPERIMENT_NAME
echo $MODEL_NAME
echo $timestamp2

echo "$timestamp2,started," >> exp_db.csv

echo "Model name:       $MODEL_NAME"
echo "Prompt type:      $PROMPT_TYPE"
echo "Experiment Name:  $EXPERIMENT_NAME"
echo "Trial:            $TRIAL"

TIMESTAMP=$(date +"%Y-%m-%d_%H:%M:%S") # Format: YYYY-MM-DD_HH-MM-SS
echo $TIMESTAMP

cd ${LOCAL_HOME}/robust_kernelbench/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_robust_kernelbench 

echo "===========================HF HOME/ TORCH HOME"
echo "HF HOME=${HF_HOME}"
export HF_HOME=${LOCAL_HOME}
echo "HF HOME=${HF_HOME}"

export TORCH_HOME=${LOCAL_HOME}
echo "TORCH HOME=${TORCH_HOME}"

echo "==========================="
echo "NVIDIA ANALYSIS"
which nvidia-smi
which nvcc
nvcc --version
echo "CUDA Home:"
echo $CUDA_HOME
echo "finished analysis"
nvidia-smi

if [ "${INFERENCE}" == "1" ];then
    START_INFERENCE=$(date +"%Y-%m-%d_%H:%M:%S") # Format: YYYY-MM-DD_HH-MM-SS
    echo "============= Starting Inference"
    echo "python3 run_inference.py"
    echo ${ARGS}
    python3 robust_kernelbench/run_inference_test_time_scaling.py ${ARGS}
    END_INFERENCE=$(date +"%Y-%m-%d_%H:%M:%S") # Format: YYYY-MM-DD_HH-MM-SS

    echo "START INFERENCE: $START_INFERENCE"
    echo "END INFERENCE: $END_INFERENCE"
fi
echo "$timestamp2,inferenced," >> exp_db.csv

nvidia-smi

if [ "${COMPILE}" == "1" ];then
    echo "============= Starting CLEAN_UP"
    echo "python3 robust_kernelbench/clean_cuda_cache.py --experiment ${EXPERIMENT_NAME} --trial ${TRIAL} --remove"
    python3 robust_kernelbench/clean_cuda_cache.py --experiment ${EXPERIMENT_NAME} --trial ${TRIAL} --remove

    START_COMPILATION=$(date +"%Y-%m-%d_%H:%M:%S") # Format: YYYY-MM-DD_HH-MM-SS
    echo "============= Starting COMPILATION (PARALLEL)"
    echo "python3 robust_kernelbench/compile_all.py --experiment_name ${EXPERIMENT_NAME} --trial ${TRIAL} --num_processes ${NUM_PROCS}"
    python3 robust_kernelbench/compile_all.py --experiment_name ${EXPERIMENT_NAME} --trial ${TRIAL} --num_processes ${NUM_PROCS}
    END_COMPILATION=$(date +"%Y-%m-%d_%H:%M:%S") # Format: YYYY-MM-DD_HH-MM-SS

    echo "START COMPILATION: $START_COMPILATION"
    echo "END COMPILATION: $END_COMPILATION"
fi

if [ "${COMPILE}" == "2" ];then
    echo "============= Starting CLEAN_UP"
    echo "python3 robust_kernelbench/clean_cuda_cache.py --experiment ${EXPERIMENT_NAME} --trial ${TRIAL} --remove"
    python3 robust_kernelbench/clean_cuda_cache.py --experiment ${EXPERIMENT_NAME} --trial ${TRIAL} --remove

    START_COMPILATION=$(date +"%Y-%m-%d_%H:%M:%S") # Format: YYYY-MM-DD_HH-MM-SS
    echo "============= Starting COMPILATION (SEQUENTIAL)"
    echo "python3 robust_kernelbench/compile_all_sequential.py --experiment_name ${EXPERIMENT_NAME} --trial ${TRIAL} --num_processes ${NUM_PROCS}"
    python3 robust_kernelbench/compile_all_sequential.py --experiment_name ${EXPERIMENT_NAME} --trial ${TRIAL} --num_processes ${NUM_PROCS}
    END_COMPILATION=$(date +"%Y-%m-%d_%H:%M:%S") # Format: YYYY-MM-DD_HH-MM-SS

    echo "START COMPILATION: $START_COMPILATION"
    echo "END COMPILATION: $END_COMPILATION"
fi
echo "$timestamp2,compiled," >> exp_db.csv

nvidia-smi

if [ "${EVAL}" == "1" ];then
    echo "============= Starting CLEAN_UP"
    echo "python3 robust_kernelbench/clean_cuda_cache.py --experiment ${EXPERIMENT_NAME} --trial ${TRIAL} --remove"
    python3 robust_kernelbench/clean_cuda_cache.py --experiment ${EXPERIMENT_NAME} --trial ${TRIAL} --remove

    START_EVAL=$(date +"%Y-%m-%d_%H:%M:%S") # Format: YYYY-MM-DD_HH-MM-SS
    echo "============= Starting EVAL"
    echo "python3 evaluate_all.py --experiment_name ${EXPERIMENT_NAME} --trial ${TRIAL}"
    python3 robust_kernelbench/evaluate_all.py --experiment_name ${EXPERIMENT_NAME} --trial ${TRIAL}
    END_EVAL=$(date +"%Y-%m-%d_%H:%M:%S") # Format: YYYY-MM-DD_HH-MM-SS

    echo "START EVAL: $START_EVAL"
    echo "END EVAL: $END_EVAL"
fi

echo "$timestamp2,finished," >> exp_db.csv

TIMESTAMP3=$(date +"%Y-%m-%d_%H:%M:%S") # Format: YYYY-MM-DD_HH-MM-SS
echo "START OF SCRIPT: $TIMESTAMP"
echo "END OF SCRIPT $TIMESTAMP3"
echo "----------------"
echo "START INFERENCE: $START_INFERENCE"
echo "END INFERENCE: $END_INFERENCE"
echo "START COMPILATION: $START_COMPILATION"
echo "END COMPILATION: $END_COMPILATION"
echo "START EVAL: $START_EVAL"
echo "END EVAL: $END_EVAL"
echo "----------------"
echo "Model name:       $MODEL_NAME"
echo "Prompt type:       $PROMPT_TYPE"
echo "Experiment Name:  $EXPERIMENT_NAME"
echo "Trial:            $TRIAL"
echo "Alhamdulillah, we finished running."
