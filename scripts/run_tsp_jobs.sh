#!/bin/bash

# Define a list of tuples (represented as space-separated values in an array)
MODEL_NAME=""
# MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct"
# MODEL_NAME="Qwen/QwQ-32B"
# MODEL_NAME="Qwen/Qwen3-Coder-Next" #80B
# MODEL_NAME="cognition-ai/Kevin-32B"
# MODEL_NAME="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
# MODEL_NAME="zai-org/GLM-4.7-Flash"
# MODEL_NAME="mistralai/Devstral-Small-2-24B-Instruct-2512"
# MODEL_NAME="mistralai/Ministral-3-14B-Instruct-2512"
# MODEL_NAME="moonshotai/Kimi-Linear-48B-A3B-Instruct"


EXPERIMENT_NAME=""
# EXPERIMENT_NAME="exp_local_v6L_2_Qwen3_Coder_30B_A3B_Instruct_kernelbench_20260203_184734"
# EXPERIMENT_NAME="exp_local_v6L_2_Qwen3_Coder_30B_A3B_Instruct_kernelbench_20260205_094151"
# EXPERIMENT_NAME="exp_local_v6L_2_Qwen3_Coder_30B_A3B_Instruct_kernelbench_L3"

# EXPERIMENT_NAME=
# EXPERIMENT_NAME="exp_local_v6L_2_Qwen3_Coder_30B_A3B_Instruct_kernelbench_L3"


# MODEL_NAME="openai/gpt-5.4-nano"
# MODEL_NAME="minimax/minimax-m2.7"
# MODEL_NAME="minimax/minimax-m2.5:free"

# MODEL_NAME="z-ai/glm-5-turbo"
# MODEL_NAME="z-ai/glm-5"

# MODEL_NAME="nvidia/nemotron-3-super-120b-a12b:free"

# MODEL_NAME="openai/gpt-5.3-codex"
# MODEL_NAME="openai/gpt-oss-120b"

# MODEL_NAME="qwen/qwen3.5-35b-a3b"
# MODEL_NAME="qwen/qwen3.5-122b-a10b"
# MODEL_NAME="qwen/qwen3-next-80b-a3b-instruct:free"
# MODEL_NAME="qwen/qwen3-coder:free"
# MODEL_NAME="qwen/qwen3-coder"
# Bigger modesls
# MODEL_NAME="qwen/qwen3.5-397b-a17b"
# MODEL_NAME="qwen/qwen3-coder-next"
# MODEL_NAME="qwen/qwen3-max-thinking"
# MODEL_NAME="qwen/qwen3-coder-plus"
# MODEL_NAME="qwen/qwen3-max" #yes
# MODEL_NAME="qwen/qwen3-coder" #yes Qwen: Qwen3 Coder 480B A35B

# MODEL_NAME="nousresearch/hermes-3-llama-3.1-405b:free" #not good at all...

# MODEL_NAME="mistralai/devstral-2512"

# MODEL_NAME="deepseek/deepseek-v3.2"

LEVEL=1
# LEVEL=2
# LEVEL=3

INFERENCE=0
COMPILE=0
EVAL=0


INFERENCE=1
COMPILE=1
EVAL=1

USE_ONLINE=0
# USE_ONLINE=1

# VERSION="V8_2_1_test"
VERSION="V8_3"


# WHICH MODELS TO RUN
    # "deepseek/deepseek-v3.2"
    # "deepseek/deepseek-r1"

# CURRENTLY RUNNING (07.04.2026)
# models=(
#     # "z-ai/glm-5"
#     # z-ai/glm-4.6v
#     # "deepseek/deepseek-r1-0528"
#     # "deepseek/deepseek-v3.1-terminus"
#     # "mistralai/devstral-2512"
#     # "qwen/qwen3-coder"
#     # "qwen/qwen3.5-397b-a17b"
#     # "openai/gpt-oss-120b"
#     # "google/gemma-4-31b-it"
# )
#     # "minimax/minimax-m2.7"


# FINAL MODELS
# models=(
#     "deepseek/deepseek-v3.1-terminus"
#     "mistralai/devstral-2512"
#     "openai/gpt-oss-120b"
# )

models=(
    "Qwen/Qwen3-0.6B"
)

jobs=(
    "1 1 single_stage"
    # "1 12 single_stage"
    # "1 13 multi_stage"
    # "1 22 single_stage"
    # "1 23 multi_stage"
)


# WHICH JOBS TO RUN
# for later to try the new prompt.
# jobs=(
#     "1 1 kernelbench"
#     "1 2 single_stage"
#     "1 3 multi_stage"
#     # "1 8 kb_multi_stage_v2"
#     # "1 14 kernelbench"
#     # "1 17 kb_multi_stage"
# )

# jobs=(
#     "1 1 kernelbench"
#     "1 4 kernelbench"
#     "1 7 kb_multi_stage"
#     "4 204 kernelbench"
#     "7 207 kb_multi_stage"
#     "204 304 kernelbench"
#     "207 307 kb_multi_stage"
# )

# # OPTIONAL
# SHORT_MODEL_NAME=$(echo "$MODEL_NAME" | cut -d'/' -f2 | sed 's/[^a-zA-Z0-9]/_/g' | cut -c1-32)

# SIGNATURE="L${LEVEL}_${VERSION}_${SHORT_MODEL_NAME}_${PROMPT_TYPE}_API${USE_ONLINE}"

# if [ -z "$EXPERIMENT_NAME" ]; then
#     EXPERIMENT_NAME="exp_local_${SIGNATURE}"
# fi

NUM_ITEMS=""
NUM_ITEMS=1

# Iterate through each tuple
for model in "${models[@]}"; do
    read -r concrete_model <<< "$model"

    # FIXING THE EXPERIMENT NAME
    SHORT_MODEL_NAME=$(echo "$concrete_model" | cut -d'/' -f2 | sed 's/[^a-zA-Z0-9]/_/g' | cut -c1-32)
    SIGNATURE="L${LEVEL}_${VERSION}_${SHORT_MODEL_NAME}_${PROMPT_TYPE}_API${USE_ONLINE}"
    EXPERIMENT_NAME="exp_local_${SIGNATURE}"

    for tuple in "${jobs[@]}"; do

        # Split the tuple into individual components
        read -r previous new prompt <<< "$tuple"

        # Call another bash script with the tuple components as arguments
        ./scripts/generate_tsp_job.sh "$previous" "$new" "$prompt" "$EXPERIMENT_NAME" "$concrete_model" "$INFERENCE" "$COMPILE" "$EVAL" "$LEVEL" "$VERSION" "$USE_ONLINE" $NUM_ITEMS
        
        sleep 0.5
    done
done