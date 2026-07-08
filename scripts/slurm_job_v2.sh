#!/bin/bash
# LOCAL_HOME="/ukp-storage-1/rozanov"
# LOCAL_HOME="/storage/ukp/work/rozanov"
LOCAL_HOME="~"

QUEUE_NAME=""
PARTITION_NAME=""

#SBATCH -p gpu
#SBATCH -q gpu-small


NUM_GPUS=2
GPU_MEM="40gb"

NUM_GPUS=1
GPU_MEM="80gb"

WALLTIME=15:00:00
NUM_PROCS=4

PREVIOUS_TRIAL=${1:-1}
NEW_TRIAL=${2:-1}
PROMPT_TYPE=${3:-"kernelbench"}


ONLINE_SERVICE_URL=https://openrouter.ai/api/v1
USE_ONLINE=${11:-0}


EXPERIMENT_NAME=${4:-""}
MODEL_NAME=${5:-"Qwen/Qwen3-Coder-30B-A3B-Instruct"}

INFERENCE=${6:-1}
COMPILE=${7:-1}
EVAL=${8:-1}

LEVEL=${9:-1}


VERSION=${10:-"V8"}


# FOR A QUICK TEST:
NUM_ITEMS=${12:-}


# PROMPT_TYPE="kb_multi_stage"
# PROMPT_TYPE="kernelbench"
# PROMPT_TYPE="multi_stage"
# PROMPT_TYPE="single_stage"
# PROMPT_TYPE="normal"

#1-base
#2-single_stage
#3-multi_stage
#4-kernelbench
#7-kb_multi_stage

timestamp=$(date +%Y%m%d_%H%M%S)
timestamp2=$(date +%Y-%m-%d_%H:%M:%S)


# Walltime
# WALLTIME=8:00:00
# WALLTIME=01:00:00

# KEY PARAMS
PREVIOUS_TRIAL=$PREVIOUS_TRIAL
TRIAL=$NEW_TRIAL


# PARAMS
FILTER=false
NUM_SAMPLES=1

MAX_MEM_UTIL=0.90
MAX_MODEL_LEN=16000
MAX_TOKENS=10000





# ###########################
# MAIN SCRIPT
# ###########################

SHORT_MODEL_NAME=$(echo "$MODEL_NAME" | cut -d'/' -f2 | sed 's/[^a-zA-Z0-9]/_/g' | cut -c1-32)

SIGNATURE="L${LEVEL}_${VERSION}_${SHORT_MODEL_NAME}_${PROMPT_TYPE}_API${USE_ONLINE}"

if [ -z "$EXPERIMENT_NAME" ]; then
    EXPERIMENT_NAME="exp_local_${SIGNATURE}"
fi

# ###########
# FILE NAME & ENV
output_folder="job_scripts"
mkdir -p "${output_folder}"
output_file="temp_slurm_tts_${timestamp}_E${SIGNATURE}_T${TRIAL}_P${PREVIOUS_TRIAL}.sh" #needs to be unique...
JOB_NAME="kernelbench_${VERSION}_t${NEW_TRIAL}"

ENV="env_robust_kernelbench"
# ###########

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
GPU_MEM=${GPU_MEM:-"40gb"}
# GPU_MEM="80gb" #80gb

if [ -n "${NUM_GPUS}" ];then
    ARGS="$ARGS --tensor_parallel_size ${NUM_GPUS}"
fi


# SBATCH COMMANDS WE DO NOT NEED FOR NOW
#SBATCH --constraint="gpu_mem:${GPU_MEM}" #this is specific to run GPU_MEM GB gpus...
#SBATCH --mem=64G #128G is not working #requesting more than 128 leads to an error.
#SBATCH -q ${QUEUE_NAME}

cat << EOF > "$output_folder/$output_file"
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:${NUM_GPUS}

#SBATCH --cpus-per-task=16 #this should be the number of cores. So I guess we can request 32

#SBATCH -t $WALLTIME

#SBATCH -p ${PARTITION_NAME}


#SBATCH -o ${LOCAL_HOME}/logs/${JOB_NAME}/run_%j_${SHORT_MODEL_NAME}_${PROMPT_TYPE}_%N_%t_${EXPERIMENT_NAME}.log
#SBATCH -e ${LOCAL_HOME}/logs/${JOB_NAME}/run_%j_${SHORT_MODEL_NAME}_${PROMPT_TYPE}_%N_%t_${EXPERIMENT_NAME}.log


mkdir -p ${LOCAL_HOME}/logs
mkdir -p ${LOCAL_HOME}/logs/${JOB_NAME}

cd ${LOCAL_HOME}/robust_kernelbench

python3 robust_kernel_bench/run_main.py

echo "\$SLURM_JOB_ID,finished," >> exp_db.csv

TIMESTAMP3=\$(date +"%Y-%m-%d_%H:%M:%S") # Format: YYYY-MM-DD_HH-MM-SS
echo "SUBMISSION OF SCRIPT: $timestamp2"
echo "START OF SCRIPT: \$TIMESTAMP"
echo "END OF SCRIPT \$TIMESTAMP3"
echo "----------------"
echo "START INFERENCE: \$START_INFERENCE"
echo "END INFERENCE: \$END_INFERENCE"
echo "START COMPILATION: \$START_COMPILATION"
echo "END COMPILATION: \$END_COMPILATION"
echo "START EVAL: \$START_EVAL"
echo "END EVAL: \$END_EVAL"
echo "----------------"
echo "Model name:       $MODEL_NAME"
echo "Prompt type:       $PROMPT_TYPE"
echo "Experiment Name:  $EXPERIMENT_NAME"
echo "Trial:            $TRIAL"
echo "Alhamdulillah, we finished running."
EOF

echo -e "\n=== Job Configuration ==="
echo "Created job file: $output_file"
echo "Experiment Name: $EXPERIMENT_NAME"
echo "Wall Time: $WALLTIME"
echo "NUM GPUS: $NUM_GPUS"
echo "GPU MEM: $GPU_MEM"
echo "Model Name: $MODEL_NAME"
echo "Short Model name: ${SHORT_MODEL_NAME}"
echo "Prompt Type: ${PROMPT_TYPE}"
echo "Old Trial: ${PREVIOUS_TRIAL}"
echo "New Trial: ${TRIAL}"
echo "ARGS: $ARGS"
echo "ENV: $ENV"
echo "INFERENCE: $INFERENCE"
echo "COMPILE: $COMPILE"
echo "EVAL: $EVAL"
echo "Timestamp: $timestamp2"
echo "=========================="
echo "Submitting: $output_file"
SBATCH_OUTPUT=$(sbatch --parsable "$output_folder/$output_file")
echo "Submitted: $SBATCH_OUTPUT"
echo "$SBATCH_OUTPUT,submitted,'${SHORT_MODEL_NAME} ${PROMPT_TYPE} ${PREVIOUS_TRIAL} ${TRIAL}'" >> exp_db.csv
