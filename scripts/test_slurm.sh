#!/bin/bash
LOCAL_HOME="/ukp-storage-1/rozanov"


timestamp=$(date +%Y%m%d_%H%M%S)
timestamp2=$(date +%Y-%m-%d_%H:%M:%S)

VERSION="v4_2"
output_file="temp_slurm_test.sh"
EXPERIMENT_NAME="exp_slurm_${VERSION}_${timestamp}"
JOB_NAME="test_${VERSION}"

ENV="env_robust_kernelbench"

NUM_PROCS=8

# Walltime
# WALLTIME=8:00:00
WALLTIME=00:20:00

# KEY PARAMS
TRIAL=1
LEVEL=1

# PARAMS
FILTER=false
# NUM_ITEMS=3
NUM_SAMPLES=1

MAX_MEM_UTIL=0.8

# Qwen2.5 Model
MAX_MODEL_LEN=32000
MAX_TOKENS=16000
MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
# Qwen3 Model
# MODEL_NAME="Qwen/Qwen3-8B"
# TOKENIZER_NAME="cognition-ai/Kevin-32B"
# MODEL_NAME="./local_models/cognition-ai_Kevin-32B-Q6_K_L.gguf"
# wget https://huggingface.co/bartowski/cognition-ai_Kevin-32B-GGUF/resolve/main/cognition-ai_Kevin-32B-Q6_K_L.gguf

# Qwen3 models...
# MAX_MODEL_LEN=64000
# MAX_TOKENS=32000
# MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"
# MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"

# Larger Models (MoE)
MAX_MODEL_LEN=8000
MAX_TOKENS=6000
MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"


ARGS_TEST="--model_name ${MODEL_NAME}"

ARGS="--experiment_name ${EXPERIMENT_NAME} \
--model_name ${MODEL_NAME} \
--max_model_len ${MAX_MODEL_LEN} \
--max_tokens ${MAX_TOKENS}"

if [ -n "$TOKENIZER_NAME" ]; then
    ARGS="$ARGS --tokenizer $TOKENIZER_NAME"
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

if [ -n "${LEVEL}" ];then
    ARGS="$ARGS --level ${LEVEL}"
fi

if [ -n "${MAX_MEM_UTIL}" ];then
    ARGS="$ARGS --max_mem_util ${MAX_MEM_UTIL}"
fi

cat << EOF > "$output_file"
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_mem:80gb" #this is specific to run 80GB gpus...

#SBATCH --cpus-per-task=2 #this should be the number of cores. So I guess we can request 32
#SBATCH --mem=32G #128G is not working #requesting more than 128 leads to an error.

#SBATCH -t $WALLTIME

#SBATCH -p gpu

#SBATCH -o ${LOCAL_HOME}/logs/${JOB_NAME}/run_%j_%N_%t_${EXPERIMENT_NAME}.log
#SBATCH -e ${LOCAL_HOME}/logs/${JOB_NAME}/run_%j_%N_%t_${EXPERIMENT_NAME}.log


mkdir -p ${LOCAL_HOME}/logs
mkdir -p ${LOCAL_HOME}/logs/${JOB_NAME}

# Specific Issue with VLLM : https://github.com/vllm-project/vllm/issues/5222
#  #SBATCH --exclude=gpu-01
#  #SBATCH --cpus-per-task=16
#  #SBATCH --gpus-per-node=1 #8



# Print SLURM node info
echo "Welcome to the script..."
echo $WALLTIME
echo $EXPERIMENT_NAME
echo $MODEL_NAME
echo $timestamp2


TIMESTAMP=\$(date +"%Y-%m-%d_%H:%M:%S") # Format: YYYY-MM-DD_HH-MM-SS
echo \$TIMESTAMP

echo "SLURM_JOB_CPUS_PER_NODE: \$SLURM_JOB_CPUS_PER_NODE"
echo "SLURM_NODELIST: \$SLURM_NODELIST"
echo "SLURM_NNODES: \$SLURM_NNODES"
echo "SLURM_JOB_ID: \$SLURM_JOB_ID"
echo "SLURM_NODEID: \$SLURM_NODEID"
scontrol show hostname \$SLURM_NODELIST

# export NCCL_P2P_DISABLE=1 

echo "===========================GPU VISIBLE DEVICES"
# export CUDA_VISIBLE_DEVICES=\$SLURM_JOB_GPUS
export CUDA_VISIBLE_DEVICES=0
echo "SLURM JOB GPUS: \$SLURM_JOB_GPUS"
echo CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}

cd ${LOCAL_HOME}/robust_kernelbench/


# source env_robust_kernelbench/bin/activate
eval "\$(conda shell.bash hook)"
echo "Activated Conda"
conda --version
conda activate $ENV
echo "Activated Env $ENV"

echo "===========================HF HOME/ TORCH HOME"
echo "HF HOME=\${HF_HOME}"
export HF_HOME=${LOCAL_HOME}
echo "HF HOME=\${HF_HOME}"

export TORCH_HOME=${LOCAL_HOME}
echo "TORCH HOME=\${TORCH_HOME}"

# Set PyTorch Inductor cache directory to a writable location
# export TMPDIR=${LOCAL_HOME}/tmp
# export TORCHINDUCTOR_CACHE_DIR=${LOCAL_HOME}/tmp/torch_inductor_cache
# export HOME=${LOCAL_HOME}
# echo HOME=${HOME}

export VLLM_CACHE_ROOT=${LOCAL_HOME}/.cache/vllm/
export VLLM_CONFIG_ROOT=${LOCAL_HOME}/.config/vllm/
export TRITON_CACHE_DIR=${LOCAL_HOME}/.cache/triton/
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "==========================="
echo "NVIDIA ANALYSIS"
which nvidia-smi
which nvcc
nvcc --version
echo "CUDA Home:"
echo \$CUDA_HOME
echo "finished analysis"
nvidia-smi

echo "============= Starting Test"
echo "python3 scripts/test.py"
echo ${ARGS_TEST}
python3 scripts/test.py ${ARGS_TEST}



TIMESTAMP3=\$(date +"%Y-%m-%d_%H:%M:%S") # Format: YYYY-MM-DD_HH-MM-SS
echo $timestamp2
echo \$TIMESTAMP
echo \$TIMESTAMP3
echo "Alhamdulillah, we finished running."
EOF

echo -e "\n=== Job Configuration ==="
echo "Created job file: $output_file"
echo "Experiment Name: $EXPERIMENT_NAME"
echo "Wall Time: $WALLTIME"
echo "Model Name: $MODEL_NAME"
echo "ARGS: $ARGS"
echo "ENV: $ENV"
echo "Timestamp: $timestamp2"
echo "=========================="
sbatch $output_file