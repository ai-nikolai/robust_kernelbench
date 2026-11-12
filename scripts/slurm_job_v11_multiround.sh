#!/bin/bash
#SBATCH --job-name=kb_v11_0_multiround
#SBATCH --chdir=/hx2-weka/home/nr1713/robust_kernelbench
#SBATCH --output=/hx2-weka/home/nr1713/robust_kernelbench/logs/kb_%j_v11_0_multiround.out
#SBATCH --error=/hx2-weka/home/nr1713/robust_kernelbench/logs/kb_%j_v11_0_multiround.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3,scratch:200g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G

# Five-round robust_kernelbench run for comparison with optimizer-agent:
#   trial 1: one-shot KernelBench prompt
#   trials 2..5: kb_multi_stage repair prompts using the previous trial
# Same model, level 1, 100 problems, one sample per problem.

set -uo pipefail

cd /hx2-weka/home/nr1713/robust_kernelbench
mkdir -p logs
JOB_ID="${SLURM_JOB_ID:-local_$(date +%s)}"
SCRATCH="${TMPDIR:-/tmp}/robust_kb_${JOB_ID}"
mkdir -p "${SCRATCH}/tmp" "${SCRATCH}/torch_extensions" "${SCRATCH}/triton" "${SCRATCH}/torchinductor" "${SCRATCH}/xdg" "${SCRATCH}/hf_datasets"
export TMPDIR="${SCRATCH}/tmp"
export TORCH_EXTENSIONS_DIR="${SCRATCH}/torch_extensions"
export TRITON_CACHE_DIR="${SCRATCH}/triton"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH}/torchinductor"
export XDG_CACHE_HOME="${SCRATCH}/xdg"
export HF_DATASETS_CACHE="${SCRATCH}/hf_datasets"
source /hx2-weka/home/nr1713/miniconda3/etc/profile.d/conda.sh
conda activate env_vllm30

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
VERSION="${VERSION:-v11_0}"

export CUDA_VISIBLE_DEVICES=0,1

VLLM_ARGS=(
    --model "${MODEL_NAME}"
    --served-model-name "${MODEL_NAME}"
    --tensor-parallel-size 2
    --host 0.0.0.0
    --port 30000
    --max-model-len 32768
    --gpu-memory-utilization 0.90
    --dtype bfloat16
)

setsid python3 -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" &
SERVER_PID=$!

cleanup() {
    echo "Stopping server (process group ${SERVER_PID})..."
    kill -- "-${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

export MAX_JOBS=4
export OMP_NUM_THREADS=8

wait_for_server() {
    local base="http://localhost:30000"
    local max_attempts="3600"
    local attempt=0
    echo "Waiting for vLLM server to be ready..."
    while true; do
        if curl -s -f "$base/health" > /dev/null 2>&1 || curl -s -f "$base/v1/models" > /dev/null 2>&1; then
            echo "Server is ready."
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "ERROR: vLLM exited during startup." >&2
            return 1
        fi
        attempt=$((attempt + 1))
        if [ "$attempt" -ge "$max_attempts" ]; then
            echo "ERROR: Server did not become ready in time." >&2
            return 1
        fi
        sleep 1
    done
}

wait_for_server || exit 1

conda deactivate
conda activate env_kb2
export CUDA_VISIBLE_DEVICES=2

jobs=(
    "1 1 kernelbench"
    "1 2 kb_multi_stage"
    "2 3 kb_multi_stage"
    "3 4 kb_multi_stage"
    "4 5 kb_multi_stage"
)

PIPELINE_RC=0
python3 -m robust_kernelbench.run_main \
    --version "${VERSION}" \
    --model_name "${MODEL_NAME}" \
    --parent_prompt_type "kernelbench" \
    --serving_backend "vllm" \
    --tensor_parallel_size 2 \
    --online_service_url "http://localhost:30000/v1" \
    --max_model_len 32768 \
    --max_tokens 16384 \
    --max_mem_util 0.90 \
    --num_samples 1 \
    --num_procs 4 \
    --eval_timeout 1800 \
    --num_correct_trials 5 \
    --num_perf_trials 3 \
    --log-level INFO \
    "${jobs[@]}"
PIPELINE_RC=$?

if [ "${PIPELINE_RC}" -ne 0 ]; then
    echo "Pipeline FAILED with exit code ${PIPELINE_RC}; see the log above." >&2
    exit "${PIPELINE_RC}"
fi
echo "Benchmark finished, server terminated."
exit 0
