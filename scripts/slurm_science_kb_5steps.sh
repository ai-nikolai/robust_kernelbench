#!/bin/bash
#SBATCH --job-name=science_kb_5steps
#SBATCH --chdir=/hx2-weka/home/nr1713/robust_kernelbench
#SBATCH --output=/hx2-weka/home/nr1713/robust_kernelbench/logs/science_kb_5steps_%j.out
#SBATCH --error=/hx2-weka/home/nr1713/robust_kernelbench/logs/science_kb_5steps_%j.err
#SBATCH --time=23:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G

# ScienceKernelBench benchmark run:
#   * baseline + 5 repair/improvement rounds
#   * two prompt families: kernelbench and kb_multi_stage
#   * every level in the dataset (1, 2, 3)
#   * local Qwen3-Coder-30B-A3B-Instruct on GPUs 0,1; evaluation on GPU 2
#
# Each (level, prompt family) gets its own experiment directory:
#   experiments/exp_science_v0_level_1_kernelbench/trial_1 ... trial_6
#   experiments/exp_science_v0_level_1_kb_multi_stage/trial_1 ... trial_6
#
# Override from the environment:
#   MODEL_NAME=... LEVELS="1 2" MODES="kernelbench" RESET_EXPERIMENTS=1
#   DATASET_REVISION=v0.1.0 DATASET_VERSION=0.1.0

set -uo pipefail

REPO_DIR="${REPO_DIR:-/hx2-weka/home/nr1713/robust_kernelbench}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
DATASET="${DATASET:-ai-nikolai/ScienceKernelBench}"
DATASET_REVISION="${DATASET_REVISION:-v0.1.0}"
DATASET_VERSION="${DATASET_VERSION:-0.1.0}"
DATASET_NAME="${DATASET_NAME:-ScienceKernelBench}"
VERSION="${VERSION:-science_v0}"
LEVELS="${LEVELS:-1 2 3}"
MODES="${MODES:-kernelbench kb_multi_stage}"
SEED="${SEED:-42}"
RESET_EXPERIMENTS="${RESET_EXPERIMENTS:-0}"

CONDA_SH="${CONDA_SH:-/hx2-weka/home/nr1713/miniconda3/etc/profile.d/conda.sh}"
ENV_MODEL="${ENV_MODEL:-env_vllm30}"
ENV_KB="${ENV_KB:-env_kb2}"
SERVER_PORT="${SERVER_PORT:-30000}"
SERVER_TIME_OUT_SECONDS="${SERVER_TIME_OUT_SECONDS:-3600}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-16384}"

export MAX_JOBS="${MAX_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${REPO_DIR}"
mkdir -p logs

source "${CONDA_SH}"
conda activate "${ENV_MODEL}"

export CUDA_VISIBLE_DEVICES=0,1
setsid python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --host 0.0.0.0 \
    --port "${SERVER_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --dtype bfloat16 \
    --disable-log-requests \
    > logs/science_kb_5steps_server_${SLURM_JOB_ID:-local}.log 2>&1 &
SERVER_PID=$!

cleanup() {
    echo "Stopping vLLM server (process group ${SERVER_PID})..."
    kill -- "-${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_server() {
    local base="http://localhost:${SERVER_PORT}"
    local attempt=0
    echo "Waiting for vLLM at ${base} ..."
    while true; do
        if curl -s -f "${base}/health" >/dev/null 2>&1 || curl -s -f "${base}/v1/models" >/dev/null 2>&1; then
            echo "Server is ready."
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "ERROR: vLLM exited during startup; see logs/science_kb_5steps_server_${SLURM_JOB_ID:-local}.log" >&2
            return 1
        fi
        attempt=$((attempt + 1))
        if [ "${attempt}" -ge "${SERVER_TIME_OUT_SECONDS}" ]; then
            echo "ERROR: server did not become ready in ${SERVER_TIME_OUT_SECONDS}s." >&2
            return 1
        fi
        sleep 1
    done
}

wait_for_server || exit 1

conda deactivate
conda activate "${ENV_KB}"
export CUDA_VISIBLE_DEVICES=2

for LEVEL in ${LEVELS}; do
    for MODE in ${MODES}; do
        EXP="exp_${VERSION}_level_${LEVEL}_${MODE}"
        JOBS=(
            "1 1 ${MODE}"
            "1 2 ${MODE}"
            "2 3 ${MODE}"
            "3 4 ${MODE}"
            "4 5 ${MODE}"
            "5 6 ${MODE}"
        )
        EXTRA=()
        if [ "${RESET_EXPERIMENTS}" = "1" ]; then
            EXTRA+=(--reset_experiments)
        fi

        echo "============================================================"
        echo "experiment=${EXP} level=${LEVEL} mode=${MODE}"
        echo "dataset=${DATASET} revision=${DATASET_REVISION} version=${DATASET_VERSION}"
        echo "rounds: ${JOBS[*]}"
        echo "============================================================"

        if ! python3 -m robust_kernelbench.run_main \
            --experiment_name "${EXP}" \
            --dataset "${DATASET}" \
            --dataset_revision "${DATASET_REVISION}" \
            --dataset_version "${DATASET_VERSION}" \
            --dataset_split "level_${LEVEL}" \
            --dataset_name "${DATASET_NAME}" \
            --version "${VERSION}" \
            --model_name "${MODEL_NAME}" \
            --parent_prompt_type kernelbench \
            --serving_backend vllm \
            --online_service_url "http://localhost:${SERVER_PORT}/v1" \
            --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
            --max_model_len "${MAX_MODEL_LEN}" \
            --max_tokens "${MAX_TOKENS}" \
            --max_mem_util "${GPU_MEM_UTIL}" \
            --num_samples 1 \
            --seed "${SEED}" \
            --num_procs 4 \
            --eval_timeout 1800 \
            --eval_mem_gb 24 \
            --num_correct_trials 5 \
            --num_perf_trials 3 \
            --level "${LEVEL}" \
            --log-level INFO \
            ${EXTRA[@]+"${EXTRA[@]}"} \
            "${JOBS[@]}"; then
            echo "ERROR: ${EXP} failed; stopping the benchmark run." >&2
            exit 1
        fi
    done
done

echo "ScienceKernelBench run complete."
