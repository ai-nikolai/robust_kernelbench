#!/usr/bin/env bash
# ----------------------------------------------------------------------
# Local (single node) end-to-end run: serve the model with vLLM on GPUs 0..N
# and run the pipeline on the remaining GPU(s).
#
# Every setting below can be overridden from the environment, e.g.
#   MODEL_NAME=Qwen/Qwen3-Coder-30B-A3B-Instruct NUM_ITEMS=5 bash scripts/run_main.sh
#
# Co-located (single-GPU) mode, where the server and the pipeline share one GPU:
#   SERVER_GPUS=0 EVAL_GPU=0 GPU_MEM_UTIL=0.55 bash scripts/run_main.sh
# In that mode leave SLEEP_MODE=0 (vLLM only supports /sleep with --enable-sleep-mode,
# and enabling it costs extra host memory).

#
# Cluster memory rules:
#   * gpu partition    -> ask Slurm for all node RAM (MEM=0) and keep MAX_JOBS set.
#   * interactive-gpu  -> QoS caps a job at 1 GPU and 64 GB RAM, so keep
#                         NUM_PROCS/MAX_JOBS low and size EVAL_MEM_GB to fit.
#   * never leave MAX_JOBS unset: it stops nvcc from spawning one cudafe++ per
#     core (~2 GB each), which OOM-killed a run inside a 16 GB cgroup.
# ----------------------------------------------------------------------
set -euo pipefail

# ----------------------------------------------------------------------
# 0. Parameters (environment-overridable)
# ----------------------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
VERSION="${VERSION:-v10_1}"
PARENT_PROMPT_TYPE="${PARENT_PROMPT_TYPE:-kernelbench}"

# Empty => run the whole level. Set e.g. NUM_ITEMS=2 for a smoke test.
NUM_ITEMS="${NUM_ITEMS:-}"

SERVER_GPUS="${SERVER_GPUS:-0,1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
EVAL_GPU="${EVAL_GPU:-2}"
SERVER_PORT="${SERVER_PORT:-30000}"
ONLINE_SERVICE_URL="${ONLINE_SERVICE_URL:-http://localhost:${SERVER_PORT}/v1}"

# vLLM replaces SGLang: no --enable-memory-saver (it crashed sglang 0.5.16 with
# KeyError: 'kv_cache'); memory is bounded by --gpu-memory-utilization instead.
SERVING_BACKEND="${SERVING_BACKEND:-vllm}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-16384}"        # must stay < MAX_MODEL_LEN
DTYPE="${DTYPE:-bfloat16}"
SERVER_STARTUP_TIMEOUT="${SERVER_STARTUP_TIMEOUT:-3600}"   # 30B load can be slow

# vLLM /sleep (pause+resume) is only available with --enable-sleep-mode.
SLEEP_MODE="${SLEEP_MODE:-0}"

NUM_CORRECT_TRIALS="${NUM_CORRECT_TRIALS:-5}"
NUM_PERF_TRIALS="${NUM_PERF_TRIALS:-3}"
NUM_PROCS="${NUM_PROCS:-4}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}"
EVAL_MEM_GB="${EVAL_MEM_GB:-24}"     # address-space cap per eval child
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Deleting an existing experiment directory is destructive, so it is opt-in.
RESET_EXPERIMENTS="${RESET_EXPERIMENTS:-0}"

# Optional tolerance overrides. Leave empty to use the dtype-aware defaults;
# ATOL=1e-2 RTOL=1e-2 reproduces the historical fixed tolerance.
ATOL="${ATOL:-}"
RTOL="${RTOL:-}"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
ENV_MODEL="${ENV_MODEL:-env_vllm30}"
ENV_KB="${ENV_KB:-env_kb2}"

# Bound nvcc/ninja parallelism: torch's cpp_extension spawns one cudafe++ per job and
# several of them (plus the model server) is what blew through the 16 GB cgroup limit.
export MAX_JOBS="${MAX_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

JOBS=(
 "${JOB_1:-1 1 kernelbench}"
 "${JOB_2:-1 4 kernelbench}"
 "${JOB_3:-1 5 kb_multi_stage}"
 "${JOB_4:-4 204 kernelbench}"
 "${JOB_5:-5 205 kb_multi_stage}"
)

# ----------------------------------------------------------------------
# Build the argument lists
#
# NOTE: NUM_ITEMS must be tested here, *after* its default is assigned above
# (it used to be assigned below this `if`, so the test always saw an empty value).
# ----------------------------------------------------------------------
MODEL_ARGS=()
PIPELINE_ARGS=()

if [ -n "${NUM_ITEMS}" ]; then
    PIPELINE_ARGS+=("--num_items" "${NUM_ITEMS}")
fi

PIPELINE_ARGS+=(
    "--version" "${VERSION}"
    "--model_name" "${MODEL_NAME}"
    "--parent_prompt_type" "${PARENT_PROMPT_TYPE}"
    "--serving_backend" "${SERVING_BACKEND}"
    "--tensor_parallel_size" "${TENSOR_PARALLEL_SIZE}"
    "--online_service_url" "${ONLINE_SERVICE_URL}"
    "--max_model_len" "${MAX_MODEL_LEN}"
    "--max_tokens" "${MAX_TOKENS}"
    "--max_mem_util" "${GPU_MEM_UTIL}"
    "--num_procs" "${NUM_PROCS}"
    "--eval_timeout" "${EVAL_TIMEOUT}"
    "--eval_mem_gb" "${EVAL_MEM_GB}"
    "--num_correct_trials" "${NUM_CORRECT_TRIALS}"
    "--num_perf_trials" "${NUM_PERF_TRIALS}"
    "--log-level" "${LOG_LEVEL}"
)

if [ "${SLEEP_MODE}" = "1" ]; then
    PIPELINE_ARGS+=("--enable_sleep")
fi
if [ "${RESET_EXPERIMENTS}" = "1" ]; then
    PIPELINE_ARGS+=("--reset_experiments")
fi
if [ -n "${ATOL}" ]; then PIPELINE_ARGS+=("--atol" "${ATOL}"); fi
if [ -n "${RTOL}" ]; then PIPELINE_ARGS+=("--rtol" "${RTOL}"); fi

echo "=== limits: MAX_JOBS=${MAX_JOBS} OMP_NUM_THREADS=${OMP_NUM_THREADS} GPU_MEM_UTIL=${GPU_MEM_UTIL} MAX_MODEL_LEN=${MAX_MODEL_LEN} EVAL_TIMEOUT=${EVAL_TIMEOUT}s EVAL_MEM_GB=${EVAL_MEM_GB} NUM_PROCS=${NUM_PROCS}"

# ----------------------------------------------------------------------
# 1. Launch the vLLM server
# ----------------------------------------------------------------------
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${ENV_MODEL}"

export CUDA_VISIBLE_DEVICES="${SERVER_GPUS}"

VLLM_SERVER_ARGS=(
    --model "${MODEL_NAME}"
    --served-model-name "${MODEL_NAME}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --host 0.0.0.0
    --port "${SERVER_PORT}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --dtype "${DTYPE}"
)
if [ "${SLEEP_MODE}" = "1" ]; then
    VLLM_SERVER_ARGS+=(--enable-sleep-mode)
fi

setsid python3 -m vllm.entrypoints.openai.api_server "${VLLM_SERVER_ARGS[@]}" &
SERVER_PID=$!

cleanup() {
    echo "Stopping server (process group ${SERVER_PID})..."
    # vLLM forks worker processes; kill the whole group so no GPU memory is leaked.
    kill -- "-${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ----------------------------------------------------------------------
# 2. Wait for the server
# ----------------------------------------------------------------------
wait_for_server() {
    local base="http://localhost:${SERVER_PORT}"
    local max_attempts="${SERVER_STARTUP_TIMEOUT}"
    local attempt=0
    echo "Waiting for vLLM at ${base} ..."
    while true; do
        if curl -s -f "${base}/health" > /dev/null 2>&1 || curl -s -f "${base}/v1/models" > /dev/null 2>&1; then
            echo "Server is ready."
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "ERROR: vLLM exited during startup; see the log above." >&2
            return 1
        fi
        attempt=$((attempt + 1))
        if [ "${attempt}" -ge "${max_attempts}" ]; then
            echo "ERROR: server did not become ready in ${max_attempts}s." >&2
            return 1
        fi
        sleep 1
    done
}

wait_for_server || exit 1

# ----------------------------------------------------------------------
# 3. Run the pipeline on the evaluation GPU
# ----------------------------------------------------------------------
conda deactivate
conda activate "${ENV_KB}"

export CUDA_VISIBLE_DEVICES="${EVAL_GPU}"

python3 -m robust_kernelbench.run_main "${PIPELINE_ARGS[@]}" "${JOBS[@]}"
