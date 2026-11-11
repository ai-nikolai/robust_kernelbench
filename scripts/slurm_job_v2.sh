#!/usr/bin/env bash
# ----------------------------------------------------------------------
# Submit the pipeline as a SLURM job:
#   * a SGLang server occupies the first (NUM_GPUS - P1) GPUs
#   * the pipeline runs on the remaining GPU(s)
#
# Everything below is environment-overridable, e.g.
#   MODEL_NAME=Qwen/Qwen3-Coder-30B-A3B-Instruct VERSION=v9_8 bash scripts/slurm_job_v2.sh
# ----------------------------------------------------------------------
set -euo pipefail

# ----------------------------------------------------------------------
# SCRIPT PARAMETERS
# ----------------------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
VERSION="${VERSION:-v10_1}"
WALLTIME="${WALLTIME:-15:00:00}"
NUM_GPUS="${NUM_GPUS:-3}"
PARTITION="${PARTITION:-gpu}"
# 0 means "all memory on the node" (the gpu partition nodes have ~1.5 TB).
# Cluster rules:  gpu partition  -> MEM=0 (all RAM, 4x H200 per node)
#                 interactive-gpu -> QoS caps 1 GPU AND 64 GB (gres/gpu=1, mem=64G)
MEM="${MEM:-0}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"

# GPUs reserved for the pipeline (P1); the rest go to the model server.
# With NUM_GPUS=3 and P1=1: server gets GPUs 0,1 (TP=2), pipeline gets GPU 2.
P1="${P1:-1}"
SERVER_GPUS=$((NUM_GPUS - P1))
if [ "${SERVER_GPUS}" -lt 1 ]; then
    echo "ERROR: NUM_GPUS=${NUM_GPUS} with P1=${P1} leaves no GPU for the model server." >&2
    exit 1
fi

MODEL_VISIBLE_DEVICES=$(seq -s, 0 $((SERVER_GPUS - 1)))
KB_VISIBLE_DEVICES=$((NUM_GPUS - 1))
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${SERVER_GPUS}}"

SERVER_TIME_OUT_SECONDS="${SERVER_TIME_OUT_SECONDS:-3600}" # 1 hour (30B weights + TP init)
SERVER_PORT="${SERVER_PORT:-30000}"

ENV_MODEL="${ENV_MODEL:-env_vllm30}"
ENV_KB="${ENV_KB:-env_kb2}"
REPO_DIR="${REPO_DIR:-/hx2-weka/home/nr1713/robust_kernelbench}"
CONDA_SH="${CONDA_SH:-/hx2-weka/home/nr1713/miniconda3/etc/profile.d/conda.sh}"

SERVING_BACKEND="${SERVING_BACKEND:-vllm}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-16384}"   # must stay < MAX_MODEL_LEN
DTYPE="${DTYPE:-bfloat16}"
SLEEP_MODE="${SLEEP_MODE:-0}"       # vLLM /sleep needs --enable-sleep-mode
NUM_PROCS="${NUM_PROCS:-4}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}"
EVAL_MEM_GB="${EVAL_MEM_GB:-24}"       # address-space cap per eval child
MAX_JOBS="${MAX_JOBS:-4}"             # bounds nvcc/cudafe++ fan-out
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
NUM_CORRECT_TRIALS="${NUM_CORRECT_TRIALS:-5}"
NUM_PERF_TRIALS="${NUM_PERF_TRIALS:-3}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Optional correctness tolerance overrides (empty => dtype-aware default).
ATOL="${ATOL:-}"
RTOL="${RTOL:-}"

PARENT_PROMPT_TYPE="${PARENT_PROMPT_TYPE:-kernelbench}"

# Only pass --enable_sleep when the server really was started with sleep support.
if [ "${SLEEP_MODE}" = "1" ]; then SLEEP_FLAG="--enable_sleep"; else SLEEP_FLAG=""; fi

# MODE selects which prompt family the rounds use.
MODE="${MODE:-kernelbench}" # kernelbench | normal
if [ "${MODE}" = "kernelbench" ]; then
    JOBS=(
        "1 1 kernelbench"
        "1 4 kernelbench"
        "1 5 kb_multi_stage"
        "4 204 kernelbench"
        "5 205 kb_multi_stage"
    )
else
    JOBS=(
        "1 1 single_stage"
        "1 2 single_stage"
        "1 3 multi_stage"
        "4 202 single_stage"
        "5 203 multi_stage"
    )
fi

# JOBS_OVERRIDE replaces the list above, e.g. to run just the first round:
#   JOBS_OVERRIDE="1 1 kernelbench"
# Use ';' to separate several rounds:
#   JOBS_OVERRIDE="1 1 kernelbench;1 4 kernelbench"
if [ -n "${JOBS_OVERRIDE:-}" ]; then
    JOBS=()
    while IFS= read -r _job; do
        if [ -n "${_job}" ]; then
            JOBS+=("${_job}")
        fi
    done <<< "$(printf '%s' "${JOBS_OVERRIDE}" | tr ';' '\n')"
fi

timestamp=$(date +%Y%m%d_%H%M%S)

SHORT_MODEL_NAME=$(echo "${MODEL_NAME}" | cut -d'/' -f2 | sed 's/[^a-zA-Z0-9]/_/g' | cut -c1-32)
SIGNATURE="${VERSION}_${SHORT_MODEL_NAME}_${PARENT_PROMPT_TYPE}"

output_folder="job_scripts"
mkdir -p "${output_folder}"
output_file="temp_slurm_v2_${timestamp}_${SIGNATURE}.sh"

# Optional pipeline flags, assembled as a bash array so no word-splitting surprises.
PIPELINE_ARGS=()
if [ -n "${NUM_ITEMS:-}" ]; then
    PIPELINE_ARGS+=("--num_items" "${NUM_ITEMS}")
fi
if [ -n "${ATOL}" ]; then PIPELINE_ARGS+=("--atol" "${ATOL}"); fi
if [ -n "${RTOL}" ]; then PIPELINE_ARGS+=("--rtol" "${RTOL}"); fi

{
cat << EOF
#!/bin/bash
#SBATCH --job-name=kb_${SIGNATURE}
#SBATCH --output=logs/kb_%j_${SIGNATURE}.out
#SBATCH --error=logs/kb_%j_${SIGNATURE}.err
#SBATCH --time=${WALLTIME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}

set -uo pipefail

cd ${REPO_DIR}
mkdir -p logs
source ${CONDA_SH}
conda activate ${ENV_MODEL}

# ----------------------------------------------------------------------
# 1. Launch the vLLM server on the first (NUM_GPUS - P1) GPUs
#
# vLLM replaces SGLang here: `--enable-memory-saver` crashed sglang 0.5.16
# (KeyError: 'kv_cache' -> CUDA error: invalid argument -> abort) at both TP=1
# and TP=2. Memory is bounded by --gpu-memory-utilization instead, and CPU
# memory use is bounded by MAX_JOBS (see below).
# ----------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=${MODEL_VISIBLE_DEVICES}

MODEL_NAME="\${MODEL_NAME:-${MODEL_NAME}}"

VLLM_ARGS=(
    --model "\${MODEL_NAME}"
    --served-model-name "\${MODEL_NAME}"
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
    --host 0.0.0.0
    --port ${SERVER_PORT}
    --max-model-len ${MAX_MODEL_LEN}
    --gpu-memory-utilization ${GPU_MEM_UTIL}
    --dtype ${DTYPE}
)
if [ "${SLEEP_MODE}" = "1" ]; then
    VLLM_ARGS+=(--enable-sleep-mode)
fi

setsid python3 -m vllm.entrypoints.openai.api_server "\${VLLM_ARGS[@]}" &
SERVER_PID=\$!

cleanup() {
    echo "Stopping server (process group \${SERVER_PID})..."
    # vLLM forks workers; kill the group so no GPU memory is leaked.
    kill -- "-\${SERVER_PID}" 2>/dev/null || kill "\${SERVER_PID}" 2>/dev/null || true
    wait "\${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Bound nvcc/ninja fan-out: each cudafe++ costs ~2 GB RSS and the compile pool
# plus the model server is what tripped the cgroup OOM killer.
export MAX_JOBS=${MAX_JOBS}
export OMP_NUM_THREADS=${OMP_NUM_THREADS}
# Reduces allocator fragmentation for long-lived kernels (helps avoid OOM).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ----------------------------------------------------------------------
# 2. Wait for the server to be ready
# ----------------------------------------------------------------------
wait_for_server() {
    local base="http://localhost:${SERVER_PORT}"
    local max_attempts="${SERVER_TIME_OUT_SECONDS}"
    local attempt=0
    echo "Waiting for vLLM server to be ready..."
    while true; do
        if curl -s -f "\$base/health" > /dev/null 2>&1 || curl -s -f "\$base/v1/models" > /dev/null 2>&1; then
            echo "Server is ready."
            return 0
        fi
        if ! kill -0 "\${SERVER_PID}" 2>/dev/null; then
            echo "ERROR: vLLM exited during startup (see the log above)." >&2
            return 1
        fi
        attempt=\$((attempt + 1))
        if [ "\$attempt" -ge "\$max_attempts" ]; then
            echo "ERROR: Server did not become ready in time." >&2
            return 1
        fi
        sleep 1
    done
}

wait_for_server || exit 1

conda deactivate
conda activate ${ENV_KB}

# ----------------------------------------------------------------------
# 3. Run the pipeline on the remaining GPU(s)
# ----------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=${KB_VISIBLE_DEVICES}

jobs=(
$(printf '    "%s"\n' "${JOBS[@]}")
)

extras=(
$(for arg in "${PIPELINE_ARGS[@]+"${PIPELINE_ARGS[@]}"}"; do printf '    "%s"\n' "${arg}"; done)
)

PIPELINE_RC=0
python3 -m robust_kernelbench.run_main \\
    --version "${VERSION}" \\
    --model_name "\${MODEL_NAME}" \\
    --parent_prompt_type "${PARENT_PROMPT_TYPE}" \\
    --serving_backend "${SERVING_BACKEND}" \\
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \\
    --online_service_url "http://localhost:${SERVER_PORT}/v1" \\
    --max_model_len ${MAX_MODEL_LEN} \\
    --max_tokens ${MAX_TOKENS} \\
    --max_mem_util ${GPU_MEM_UTIL} \\
    --num_procs ${NUM_PROCS} \\
    --eval_timeout ${EVAL_TIMEOUT} \\
    --eval_mem_gb ${EVAL_MEM_GB} \\
    --num_correct_trials ${NUM_CORRECT_TRIALS} \\
    --num_perf_trials ${NUM_PERF_TRIALS} \\
    --log-level ${LOG_LEVEL} ${SLEEP_FLAG} \\
    "\${extras[@]+"\${extras[@]}"}" \\
    "\${jobs[@]}"
PIPELINE_RC=\$?

# ----------------------------------------------------------------------
# 4. Cleanup happens via the EXIT trap
#
# The run's status is propagated: previously this script always ended with
# `exit 0`, so a job whose every inference request failed (job 1650138) was
# reported as COMPLETED.
# ----------------------------------------------------------------------
if [ "\${PIPELINE_RC}" -ne 0 ]; then
    echo "Pipeline FAILED with exit code \${PIPELINE_RC}; see the log above." >&2
    exit "\${PIPELINE_RC}"
fi
echo "Benchmark finished, server terminated. Alhamdulillah!"
exit 0
EOF
} > "${output_folder}/${output_file}"

chmod +x "${output_folder}/${output_file}"

echo -e "\n=== Job Configuration ==="
echo "Created job file: ${output_file}"
echo "SIGNATURE: ${SIGNATURE}"
echo "Model: ${MODEL_NAME}"
echo "Wall Time: ${WALLTIME}"
echo "NUM GPUS: ${NUM_GPUS} (server=${MODEL_VISIBLE_DEVICES}, pipeline=${KB_VISIBLE_DEVICES})"
echo "JOBS: ("
printf ' "%s"\n' "${JOBS[@]}"
echo ")"
echo "Parent prompt: ${PARENT_PROMPT_TYPE}"
echo "=========================="
echo "Submitting: ${output_file}"
SBATCH_OUTPUT=$(sbatch --parsable "${output_folder}/${output_file}")
echo "Submitted: ${SBATCH_OUTPUT}"
# One row per submission, in explicit named columns (the old row was a single
# space-joined blob, e.g. '1650138 Qwen3_Coder_30B_A3B_Instruct kernelbench').
printf '%s,%s,%s,%s,%s,%s\n' \
    "$(date +%Y-%m-%d_%H:%M:%S)" "submitted" "${SBATCH_OUTPUT}" \
    "${VERSION}" "${SHORT_MODEL_NAME}" "${PARENT_PROMPT_TYPE}" >> exp_db.csv
