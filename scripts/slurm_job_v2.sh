# SCRIPT

VERSION="v9_6_test2"
WALLTIME="15:00:00"
NUM_GPUS=3

P1=2
P2=1
# OR, if GPU Collacation (seems like we can pause the sglang server...)
P1=1
P2=1 

MODEL_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-P2)))
KB_VISIBLE_DEVICES=$((NUM_GPUS-P1))
TENSOR_PARALLEL_SIZE=$((NUM_GPUS-P1))

SERVER_TIME_OUT_SECONDS=1800 #30minutes

ENV_MODEL="env_model2"
ENV_KB="env_kb2"

# NUM_ITEMS=1


# MODEL_NAME="sgl-project/DeepSeek-V4-Flash-FP8"
# MODEL_NAME="deepreinforce-ai/Ornith-1.0-397B-FP8"
# MODEL_NAME="deepreinforce-ai/Ornith-1.0-35B"

    # --tool-call-parser qwen3_coder \
    # --reasoning-parser qwen3 \

# MODEL_NAME="Qwen/Qwen3-Coder-Next" #removed from cache...
MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct"

PARENT_PROMPT_TYPE="kernelbench"
JOBS=(
    "1 1 kernelbench"
    "1 4 kernelbench"
    "1 5 kb_multi_stage"
    "4 204 kernelbench"
    "5 205 kb_multi_stage"
)

# PARENT_PROMPT_TYPE="normal"
# JOBS=(
#     "1 1 single_stage"
#     "1 2 single_stage"
#     "1 3 multi_stage"
#     "4 202 single_stage"
#     "5 203 multi_stage"
# )

timestamp=$(date +%Y%m%d_%H%M%S)


SHORT_MODEL_NAME=$(echo "$MODEL_NAME" | cut -d'/' -f2 | sed 's/[^a-zA-Z0-9]/_/g' | cut -c1-32)
SIGNATURE="${VERSION}_${SHORT_MODEL_NAME}_${PARENT_PROMPT_TYPE}"

output_folder="job_scripts"
mkdir -p "${output_folder}"
output_file="temp_slurm_v2_${timestamp}_${SIGNATURE}.sh" 


ARGS=""
if [ -n "${NUM_ITEMS}" ];then
    ARGS="$ARGS --num_items ${NUM_ITEMS}"
fi

cat << EOF > "$output_folder/$output_file"
#!/bin/bash
#SBATCH --job-name=kb_${SIGNATURE}
#SBATCH --output=logs/kb_%j_${SIGNATURE}.out
#SBATCH --error=logs/kb_%j_${SIGNATURE}.err
#SBATCH --time=${WALLTIME}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G



cd /hx2-weka/home/nr1713/robust_kernelbench
mkdir -p logs
source /hx2-weka/home/nr1713/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_MODEL}
# ----------------------------------------------------------------------
# 1. Launch SGLang server on GPUs first NUM_GPUS-1 GPUs.
# ----------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=${MODEL_VISIBLE_DEVICES}
export SGLANG_DSV4_FP4_EXPERTS=0 #maybe needed for SGLang on H200?

MODEL_NAME=${MODEL_NAME}

# Start server in background; save its PID for cleanup
python3 -m sglang.launch_server \
    --model-path  \${MODEL_NAME} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --host 0.0.0.0 \
    --port 30000 \
    --enable-memory-saver &
SERVER_PID=\$!

# ----------------------------------------------------------------------
# 2. Wait for server to be ready
# ----------------------------------------------------------------------
wait_for_server() {
    local url="http://localhost:30000/health"
    local max_attempts=${SERVER_TIME_OUT_SECONDS}
    local attempt=0
    echo "Waiting for SGLang server to be ready..."
    while true; do
        if curl -s -f "\$url" > /dev/null 2>&1; then
            echo "Server is ready."
            return 0
        fi
        attempt=\$((attempt + 1))
        if [ \$attempt -ge \$max_attempts ]; then
            echo "ERROR: Server did not become ready in time."
            return 1
        fi
        sleep 1
    done
}

wait_for_server || { kill \$SERVER_PID; exit 1; }


conda deactivate
conda activate ${ENV_KB}

# ----------------------------------------------------------------------
# 3. Run the benchmark script on last GPU
# ----------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=${KB_VISIBLE_DEVICES}

jobs=(
$(printf ' "%s"\n' "${JOBS[@]}")
)

python3 robust_kernelbench/run_main.py \
    --version "${VERSION}" \
    --model ${SHORT_MODEL_NAME} \
    --parent_prompt_type "${PARENT_PROMPT_TYPE}" \
    $ARGS \
    "\${jobs[@]}"

# ----------------------------------------------------------------------
# 4. Cleanup: kill the server when done
# ----------------------------------------------------------------------
kill $SERVER_PID; 

echo "Benchmark finished, server terminated. Alhamdulillah!"

exit 1;
EOF
echo -e "\n=== Job Configuration ==="
echo "Created job file: $output_file"
echo "SIGNATURE: $SIGNATURE"
echo "Wall Time: $WALLTIME"
echo "NUM GPUS: $NUM_GPUS"
echo "JOBS: (
$(printf ' "%s"\n' "${JOBS[@]}")
)"
echo "Parent prompt: $PARENT_PROMPT_TYPE"
echo "=========================="
echo "Submitting: $output_file"
SBATCH_OUTPUT=$(sbatch --parsable "$output_folder/$output_file")
echo "Submitted: $SBATCH_OUTPUT"
echo "$SBATCH_OUTPUT,submitted,'${SIGNATURE}'" >> exp_db.csv






# With FILTER
# python3 robust_kernelbench/run_main.py \
#     --version "v9_1" \
#     --model qwen_coder_next \
#     --parent_prompt_type "${PARENT_PROMPT_TYPE}" \
#     --filter \
#     "\${jobs[@]}"

# Test job
# python3 robust_kernelbench/run_main.py \
#     --model default2 \
#     --reset_experiments \
#     --num_items 1 \
#     "\${jobs[@]}"
