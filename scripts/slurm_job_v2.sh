# SCRIPT

VERSION="v9_6"
WALLTIME="12:00:00"
NUM_GPUS=3

ENV_MODEL="env_kb"
ENV_KB="env_kb2"

MODEL_NAME="Qwen/Qwen3-Coder-Next"
# MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct"

# PARENT_PROMPT_TYPE="kernelbench"
# JOBS=(
#     "1 1 kernelbench"
#     "1 4 kernelbench"
#     "1 5 kb_multi_stage"
#     "4 204 kernelbench"
#     "5 205 kb_multi_stage"
# )

PARENT_PROMPT_TYPE="normal"
JOBS=(
    "1 1 single_stage"
    "1 4 single_stage"
    "1 5 multi_stage"
    "4 204 single_stage"
    "5 205 multi_stage"
)

timestamp=$(date +%Y%m%d_%H%M%S)


SHORT_MODEL_NAME=$(echo "$MODEL_NAME" | cut -d'/' -f2 | sed 's/[^a-zA-Z0-9]/_/g' | cut -c1-32)
SIGNATURE="${VERSION}_${SHORT_MODEL_NAME}_${PARENT_PROMPT_TYPE}"

output_folder="job_scripts"
mkdir -p "${output_folder}"
output_file="temp_slurm_v2_${timestamp}_${SIGNATURE}.sh" 

cat << EOF > "$output_folder/$output_file"
#!/bin/bash
#SBATCH --job-name=kb_${SIGNATURE}
#SBATCH --output=logs/kb_%j.out
#SBATCH --error=logs/kb_%j.err
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
# 1. Launch SGLang server on GPUs 0 and 1
# ----------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1

MODEL_NAME=${MODEL_NAME}

# Start server in background; save its PID for cleanup
python3 -m sglang.launch_server \
    --model-path  \${MODEL_NAME} \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 30000 &
SERVER_PID=\$!

# ----------------------------------------------------------------------
# 2. Wait for server to be ready
# ----------------------------------------------------------------------
wait_for_server() {
    local url="http://localhost:30000/health"
    local max_attempts=30
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
        sleep 5
    done
}

wait_for_server || { kill \$SERVER_PID; exit 1; }


conda deactivate
conda activate ${ENV_KB}

# ----------------------------------------------------------------------
# 3. Run the benchmark script on GPU 2
# ----------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=2

jobs=(
$(printf ' "%s"\n' "${JOBS[@]}")
)

python3 robust_kernelbench/run_main.py \
    --version "${VERSION}" \
    --model ${SHORT_MODEL_NAME} \
    --parent_prompt_type "${PARENT_PROMPT_TYPE}" \
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
