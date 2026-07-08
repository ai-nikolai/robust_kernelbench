#!/bin/bash
#SBATCH --job-name=kb
#SBATCH --output=logs/sglang_benchmark_%j.out
#SBATCH --error=logs/sglang_benchmark_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G



cd /hx2-weka/home/nr1713/robust_kernelbench
mkdir -p logs
source /hx2-weka/home/nr1713/miniconda3/etc/profile.d/conda.sh
conda activate env_kb
# ----------------------------------------------------------------------
# 1. Launch SGLang server on GPUs 0 and 1
# ----------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1

# Start server in background; save its PID for cleanup
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-Coder-Next \
    --tensor-parallel-size 2 \
    --tool-call-parser qwen3_coder \
    --host 0.0.0.0 \
    --port 30000 &
SERVER_PID=$!

# ----------------------------------------------------------------------
# 2. Wait for server to be ready
# ----------------------------------------------------------------------
wait_for_server() {
    local url="http://localhost:30000/health"
    local max_attempts=30
    local attempt=0
    echo "Waiting for SGLang server to be ready..."
    while true; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo "Server is ready."
            return 0
        fi
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            echo "ERROR: Server did not become ready in time."
            return 1
        fi
        sleep 5
    done
}

wait_for_server || { kill $SERVER_PID; exit 1; }


conda deactivate
conda activate env_robust_kernelbench

# ----------------------------------------------------------------------
# 3. Run the benchmark script on GPU 2
# ----------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=2

parent_kernel_type="kernelbench"
jobs=(
    "1 1 kernelbench"
    "1 4 kernelbench"
    "1 5 kb_multi_stage"
    "4 204 kernelbench"
    "5 205 kb_multi_stage"
)

python3 robust_kernelbench/run_main.py \
    --version "v9" \
    --model qwen_coder_next \
    --parent_prompt_type "${parent_kernel_type}" \
    --filter \
    "${jobs[@]}"

# Test job
# python3 robust_kernelbench/run_main.py \
#     --model default2 \
#     --reset_experiments \
#     --num_items 1 \
#     "${jobs[@]}"

# ----------------------------------------------------------------------
# 4. Cleanup: kill the server when done
# ----------------------------------------------------------------------
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "Benchmark finished, server terminated. Alhamdulillah!"