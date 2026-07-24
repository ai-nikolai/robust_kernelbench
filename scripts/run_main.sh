# ----------------------------------------------------------------------
# 0. PARAMS to Set
# ----------------------------------------------------------------------

MODEL_NAME=Qwen/Qwen3-Coder-Next
TOOL_PARSER="qwen3_coder"

MODEL_ARGS=""
if [ -n "${NUM_ITEMS}" ];then
    MODEL_ARGS="$MODEL_ARGS --tool-call-parser ${TOOL_PARSER}"
fi


VERSION=v9_6

PARENT_PROMPT_TYPE="kernelbench"

JOBS=(
 "1 1 kernelbench"
 "1 4 kernelbench"
 "1 5 kb_multi_stage"
 "4 204 kernelbench"
 "5 205 kb_multi_stage"
)

NUM_ITEMS=1

ARGS=""
if [ -n "${NUM_ITEMS}" ];then
    ARGS="$ARGS --num_items ${NUM_ITEMS}"
fi





# ----------------------------------------------------------------------
# 1. Launch SGLang server on GPUs 0 and 1
# ----------------------------------------------------------------------
source /hx2-weka/home/nr1713/miniconda3/etc/profile.d/conda.sh
conda activate env_kb

export CUDA_VISIBLE_DEVICES=0,1

# Start server in background; save its PID for cleanup
python3 -m sglang.launch_server \
    --model-path ${MODEL_NAME} \
    --tensor-parallel-size 2 \
    $MODEL_ARGS \
    --host 0.0.0.0 \
    --port 30000

SERVER_PID=$!

# ----------------------------------------------------------------------
# 2. Wait for server to be ready
# ----------------------------------------------------------------------
wait_for_server() {
    local url="http://localhost:30000/health"
    local max_attempts=600
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
        sleep 1
    done
}

wait_for_server || { kill $SERVER_PID; exit 1; }


conda deactivate
conda activate env_kb2

# ----------------------------------------------------------------------
# 3. Run the benchmark script on GPU 2
# ----------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=2

python3 robust_kernelbench/run_main.py \
    --version ${VERSION} \
    --model ${MODEL_NAME} \
   --parent_prompt_type ${PARENT_PROMPT_TYPE} \
    --reset_experiments \
    ${ARGS} \
    "${JOBS[@]}"
