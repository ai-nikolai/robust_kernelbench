
python3 -m sglang.launch_server \
--model-path Qwen/Qwen3-Coder-Next \
--tensor-parallel-size 2 \
--tool-call-parser qwen3_coder \
--host 0.0.0.0 \
--port 30000

jobs=(
    "1 1 kernelbench"
    "1 4 kernelbench"
)

CUDA_VISIBLE_DEVICES=2 python3 robust_kernelbench/run_main.py --model default2 \
--reset_experiments \
--num_items 1 \
"${jobs[@]}"
