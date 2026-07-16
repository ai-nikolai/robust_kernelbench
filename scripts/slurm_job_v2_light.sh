# SCRIPT

VERSION="v9_4"
WALLTIME="4:00:00"

# MODEL_NAME="Qwen/Qwen3-Coder-Next"
MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct"


PARENT_PROMPT_TYPE="kernelbench"
JOBS=(
    "1 1 kernelbench"
    # "1 4 kernelbench"
    # "1 5 kb_multi_stage"
    # "4 204 kernelbench"
    # "5 205 kb_multi_stage"
)

# PARENT_PROMPT_TYPE="normal"
# JOBS=(
#     "1 1 single_stage"
#     "1 4 single_stage"
#     "1 5 multi_stage"
#     "4 204 single_stage"
#     "5 205 multi_stage"
# )

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
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo "==========================="
echo "NVIDIA ANALYSIS"
which nvidia-smi
which nvcc
nvcc --version
echo "CUDA Home:"
echo \$CUDA_HOME
echo "finished analysis"
nvidia-smi


echo "==========================="
cd /hx2-weka/home/nr1713/robust_kernelbench
mkdir -p logs
source /hx2-weka/home/nr1713/miniconda3/etc/profile.d/conda.sh

conda activate env_robust_kernelbench

# ----------------------------------------------------------------------
# 3. Run the benchmark script
# ----------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0

jobs=(
$(printf ' "%s"\n' "${JOBS[@]}")
)

python3 robust_kernelbench/run_main.py \
    --inference 0 \
    --version "${VERSION}" \
    --model ${SHORT_MODEL_NAME} \
    --parent_prompt_type "${PARENT_PROMPT_TYPE}" \
    --num_samples 1 \
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
