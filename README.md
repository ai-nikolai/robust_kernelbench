# Robust KernelBench

This is generally a robust version inspired by KernelBench (https://github.com/ScalingIntelligence/KernelBench). 
This code fixes a lot of the runtime problems the original code had: (Like fixing the GPU errors that go unoticed and corrupt the final results, adding distributed compilation, and adding more detailed metrics.)

This repo is also for the paper: "FIL Hypothesis"... (Citation coming soon). 

## Getting started:
1. Installation
```bash
# conda create -n env_robust_kernelbench python=3.11
# conda activate env_robust_kernelbench
pip3 install -r requirements.txt
conda install cuda-nvcc_linux-64 -c conda-forge # in case you are on conda you need this. 
# conda install -c conda-forge gxx_linux-64 gcc_linux-64 make #you don't really need this, but it should resolve the cxx warning.
# export PIP_CACHE_DIR=/workspace/cache
# export TMPDIR=/workspace/cache
```

## Running the code:
1. End-2-end scripts: (1. Generate Kernels, 2. Compile Kernels (distributed), 3. Evaluate Kernels)
```bash
bash ./scripts/run_tsp_jobs.sh #local server (see task-spooler below)
# OR
bash ./scripts/run_slurm_jobs.sh
```

### Sequentially running each part seperately
1. Model Inference scripts (1. Generate Kernels)
```bash
#This is the command that will be run...
python3 robust_kernelbench/run_inference_test_time_scaling.py --experiment_name "exp_local_1" --num_items 3 --trial 1
```

2. Compilation: (2. Compile Kernels (distributed / sequential))
```bash
python3 robust_kernelbench/compile_all_sequential.py --experiment_name exp_local_1 --trial 1 --num_processes 8
```

```bash
python3 robust_kernelbench/compile_all.py --experiment_name exp_local_1 --trial 1 --num_processes 8
```

3. Evaluation: (3. Benchmark and Evaluate Kernels)
```bash
python3 robust_kernelbench/evaluate_all.py --experiment_name "exp_local_1" --trial 1
```


## Running the Analysis:
1. Installation:
```bash
pip3 install -r requirements_analysis.txt
```

2. Running:
See `README_ANALYSIS.md`


---
## Task Spooler (local server SLURM alternative)


1. Installation
```bash
apt update
apt install task-spooler
```

2. Usage:
```bash
 tsp timeout 3600 python3 python_script.py \ 
                        --param_1 "experiment_name" 
#this will timeout in 60 minutes (3600 seconds)
```

```bash
tsp #this shows current available jobs
tsp -c #shows you the output of the last job
```
### References :
- https://github.com/bstee615/shared-task-spooler
- https://github.com/justanhduc/task-spooler


---
## CONDA INSTALL LINUX:
https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2 
```bash
# cd /workspace
mkdir miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda3/miniconda.sh
bash ./miniconda3/miniconda.sh -b -u -p ./miniconda3
```

To activate it next time
```bash
source /workspace/miniconda3/etc/profile.d/conda.sh
# OR
source ~/miniconda3/etc/profile.d/conda.sh

conda activate env_robust_kernelbench 

conda activate env_kb
#it was created using `conda create -n env_robust_kernelbench python=3.11
```

### Slurm:
1. Job
```bash
sbatch scripts/slurm_job_v2.sh
```

2. Interactive
```bash
# salloc --partition=interactive-gpu --gres=gpu:h200:1 --time=01:00:00 --ntasks=1
salloc --partition=interactive-gpu --gres=gpu:h200:3 --time=08:00:00 --ntasks=3
srun --pty --overlap --jobid=100854 bash
tmux attach -t 0
tmux capture-pane -t 0:0.0 -S - && tmux save-buffer ./output.txt
```

## Tmux:
https://tmux.info/docs/cheatsheet

1. Getting it running
```bash
tmux
```

2. Switching between windows:
```
CTRL+b w #switch between windows
CTRL+b c #create new window
Ctrl+b " #split vertical
Ctrl+b d #detach from tmux
```

## LLM Serving

1. Serving LLM
```bash
python3 -m sglang.launch_server \
--model-path Qwen/Qwen3-Coder-Next \
--tensor-parallel-size 2 \
--tool-call-parser qwen3_coder \
--host 0.0.0.0 \
--port 30000
```


2. Testing it is running:
```bash
curl -v http://0.0.0.0:30000/health
curl -v http://0.0.0.0:30000/model_info
```

---
## (C) Nikolai Rozanov - 2025 - Present

<!-- 
read train.py and split it into two files. evaluate.py and model.py evaluate should handel
 all data loading, and the actual training loop and reporting of results. Model.py should have 
all the model and optimiser code. 
-->

<!-- 
python -c "import mini_code_cli; print('OK')"
 -->

<!-- 1. Running sglang
```bash
# create config
cat > config.yaml << EOF
model-path: Qwen/Qwen3-Coder-Next
host: 0.0.0.0
port: 30000
tensor-parallel-size: 2
enable-metrics: true
log-requests: true
EOF

# run server
python -m sglang.launch_server --config config.yaml
``` -->

<!-- Example command to run Qwen-Max-Plus on two GPUs.

    python3 robust_kernelbench/run_inference_test_time_scaling_v2.py --experiment_name exp_local_L1_V8_3_Qwen3_Coder_Next__API0 --model_name Qwen/Qwen3-Coder-Next --max_model_len 16000 --max_tokens 10000 --prompt_type single_stage --num_items 1 --num_samples 1 --trial 1 --previous_trial 1 --level 1 --max_mem_util 0.90 --tensor_parallel_size 2

 -->

<!-- 
git config --global user.name "Nikolai Rozanov"
git config --global user.email "nikolai.rozanov@gmail.com" 
-->