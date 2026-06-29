# Robust KernelBench

This is generally a robust version inspired by KernelBench (https://github.com/ScalingIntelligence/KernelBench). 
This code fixes a lot of the runtime problems the original code had: (Like fixing the GPU errors that go unoticed and corrupt the final results, adding distributed compilation, and adding more detailed metrics.)

This repo is also for the paper: "FIL Hypothesis"... (Citation coming soon). 

## Getting started:
1. Installation
```bash
# conda create -n env_robust_kernelbench python=3.11
pip3 install -r requirements.txt
conda install cuda-nvcc_linux-64 -c conda-forge # in case you are on conda you need this. 
# conda install -c conda-forge gxx_linux-64 gcc_linux-64 make
# export PIP_CACHE_DIR=/workspace/cache
# export TMPDIR=/workspace/cache
```

## Running the code:
1. End-2-end scripts: (1. Generate Kernels, 2. Compile Kernels (distributed), 3. Evaluate Kernels)
```bash
bash ./scripts/run_tsp_jobs.sh #local serverr (see task-spooler below)
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

# conda activate env_robust_kernelbench #it was created using `conda create -n env_robust_kernelbench python=3.11
```


---
## (C) Nikolai Rozanov - 2025 - Present
