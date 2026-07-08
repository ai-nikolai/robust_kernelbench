#!/usr/bin/env python3
"""
Minimal Python port of the robust_kernelbench pipeline.
Usage:
jobs = (
    "1 1 kernelbench"
    "1 4 kernelbench"
)
python3 run_main.py --parent_prompt_type kernelbench --filter $(jobs[@])
python3 robust_kernelbench/run_main.py --reset_experiments --num_items 1 --filter

Assumes:
- Conda environment is already activated (or `python3` points to the right interpreter).
- API keys (if needed) are already set in the environment.
- Current working directory is the parent of `robust_kernelbench/`.
"""

import subprocess
import datetime
import re
import os

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run robust_kernelbench pipeline')

    # Experiment settings
    parser.add_argument('--previous_trial', type=int, default=1, help='Previous trial number')
    parser.add_argument('--new_trial', type=int, default=1, help='New trial number')
    parser.add_argument('--prompt_type', type=str, default='kernelbench', help='Type of prompt')
    parser.add_argument('--experiment_name', type=str, default='', help='Experiment name (auto-generated if empty)')
    parser.add_argument('--model_name', type=str, default='default', help='Model name')
    parser.add_argument('--parent_prompt_type', type=str, default='kernelbench', help='Parent prompt type')

    # Pipeline stages
    parser.add_argument('--inference', type=int, default=1, choices=[0, 1], help='Run inference (1) or skip (0)')
    parser.add_argument('--compile', type=int, default=1, choices=[0, 1, 2], help='0=skip, 1=parallel, 2=sequential')
    parser.add_argument('--eval', type=int, default=1, choices=[0, 1], help='Run evaluation (1) or skip (0)')

    # Configuration
    parser.add_argument('--level', type=int, default=1, help='Level')
    parser.add_argument('--version', type=str, default='V8', help='Version string')
    parser.add_argument('--use_online', type=int, default=1, choices=[0, 1], help='Use online service (1) or offline (0)')
    parser.add_argument('--num_items', type=int, default=None, help='Limit number of items (None for all)')
    parser.add_argument('--max_mem_util', type=float, default=0.90, help='Maximum memory utilization')
    parser.add_argument('--max_model_len', type=int, default=16000, help='Maximum model length')
    parser.add_argument('--max_tokens', type=int, default=10000, help='Maximum tokens')
    parser.add_argument('--filter', action='store_true',  help='Enable filtering')
    # parser.add_argument('--no_filter', action='store_false', dest='filter', help='Disable filtering')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples')
    parser.add_argument('--num_procs', type=int, default=4, help='Number of processes for parallel compilation')
    parser.add_argument('--online_service_url', type=str, default='http://localhost:30000/v1', help='URL for online service')
    parser.add_argument('--reset_experiments', action='store_true', default=False, help='Reset experiments directory')


    parser.add_argument('jobs', nargs='*', help='List of job strings')
    return parser.parse_args()

args = parse_args()

# Set global variables from args
PREVIOUS_TRIAL = args.previous_trial
NEW_TRIAL = args.new_trial
PROMPT_TYPE = args.prompt_type
EXPERIMENT_NAME = args.experiment_name
MODEL_NAME = args.model_name
PARENT_PROMPT_TYPE = args.parent_prompt_type

INFERENCE = args.inference
COMPILE = args.compile
EVAL = args.eval

LEVEL = args.level
VERSION = args.version
USE_ONLINE = args.use_online

NUM_ITEMS = args.num_items

MAX_MEM_UTIL = args.max_mem_util
MAX_MODEL_LEN = args.max_model_len
MAX_TOKENS = args.max_tokens
FILTER = args.filter
NUM_SAMPLES = args.num_samples
NUM_PROCS = args.num_procs
ONLINE_SERVICE_URL = args.online_service_url

RESET_EXPERIMENTS = args.reset_experiments


# ============================================================================
#  Build experiment name (if not provided)
# ============================================================================
short_model = re.sub(r'[^a-zA-Z0-9]', '_', MODEL_NAME.split('/')[-1])[:32]


# ============================================================================
#  Helper functions.
# ============================================================================

def get_signature(parent_prompt_type):
    signature = f"L{LEVEL}_{VERSION}_{short_model}_{parent_prompt_type}_API{USE_ONLINE}"
    return signature

def run(cmd, desc):
    print(f"\n============= {desc}")
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)



# ============================================================================
#  Common arguments for all Python scripts
# ============================================================================

def build_args(previous_trial=PREVIOUS_TRIAL, new_trial=NEW_TRIAL, prompt_type=PROMPT_TYPE, experiment_name=""): 
    base_args = [
        "--experiment_name", experiment_name,
        "--model_name", MODEL_NAME,
        "--max_model_len", str(MAX_MODEL_LEN),
        "--max_tokens", str(MAX_TOKENS),
        "--prompt_type", prompt_type,
        "--trial", str(new_trial),
        "--previous_trial", str(previous_trial),
        "--level", str(LEVEL),
        "--max_mem_util", str(MAX_MEM_UTIL),
        "--tensor_parallel_size", "1",          # adjust if you have multiple GPUs
        "--num_samples", str(NUM_SAMPLES),
    ]

    if USE_ONLINE == 1:
        base_args.extend(["--online_service_url", ONLINE_SERVICE_URL])
    if FILTER:
        base_args.append("--filter")
    if NUM_ITEMS is not None:
        base_args.extend(["--num_items", str(NUM_ITEMS)])

    return base_args

# ============================================================================
#  Run the pipeline
# ============================================================================

def main_loop(run_config, parent_prompt_type):

    signature = get_signature(parent_prompt_type)

    global EXPERIMENT_NAME

    if not EXPERIMENT_NAME:
        EXPERIMENT_NAME = f"exp_local_{signature}"

    if RESET_EXPERIMENTS:
        import shutil
        target_dir = os.path.join("./experiments", EXPERIMENT_NAME)
        if os.path.isdir(target_dir):
            print(f"RESET_EXPERIMENTS flag is set. Removing existing experiment directory: {target_dir}")
            shutil.rmtree(target_dir)
            print(f"Successfully removed: {target_dir}")

    count = 0
    total_len = len(run_config)
    for previous_trial, new_trial, prompt_type in run_config:
        count += 1
        print("="*60)
        print(f"RUNNING TRIAL: {count}/{total_len} with prev: {previous_trial}, new: {new_trial}, prompt: {prompt_type}")
        
        base_args = build_args(previous_trial, new_trial, prompt_type, EXPERIMENT_NAME)
        # Inference
        if INFERENCE == 1:
            try:
                run(["python3", "robust_kernelbench/run_inference_test_time_scaling_v2.py"] + base_args,
                    "Inference")
            except Exception as e:
                print(f"INFERENCE DID NOT WORK. {e}")
                print(f"Skipping this one: prev: {previous_trial}, new: {new_trial}, prompt: {prompt_type}")
                continue

        # Compilation (clean + compile)
        if COMPILE in (1, 2):
            # Clean cache first
            try:
                run(["python3", "robust_kernelbench/clean_cuda_cache.py",
                    "--experiment", EXPERIMENT_NAME, "--trial", str(new_trial), "--remove"],
                    "Clean CUDA cache")
            except Exception as e:
                print(f"Cache clean-up did not work. {e}")

            if COMPILE == 1:
                script = "robust_kernelbench/compile_all.py"
                mode = "PARALLEL"
            else:
                script = "robust_kernelbench/compile_all_sequential.py"
                mode = "SEQUENTIAL"

            run(["python3", script,
                "--experiment_name", EXPERIMENT_NAME,
                "--trial", str(new_trial),
                "--num_processes", str(NUM_PROCS)],
                f"Compilation ({mode})")

        # Evaluation
        if EVAL == 1:
            # Clean again before eval
            try:
                run(["python3", "robust_kernelbench/clean_cuda_cache.py",
                    "--experiment", EXPERIMENT_NAME, "--trial", str(new_trial), "--remove"],
                    "Clean CUDA cache (pre‑eval)")
            except Exception as e:
                print(f"Cuda cache clean-up did not work before eval. {e}")

            run(["python3", "robust_kernelbench/evaluate_all.py",
                "--experiment_name", EXPERIMENT_NAME,
                "--trial", str(new_trial)],
                "Evaluation")

    print("\nAlhamdulillah, we finished running.")


if __name__=="__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--parent_prompt_type', type=str, help="What the parent prompt type will be..")
    # parser.add_argument('jobs', nargs='*', help='List of job strings')
    # args = parser.parse_args()

    # for job in args.jobs:
    #     run, node, name = job.split()
    #     print(f"Run: {run}, Node: {node}, Name: {name}")
    
    run_config = [
        (1,1,"kernelbench"),
        (1,4,"kernelbench")
    ]

    main_loop(run_config, "kernelbench")




# jobs=(
#     "1 1 single_stage"
#     # "1 12 single_stage"
#     # "1 13 multi_stage"
#     # "1 22 single_stage"
#     # "1 23 multi_stage"
# )

# jobs=(
#     "1 1 kernelbench"
#     "1 2 single_stage"
#     "1 3 multi_stage"
#     "1 8 kb_multi_stage_v2"
#     "1 14 kernelbench"
#     "1 17 kb_multi_stage"
# )