# Previously Working Version 05.11.2025 (without subprocess). Current version from 05.11.2025
import os
import json
import io
import sys

import subprocess

from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

import gc

from datasets import load_dataset

from pydantic import BaseModel

from tqdm import tqdm

import pandas as pd

import shutil

from datetime import datetime

import collections

import multiprocessing as mp
from multiprocessing import (
    Pool,
    Lock, #requires global values... https://stackoverflow.com/questions/28267972/python-multiprocessing-locks
    Value
)

import traceback
from typing import Any


from utils import (
    get_folder_path,
    get_file_name,
    get_problem_id_from_file_name
)

from evaluate_single import (
    memory_cleanup, 
    eval_kernel_against_ref,
    get_dataframe_results,
    read_file,
    
)

from utils_compile import (
    run_compilation
)

from utils_data import (
    CompileResult,
    KernelExecResult
)

from evaluate_all import (
    find_ending_digit_and_string,
    construct_generation_dict
)

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)  # Ensure spawn is used

global_counter =None
global_lock = None
def job_init(local_lock, local_counter):
    global global_lock, global_counter
    global_lock = local_lock
    global_counter = local_counter

def single_compile_job(device, custom_model_src, build_dir_new, trial, problem_id, sample_id, experiment_name, len_list_of_solutions):
    print("\n=====s")
    print(f"[COMPILATION - compile_all.single_compile_job] Entering single_compile_job.")
    global global_lock, global_counter
    try:
        try:
            with global_counter.get_lock():
                global_counter.value += 1
            print(f"[COMPILATION - compile_all.single_compile_job] Parallel:{global_counter.value} ProblemID: {problem_id} -- compiling job for solution id: {sample_id+1} of {len_list_of_solutions}")
            print(f"[COMPILATION - compile_all.single_compile_job]\nbuild_dir_new:\t{build_dir_new}")
            context = {}
            start = datetime.now()
            # torch.cuda.set_device(device)
            compile_results = run_compilation(device, custom_model_src, context, build_dir_new,)
            end = datetime.now()
            print(f"[COMPILATION - compile_all.single_compile_job] Finished. {end-start}")

            output = compile_results.__dict__
            output["problem_id"] = problem_id
            output["sample_id"] = sample_id
            out_results= [output]       

        except Exception as e:
            print(f"[COMPILATION - compile_all.single_compile_job] t:{trial} p:{problem_id} s:{sample_id}] No Result.")
            print(e)
            output = CompileResult().__dict__
            output["problem_id"] = problem_id
            output["sample_id"] = sample_id
            out_results= [output] 

        finally:
            with global_counter.get_lock():
                global_counter.value -= 1 
            print(f"[COMPILATION - compile_all.single_compile_job] This is how many parallel jobs remain: {global_counter.value}")

        print(f"[COMPILATION - compile_all.single_compile_job] Writing to file...")
        folder_path = get_folder_path(experiment_name, trial=trial)
        eval_filename = os.path.join(folder_path,'compilations.csv')

        results_df = get_dataframe_results(out_results)

        with global_lock:
            print("[COMPILATION - compile_all.single_compile_job] Writing File with Lock")
            if os.path.exists(eval_filename):
                results_df.to_csv(os.path.join(eval_filename), index=False, header=False, mode="a")
            else:
                results_df.to_csv(os.path.join(eval_filename), index=False)
        print("[COMPILATION - compile_all.single_compile_job] Exited Lock")
        
        print(f"[COMPILATION - compile_all.single_compile_job] Success. Exiting single_compile_job.")
        print("-----e")

    except Exception as e:
        end = datetime.now()
        print(f"[COMPILATION - compile_all.single_compile_job] FAIL t:{trial} p:{problem_id} s:{sample_id}. Took: {end-start}")
        print(e)


def compile_all(
    experiment_name, 
    clean_build_dir=True, 
    trial=None, 
    verbose=True, 
    num_processes=8, 
    target_problem_id = None,
    pid_sid_set={},
    ):
    """Main Compile Loop"""

    # folder_path = get_folder_path(experiment_name, trial)
    folder_path_code = get_folder_path(experiment_name, trial, postfix="code")
    folder_path_kernel = get_folder_path(experiment_name, trial, postfix="kernel")

    dict_of_files_to_evaluate = construct_generation_dict(folder_path_code)
    print(f"[Main Compile] Found {len(dict_of_files_to_evaluate)} problems (file_dict keys) to compile.")
    # out_results = []

    # Path("kernels").mkdir(parents=True, exist_ok=True)
    # build_dir = os.path.join("kernels", f"ref_kernels_{CURRENT_DATETIME}")
    
    # MULTIPROCESSING INIT
    lock = Lock()
    counter = Value('i', 0)

    jobs = []
    try:
        # device = "cuda"
        device = torch.cuda.current_device() if torch.cuda.is_available() else None

        # device_name = torch.cuda.get_device_name(device=device)
        for sample_file_name, list_of_solutions in dict_of_files_to_evaluate.items():
                
            problem_id = get_problem_id_from_file_name(sample_file_name)
            # print(f"[Main Compile] Currently compiling problem_id: {problem_id} with {len(list_of_solutions)} solutions.")
            
            if target_problem_id:
                if not target_problem_id == problem_id:
                    print(f"[Main Compile] Skipping {problem_id}")
                    continue

            len_list_of_solutions = len(list_of_solutions)
            
            for sample_id, solution in enumerate(list_of_solutions):
                if (problem_id, sample_id) in pid_sid_set:
                    print(f"[Main Compile] Skipping: pid_sid=({problem_id},{sample_id})")
                    continue

                file_name_solution = get_file_name(problem_id, sample_id)
                solution_kernel_folder_name = file_name_solution+"_solution_kernel"

                build_dir_new = os.path.join(folder_path_kernel, solution_kernel_folder_name)               
                Path(build_dir_new).mkdir(parents=True, exist_ok=True)

                code_path = os.path.join(folder_path_code, solution)
                custom_model_src = read_file(code_path)

                # NOTE: change here device_name
                jobs.append((device, custom_model_src, build_dir_new, trial, problem_id, sample_id, experiment_name, len_list_of_solutions))

        print(f"[Main Compile] We have {len(jobs)} jobs to compile...")
        total_threads = len(os.sched_getaffinity(0))
        # useful_threads = max(total_threads - 20, total_threads / 2, 1)
        selected_threads = min(num_processes, len(jobs))
        print(f"[Main Compile] Launching Compile with num_process:{num_processes}  total_threads:{total_threads}.")
        start = datetime.now()
        
        # MP Approach... (want to also investigate an approach with: from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError)
        with Pool(processes=num_processes, initializer=job_init, initargs=(lock,counter, )) as pool:
            print("[Main Compile] Starting pool...")
            results = pool.starmap(single_compile_job, jobs)
            # results = collections.deque()
            print("[Main Compile] Pool finished...")

            # async_result.get(timeout=) #joined timeout...
        print(results)
        end = datetime.now()
        print(f"[Main Compile] Finished compiling. Overall it took: {end-start}")
        # DEBUGGING A SINGLE JOB
        # exec(jobs[0][1],{})
        # print("[Main] Exec finished")
        # torch.cuda.synchronize(device)

    except Exception as e:
        print("[Main Compile] Totally crashed...")
        print(e)

    print("[Main Compile] Finished.")




def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1, help="Num of Parallel responses")
    parser.add_argument("--prompt_type", type=str, default="normal", help="Which prompt to use...")
    parser.add_argument("--trial", type=int, default=1, help="Which trial you are currently running...")
    parser.add_argument("--experiment_name", type=str, default="exp_v3", help="experiment names")
    parser.add_argument("--temperature", type=float, default=0.6, help="temp for LLM")
    parser.add_argument("--max_tokens", type=str, default=2000, help="max tokens")
    parser.add_argument("--num_items", type=int, help="How many data points to run...")
    parser.add_argument("--level", type=int, default=1, help="Which level to run...")
    parser.add_argument("--problem_id", type=int, help="if you want to run only one problem id...")


    parser.add_argument("--num_processes", type=int, default=8, help="How many parallel compilations to run.")

    # TODO: add lora support

    args = parser.parse_args()
    return args

def get_compiled_pid_sid(experiment_name, trial):
    pid_sid_list = {}
    try:
        folder_path = get_folder_path(experiment_name, trial=trial)
        eval_filename = os.path.join(folder_path,'compilations.csv')

        if os.path.exists(eval_filename):
            existing_compilations = pd.read_csv(eval_filename)
            pid_sid_list = set(
                [
                    x for x in \
                    existing_compilations[["problem_id", "sample_id"]].itertuples(index=False, name=None)
                ]
            )
    
    except Exception as e:
        print(f"[get_compiled pid_sid] failed::{e}")
    
    return pid_sid_list


if __name__=="__main__":
    args = get_args()

    experiment_name = args.experiment_name

    pid_sid_list = get_compiled_pid_sid(experiment_name, args.trial)

    print(f"[Main Compile] Found {len(pid_sid_list)} existing compilations.")

    # print(pid_sid_list)

    results = compile_all(experiment_name=experiment_name, trial=args.trial, num_processes=args.num_processes, target_problem_id = args.problem_id, pid_sid_set=pid_sid_list)
    
    # results_df = get_dataframe_results(results)
    # results_df.head()

# Example shell commands to invoke this file:
#
# Basic usage:
# python robust_kernelbench/compile_all.py --experiment_name exp_v3 --trial 1 --problem_id 88

# python robust_kernelbench/compile_all.py --experiment_name exp_v3 --trial 1 --problem_id 98