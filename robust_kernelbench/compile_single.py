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

def single_compile_job(custom_model_src, build_dir_new, trial, problem_id, sample_id, experiment_name):
    print(f"[COMPILATION] Entering single_compile_job.")
    try:
        start = datetime.now()
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        try:
            start = datetime.now()
            print(f"-------\n[COMPILATION] ProblemID: {problem_id} -- compiling job for solution id: {sample_id}.")
            print(f"[COMPILATION]\nbuild_dir_new:\t{build_dir_new}")
            context = {}
            # torch.cuda.set_device(device)
            compile_results = run_compilation(device, custom_model_src, context, build_dir_new,)
            end = datetime.now()
            print(f"[COMPILATION] Finished. {end-start}")

            output = compile_results.__dict__
            output["problem_id"] = problem_id
            output["sample_id"] = sample_id
            out_results= [output]       

        except Exception as e:
            print(f"[COMPILATION t:{trial} p:{problem_id} s:{sample_id}] No Result.")
            print(e)
            output = CompileResult().__dict__
            output["problem_id"] = problem_id
            output["sample_id"] = sample_id
            out_results= [output] 

        print(f"[COMPILATION] Writing to file...")
        folder_path = get_folder_path(experiment_name, trial=trial)
        eval_filename = os.path.join(folder_path,'compilations.csv')

        results_df = get_dataframe_results(out_results)

        print("[Compilation] Writing File")
        if os.path.exists(eval_filename):
            results_df.to_csv(os.path.join(eval_filename), index=False, header=False, mode="a")
        else:
            results_df.to_csv(os.path.join(eval_filename), index=False)
        
        print(f"[COMPILATION] Success. Exiting single_compile_job.")

    except Exception as e:
        end = datetime.now()
        print(f"[COMPILATION] FAIL t:{trial} p:{problem_id} s:{sample_id}. Took: {end-start}")
        print(e)




def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1, help="Which level to run...")
    parser.add_argument("--trial", type=int, default=1, help="Which trial you are currently running...")
    parser.add_argument("--experiment", type=str, default="exp_v3", help="experiment names")

    # Important
    parser.add_argument("--code_path", type=str, help="The code path...")
    parser.add_argument("--build_dir_new", type=str, help="the build dir")

    parser.add_argument("--problem_id", type=int, default=None, help="Problem ID")
    parser.add_argument("--sample_id", type=int, default=0, help="Sample ID")

    parser.add_argument("--num_processes", type=int, default=8, help="How many parallel compilations to run.")

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()

    experiment_name = args.experiment

    custom_model_src = read_file(args.code_path)

    single_compile_job(
        custom_model_src=custom_model_src, 
        build_dir_new=args.build_dir_new,
        experiment_name=experiment_name,
        trial=args.trial,
    )
    print("[Compile Single Main] Exiting...")

# Example shell commands to invoke this file:
#
# Basic usage:
# python robust_kernelbench/compile_all.py --experiment_name exp_v3 --trial 1 --problem_id 88

# python robust_kernelbench/compile_all.py --experiment_name exp_v3 --trial 1 --problem_id 98