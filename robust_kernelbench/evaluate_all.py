# Previously Working Version 05.11.2025 (without subprocess). Current version from 05.11.2025
import os
import json
import io
import sys

import subprocess

from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.cpp_extension
import numpy as np

import gc

from datasets import load_dataset

from pydantic import BaseModel

from tqdm import tqdm

import pandas as pd

import shutil

from datetime import datetime

import multiprocessing as mp
from multiprocessing import Pool,Lock

import traceback

from typing import Any


from utils.utils import (
    get_folder_path,
    get_file_name,
    get_problem_id_from_file_name
)

from evaluate_single import (
    memory_cleanup, 
    eval_kernel_against_ref,
    get_dataframe_results,
    read_file,
    evaluate_single,
    eval_wrapper
)

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)  # Ensure spawn is used


# BASIC DEFINITIONS
level1_representative_subset_problem_ids = [1, 3, 6, 18, 23, 26, 33, 36, 40, 42, 48, 54, 57, 65, 77, 82, 86, 87]
level2_representative_subset_problem_ids = [1, 2, 8, 18, 23, 28, 33, 43]
level3_representative_subset_problem_ids = [1, 5, 8, 11, 20, 33, 38, 43]

CURRENT_DATETIME=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def find_ending_digit_and_string(name):
    """Extracts the name when it is: name_9812"""
    out_digit = []
    out_name = ""
    for idx, c in enumerate(name[::-1]):
        if c in "0123456789":
            out_digit.insert(0,c)
        else:
            break
    out_name = name[:-idx]

    return out_name, "".join(out_digit)

# %%
def construct_generation_dict(folder):
    """
    This functions assumes that inside 'folder' there will be files named according to the KernelBench problem names. 

    E.g. 1_Square_matrix_multiplication_{idx}.py

    It will return:
    {
        '1_Square_matrix_multiplication_' : [<list of matching files>],
    }

    Where hopefully is also inside KernelBench...
    """
    files = os.listdir(folder)

    out_dict = {}

    for file in files:
        if file.endswith(".py"):
            file_basename = file.split(".py")[0]
            file_basename_root, file_digit = find_ending_digit_and_string(file_basename)
            # tmp = file_basename.split("_")
            # if len(tmp) == 1:
            #     file_basename = tmp[0]
            # else:
            #     file_basename = "_".join(tmp[:-1]) + "_"
            
            if out_dict.get(file_basename_root):
                out_dict[file_basename_root].append(file)
            else:
                out_dict[file_basename_root] = [file]
    
    return out_dict

def write_code(fullpath, code_src):
    with open(fullpath,"w") as file:
        file.write(code_src)

def eval_main_loop(
    dataset, 
    experiment_name, 
    clean_build_dir=True, 
    trial=None, 
    verbose=True, 
    compiled_problem_ids=None, 
    level=None, 
    already_evaluated=None,
    pid_sid_set={},
    ):
    """Main Eval Loop"""
    print("[Main Evaluate] Entered `eval_main_loop()`")
    captured_text = "No Capture."
    device = torch.cuda.current_device() if torch.cuda.is_available() else None


    # folder_path = get_folder_path(experiment_name, trial)
    folder_path_code = get_folder_path(experiment_name, trial, postfix="code")
    folder_path_kernel = get_folder_path(experiment_name, trial, postfix="kernel")

    dict_of_files_to_evaluate = construct_generation_dict(folder_path_code)

    out_results = []

    # Path("kernels").mkdir(parents=True, exist_ok=True)
    # build_dir = os.path.join("kernels", f"ref_kernels_{CURRENT_DATETIME}")


    for sample_file_name, list_of_solutions in (pbar := tqdm(dict_of_files_to_evaluate.items())):
        problem_id = get_problem_id_from_file_name(sample_file_name)
        print("\n======Start\n[Main] Main Loop start...")
        if compiled_problem_ids:
            if not problem_id in compiled_problem_ids:
                print(f"Skipping problem_id {problem_id} as it's not in the compiled list.")
                continue

        print(f"[Main Evaluate] Currently evaluating problem_id: {problem_id} with {len(list_of_solutions)} solutions.")

        file_name_reference = get_file_name(problem_id)
        ref_kernel_folder_name = file_name_reference+"_ref_kernel"

        build_dir = os.path.join(folder_path_kernel, ref_kernel_folder_name)
        Path(build_dir).mkdir(parents=True, exist_ok=True)

        tmp = dataset.filter(lambda x: x["problem_id"] == problem_id)
        if tmp:
            sample = tmp[0]
        else:
            print(f"Problem with problem_id {problem_id} is not in the dataset. Skipping.")
            continue

        reference_code_path = os.path.join(build_dir,f"{sample['name']}.py")
        write_code(reference_code_path,sample["code"])


        for sample_id, solution in (pbar2 := tqdm(enumerate(list_of_solutions))):
            if (problem_id, sample_id) in pid_sid_set:
                print(f"[Main Evaluate] Skipping, pid_sid already evaluated: {(problem_id, sample_id)}")
                continue
            
            # pbar2.set_description(f"-------\nProblemID: {problem_id} -- evaluating specific solution id: {sample_id+1} of {len(list_of_solutions)}")
            print(f"-------\n[Main Evaluate] ProblemID: {problem_id} -- evaluating specific solution id: {sample_id+1} of {len(list_of_solutions)}")
            try:
                file_name_solution = get_file_name(problem_id, sample_id)
                solution_kernel_folder_name = file_name_solution+"_solution_kernel"

                build_dir_new = os.path.join(folder_path_kernel, solution_kernel_folder_name)               
                Path(build_dir_new).mkdir(parents=True, exist_ok=True)

                code_path = os.path.join(folder_path_code, solution)

                current_file_path = os.path.dirname(os.path.abspath(__file__))
                eval_file_path = os.path.join(current_file_path,'evaluate_single.py')

                ##################################
                # WITH SUBPROCESS

                python_command_to_run = [
                        'python', 
                        eval_file_path,
                        # Path
                        "--original_path", f"{reference_code_path}",
                        "--new_path", f"{code_path}",
                        "--build_dir", f"{build_dir}",
                        "--build_dir_new", f"{build_dir_new}",
                        # Settings
                        "--experiment", f"{experiment_name}",
                        "--problem_id", f"{problem_id}",
                        "--sample_id", f"{sample_id}",
                        "--level", f"{level}",
                        "--trial", f"{trial}",
                        # other
                    ]
                # kwargs = {
                #     "original_path":reference_code_path, 
                #     "new_path":code_path, 
                #     "build_dir":build_dir, 
                #     "build_dir_new":build_dir_new, 
                #     # other
                #     "device":device, # have to run on GPU
                #     # Admin:
                #     "experiment_folder":experiment_name,
                #     "level":level,
                #     "trial":trial,
                #     "problem_id":problem_id,
                #     "sample_id":sample_id,
                # }

                python_command_string = " ".join(python_command_to_run)

                print(f"[Main] Attempting to run subprocess:\n---\n{python_command_string}\n---\n")
                print(f"[Main] This is the cache location: {torch.utils.cpp_extension._get_build_directory(eval_file_path, verbose=True)}")

                start = datetime.now()
                captured_text = subprocess.run(
                    python_command_to_run,
                    # check=True,
                    capture_output=True,     # capture stdout and stderr
                    text=True,               # decode bytes to str
                    timeout=600 #timeout after 10 minutes.
                )
                end = datetime.now()
                duration = end - start
                print(f"[Main] Subprocess finished. It took {duration}")
                if verbose:
                    print(f"[Main] captured_text:\n----Start----\n\n{captured_text}\n\n----End----") 

                ##################################
                # WITH Multi Processing
                # reference_src = read_file(reference_code_path)
                # custom_src = read_file(code_path)

                # print(f"\n---:\n[Main Evaluate] Running Actual Evaluation now for l:{level}, t: {trial}, p:{problem_id}, s:{sample_id}")
                # start = datetime.now()

                # kwargs = {
                #     "original_path":reference_code_path, 
                #     "new_path":code_path, 
                #     "build_dir":build_dir, 
                #     "build_dir_new":build_dir_new, 
                #     # other
                #     "device":device, # have to run on GPU
                #     # Admin:
                #     "experiment_folder":experiment_name,
                #     "level":level,
                #     "trial":trial,
                #     "problem_id":problem_id,
                #     "sample_id":sample_id,
                # }

                # with Pool(processes=1) as pool:
                #     print(f"Entering Pool {kwargs}")
                #     result = pool.apply_async(evaluate_single, kwds=kwargs)
                #     result.get(timeout=600)  # 10 minutes timeout
        
                # end = datetime.now()
                # print(f"[Main Evaluate] Got Results. Took time: {end-start}") 
                
                # NOTE: OLD CODE
                # custom_src = read_file(code_path)
                
                # print("\n---:\n[Main] Running Actual Evaluation now:")
                # results = eval_kernel_against_ref(
                #     original_model_src = sample["code"],
                #     custom_model_src = custom_src,
                #     seed_num = 42,
                #     num_correct_trials = 3,
                #     num_perf_trials = 3,
                #     verbose  = True,
                #     measure_performance = True,
                #     build_dir = build_dir,
                #     build_dir_new = build_dir_new,
                #     device = torch.cuda.current_device() if torch.cuda.is_available() else None, # have to run on GPU
                # )

                # memory_cleanup(custom_string=f"Final Clean-up: problem_id: {problem_id}, sample_id: {sample_id}", verbose=True)
                # Cleaning-up Evaluation
                # if clean_build_dir:
                #     pass
                #     Skipping this for now...
                #     shutil.rmtree(build_dir_new)
                
                # END OF CLEAN_UP
                # END OF OLD CODE 


            except Exception as e:
                print(f"[Main] THIS EVAL for problem_id:{problem_id} and sample_id:{sample_id} FULLY FAILED")
                print(captured_text)
                print("----")
                print(e)
                # memory_cleanup(custom_string=f"Final Clean-up:(fail) problem_id: {problem_id}, sample_id: {sample_id}", verbose=True)

    print("Done overall with eval loop.")
    return out_results



def get_successful_compilation(experiment_name, trial):
    # Load compilations.csv and filter for successful compilations
    compilations_csv_path = os.path.join(get_folder_path(experiment_name, trial=trial), 'compilations.csv')
    successful_problem_sample_pairs = None

    if os.path.exists(compilations_csv_path):
        compilations_df = pd.read_csv(compilations_csv_path)

        # Filter for rows where model_new_available is True
        successful_compilations = compilations_df[compilations_df['model_new_available'] == True]

        # Extract unique problem_ids and sample_ids
        successful_problem_sample_pairs = list(zip(successful_compilations['problem_id'], successful_compilations['sample_id']))

        print(f"[Main] Found {len(successful_problem_sample_pairs)} successful compilations with model_new_available=True")
    else:
        print(f"[Main] Warning: compilations.csv not found at {compilations_csv_path}. Proceeding with full dataset.")   

    return successful_problem_sample_pairs


def get_evaluation_pid_sid(experiment_name, trial):
    pid_sid_list = {}
    try:
        folder_path = get_folder_path(experiment_name, trial=trial)
        eval_filename = os.path.join(folder_path,'evaluations.csv')

        if os.path.exists(eval_filename):
            existing_compilations = pd.read_csv(eval_filename)
            pid_sid_list = set(
                [
                    x for x in \
                    existing_compilations[["problem_id", "sample_id"]].itertuples(index=False, name=None)
                ]
            )
    
    except Exception as e:
        print(f"[get_evaluations pid_sid] failed::{e}")
    
    return pid_sid_list

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1, help="Num of Parallel responses")
    parser.add_argument("--prompt_type", type=str, default="normal", help="Which prompt to use...")
    parser.add_argument("--trial", type=int, default=1, help="Which trial you are currently running...")
    parser.add_argument("--experiment_name", type=str, default="exp_test_run", help="experiment names")
    parser.add_argument("--temperature", type=float, default=0.6, help="temp for LLM")
    parser.add_argument("--max_tokens", type=str, default=2000, help="max tokens")
    parser.add_argument("--num_items", type=int, help="How many data points to run...")
    parser.add_argument("--level", type=int, default=1, help="Which level to run...")

    # TODO: add lora support

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()

    ds = load_dataset("ai-nikolai/KernelBench")
    level=args.level
    actual_ds = ds[f"level_{level}"]

    experiment_name = args.experiment_name

    successful_problem_sample_pairs = get_successful_compilation(experiment_name, args.trial)
    # Filter dataset to only include successful compilations
    if successful_problem_sample_pairs and len(successful_problem_sample_pairs) > 0:
        successful_problem_ids = set([pid for pid, _ in successful_problem_sample_pairs])
        actual_ds = actual_ds.filter(lambda x: x["problem_id"] in successful_problem_ids)
        print(f"[Main] Filtered dataset to {len(actual_ds)} problems with successful compilations")

        pid_sid_set = get_evaluation_pid_sid(experiment_name,args.trial)
        print(f"[Main] Found already existing evaluations: {len(pid_sid_set)}")
        results = eval_main_loop(actual_ds, experiment_name=experiment_name, trial=args.trial, level=args.level,pid_sid_set=pid_sid_set)
        print("[Main] Done with evaluate all.")
    else:
        print("[Main] Warning: No successful compilations found.")


    try:
        # Ensure compilations.csv and evaluations.csv exist if not present
        output_dir = get_folder_path(experiment_name, args.trial)
        # Path(output_dir).mkdir(parents=True, exist_ok=True)

        compilations_csv = os.path.join(output_dir, 'compilations.csv')
        if os.path.exists(compilations_csv):
            evaluations_csv = os.path.join(output_dir, 'evaluations.csv')
            if not os.path.exists(evaluations_csv):
                print("[Main] WARNING: GENERATING EMPTY evaluations.csv, since compilations.csv exists, but no evaluations.csv.")
                empty_evaluations_df = pd.DataFrame(columns=[
                    'problem_id', 
                    'sample_id', 
                    'cuda_success', 
                    'compiled', 
                    'runtime_success', 
                    'loading_success', 
                    'correctness_success', 
                    'timing_success', 
                    'correctness', 
                    'metadata', 
                    'runtime', 
                    'runtime_stats', 
                    'runtime_original', 
                    'runtime_original_stats', 
                    'main_output', 
                    'main_error', 
                    'main_traceback'
                ])
                empty_evaluations_df.to_csv(evaluations_csv, index=False)
                print(f"[Main] Created empty evaluations.csv at {evaluations_csv}")
            else:
                print(f"[Main] compilations.csv and evaluations.csv files exist. All seems ok.")
    except Exception as e:
        print(f"[Main] Filling in the blanks, did not work for evaluations.csv. {e}")  
    
    print("[Main] Exiting...")

    # results_df = get_dataframe_results(results)
    # results_df.head()

    # Path("out").mkdir(parents=True, exist_ok=True)
    # results_df.to_csv(f'out/evaluations_{experiment_name}_{CURRENT_DATETIME}.csv', index=False)

    # folder_path = get_folder_path(experiment_name, trial=args.trial)
    # results_df.to_csv(os.path.join(folder_path,'evaluations.csv'), index=False)
    