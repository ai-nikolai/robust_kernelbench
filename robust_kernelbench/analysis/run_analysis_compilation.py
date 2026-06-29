"""
Usage:

python3 robust_kernelbench/analysis/run_analysis_compilation.py --experiment v4_6
"""

import os
import pandas as pd
import json
from pprint import pprint

from tqdm import tqdm

from ..utils.utils import (
    get_folder_path,
    # get_file_name,
    # get_problem_id_from_file_name
)

# TODO
def find_experiments_many(base_experiment_name, base_experiments_folder="experiments"):
    """e.g. exp_slurm_v1"""
    out_list = []
    
    for experiment_name in os.listdir(base_experiments_folder):
        # print("---")
        # print(f"0 - {experiment_name}")
        if os.path.isdir(os.path.join(os.path.join(base_experiments_folder,experiment_name))):
            # print(f"A - {experiment_name} vs. {base_experiment_name}")
            if base_experiment_name in experiment_name:
                # print(f"B - ")
                out_list.append(experiment_name)
        else:
            print(f"0 FAIL- {experiment_name} not dir?")

    return out_list

def read_json(filename):
    with open(filename) as file:
        data = json.load(file)
    return data


    # pre_compiled: bool = False #whether compile() was successful
    # compiled: bool = False #whether when exec() failed with 'Error building extension'
    # runtime_success: bool = False #whether exec() was successful
    # model_new_available: bool = False #whether ModelNew is available.
    # metadata: dict = {}
    # main_output: str = ""
    # main_error: str = ""
    # main_traceback: str = ""
def output_analysis_compilation(df):
    """Outputs analysis results"""
    out_dict = {}
    df = df.astype({
        "pre_compiled":int, 
        "compiled":int, 
        "runtime_success" : int,
        "model_new_available" : int,
    })
    out_dict["count"] = df["compiled"].count()
    out_dict["pre_compiled"] = df["pre_compiled"].mean()
    out_dict["compiled"] = df["compiled"].mean()
    out_dict["runtime_success"] = df["runtime_success"].mean()
    out_dict["model_new_available"] = df["model_new_available"].mean()

    out_dict["pre_compiled_count"] = df["pre_compiled"].sum()
    out_dict["compiled_count"] = df["compiled"].sum()
    out_dict["runtime_success_count"] = df["runtime_success"].sum()
    out_dict["model_new_available_count"] = df["model_new_available"].sum()

    out_dict["pre_compiled_global"] = df["pre_compiled"].sum() / 100
    out_dict["compiled_global"] = df["compiled"].sum() / 100
    out_dict["runtime_success_global"] = df["runtime_success"].sum() / 100
    out_dict["model_new_available_global"] = df["model_new_available"].sum() / 100

    return out_dict

def full_analysis(base_experiment_name, trial=1):
    """Running the full analysis..."""
    COMPILATIONS_CSV = "compilations.csv"
    METADATA_JSON = "metadata.json"

    full_analysis_list = []

    experiments = find_experiments_many(base_experiment_name)

    print(f"[main: analysis] We found {len(experiments)} experiments")

    for experiment_name in tqdm(experiments):
        try:
            folder_path = get_folder_path(experiment_name, trial)

            csv_name = os.path.join(folder_path, COMPILATIONS_CSV)
            json_name = os.path.join(folder_path, METADATA_JSON)
            df = pd.read_csv(csv_name)
            analysis = output_analysis_compilation(df)
            metadata = read_json(json_name)
            analysis.update(metadata)
            full_analysis_list.append(analysis)
        except Exception as e:
            print("-------")
            print(e)
            print(f"Skipping {folder_path}.")

    df_out = pd.DataFrame(full_analysis_list)

    return df_out



def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=1, help="Which trial you are currently running...")
    parser.add_argument("--experiment_name", type=str,  help="experiment names")
    parser.add_argument("--level", type=int, default=1, help="Which level to run...")

    # TODO: add lora support

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    # folder_path = get_folder_path(args.experiment_name, args.trial)
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)

    df_analysis = full_analysis(args.experiment_name, args.trial)
    # print(df_analysis.head())
    file_name = os.path.join(output_dir,f"{args.experiment_name}_analysis_compilation_{args.trial}.csv")
    df_analysis.to_csv(file_name, index=False)


    df_summary_analysis = df_analysis[['model_name','prompt_type','count','pre_compiled_global','compiled_global','runtime_success_global','model_new_available_global']]
    print(f"Summary Table:\n{df_summary_analysis}")

    file_name_2 = os.path.join(output_dir,f"{args.experiment_name}_analysis_compilation_summary_{args.trial}.csv") 
    df_summary_analysis.to_csv(file_name_2, index=False)

