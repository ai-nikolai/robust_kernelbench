"""
Usage:

python3 robust_kernelbench/analysis/run_analysis.py --experiment v4_6
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

def output_analysis(df):
    """Outputs analysis results"""
    out_dict = {}
    df = df.astype({
        "compiled":int, 
        "correctness":int, 
        "runtime_success" : int,
        "loading_success" : int,
        "correctness_success": int,
        "timing_success": int,
    })
    out_dict["count"] = df["compiled"].count()
    out_dict["compiled"] = df["compiled"].mean()

    out_dict["runtime_success"] = df["runtime_success"].mean()
    out_dict["loading_success"] = df["loading_success"].mean()
    out_dict["correctness_success"] = df["correctness_success"].mean()
    out_dict["timing_success"] = df["timing_success"].mean()

    out_dict["correctness"] = df["correctness"].mean()

    df_compiled = df[df["compiled"] == 1]
    # out_dict["compiled_count"] = len(df_compiled)
    out_dict["runtime_mean_compiled"] = df_compiled["runtime"].mean() if len(df_compiled) > 0 else 0
    out_dict["runtime_original_mean_compiled"] = df_compiled["runtime_original"].mean() if len(df_compiled) > 0 else 0
    out_dict["runtime_sum_compiled"] = df_compiled["runtime"].sum() if len(df_compiled) > 0 else 0
    out_dict["runtime_original_sum_compiled"] = df_compiled["runtime_original"].sum() if len(df_compiled) > 0 else 0

    # out_dict["runtime_mean"] = df["runtime"].mean()
    # out_dict["runtime_original_mean"] = df["runtime_original"].mean()
    # out_dict["runtime_sum"] = df["runtime"].sum()
    # out_dict["runtime_original_sum"] = df["runtime_original"].sum()

    # index = df["runtime"]>0
    EPSILON=1e-8
    total_count = out_dict["count"]
    out_dict["speedup_overall_count"] = df.apply(
        lambda row: float(row["runtime_original"] / (row["runtime"] + EPSILON)) if row["runtime"] > 0 else 0.0,
        axis=1
    ).sum()

    out_dict["fast_0_count"] = df.apply(
        lambda row: int(bool( 0 < (float(row["runtime_original"] / (row["runtime"] + EPSILON))) ) ) if row["runtime"] > 0 else 0.0,
        axis=1
    ).sum() 

    out_dict["fast_1_count"] = df.apply(
        lambda row: int(bool( 1 <= (float(row["runtime_original"] / (row["runtime"] + EPSILON))) ) ) if row["runtime"] > 0 else 0.0,
        axis=1
    ).sum() 

    out_dict["fast_2_count"] = df.apply(
        lambda row: int(bool( 2 <= (float(row["runtime_original"] / (row["runtime"] + EPSILON))) ) ) if row["runtime"] > 0 else 0.0,
        axis=1
    ).sum()
    

    # normalised , overall
    out_dict["speedup_overall_normalised"] = out_dict["speedup_overall_count"] / total_count
    out_dict["speedup_overall_global"] = out_dict["speedup_overall_count"] / 100

    out_dict["fast_0_normalised"] = out_dict["fast_0_count"] / total_count
    out_dict["fast_0_global"] = out_dict["fast_0_count"] / 100

    out_dict["fast_1_normalised"] = out_dict["fast_1_count"] / total_count
    out_dict["fast_1_global"] = out_dict["fast_1_count"] / 100

    out_dict["fast_2_normalised"] = out_dict["fast_2_count"] / total_count
    out_dict["fast_2_global"] = out_dict["fast_2_count"] / 100

    out_dict["compiled_global"] = out_dict["count"] / 100

    return out_dict

def full_analysis(base_experiment_name, trial=1):
    """Running the full analysis..."""
    EVALUATIONS_CSV = "evaluations.csv"
    METADATA_JSON = "metadata.json"

    full_analysis_list = []

    experiments = find_experiments_many(base_experiment_name)

    print(f"[main: analysis] We found {len(experiments)} experiments")

    for experiment_name in tqdm(experiments):
        try:
            folder_path = get_folder_path(experiment_name, trial)

            csv_name = os.path.join(folder_path, EVALUATIONS_CSV)
            json_name = os.path.join(folder_path, METADATA_JSON)
            df = pd.read_csv(csv_name)
            analysis = output_analysis(df)
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
    file_name = os.path.join(output_dir,f"{args.experiment_name}_analysis_{args.trial}.csv")
    df_analysis.to_csv(file_name, index=False)

    df_summary_analysis = df_analysis[['model_name','prompt_type','compiled_global','fast_0_global','fast_1_global','fast_2_global','fast_0_normalised','fast_1_normalised','fast_2_normalised']]
    print(f"Summary Table:\n{df_summary_analysis}")

    file_name_2 = os.path.join(output_dir,f"{args.experiment_name}_analysis_summary_{args.trial}.csv")
    df_summary_analysis.to_csv(file_name_2, index=False)
