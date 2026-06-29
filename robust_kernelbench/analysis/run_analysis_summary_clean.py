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
        print("---")
        print(f"0 - {experiment_name}")
        if os.path.isdir(os.path.join(os.path.join(base_experiments_folder,experiment_name))):
            print(f"A - {experiment_name} vs. {base_experiment_name}")
            if base_experiment_name in experiment_name:
                print(f"B - ")
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


    out_dict["compiled_global"] = df["compiled"].sum() / 100
    out_dict["runtime_success_global"] = df["runtime_success"].sum()/ 100
    out_dict["loading_success_global"] = df["loading_success"].sum()/ 100
    out_dict["correctness_success_global"] = df["correctness_success"].sum()/ 100
    out_dict["timing_success_global"] = df["timing_success"].sum()/ 100
    out_dict["correctness_global"] = df["correctness"].sum()/ 100

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
    out_dict["speedup_overall_global"] = out_dict["speedup_overall_count"] / 100.0

    out_dict["fast_0_normalised"] = out_dict["fast_0_count"] / total_count
    out_dict["fast_0_global"] = out_dict["fast_0_count"] / 100.0

    out_dict["fast_1_normalised"] = out_dict["fast_1_count"] / total_count
    out_dict["fast_1_global"] = out_dict["fast_1_count"] / 100.0

    out_dict["fast_2_normalised"] = out_dict["fast_2_count"] / total_count
    out_dict["fast_2_global"] = out_dict["fast_2_count"] / 100.0

    out_dict["compiled_global"] = out_dict["count"] / 100.0

    return out_dict

def output_analysis_compilation(df):
    """Outputs analysis results"""
    out_dict = {}
    df = df.astype({
        "format_passed":int,
        "pre_compiled":int, 
        "compiled":int, 
        "runtime_success" : int,
        "model_new_available" : int,
    })
    out_dict["count"] = df["compiled"].count()
    try:
        out_dict["format_passed"] = df["format_passed"].mean()
        out_dict["format_passed_count"] = df["format_passed"].sum()
        out_dict["format_passed_global"] = df["format_passed"].sum() / 100.0

    except:
        print("[`format_passed` not present]")

    out_dict["pre_compiled"] = df["pre_compiled"].mean()
    out_dict["compiled_compiled"] = df["compiled"].mean()
    out_dict["runtime_success_compiled"] = df["runtime_success"].mean()
    out_dict["model_new_available"] = df["model_new_available"].mean()

    out_dict["pre_compiled_count"] = df["pre_compiled"].sum()
    out_dict["compiled_count"] = df["compiled"].sum()
    out_dict["runtime_success_count"] = df["runtime_success"].sum()
    out_dict["model_new_available_count"] = df["model_new_available"].sum()

    out_dict["pre_compiled_global"] = df["pre_compiled"].sum() / 100.0
    out_dict["compiled_global_compiled"] = df["compiled"].sum() / 100.0
    out_dict["runtime_success_global_compiled"] = df["runtime_success"].sum() / 100.0
    out_dict["model_new_available_global"] = df["model_new_available"].sum() / 100.0

    return out_dict

# def full_analysis(base_experiment_name, trial=1):
#     """Running the full analysis..."""
#     EVALUATIONS_CSV = "evaluations.csv"
#     METADATA_JSON = "metadata.json"

#     full_analysis_list = []

#     experiments = find_experiments_many(base_experiment_name)

#     print(f"[main: analysis] We found {len(experiments)} experiments")

#     for experiment_name in tqdm(experiments):
#         try:
#             folder_path = get_folder_path(experiment_name, trial)

#             csv_name = os.path.join(folder_path, EVALUATIONS_CSV)
#             json_name = os.path.join(folder_path, METADATA_JSON)
#             df = pd.read_csv(csv_name)
#             analysis = output_analysis(df)
#             metadata = read_json(json_name)
#             analysis.update(metadata)
#             full_analysis_list.append(analysis)
#         except Exception as e:
#             print("-------")
#             print(e)
#             print(f"Skipping {folder_path}.")

#     df_out = pd.DataFrame(full_analysis_list)

#     return df_out

def load_trial_data(base_experiment_name, trial, base_experiments_folder="experiments"):
    """Load evaluation data for a specific trial"""
    print(f"[Data Loading] Loading data. {base_experiment_name} trial={trial}")
    EVALUATIONS_CSV = "evaluations.csv"
    COMPILATIONS_CSV = "compilations.csv"
    METADATA_JSON = "metadata.json"

    experiments = find_experiments_many(base_experiment_name,base_experiments_folder=base_experiments_folder)
    all_data = []

    for experiment_name in tqdm(experiments, desc=f"Loading trial {trial}"):
        try:
            evaluations_dict = {}
            compilations_dict = {}

            folder_path = get_folder_path(experiment_name, trial, prefix=base_experiments_folder)
            json_name = os.path.join(folder_path, METADATA_JSON)

            csv_name = os.path.join(folder_path, EVALUATIONS_CSV)
            # TODO: FIX thiS
            df = pd.read_csv(csv_name)
            evaluations_dict = output_analysis(df)
            metadata = read_json(json_name)
            evaluations_dict.update(metadata)

            csv_name2 = os.path.join(folder_path, COMPILATIONS_CSV)
            df2 = pd.read_csv(csv_name2)
            compilations_dict = output_analysis_compilation(df2)
            evaluations_dict.update(compilations_dict)

            # df['experiment_name'] = experiment_name
            # df['trial'] = trial
            evaluations_dict['trial'] = trial


            # df_combined = pd.merge(
            #     df,
            #     df2,
            #     on=['problem_id', 'sample_id'],
            #     how='outer',
            # )

            all_data.append(evaluations_dict)
        except Exception as e:
            print(f"[Data Loading] Error loading {experiment_name} trial {trial}: {e}")

    print(f'[Data Loading] type({type(all_data)}) ')

    out_df = pd.DataFrame(all_data)
    return out_df

def compare_trials(base_experiment_name, trial1=1, trial2=2, base_experiments_folder="experiments"):
    """Compare two trials based on problem_id grouping"""

    trials_different = True
    if trial1 == trial2:
        trials_different = False

    df_trial1 = load_trial_data(base_experiment_name, trial1,base_experiments_folder=base_experiments_folder)

    if trials_different:
        df_trial2 = load_trial_data(base_experiment_name, trial2,base_experiments_folder=base_experiments_folder)
    else:
        df_trial2 = pd.DataFrame()
    


    print("[COMPARE ANALYSIS]")
    print(f"----1 ({trial1})")
    print(df_trial1.head())
    if trials_different:
        print(f"----2 ({trial2})")
        print(df_trial2.head())

    if trials_different:
        required_columns = [
            # CONFIG
            "model_name",
            "method_name", #NEW
            # "prompt_type",
            # Compilations
            "format_passed_global",
            "pre_compiled_global",
            "compiled_global_compiled",
            "runtime_success_global_compiled",
            "model_new_available_global",
            # Evaluations
            # "cuda_success_global",
            "compiled_global",
            "runtime_success_global",
            "loading_success_global",
            "correctness_success_global",
            # "timing_success_global",
            "correctness_global",
            # "runtime",
            # "runtime_original",
            # "speedup_overall_global",
            "fast_0_global",
            "fast_1_global",
            "fast_2_global"
            ]
    else:
        required_columns = [
            # CONFIG
            "model_name",
            "method_name", #NEW
            # "prompt_type",
            # Compilations
            "format_passed_global",
            "pre_compiled_global",
            "compiled_global_compiled",
            "runtime_success_global_compiled",
            "model_new_available_global",
            # Evaluations
            # "cuda_success_global",
            "compiled_global",
            "runtime_success_global",
            "loading_success_global",
            "correctness_success_global",
            # "timing_success_global",
            "correctness_global",
            # "runtime",
            # "runtime_original",
            # "speedup_overall_global",
            "fast_0_global",
            "fast_1_global",
            "fast_2_global"
        ]

    for col in required_columns:
        if col not in df_trial1.columns:
            df_trial1[col] = None

    df_trial1 = df_trial1[required_columns]
    df_trial1["method_name"] = "iterative_refinement"
    if trials_different:
        for col in required_columns:
            if col not in df_trial2.columns:
                df_trial2[col] = None

        df_trial2 = df_trial2[required_columns]
        df_trial2["method_name"] = "+inductive"


    combined = pd.concat([df_trial1,df_trial2], axis=0)

    return combined

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=1, help="Which trial you are currently running...")
    parser.add_argument("--trial1", type=int, default=1, help="First trial to compare")
    parser.add_argument("--trial2", type=int, help="Second trial to compare (or None)")
    parser.add_argument("--trial1_name", type=str, default=None, help="Optional name for trial 1 (for plot labels only)")
    parser.add_argument("--trial2_name", type=str, default=None, help="Optional name for trial 2 (for plot labels only)")
    parser.add_argument("--experiment_name", type=str,  help="experiment names")
    parser.add_argument("--base_experiments_folder", type=str, default="experiments", help="experiment names")
    parser.add_argument("--level", type=int, default=1, help="Which level to run...")
    
    # parser.add_argument("--compare_trials", action='store_true', help="Enable trial comparison mode")

    # TODO: add lora support

    args = parser.parse_args()
    return args

if __name__=="__main__":
    # import matplotlib.pyplot as plt
    import numpy as np

    args = get_args()

    if not args.trial2:
        actual_trial2 = args.trial1
        trials_different = False
    else:
        actual_trial2 = args.trial2
        if actual_trial2 == args.trial1:
            trials_different = False
        else:
            trials_different = True

    # Determine display names for trials
    trial1_display = args.trial1_name if args.trial1_name else f'Trial {args.trial1}'
    trial2_display = args.trial2_name if args.trial2_name else f'Trial {actual_trial2}'

    # if args.compare_trials:
    print(f"Comparing trial {args.trial1} and trial {actual_trial2} for {args.experiment_name} in {args.base_experiments_folder}")
    df_comparison = compare_trials(args.experiment_name, args.trial1, actual_trial2, base_experiments_folder=args.base_experiments_folder)

    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    if df_comparison is not None:
        output_file = os.path.join(output_dir,f"{args.experiment_name}_comparison_clean_t{args.trial1}_vs_t{actual_trial2}.csv")
        df_comparison.to_csv(output_file, index=False)
        
        print(f"Comparison saved to {output_file}")
        print(f"Total problems compared: {len(df_comparison)}")
        print(f"\nFirst few rows:")
        print(df_comparison.head())


"""
USAGE:

###############################
# FINAL ANALYSIS
python3 robust_kernelbench/analysis/run_analysis_summary_clean.py --experiment V8_3 --trial1 2 --trial2 3 --base_experiments_folder experiments_backup2
python3 robust_kernelbench/analysis/run_analysis_summary_clean.py  --experiment V8_3 --trial1 12 --trial2 13 --base_experiments_folder experiments_backup2
python3 robust_kernelbench/analysis/run_analysis_summary_clean.py  --experiment V8_3 --trial1 22 --trial2 23 --base_experiments_folder experiments_backup2


"""