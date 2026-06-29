"""
USAGE:

python3 robust_kernelbench/analysis/run_analysis_test_time_scaling.py --experiment v4_5 --trial1 5 --trial2 4 --trial1_name "single stage" --trial2_name "multi stage"
python3 robust_kernelbench/analysis/run_analysis_test_time_scaling.py --experiment v4_5 --trial1 1 --trial2 4 --trial1_name "no test time" --trial2_name "multi stage"
python3 robust_kernelbench/analysis/run_analysis_test_time_scaling.py --experiment v4_7 --trial1 2 --trial2 3



python3 robust_kernelbench/analysis/run_analysis_test_time_scaling.py --experiment exp_slurm_v4_8_Qwen2_5_Coder_7B_Instruct_normal_20260112_200549 --trial1 4 --trial2 14
python3 robust_kernelbench/analysis/run_analysis_test_time_scaling.py --experiment exp_slurm_v4_8_Qwen2_5_Coder_7B_Instruct_kernelbench_20260112_200538 --trial1 204 --trial2 207


python3 robust_kernelbench/analysis/run_analysis_test_time_scaling.py --experiment exp_local_L1_V8_3_deepseek_v3_1_terminus__API1 --trial1 1 --trial2 4
python3 robust_kernelbench/analysis/run_analysis_test_time_scaling.py --experiment exp_local_L1_V8_3_deepseek_v3_1_terminus__API1 --trial1 1 --trial2 7


python3 robust_kernelbench/analysis/run_analysis_test_time_scaling.py --experiment exp_local_L1_V8_3_glm_5__API1 --trial1 1 --trial2 4
python3 robust_kernelbench/analysis/run_analysis_test_time_scaling.py --experiment exp_local_L1_V8_3_glm_5__API1 --trial1 1 --trial2 7


# 1base
# 2single
# 3multi
# 4kb
# 5multi (2)
# 6single (2)
# 7kb_multi

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

def add_fast(row):
    """ """
    # print(row)
    try:
        EPSILON=1e-8
        fast_1 = int(bool( 1 <= ( float(row["runtime_original"]) / float(row["runtime"] + EPSILON) ) ) )  if row["runtime"] > 0 else int(0)
        fast_2 = int(bool( 2 <= ( float(row["runtime_original"]) / float(row["runtime"] + EPSILON) ) ) )  if row["runtime"] > 0 else int(0)
        # print(fast_1)
        # print(type(fast_1))
        return {
            "fast_1" : fast_1,
            "fast_2" : fast_2
        }
    except Exception as e:
        print("---\n[add fast] Exception:")
        print(e)
        print(row)
        print()
        return None

def output_evaluation_formatting(df):
    """Outputs analysis results"""

    df[["fast_1","fast_2"]] = df.apply(add_fast,axis=1, result_type="expand")
    # print(df.head())
    # input()
    df = df.astype({
        "compiled":int, 
        "correctness":int, 
        "runtime_success" : int,
        "loading_success" : int,
        "correctness_success": int,
        "timing_success": int,
        "fast_1": int,
        "fast_2": int,
    })
    return df

def output_compilation_formatting(df):
    """Outputs analysis results"""
    df = df.astype({
        "pre_compiled":int, 
        "compiled":int, 
        "runtime_success" : int,
        "model_new_available" : int,
    })

    return df

COMP_ENDING="__comp"
EVAL_ENDING="__eval"
T1_ENDING="__t1"
T2_ENDING="__t2"

COLUMNS_OF_INTEREST = [
    f"pre_compiled", 
    f"compiled{COMP_ENDING}", #double
    f"runtime_success{COMP_ENDING}",#double
    f"model_new_available",
    f"compiled{EVAL_ENDING}", #double
    f"runtime_success{EVAL_ENDING}",#double
    f"loading_success",
    f"correctness_success",
    # f"timing_success",
    f"correctness",
    "fast_1",
    "fast_2"
]

def find_highest_key(row):
    """Helper function to find highest key."""
    # print(row)
    try:
        for key in COLUMNS_OF_INTEREST[::-1]:
            if row[key] and int(row[key])!=0:
                return {"highest_key" : key}
        print(f"\n\n===s\n{key}\n--{row}\n++")
        return {"highest_key" : key+"_failed"}
    except Exception as e:
        print("---\n[find highest] Exception:")
        print(e)
        print(row)
        print()
        return {"highest_key" : "NONE"}

def load_trial_data(base_experiment_name, trial):
    """Load evaluation data for a specific trial"""
    print(f"[Data Loading] Loading data. {base_experiment_name} trial={trial}")
    EVALUATIONS_CSV = "evaluations.csv"
    COMPILATIONS_CSV = "compilations.csv"
    METADATA_JSON = "metadata.json"

    experiments = find_experiments_many(base_experiment_name)
    all_data = []

    for experiment_name in tqdm(experiments, desc=f"Loading trial {trial}"):
        try:
            evaluations_dict = {}
            compilations_dict = {}

            folder_path = get_folder_path(experiment_name, trial)
            json_name = os.path.join(folder_path, METADATA_JSON)

            csv_name = os.path.join(folder_path, EVALUATIONS_CSV)
            df = pd.read_csv(csv_name)
            df = output_evaluation_formatting(df)

            metadata = read_json(json_name)
            for key, value in metadata.items():
                df[key] = value
            # evaluations_dict = output_analysis(df)

            csv_name2 = os.path.join(folder_path, COMPILATIONS_CSV)
            df2 = pd.read_csv(csv_name2)
            df2 = output_compilation_formatting(df2)
            for key, value in metadata.items():
                df2[key] = value
            # evaluations_dict.update(compilations_dict)

            # df['experiment_name'] = experiment_name
            # df['trial'] = trial
            df['trial'] = trial
            df2['trial'] = trial

            if (not df.empty) and (not df2.empty):
                df_combined = pd.merge(
                    df,
                    df2,
                    on=['problem_id', 'sample_id', 'model_name'],
                    suffixes=(COMP_ENDING, EVAL_ENDING),
                    how='outer',

                )
                # Ensure all columns from COLUMNS_OF_INTEREST are present
                for col in COLUMNS_OF_INTEREST:
                    if col not in df_combined.columns:
                        df_combined[col] = 0

                # df_combined.fillna(int(0))
                df_combined = df_combined.fillna(0)

                df_combined["highest_key"] = df_combined.apply(find_highest_key, axis=1, result_type="expand")
                all_data.append(df_combined)
            else:
                print(f"[Data LOADING] Skipping...{df.empty} and {df2.empty}")
                print(df)
                print("===")
                print(df2)
        except Exception as e:
            print(f"[Data Loading] Error loading {experiment_name} trial {trial}: {e}")

    print(f'[Data Loading] type({type(all_data)}) ')

    out_df = pd.concat(all_data, ignore_index=True)
    return out_df


def count_movement(row):
    """Helper function to find highest key."""
    # print(row)
    try:
        model_name = row["model_name"]
        key1 = row[f"highest_key{T1_ENDING}"]
        key2 = row[f"highest_key{T2_ENDING}"]

        return {
            model_name : {
                "transition" : {f"{key1}->{key2}" : 1},
                "problem_id" : row["problem_id"],
            }
        }
    except Exception as e:
        print("---\n[count movement] Exception:")
        print(e)
        print(row)
        print()
        return None


def compare_trials(base_experiment_name, trial1=1, trial2=2):
    """Compare two trials based on problem_id grouping"""

    df_trial1 = load_trial_data(base_experiment_name, trial1)
    df_trial2 = load_trial_data(base_experiment_name, trial2)

    print("[COMPARE ANALYSIS]")
    print(f"----1 ({trial1})")
    print(df_trial1.head())
    print(f"----2 ({trial2})")
    print(df_trial2.head())

    # joined_df = pd.concat([df_trial1,df_trial2], ignore_index=True)
    
    # todo... do actual analysis...

    # Group by problem_id and model_name, then calculate confusion-like matrices
    # For each column, we'll create a cross-tabulation between trial1 and trial2 values

    # # First, separate trial1 and trial2 data
    # df_t1 = joined_df[joined_df['trial'] == trial1].copy()
    # df_t2 = joined_df[joined_df['trial'] == trial2].copy()

    # Merge on problem_id, sample_id, and model_name to align corresponding rows
    merged = pd.merge(
        df_trial1,
        df_trial2,
        on=['problem_id', 'model_name'],
        suffixes=(T1_ENDING, T2_ENDING),
        how='inner'
    )

    counts = merged.apply(count_movement, axis=1)

    return merged, counts

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=1, help="Which trial you are currently running...")
    parser.add_argument("--trial1", type=int, default=1, help="First trial to compare")
    parser.add_argument("--trial2", type=int, default=2, help="Second trial to compare")
    parser.add_argument("--trial1_name", type=str, default=None, help="Optional name for trial 1 (for plot labels only)")
    parser.add_argument("--trial2_name", type=str, default=None, help="Optional name for trial 2 (for plot labels only)")
    parser.add_argument("--experiment_name", type=str,  help="experiment names")
    parser.add_argument("--level", type=int, default=1, help="Which level to run...")
    # parser.add_argument("--compare_trials", action='store_true', help="Enable trial comparison mode")

    # TODO: add lora support

    args = parser.parse_args()
    return args



if __name__=="__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    args = get_args()

    # Determine display names for trials
    trial1_display = args.trial1_name if args.trial1_name else f'Trial {args.trial1}'
    trial2_display = args.trial2_name if args.trial2_name else f'Trial {args.trial2}'

    # if args.compare_trials:
    print(f"Comparing trial {args.trial1} and trial {args.trial2} for {args.experiment_name}")
    merged_data, counts = compare_trials(args.experiment_name, args.trial1, args.trial2)

    print("[ANALYSIS] Finished")
    # Convert counts to DataFrame and aggregate
    # First, we need to accumulate counts across all model names
    model_counts = {}
    problem_ids_and_transitions = {}
    # print(counts)
    for count_dict in counts:
        print("====1")
        print(count_dict)
        if count_dict is not None:
            for model_name, model_data in count_dict.items():
                if model_name not in model_counts:
                    model_counts[model_name] = {}
                if model_name not in problem_ids_and_transitions:
                    problem_ids_and_transitions[model_name] = {}
                for transition, count in model_data["transition"].items():
                    if transition not in model_counts[model_name]:
                        model_counts[model_name][transition] = 0
                    model_counts[model_name][transition] += count

                    if transition not in problem_ids_and_transitions[model_name]:
                        problem_ids_and_transitions[model_name][transition] = []
                    problem_ids_and_transitions[model_name][transition].append(model_data["problem_id"])
    # print("\n=== Movement Summary ===")
    # print(counts_summary)

    # Define ordering based on COLUMNS_OF_INTEREST (higher index = better state)
    stage_order = {col: idx for idx, col in enumerate(COLUMNS_OF_INTEREST)}

    def transition_score(trans):
        """Calculate score for transition ordering. Higher is better improvement."""
        if '->' in trans:
            from_stage, to_stage = trans.split('->')
            from_score = stage_order.get(from_stage, -1)
            to_score = stage_order.get(to_stage, -1)
            # Prioritize by from_stage (initial state), then by to_stage (final state)
            return (from_score, to_score)
        return (-1, -1)

    def is_improvement(trans):
        """Determine if transition is an improvement (True), regression (False), or neutral (None)."""
        if '->' in trans:
            from_stage, to_stage = trans.split('->')
            from_score = stage_order.get(from_stage, -1)
            to_score = stage_order.get(to_stage, -1)
            if to_score > from_score:
                return True  # Improvement
            elif to_score < from_score:
                return False  # Regression
        return None  # Neutral or unknown

    # Create a plot for each model
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)

    for model_name, transitions in model_counts.items():
        counts_summary = pd.Series(transitions).sort_values(ascending=False)
        # Sort counts_summary by transition score (descending for better states first)
        counts_summary_sorted = counts_summary.iloc[
            sorted(range(len(counts_summary)),
                   key=lambda i: transition_score(counts_summary.index[i]),
                   reverse=False)
        ]

        print(f"\n=== Movement Summary for {model_name} ===")
        print(counts_summary_sorted)



        # Create visualization
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 6))
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Bar plot of movements with color coding
        total_improvements = 0
        total_regressions = 0
        total_same = 0
        if not counts_summary_sorted.empty:
            # Determine colors based on improvement
            colors = []
            # for trans in counts_summary_sorted.index:
            for idx, value in enumerate(counts_summary_sorted):
                trans = counts_summary_sorted.index[idx]
                improvement = is_improvement(trans)
                if improvement is True:
                    total_improvements+=value
                    colors.append('lightseagreen')
                elif improvement is False:
                    total_regressions+=value
                    colors.append('coral')
                else:
                    total_same+=value
                    colors.append('gray')

            total_count = total_same + total_regressions + total_improvements

            counts_summary_sorted.plot(kind='bar', ax=ax1, color=colors, legend=False)
            ax1.set_title(f'Success Stage Transitions - {model_name}\n{trial1_display} → {trial2_display} (Total="{total_count}")', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Transition', fontsize=12)
            ax1.set_ylabel('Count', fontsize=12)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            ax1.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for i, v in enumerate(counts_summary_sorted):
                ax1.text(i, v + max(counts_summary_sorted) * 0.01, str(int(v)),
                        ha='center', va='bottom', fontsize=9)

            # Add legend for color coding
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightseagreen', label=f'Improvement={total_improvements}'),
                Patch(facecolor='coral', label=f'Regression={total_regressions}'),
                Patch(facecolor='gray', label=f'Neutral={total_same}')
            ]
            ax1.legend(handles=legend_elements, loc='upper left')

            # Save figure
            safe_model_name = model_name.replace('/', '_').replace('\\', '_')
            output_path = os.path.join(output_dir, f'{args.experiment_name}_{safe_model_name}_trial{args.trial1}_vs_trial{args.trial2}_transitions.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n[VISUALIZATION] Saved to: {output_path}")
            plt.close(fig)
    
    pprint(problem_ids_and_transitions)


