"""
USAGE:

python3 analysis/run_analysis_plotting.py --file_path analysis_output/V8_3_comparison_statistical_analysis_2_3.csv
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="analysis_output/V8_3_comparison_statistical_analysis_2_3.csv",help="Path")
    args = parser.parse_args()
    return args

args = get_args()

FILE_PATH = args.file_path
OUT_FILE_PATH = FILE_PATH.replace(".csv","_plotting.png")

# Load the data
df = pd.read_csv(FILE_PATH)


# special_key="Method Name"
special_key="Prompt Type"
# Rename columns for clarity (keeping both mean and std)
required_columns_mean = {
    "model_name": "Model Name",
    # "method_name": "Method Name",
    "prompt_type" : "Prompt Type",
    "format_passed_global_mean": "Format Success",
    "pre_compiled_global_mean": "Syntax Success",
    "compiled_global_compiled_mean": "Compilation Step 1 Success",
    "runtime_success_global_compiled_mean": "Compilation Step 2 Success",
    "model_new_available_global_mean": "Full Compilation Success",
    "compiled_global_mean": "Model Available",
    "runtime_success_global_mean": "Model Init Success",
    "loading_success_global_mean": "Model Loading Success",
    "correctness_success_global_mean": "Full Runtime Success",
    "correctness_global_mean": "Correctness",
    "fast_0_global_mean": "fast 0",
    "fast_1_global_mean": "fast 1",
    "fast_2_global_mean": "fast 2"
}
df.rename(columns=required_columns_mean, inplace=True)

# Melt the dataframe: each metric as a separate row (only mean values)
# id_vars = ["Model Name", "Method Name"]
id_vars = ["Model Name", "Prompt Type"]

value_vars = [v for k, v in required_columns_mean.items() if k not in id_vars]
df_melted = df.melt(id_vars=id_vars, value_vars=value_vars,
                    var_name="Metric", value_name="Mean Value")

# Get unique models
models = df["Model Name"].unique()
print(models)

# HARD CODE MODELS:
# models = [  
    # 'Qwen/Qwen3-4B',
    # 'Qwen/Qwen3-8B',
    # 'Qwen/Qwen3-14B',
    # 'Qwen/Qwen3-32B',
    # 'Qwen/Qwen3-30B-A3B-Instruct-2507',
    # 'Qwen/Qwen3-Coder-30B-A3B-Instruct',
    # 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
    # 'deepseek/deepseek-v3.1-terminus',
    # 'mistralai/devstral-2512',
    # 'openai/gpt-oss-120b'
# ]
NCOLS=2

# models = [  
#     'Qwen/Qwen3-14B',
#     'Qwen/Qwen3-32B',
#     'Qwen/Qwen3-30B-A3B-Instruct-2507',
#     'deepseek/deepseek-v3.1-terminus',
#     'mistralai/devstral-2512',
#     'openai/gpt-oss-120b'
# ]
# NCOLS=3

# Define a nice color palette
palette = sns.color_palette("Set2", 2)

# Create a grid of subplots: one row per model
ncols = NCOLS  # can adjust if you want more columns; here we use 2 for narrower bars
nrows = int(np.ceil(len(models) / ncols))
print(f"NROWS: {nrows}")
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 5 * nrows), sharey=False)

# Flatten axes for easier indexing
axes_flat = axes.flatten() #if nrows > 1 else [axes]

# Sort metrics by the order they appear in the original CSV for consistency
all_metrics = [v for k, v in required_columns_mean.items() if v not in id_vars]
# Remove duplicates while preserving order (just in case)
ordered_metrics = list(dict.fromkeys(all_metrics))

for i, model in enumerate(models):
    ax = axes_flat[i]
    model_data = df_melted[df_melted["Model Name"] == model]
    # Ensure Metric column is categorical with the desired order
    model_data = model_data.copy()
    model_data["Metric"] = pd.Categorical(model_data["Metric"], categories=ordered_metrics, ordered=True)
    # Rename method names and set order
    # method_order = ["IR", "+IB"]
    # method_order = df[f"{special_key}"].unique()
    method_order = ["single_stage","multi_stage", "kernelbench", "kb_multi_stage"]
    model_data[f"{special_key}"] = model_data[f"{special_key}"].replace({
        "iterative_refinement": "IR",
        "+inductive": "+IB"
    })
    # Ensure the hue order
    model_data[f"{special_key}"] = pd.Categorical(model_data[f"{special_key}"],
                                               categories=method_order, ordered=True)

    sns.barplot(data=model_data, x="Metric", y="Mean Value", hue=f"{special_key}",
                palette=palette, ax=ax, ci=None, edgecolor='black', linewidth=0.5,
                hue_order=method_order)

    # Rotate x-axis labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_title(model, fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean Value", fontsize=10)
    ax.set_xlabel("")
    ax.legend(title="Method", fontsize=8, title_fontsize=9)
    ax.set_ylim(0, 1.05)  # since metrics are between 0 and 1

# Hide any unused subplots
for j in range(len(models), len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUT_FILE_PATH}', dpi=300, bbox_inches='tight')
plt.show()