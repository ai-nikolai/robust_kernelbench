"""
USAGE:

python3 analysis/run_analysis_statistical.py analysis_output/V8_3_comparison_clean_t2_vs_t3.csv analysis_output/V8_3_comparison_clean_t12_vs_t13.csv analysis_output/V8_3_comparison_clean_t22_vs_t23.csv analysis_output/V8_3_comparison_statistical_analysis_2_3.csv

"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

def aggregate_across_files(file_paths, special_key2="prompt_type"):
    """
    Accepts a list of CSV file paths, aggregates across models and methods,
    and outputs a CSV with mean and standard deviation for each metric.
    """
    # Read all CSV files into a list of DataFrames
    dataframes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        # Ensure required columns exist
        if 'model_name' not in df.columns:
            raise ValueError(f"File {file_path} must contain 'model_name' column")
        if f'{special_key2}' not in df.columns:
            raise ValueError(f"File {file_path} must contain '{special_key2}' column")
        # Set multi-index for consistent merging
        df = df.set_index(['model_name', f'{special_key2}'])
        dataframes.append(df)

    # Combine all DataFrames by concatenating along rows (different observations)
    combined = pd.concat(dataframes, axis=0)

    # Group by model_name and method_name, compute mean and std for each metric
    mean_df = combined.groupby(level=['model_name', f'{special_key2}']).mean()
    std_df = combined.groupby(level=['model_name', f'{special_key2}']).std()

    # Rename columns to indicate mean or std
    mean_df = mean_df.add_suffix('_mean')
    std_df = std_df.add_suffix('_std')

    # Concatenate mean and std DataFrames along columns
    result = pd.concat([mean_df, std_df], axis=1)

    # Reorder columns to interleave mean and std for each metric
    base_metrics = [col.replace('_mean', '') for col in mean_df.columns]

    ordered_columns = []
    for metric in base_metrics:
        ordered_columns.append(f"{metric}_mean")
        ordered_columns.append(f"{metric}_std")

    result = result[ordered_columns]

    # Reset index to make model_name and method_name columns again
    result = result.reset_index()

    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate_analysis.py <input_files...>")
        print("Example: python aggregate_analysis.py file1.csv file2.csv")
        sys.exit(1)

    # All arguments except the last one are input files
    input_files = sys.argv[1:]
    # output_file = sys.argv[-1]
    output_file = input_files[0].replace("_clean","_statistical_analysis")

    if not input_files:
        print("Error: No input CSV files provided.")
        sys.exit(1)

    # Check that all input files exist
    for file_path in input_files:
        if not Path(file_path).exists():
            print(f"Error: Input file {file_path} does not exist.")
            sys.exit(1)

    # Perform aggregation
    try:
        result = aggregate_across_files(input_files)
        # Save to output CSV
        result = result.round(2)
        result.to_csv(output_file, index=False, sep=',')
        
        print(f"Aggregated results saved to {output_file}")
        print(f"Processed {len(input_files)} files for {len(result)} model-method combinations.")
    except Exception as e:
        print(f"Error during aggregation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

