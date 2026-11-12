"""
USAGE:

python3 analysis/run_analysis_statistical.py analysis_output/V8_3_comparison_clean_t2_vs_t3.csv analysis_output/V8_3_comparison_clean_t12_vs_t13.csv analysis_output/V8_3_comparison_clean_t22_vs_t23.csv analysis_output/V8_3_comparison_statistical_analysis_2_3.csv

"""

import sys
from pathlib import Path

import pandas as pd

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

    if not dataframes:
        raise ValueError("no input files")

    # Combine all DataFrames by concatenating along rows (different observations)
    combined = pd.concat(dataframes, axis=0)
    grouped = combined.groupby(level=["model_name", special_key2])

    # Numeric-only keeps string columns (e.g. prompt_type) from breaking the
    # aggregation on newer pandas versions.
    mean_df = grouped.mean(numeric_only=True).add_suffix("_mean")
    std_df = grouped.std(numeric_only=True).add_suffix("_std")
    counts = grouped.size().rename("n_runs")

    result = pd.concat([mean_df, std_df, counts], axis=1)

    # Reorder columns to interleave mean and std for each metric.
    base_metrics = [col.replace("_mean", "") for col in mean_df.columns]
    ordered_columns = []
    for metric in base_metrics:
        ordered_columns.append(f"{metric}_mean")
        ordered_columns.append(f"{metric}_std")
    ordered_columns.append("n_runs")

    result = result[ordered_columns].reset_index()
    return result

def parse_args(argv=None):
    """CLI: positional inputs (launcher form) or repeatable --file_path."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate comparison CSVs into mean/std per model and prompt type.",
    )
    parser.add_argument("files", nargs="*", help="input *_comparison_clean_*.csv files")
    parser.add_argument(
        "--file_path",
        action="append",
        default=None,
        metavar="CSV",
        help="input CSV; may be repeated (alternative to the positional form)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Positional form kept for scripts/run_analysis.sh; --file_path for clarity.
    input_files = list(args.file_path or []) + list(args.files)

    if not input_files:
        print("Usage: python -m analysis.run_analysis_statistical <input_files...>")
        print("Example: python -m analysis.run_analysis_statistical file1.csv file2.csv")
        return 1

    output_file = input_files[0].replace("_clean", "_statistical_analysis")

    # Check that all input files exist
    for file_path in input_files:
        if not Path(file_path).exists():
            print(f"Error: Input file {file_path} does not exist.")
            return 1

    # Perform aggregation
    try:
        result = aggregate_across_files(input_files)
        # Keep full precision: the LaTeX step performs the significance tests and
        # rounding here can move a p-value across the threshold.
        result.to_csv(output_file, index=False, sep=",", float_format="%.6g")
        
        print(f"Aggregated results saved to {output_file}")
        print(f"Processed {len(input_files)} files for {len(result)} model-method combinations.")
    except Exception as e:
        print(f"Error during aggregation: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
