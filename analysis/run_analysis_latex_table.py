#!/usr/bin/env python3
"""
Generate a LaTeX results table with statistical significance marking and best/second‑best highlighting.

Usage:

python robust_kernelbench/analysis/run_analysis_latex_table.py --csv analysis_output/V8_3_comparison_statistical_analysis_2_3.csv --output V8_3_table.tex

The user should edit the MODELS and METRICS lists in the script to select which models/metrics to include
and their display names.
"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict

# =============================================================================
# CONFIGURATION – modify these lists to match your data and desired LaTeX output
# =============================================================================

# List of (csv_model_name, display_name) for the models to include in the table
MODELS = [
    # ("Qwen/Qwen3-4B", "Qwen3-4B"),
    # ("Qwen/Qwen3-8B", "Qwen3-8B"),
    # ("Qwen/Qwen3-14B", "Qwen3-14B"),
    # ("Qwen/Qwen3-32B", "Qwen3-32B"),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen3-30B-A3B"),
    ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen3-Coder-30B-A3B"),
    # ("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "DeepSeek-Coder"),
    # ("deepseek/deepseek-v3.1-terminus", "deepseek-v3.1"),
    # ("mistralai/devstral-2512", "devstral-2512"),
    # ("openai/gpt-oss-120b", "gpt-oss-120b"),
]

# List of (csv_metric_key, display_name_in_table)
# The script will automatically append '_mean' and '_std' to the key.
METRICS = [
    ("compiled_global", "COMP"),
    ("runtime_success_global", "RUN"),
    ("correctness_global", "CORR"),
    ("fast_1_global", "Fast1"),
    ("fast_2_global", "Fast2"),
]

# Number of runs (used in Welch t‑test denominator)
N_RUNS = 3

# Significance threshold (one‑tailed p‑value)
ALPHA = 0.05

# Method mapping: CSV method_name → LaTeX label
METHOD_MAP = {
    "iterative_refinement": "IR",
    "+inductive": "+IB",
}

# =============================================================================
# Helper functions
# =============================================================================

def welch_ttest(mean_ir, std_ir, mean_ib, std_ib, n=N_RUNS):
    """
    One‑tailed Welch's t‑test for H0: mu_IB <= mu_IR.
    Returns (t_stat, df, p_value).
    If both standard deviations are zero and means differ, returns (inf, nan, 0.0).
    If both are zero and means equal, returns (nan, nan, 0.5).
    """
    if std_ir == 0.0 and std_ib == 0.0:
        if mean_ib > mean_ir:
            return (float('inf'), float('nan'), 0.0)   # significant (no variance)
        elif mean_ib == mean_ir:
            return (float('nan'), float('nan'), 0.5)
        else:
            return (float('-inf'), float('nan'), 1.0)   # can't happen because we test only improvement

    v_ir = std_ir**2 / n
    v_ib = std_ib**2 / n
    se = np.sqrt(v_ir + v_ib)
    if se == 0.0:
        # this case should be covered above
        return (float('nan'), float('nan'), 0.5)
    t_stat = (mean_ib - mean_ir) / se

    # Welch–Satterthwaite degrees of freedom
    num = (v_ir + v_ib)**2
    denom = (v_ir**2 / (n-1)) + (v_ib**2 / (n-1))
    if denom == 0:
        df = float('inf')
    else:
        df = num / denom

    if df == float('inf'):
        p_value = stats.norm.sf(t_stat)  # one‑tailed
    else:
        p_value = stats.t.sf(t_stat, df)

    return t_stat, df, p_value


def check_significance(mean_ir, std_ir, mean_ib, std_ib):
    """Return True if +IB significantly > IR at ALPHA level, else False or None if untestable."""
    # OLD: If both stds are zero, we call it untestable (no variance)
    # if std_ir == 0.0 and std_ib == 0.0:
    #     return None
    _, _, p = welch_ttest(mean_ir, std_ir, mean_ib, std_ib)
    if p < ALPHA:
        return True
    else:
        return False


def determine_best_second_best(all_values):
    """
    Given a list of (value, model_key, method) tuples, return sets of keys that are best and second best.
    The 'key' is something like (model_display, method_label).
    Ties for best all get bold; ties for the next value get underline (if distinct from best).
    If all values equal, no second best.
    """
    # sort descending
    sorted_items = sorted(all_values, key=lambda x: x[0], reverse=True)
    best_value = sorted_items[0][0]
    best_keys = {item[1] for item in sorted_items if item[0] == best_value}

    # find second best value
    second_best_keys = set()
    for val, key in sorted_items:
        if val < best_value:
            second_value = val
            break
    else:
        # all values equal
        return best_keys, second_best_keys

    second_best_keys = {item[1] for item in sorted_items if item[0] == second_value}
    return best_keys, second_best_keys


# =============================================================================
# Main processing
# =============================================================================

def main(csv_path, output_path):
    # Read CSV
    df = pd.read_csv(csv_path)

    # Prepare data structure: data[model_name][method_label][metric_key] = (mean, std)
    data = defaultdict(dict)

    for _, row in df.iterrows():
        csv_model = row['model_name']
        csv_method = row['method_name']
        method_label = METHOD_MAP.get(csv_method)
        if method_label is None:
            continue

        # only consider models in our list
        if csv_model not in [m[0] for m in MODELS]:
            continue

        metric_dict = {}
        for metric_key, _ in METRICS:
            mean_col = f"{metric_key}_mean"
            std_col = f"{metric_key}_std"
            if mean_col in row and std_col in row:
                metric_dict[metric_key] = (row[mean_col], row[std_col])
        data[csv_model][method_label] = metric_dict

    # We'll build rows: each row is (display_model, method_label, metric_values)
    rows = []
    model_order = [m[0] for m in MODELS]  # preserve order
    for csv_model in model_order:
        if csv_model not in data:
            continue
        display_model = dict(MODELS)[csv_model]
        for method_label in ["IR", "+IB"]:  # ensure order IR first, then +IB
            if method_label in data[csv_model]:
                rows.append((display_model, method_label, data[csv_model][method_label]))

    # Compute significance for each row that is +IB relative to its IR counterpart
    # We need to pair them.
    significance_dict = {}  # (display_model, metric_key) -> bool/None
    # Also keep (mean_ir, std_ir, mean_ib, std_ib) for potential footnote
    model_pairs = {}
    for csv_model, display_model in MODELS:
        if csv_model in data and "IR" in data[csv_model] and "+IB" in data[csv_model]:
            ir_metrics = data[csv_model]["IR"]
            ib_metrics = data[csv_model]["+IB"]
            for metric_key, _ in METRICS:
                mean_ir, std_ir = ir_metrics.get(metric_key, (None, None))
                mean_ib, std_ib = ib_metrics.get(metric_key, (None, None))
                if mean_ir is not None and mean_ib is not None:
                    sig = check_significance(mean_ir, std_ir, mean_ib, std_ib)
                    significance_dict[(display_model, metric_key)] = sig
                    model_pairs[(display_model, metric_key)] = (mean_ir, std_ir, mean_ib, std_ib)
                else:
                    significance_dict[(display_model, metric_key)] = None

    # Determine best and second best for each metric across all (model, method) entries
    best_dict = {}   # metric_key -> set of keys (display_model, method_label)
    second_dict = {}
    all_metric_values = defaultdict(list)  # metric_key -> [(value, (display_model, method_label))]
    for display_model, method_label, metric_dict in rows:
        for metric_key, _ in METRICS:
            if metric_key in metric_dict:
                mean_val, _ = metric_dict[metric_key]
                all_metric_values[metric_key].append((mean_val, (display_model, method_label)))

    for metric_key, val_list in all_metric_values.items():
        best, second = determine_best_second_best(val_list)
        best_dict[metric_key] = best
        second_dict[metric_key] = second

    # Now build the LaTeX table
    # Define the \stdpm and \sigimprove commands in the preamble (we'll output a full document or a snippet)
    # We'll output only the table (tabular + caption) for inclusion, with a note about needed packages.
    latex = []
    # latex.append("% Ensure these are in your preamble:")
    # latex.append("% \\usepackage{xcolor}")
    # latex.append("% \\newcommand{\\stdpm}[1]{\\text{\\scriptsize\\textpm\\,#1}}")
    # latex.append("% \\newcommand{\\sigimprove}[1]{\\textcolor{green!60!black}{\\textbf{#1}}}")
    # latex.append("")
    latex.append("\\begin{table*}[htbp]")
    latex.append("\\centering")
    caption = (
        "Main results table summarizing the scores. All experiments were conducted on the "
        "KernelBench GPU programming task with a fixed test set of size $N = 100$ problems from "
        "level~1. IR = Iterative Refinement; IB = Inductive Bias. "
        "Reported values are mean $\\pm$ standard deviation over 3 runs. "
        "Green bold values indicate statistically significant improvement of +IB over IR "
        "(Welch's $t$-test, one‑tailed, $p < 0.05$). "
        "\\textbf{Bold} = best overall, \\underline{underline} = second best (ties allowed). "
        "COMP = Compilation Success, RUN = Runtime Success, CORR = Correctness."
    )
    latex.append("\\caption{" + caption + "}")
    latex.append("\\label{tab:main_results}")
    latex.append("\\begin{tabular}{ll" + "c" * len(METRICS) + "}")
    latex.append("\\toprule")
    header = "\\textbf{Model} & \\textbf{Method} & " + " & ".join(
        [f"\\textbf{{{display}}}" for _, display in METRICS]
    ) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")

    # For tracking whether to add a \hline between models
    last_model = None
    for display_model, method_label, metric_dict in rows:
        if last_model is not None and display_model != last_model:
            latex.append("\\hline")
        last_model = display_model

        row_cells = [display_model, method_label]
        for metric_key, _ in METRICS:
            mean_val, std_val = metric_dict.get(metric_key, (None, None))
            if mean_val is None:
                row_cells.append("--")
                continue

            # format number to 2 decimal places
            mean_str = f"{mean_val:.2f}"
            std_str = f"{std_val:.2f}"

            # Determine if this cell should be bold, underline, or significant green
            is_best = (display_model, method_label) in best_dict.get(metric_key, set())
            is_second = (display_model, method_label) in second_dict.get(metric_key, set())
            is_sig = False
            if method_label == "+IB":
                sig_result = significance_dict.get((display_model, metric_key))
                if sig_result is True:
                    is_sig = True

            # Build cell with formatting
            cell = f"{mean_str}\\stdpm{{{std_str}}}"

            if is_sig:
                # Use \sigimprove, which makes it green and bold
                # If it's also best or second best, \sigimprove already includes \textbf,
                # so underline might need extra care. We'll apply underline outside if needed.
                if is_best:
                    # Already bold from \sigimprove; we need to add nothing else for bold,
                    # but underline for second best? But it's best, so bold is enough.
                    cell = f"\\sigimprove{{{mean_str}}}\\stdpm{{{std_str}}}"
                elif is_second:
                    # Should be both green/bold and underlined.
                    # \sigimprove applies \textbf, so we can wrap with underline.
                    cell = f"\\underline{{\\sigimprove{{{mean_str}}}\\stdpm{{{std_str}}}}}"
                else:
                    cell = f"\\sigimprove{{{mean_str}}}\\stdpm{{{std_str}}}"
            else:
                if is_best:
                    cell = f"\\textbf{{{mean_str}}}\\stdpm{{{std_str}}}"
                elif is_second:
                    cell = f"\\underline{{{mean_str}}}\\stdpm{{{std_str}}}"

            # Handle case where both best and significant? We already covered.

            # Special: if both IR and IB stds are zero, add dagger to the +IB cell for untestable
            # (but this note should only appear for DeepSeek-Coder, we'll add a generic footnote)
            # if method_label == "+IB" and significance_dict.get((display_model, metric_key)) is None:
            #     # Untestable due to zero std
            #     cell += "\\textsuperscript{\\dag}"

            row_cells.append(cell)

        latex.append(" & ".join(row_cells) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("")
    # Footnote
    # latex.append("\\vspace{4pt}")
    # latex.append("\\footnotesize")
    # latex.append(
    #     "\\textsuperscript{\\dag} Standard deviations are zero for both methods, "
    #     "making significance testing infeasible. "
    #     "All other comparisons use Welch's $t$-test (one‑tailed, unequal variances, $n=3$, "
    #     "$\\text{df}\\approx 2$--$4$, $p < 0.05$). "
    #     "\\textbf{Bold} = best, \\underline{underline} = second best (ties allowed)."
    # )
    latex.append("\\end{table*}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"LaTeX table written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LaTeX results table with significance.")
    parser.add_argument("--csv", required=True, help="Path to the CSV file")
    parser.add_argument("--output", default="table.tex", help="Output .txt file for LaTeX code")
    args = parser.parse_args()
    main(args.csv, args.output)