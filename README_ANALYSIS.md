# Running the analysis

The analysis pipeline reads the artefacts written by the main pipeline
(`experiments/<experiment>/trial_<N>/{generations,compilations,evaluations}.csv`) and
produces comparison tables, statistical tests, plots and a LaTeX table.

## Installation

```bash
pip3 install -e ".[analysis]"
```

This installs `matplotlib`, `scipy` and `seaborn`, and makes `robust_kernelbench` importable
so the scripts work from any directory.

## Running

All scripts can be run as modules from the repository root:

```bash
# 1. Aggregate the two trials into one comparison table
python -m analysis.run_analysis_summary_clean --experiment_name v9_6 --trial1 4 --trial2 5
#   -> analysis_output/v9_6_comparison_clean_t4_vs_t5.csv

# 2. Statistical analysis (depends on the output of step 1)
python -m analysis.run_analysis_statistical \
    analysis_output/v9_6_comparison_clean_t4_vs_t5.csv
#   -> analysis_output/v9_6_comparison_statistical_analysis_t4_vs_t5.csv

# 3. Plotting (depends on the output of step 2)
python -m analysis.run_analysis_plotting \
    --file_path analysis_output/v9_6_comparison_statistical_analysis_t4_vs_t5.csv

# 4. LaTeX table (depends on the output of step 2)
python -m analysis.run_analysis_latex_table \
    --csv analysis_output/v9_6_comparison_statistical_analysis_t4_vs_t5.csv \
    --output v9_6_table.tex
```

Or all four steps in one go:

```bash
VERSION=v9_7 T1=4 T2=5 bash scripts/run_analysis.sh
```

`VERSION` is matched as a substring of `experiments/exp_local_*`; the launcher prints every
directory it matched before running (so `v10_1` also picks up `..._v10_1_smoke2_...` — pass the
full directory name to analyse exactly one run). It also activates an analysis environment
(`ENV_ANALYSIS`, default `env_kb2`) and fails fast if the trials or the dependencies are
missing.

Notes:

* Use `--base_experiments_folder experiments_backup2` on
  `run_analysis_summary_clean` if the runs were archived out of `experiments/`.
* The LaTeX step needs the `MODELS` / `METRICS` lists at the top of
  `analysis/run_analysis_latex_table.py` edited to match the runs you want to report.
* `analysis.utils` is now a thin re-export of `robust_kernelbench.utils.utils`; there is a
  single definition of the experiments directory layout.

## Reading the metrics

Each kernel gets a `KernelExecResult` with stage-level flags, so a failure can be attributed
precisely rather than as a single pass/fail:

| flag | meaning |
| --- | --- |
| `compiled` | the extension built |
| `runtime_success` | the module could be `exec`'d |
| `loading_success` | `ModelNew(*init_inputs)` succeeded |
| `cuda_success` | no illegal memory access / CUDA fault was detected |
| `correctness_success` | the correctness check itself ran to completion |
| `correctness` | the outputs actually matched (see `metadata["atol"/"rtol"]`) |
| `timing_success` | a timing measurement was produced |

`metadata` also records the tolerance used, whether L2 flushing was on, and the
`correctness_trials` pass count, so old and new runs can be compared honestly.

### Column names in the aggregated tables

Aggregate columns apply the suffix `_mean` (per-row mean, as produced by the statistical step)
or `_global` (share of the evaluated rows), for example `correctness_global`.

* **`correctness_global` is the accuracy to report.** It is the share of evaluated kernels
  whose outputs actually matched the reference.
* **`check_ran_global`** (formerly `correctness_success_global`) only says the correctness
  *check executed*. It is **not** accuracy and reads far higher — for example 67.0 % vs 36.4 %
  on the same run — so it is named `check_ran` to make that impossible to misread.
* **`_global` denominators are `len(evaluations)`, not 100.** A level has 100 problems, but a
  trial can legitimately end with fewer evaluation rows (88 for `v10_1`, 94 for `v9_7`), and
  dividing by 100 silently under-reported every rate.
* `coverage_global` is the share of rows actually evaluated; it used to be written into
  `compiled_global`, hiding the real compile rate.
* `cuda_success_global` is the share of evaluated rows with no detected CUDA fault.
* The statistical step records `n_runs` per model/prompt type. The LaTeX step uses it for
  Welch's t-test and marks comparisons with fewer than two runs as untestable rather than
  treating a NaN standard deviation as a significant result.
* Plotting tolerates older statistical CSVs that predate `check_ran_global_mean` and
  `cuda_success_global_mean`; the LaTeX step reads `prompt_type` when `method_name` is not
  present.
