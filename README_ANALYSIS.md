# Specific to only running the analysis


## Installation
```bash
pip3 install requirements_analysis.txt
```

## Running: (after results are generated, see `robust_kernelbench/run_inference_test_time_scaling.py`)
```bash
# 1. Run Analysis Aggregation: (check file for more details on commandline args)
python3 analysis/run_analysis_summary_clean.py

# 2. Run Stat Analysis: (check file for more details on commandline args)
python3 analysis/run_analysis_statistical.py

# 3. Run Plotting: (check file for more details on commandline args)
python3 analysis/run_analysis_plotting.py

# 4. Run Latex Table (check file for more details on commandline args)
python3 analysis/run_analysis_latex_table.py
```

