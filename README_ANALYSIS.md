# Specific to only running the analysis


## Installation
```bash
pip3 install requirements_analysis.txt
```

## Running: (after results are generated, see `robust_kernelbench/run_inference_test_time_scaling.py`)
```bash
# 1. Run Analysis Aggregation: (check file for more details on commandline args)
# python3 analysis/run_analysis_summary_clean.py
python3 analysis/run_analysis_summary_clean.py --experiment v9_6 --trial1 4 --trial2 5


# ALL BELOW FILES depend on statistical variety...
# 2. Run Stat Analysis: (check file for more details on commandline args)
# python3 analysis/run_analysis_statistical.py
python3 analysis/run_analysis_statistical.py analysis_output/v9_6_comparison_clean_t4_vs_t5.csv


# 3. Run Plotting: (check file for more details on commandline args)
# python3 analysis/run_analysis_plotting.py
python3 analysis/run_analysis_plotting.py --file_path analysis_output/v9_6_comparison_statistical_analysis_t4_vs_t5.csv


# 4. Run Latex Table (check file for more details on commandline args)
# python3 analysis/run_analysis_latex_table.py
python robust_kernelbench/analysis/run_analysis_latex_table.py --csv analysis_output/v9_4_comparison_statistical_analysis_t4_t5.csv --output v9_4_table.tex
```

