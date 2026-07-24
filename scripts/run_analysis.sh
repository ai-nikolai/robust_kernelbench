VERSION=v9_7
T1=4
T2=5

T1=2
T2=3


python3 analysis/run_analysis_summary_clean.py --experiment ${VERSION} --trial1 ${T1} --trial2 ${T2}


# ALL BELOW FILES depend on statistical variety...
# 2. Run Stat Analysis: (check file for more details on commandline args)
# python3 analysis/run_analysis_statistical.py
python3 analysis/run_analysis_statistical.py analysis_output/${VERSION}_comparison_clean_t${T1}_vs_t${T2}.csv


# 3. Run Plotting: (check file for more details on commandline args)
# python3 analysis/run_analysis_plotting.py
python3 analysis/run_analysis_plotting.py --file_path analysis_output/${VERSION}_comparison_statistical_analysis_t${T1}_vs_t${T2}.csv


# 4. Run Latex Table (check file for more details on commandline args)
# python3 analysis/run_analysis_latex_table.py
# python robust_kernelbench/analysis/run_analysis_latex_table.py --csv analysis_output/${VERSION}_comparison_statistical_analysis_t${T1}_t${T2}.csv --output ${VERSION}_table.tex