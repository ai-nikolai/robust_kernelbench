#!/usr/bin/env bash
# ----------------------------------------------------------------------
# Post-hoc analysis pipeline for one experiment (one or two trials).
#
#   VERSION=v10_1 T1=1 bash scripts/run_analysis.sh        # single trial
#   VERSION=v9_7  T1=4 T2=5 bash scripts/run_analysis.sh   # trial comparison
#
# VERSION is matched as a *substring* of experiments/exp_local_*, so a short
# tag is fine, but the resolved directories are printed first: `v10_1` also
# matches `..._v10_1_smoke2_...`.  Pass the full experiment directory name to
# analyse exactly one run.
#
# The analysis stack (pandas / matplotlib / seaborn / scipy) lives in env_kb2,
# not in the serving environments.  Override with ENV_ANALYSIS=<env>.
# ----------------------------------------------------------------------
set -euo pipefail

VERSION="${VERSION:?set VERSION, e.g. VERSION=v10_1 (substring of experiments/exp_local_*)}"
T1="${T1:-1}"
T2="${T2:-}"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
ENV_ANALYSIS="${ENV_ANALYSIS:-env_kb2}"
BASE_EXPERIMENTS_FOLDER="${BASE_EXPERIMENTS_FOLDER:-experiments}"

# ---- resolve the experiment directories and fail fast if nothing matches
mapfile -t MATCHES < <(ls -d "${BASE_EXPERIMENTS_FOLDER}"/exp_local_*"${VERSION}"* 2>/dev/null || true)
if [ "${#MATCHES[@]}" -eq 0 ]; then
    echo "ERROR: no experiment matches '${VERSION}' under ${BASE_EXPERIMENTS_FOLDER}/" >&2
    echo "Available (first 20):" >&2
    ls -1 "${BASE_EXPERIMENTS_FOLDER}" 2>/dev/null | head -20 >&2
    exit 1
fi
echo "=== resolved ${#MATCHES[@]} experiment(s) for '${VERSION}':"
printf '  %s\n' "${MATCHES[@]}"
if [ "${#MATCHES[@]}" -gt 1 ]; then
    echo "    NOTE: more than one directory matches; their rows are aggregated together." >&2
fi

# ---- every requested trial must exist in at least one matched experiment
for t in "${T1}" ${T2:+"${T2}"}; do
    found=0
    for d in "${MATCHES[@]}"; do
        if [ -f "${d}/trial_${t}/evaluations.csv" ]; then found=1; fi
    done
    if [ "${found}" -eq 0 ]; then
        echo "ERROR: no trial_${t}/evaluations.csv under the matched experiment(s)." >&2
        echo "Available trials:" >&2
        for d in "${MATCHES[@]}"; do ls -d "${d}"/trial_* 2>/dev/null | sed 's|^|  |' >&2; done
        exit 1
    fi
done

# ---- analysis environment (deps are not in the serving envs)
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${ENV_ANALYSIS}"
if ! python3 -c "import pandas, matplotlib, seaborn, scipy" 2>/dev/null; then
    echo "ERROR: '${ENV_ANALYSIS}' lacks the analysis dependencies (pandas/matplotlib/seaborn/scipy)." >&2
    echo "       Set ENV_ANALYSIS to an environment that provides them." >&2
    exit 1
fi

TRIAL2_ARGS=()
if [ -n "${T2}" ]; then TRIAL2_ARGS=(--trial2 "${T2}"); fi

# 1. Aggregate the trial(s) into a comparison table.
#    --experiment_name (full flag, not the prefix-matched --experiment).
python3 -m analysis.run_analysis_summary_clean \
    --experiment_name "${VERSION}" \
    --trial1 "${T1}" \
    ${TRIAL2_ARGS[@]+"${TRIAL2_ARGS[@]}"} \
    --base_experiments_folder "${BASE_EXPERIMENTS_FOLDER}"

# The steps below depend on the output of step 1.
ACTUAL_T2="${T2:-${T1}}"
CLEAN_CSV="analysis_output/${VERSION}_comparison_clean_t${T1}_vs_t${ACTUAL_T2}.csv"
STAT_CSV="analysis_output/${VERSION}_comparison_statistical_analysis_t${T1}_vs_t${ACTUAL_T2}.csv"

# 2. Statistical tests.
python3 -m analysis.run_analysis_statistical "${CLEAN_CSV}"

# 3. Plots.
python3 -m analysis.run_analysis_plotting --file_path "${STAT_CSV}"

# 4. LaTeX table (edit MODELS/METRICS in the script first, see its docstring).
# python3 -m analysis.run_analysis_latex_table \
#     --csv "${STAT_CSV}" --output "${VERSION}_table.tex"
