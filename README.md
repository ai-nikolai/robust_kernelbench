# Robust KernelBench

This is generally a robust version inspired by KernelBench (https://github.com/ScalingIntelligence/KernelBench).
This code fixes a lot of the runtime problems the original code had: (Like fixing the GPU errors that go unoticed and corrupt the final results, adding isolated single-node compilation, and adding more detailed metrics.)

This repo is also for the paper: "FIL Hypothesis"... (Citation coming soon).

---

## Package layout (v0.2.0)

`robust-kernelbench` is split into two layers so a PyPI install can contain just the
evaluation harness:

| Layer | Modules | Install |
| --- | --- | --- |
| **Evaluation core** | `artifacts`, `compile_*`, `evaluate_*`, `utils` | `pip install robust-kernelbench` |
| **Agentic generation** | `robust_kernelbench.agent.*` (inference, prompts, multi-round orchestration) | `pip install "robust-kernelbench[agent]"` |
| Local engines | optional `[vllm]` / `[sglang]` extras | `pip install "robust-kernelbench[agent,vllm]"` |
| Analysis | `analysis/` (matplotlib/scipy/seaborn) | `pip install "robust-kernelbench[analysis]"` |
| Dataset sources | `data/<name>/` manifest + kernels; not packaged | local export via `to_hf.py` |

The console scripts are:

```bash
robust-kernelbench                    # agentic generate -> compile -> evaluate
robust-kernelbench-eval               # evaluate an existing trial
robust-kernelbench-eval-single        # evaluate one kernel
robust-kernelbench-compile            # parallel compilation
robust-kernelbench-compile-sequential # crash-isolated compilation
robust-kernelbench-dataset            # validate a local/HF task source
```

See `ARCHITECTURE.md` for the core/agent boundary and the migration path to a
separate `robust-kernelbench-agent` distribution.

The historical module paths (`robust_kernelbench.run_main`,
`robust_kernelbench.inference_common`, `robust_kernelbench.prompt_common`,
`robust_kernelbench.prompts_v2`, `robust_kernelbench.prompts_kb`) remain as thin
compatibility shims.

### Trust model

The compiler and evaluator `exec()` LLM-generated kernels. The subprocess isolation in
`evaluate_all` / `compile_all_sequential` protects against crashes, CUDA aborts and
runaway memory, **but it is not a security sandbox**: generated code can still read and
write files and open network connections. Run untrusted models inside a container or
namespace with a read-only filesystem and no network if that matters to you.

### Stage guarantees

A job is no longer reported as successful merely because every stage exited 0:

* inference must write at least one generation row **and** one kernel file;
* every generated kernel must have a row in `compilations.csv`;
* every successfully compiled kernel must have a row in `evaluations.csv`.

A kernel that fails to compile, or produces wrong/slow output, is a valid measurement and
is recorded as such. A *missing* row is an infrastructure failure and fails the job.
Crashed compile/evaluation children are also recorded (with their logs retained) instead
of disappearing from the CSV.

---

## What changed (v0.1.0)

The pipeline is now an installable package with a real CLI, a test suite and CI:

| Before | Now |
| --- | --- |
| `python3 robust_kernelbench/run_main.py` (from the repo root only) | `python -m robust_kernelbench.run_main` or the `robust-kernelbench` console script, after `pip install -e .` |
| `from utils.utils import ...` (cwd-relative) | package-absolute imports; works from any directory |
| one 650-line file per compile variant, two near-identical inference files | one `compile_driver` / `inference_common`; the entry points are thin wrappers |
| fixed `atol=rtol=1e-2` for every dtype | dtype-aware tolerances (fp64 1e-5, fp32 1e-4, fp16 1e-2, bf16 2e-2), overridable |
| 1 correctness trial by default | 5 correctness trials by default |
| cache-warm timing (small kernels flattered) | L2 is evicted between timing trials (`--flush_l2`, on by default) |
| `NameError` when no jobs were passed | clean usage error, exit code 2 |
| failed inference silently skipped the whole round | logged with a traceback, recorded, `--continue_on_error` to opt into skipping |
| both SGLang *and* vLLM pause/resume endpoints called every time | `--serving_backend {sglang,vllm,none}` |
| `--tensor_parallel_size` hardcoded to 1 in the orchestrator | configurable, defaults to 1 |
| no tests, no CI | `pytest` suite + GitHub Actions (GPU tests skip on CPU runners) |
| `python` from `$PATH` in subprocesses | `sys.executable`, and `-m` module invocation |

---

## Getting started

### Installation

Install just the evaluator (no serving stack):

```bash
pip install robust-kernelbench
# optional: post-hoc analysis
pip install "robust-kernelbench[analysis]"
```

Install the agentic loop with a remote OpenAI-compatible endpoint:

```bash
pip install "robust-kernelbench[agent]"
```

Install it with a local engine:

```bash
pip install "robust-kernelbench[agent,vllm]"
# or
pip install "robust-kernelbench[agent,sglang]"
```

For development from a checkout:

```bash
# ENV for the model (sglang / vllm serving)
conda create -n env_model python=3.11
conda activate env_model
pip3 install -r requirements_sglang.txt

# ENV for kernelbench (newer requirements break)
conda create -n env_kb2 python=3.11
conda activate env_kb2
pip3 install -r requirements.txt
pip3 install -e ".[agent,analysis,dev]"   # package + agentic loop + tooling
# conda install cuda-nvcc_linux-64 -c conda-forge # in case you are on conda you need this.
# Maybe needed?
# conda install cuda-toolkit
```

`pip install -e .` is what makes `python -m robust_kernelbench.<module>` work from any
directory. Running the scripts straight from a checkout still works too, as long as the
repository root is the current directory.

> **Note on invocation style.** Use `python -m robust_kernelbench.<module>` (or the
> `robust-kernelbench` console script) rather than `python3 robust_kernelbench/<module>.py`.
> When you point Python at a file, it puts that file's *directory* on `sys.path`, so
> `import robust_kernelbench` fails unless the package has been installed. With
> `pip install -e .` both styles work; without it, only `-m` from the repo root works.

---

## Running the code

### 1. Multi-round end-to-end (Slurm)

```bash
# everything can be overridden from the environment
MODEL_NAME=Qwen/Qwen3-Coder-30B-A3B-Instruct VERSION=v9_8 bash scripts/slurm_job_v2.sh
```

### 2. Single-node (local server + pipeline)

```bash
MODEL_NAME=Qwen/Qwen3-Coder-Next NUM_ITEMS=5 SERVING_BACKEND=sglang bash scripts/run_main.sh
```

### 3. All in one go (no server management)

```bash
CUDA_VISIBLE_DEVICES=2 python -m robust_kernelbench.run_main \
    --version v9_4 \
    --model_name Qwen3_Coder_30B_A3B_Instruct \
    --parent_prompt_type kernelbench \
    --num_samples 1 \
    --tensor_parallel_size 2 \
    --serving_backend sglang \
    --num_correct_trials 5 \
    "1 1 kernelbench"
```

The trailing arguments are jobs of the form `<previous_trial> <new_trial> <prompt_type>`.
Passing no jobs is a usage error (exit code 2).

### 4. Running the stages separately

```bash
# 1. Generate kernels (vLLM entry point; the _v2 variant uses SGLang)
python -m robust_kernelbench.run_inference_test_time_scaling \
    --experiment_name exp_local_1 --trial 1 --num_items 3

# 2a. Compile (parallel, multiprocessing pool)
python -m robust_kernelbench.compile_all \
    --experiment_name exp_local_1 --trial 1 --num_processes 8

# 2b. Compile (sequential, one crash-isolated subprocess per kernel)
python -m robust_kernelbench.compile_all_sequential \
    --experiment_name exp_local_1 --trial 1

# 3. Evaluate
python -m robust_kernelbench.evaluate_all \
    --experiment_name exp_local_1 --trial 1 \
    --num_correct_trials 5 --flush_l2
```

---

## Key CLI options

| Flag | Applies to | Meaning |
| --- | --- | --- |
| `--serving_backend {sglang,vllm,none}` | `run_main` | which server's pause/resume endpoints to call between stages (`none` skips) |
| `--tensor_parallel_size N` | `run_main`, inference | tensor-parallel size for the local engine |
| `--num_samples N` | `run_main`, inference | completions per problem (all backends now honour it) |
| `--seed N` | `run_main`, inference | sampling seed, recorded in `metadata.json` (`None` = engine default) |
| `--num_correct_trials N` | `run_main`, `evaluate_all`, `evaluate_single` | independently seeded correctness trials; all must match (default 5) |
| `--atol` / `--rtol` | `run_main`, `evaluate_all`, `evaluate_single` | explicit tolerances; default is chosen from the output dtype. `--atol 1e-2 --rtol 1e-2` reproduces the historical behaviour |
| `--flush_l2` / `--no-flush_l2` | `run_main`, `evaluate_all`, `evaluate_single` | evict L2 before each timed trial (default: on) |
| `--continue_on_error` | `run_main` | keep running remaining jobs after a failure (exit code still non-zero) |
| `--compile {0,1,2}` | `run_main` | 0=skip, 1=parallel, 2=sequential |
| `--log-level` | all | `DEBUG`/`INFO`/`WARNING`/... (logs go to stderr, tqdm keeps stdout) |
| `--reset_experiments` | `run_main` | delete the experiment directory first |

### Correctness and timing policy

* **Tolerances** are derived from the *reference output dtype*: `float64` 1e-5,
  `float32` 1e-4, `float16` 1e-2, `bfloat16` 2e-2, integer/bool exact. The chosen values
  are recorded in the evaluation metadata (`atol`, `rtol`, `tolerance_dtype`) so results
  stay auditable.
* **L2 flushing**: a throw-away 256 MB buffer is zeroed before every timed trial (outside
  the timed region), for both the kernel and its reference, so small kernels are not
  measured cache-warm. `metadata["flush_l2"]` records whether it was enabled.

---

## Using a custom dataset

The official KernelBench HF dataset is only the default.  Any source that provides
`problem_id`, `name` and `code` works:

```bash
# evaluator only
robust-kernelbench-eval \
  --dataset_path data/science_kernels \
  --dataset_format directory \
  --dataset_split level_1 \
  --experiment science_exp --trial 1

# agentic loop (local source)
robust-kernelbench \
  --dataset_path data/science_kernels \
  --dataset_format directory \
  --dataset_split level_1 \
  --dataset_name science-kernels \
  --version science_v0 --model_name Qwen/Qwen3-Coder-30B-A3B-Instruct \
  "1 1 kernelbench"

# same usage as KernelBench, once the science dataset is on the Hub
robust-kernelbench \
  --dataset ai-nikolai/ScienceKernelBench \
  --dataset_version 0.1.0 \
  --dataset_split level_1 \
  --version science_v0 --model_name Qwen/Qwen3-Coder-30B-A3B-Instruct \
  "1 1 kernelbench"
```

Supported formats are `hf`, `directory`, `jsonl` and `parquet`.  A directory may
contain a `manifest.json` with per-task `code_path`, provenance and optional
`backend`/`precision`/`filter` fields; without a manifest every `*.py` file is a
task.  Validate a source before using it:

```bash
robust-kernelbench-dataset validate --dataset_path data/science_kernels --dataset_format directory
robust-kernelbench-dataset validate --dataset_path data/science_kernels --dataset_format directory --execute --device cpu
robust-kernelbench-dataset stats --dataset_path data/science_kernels --dataset_format directory
robust-kernelbench-dataset list  --dataset_path data/science_kernels --dataset_format directory --domain physics
```

Use `--domain` on `run_main` / inference / `evaluate_all` to restrict a run to
one or more domains, e.g. `--domain physics --domain chemistry`.

`data/science_kernels/` is a working science-focused starter set (pairwise RBF
kernel, 2D Laplacian) plus a Hugging Face export script:

```bash
python data/science_kernels/to_hf.py --output /tmp/science_kernels_hf
```

The export mirrors KernelBench's HF layout: one `default` config with
`level_1`, `level_2` and `level_3` splits under `data/`.  Upload it to
`ai-nikolai/ScienceKernelBench` with:

```bash
python data/science_kernels/upload_to_hf.py --path /tmp/science_kernels_hf
```

See `data/science_kernels/README.md` for the schema, provenance and upload steps.

## ScienceKernelBench five-step run

The reference ScienceKernelBench run uses the full 22-task dataset across
`level_1`, `level_2` and `level_3`:

- local `Qwen/Qwen3-Coder-30B-A3B-Instruct` served with vLLM (tensor parallel 2);
- two prompt families evaluated independently: `kernelbench` and `kb_multi_stage`;
- baseline plus **five improvement rounds**, i.e. trials `1` through `6`;
- one experiment per `(level, prompt family)`.

Schedule it with:

```bash
sbatch scripts/slurm_science_kb_5steps.sh
```

The reference run is scheduled as Slurm job `1658397` on the `gpu` partition
(currently pending due to cluster priority).

The script uses dataset revision `v0.1.0`, model `Qwen/Qwen3-Coder-30B-A3B-Instruct`,
and writes experiments such as:

```text
experiments/exp_science_v0_level_1_kernelbench/trial_1 ... trial_6
experiments/exp_science_v0_level_1_kb_multi_stage/trial_1 ... trial_6
```

After the run, generate the table and plots:

```bash
python -m analysis.run_science_benchmark_report --version science_v0
```

Outputs:

```text
analysis_output/science_v0_rounds.csv
analysis_output/science_v0_summary.csv
analysis_output/science_v0_table.md
analysis_output/science_v0_correctness.png
analysis_output/science_v0_final_comparison.png
```

The results table and plots are generated from the actual run output; they are
not hard-coded in the repository.

## Running the analysis

```bash
pip3 install -e ".[analysis]"
bash scripts/run_analysis.sh                 # VERSION=v9_7 T1=4 T2=5 by default
```

See `README_ANALYSIS.md` for the individual steps.

---

## Housekeeping

`experiments/` grows without bound. Prune it — **dry-run unless `--delete`**:

```bash
python scripts/prune_experiments.py --keep-last 5                  # show what would go
python scripts/prune_experiments.py --older-than-days 30 --delete  # actually remove
```

---

## Development

```bash
pip3 install -e ".[dev]"

python -m compileall -q robust_kernelbench analysis tests conftest.py -x 'kb_prompts'
ruff check robust_kernelbench analysis tests conftest.py
pytest -q
```

`robust_kernelbench/kb_prompts/*.py` are prompt *templates* (placeholder pseudo-code that is
rendered as text and never imported), which is why `compileall`/`ruff` exclude that
directory. Tests marked `gpu` are skipped automatically when no CUDA device is present.

---

## Task Spooler (local server SLURM alternative)

1. Installation

```bash
apt update
apt install task-spooler
```

2. Usage

```bash
tsp timeout 3600 python3 -m robust_kernelbench.run_inference_test_time_scaling \
                        --experiment_name "exp_local_1" --trial 1
tsp      # shows current available jobs
tsp -c   # shows the output of the last job
```

### References

- https://github.com/bstee615/shared-task-spooler
- https://github.com/justanhduc/task-spooler

---

## CONDA INSTALL LINUX

https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2

```bash
mkdir miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda3/miniconda.sh
bash ./miniconda3/miniconda.sh -b -u -p ./miniconda3
```

To activate it next time:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_model   # serving
conda activate env_kb2     # kernelbench
```

### Slurm

```bash
sbatch scripts/slurm_job_v2.sh

# Interactive
salloc --partition=interactive-gpu --gres=gpu:h200:3 --time=08:00:00 --ntasks=3
srun --pty --overlap --jobid=100854 bash
```

Other useful commands:

```bash
scontrol ping
scontrol show partition interactive-gpu
scontrol show assoc user=$USER
```

---

## Tmux

```bash
tmux                # start
CTRL+b w            # switch windows
CTRL+b c            # new window
Ctrl+b "            # split vertical
Ctrl+b d            # detach
```

---

## LLM serving

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-Coder-Next \
    --tensor-parallel-size 2 \
    --tool-call-parser qwen3_coder \
    --host 0.0.0.0 \
    --port 30000
```

Check it is running:

```bash
curl -v http://0.0.0.0:30000/health
curl -v http://0.0.0.0:30000/model_info
```

`run_main --serving_backend {sglang,vllm}` will call the matching memory-release
endpoints (`/release_memory_occupation` + `/resume_memory_occupation` for SGLang,
`/sleep?level=1` + `/wake_up` for vLLM) around each stage.

---

## (C) Nikolai Rozanov - 2025 - Present
