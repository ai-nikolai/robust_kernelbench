---
license: agpl-3.0
language:
- en
task_categories:
- other
pretty_name: Science KernelBench
tags:
- kernelbench
- gpu
- cuda
- triton
- scientific-computing
- pde
- kernel-methods
configs:
- config_name: level_1
  data_files:
  - split: level_1
    path: level_1.parquet
- config_name: level_2
  data_files:
  - split: level_2
    path: level_2.parquet
- config_name: level_3
  data_files:
  - split: level_3
    path: level_3.parquet
---

# Science KernelBench

A science-focused task set for `robust-kernelbench`.  Each task provides a
PyTorch reference implementation of a numerical primitive; the candidate must
produce a faster, numerically equivalent GPU kernel.

This directory is the **source of truth**.  `to_hf.py` expands the manifest into
a Hugging Face dataset; the generated export is not committed.

Current starter set: **22 tasks** across physics, chemistry, biology,
statistics, mathematics and economics.

| level | focus | tasks |
| --- | --- | --- |
| `level_1` | single stencil, reduction or short elementwise expression | 8 |
| `level_2` | all-pairs interactions, batched linear algebra, multi-state updates | 7 |
| `level_3` | sequential dynamic programs, FFTs, tree backward induction | 7 |

Each task carries `domain`, `subdomain`, `level`, `tags` and a `source` block.
The reference files contain a docstring with the domain, the problem solved and
the canonical paper/source.

### Task list

| id | task | domain | level |
| --- | --- | --- | --- |
| 101 | `pairwise_rbf` | statistics | 1 |
| 102 | `laplacian_2d` | physics | 1 |
| 103 | `heat_step_2d` | physics | 1 |
| 104 | `lennard_jones_energy` | chemistry | 1 |
| 105 | `lotka_volterra_step` | biology | 1 |
| 106 | `gaussian_kde` | statistics | 1 |
| 107 | `horner_eval` | mathematics | 1 |
| 108 | `black_scholes_call` | economics | 1 |
| 201 | `nbody_acceleration` | physics | 2 |
| 202 | `coulomb_energy` | chemistry | 2 |
| 203 | `hodgkin_huxley_step` | biology | 2 |
| 204 | `kalman_update` | statistics | 2 |
| 205 | `graph_laplacian` | mathematics | 2 |
| 206 | `portfolio_variance` | economics | 2 |
| 207 | `wave_step_2d` | physics | 2 |
| 301 | `needleman_wunsch_score` | biology | 3 |
| 302 | `fft_convolution` | mathematics | 3 |
| 303 | `binomial_option_crr` | economics | 3 |
| 304 | `split_step_schrodinger` | physics | 3 |
| 305 | `ewald_real_space` | chemistry | 3 |
| 306 | `gaussian_process_posterior` | statistics | 3 |
| 307 | `floyd_warshall` | mathematics | 3 |


## Schema

Required columns:

| column | type | meaning |
| --- | --- | --- |
| `problem_id` | int | unique task id |
| `name` | string | filesystem-safe task name |
| `code` | string | self-contained reference source |

The `code` must define:

```python
class Model(nn.Module): ...
def get_inputs(): ...
def get_init_inputs(): ...
```

Optional columns understood by the harness:

| column | meaning |
| --- | --- |
| `level` | dataset split / difficulty grouping (1-3 here) |
| `domain` | physics, chemistry, biology, statistics, mathematics, economics |
| `subdomain` | finer label, e.g. `molecular dynamics`, `PDE`, `bioinformatics` |
| `filter` | include in the representative subset |
| `task_type` | `pytorch_reference` or `kernel_reference` |
| `backend` | `cuda`, `triton`, ... |
| `precision` | `fp32`, `fp16`, `bf16` |
| `tags` | searchable labels |
| `source` | provenance / license metadata |

## Local layout

```text
data/science_kernels/
├── README.md                  # this dataset card
├── manifest.json              # task list and provenance
├── kernels/
│   ├── level_1/               # 8 tasks
│   ├── level_2/               # 7 tasks
│   └── level_3/               # 7 tasks
└── to_hf.py                   # per-level export / optional push_to_hub
```

## Validate locally

```bash
# Schema only
robust-kernelbench-dataset validate \
  --dataset_path data/science_kernels \
  --dataset_format directory

# Counts by level and domain, and a browsable list
robust-kernelbench-dataset stats \
  --dataset_path data/science_kernels \
  --dataset_format directory
robust-kernelbench-dataset list \
  --dataset_path data/science_kernels \
  --dataset_format directory --domain physics

# Also run one forward pass per task
robust-kernelbench-dataset validate \
  --dataset_path data/science_kernels \
  --dataset_format directory \
  --execute --device cpu
```

## Use with robust-kernelbench

Evaluation only:

```bash
robust-kernelbench-eval \
  --dataset_path data/science_kernels \
  --dataset_format directory \
  --dataset_split level_1 \
  --experiment science_exp --trial 1
```

Agentic generate -> compile -> evaluate:

```bash
robust-kernelbench \
  --dataset_path data/science_kernels \
  --dataset_format directory \
  --dataset_split level_1 \
  --dataset_name science-kernels \
  --version science_v0 \
  --model_name Qwen/Qwen3-Coder-30B-A3B-Instruct \
  "1 1 kernelbench"
```

## Export for Hugging Face

```bash
python data/science_kernels/to_hf.py --output /tmp/science_kernels_hf
huggingface-cli upload <your-org>/science-kernels /tmp/science_kernels_hf \
  --repo-type dataset
```

Or push directly:

```bash
python data/science_kernels/to_hf.py \
  --output /tmp/science_kernels_hf \
  --repo-id <your-org>/science-kernels
```

## Provenance and licensing

Only add kernels you have the right to redistribute.  Record the source
repository, commit, and license in each task's `source` block.  Original tasks
written for this dataset are released under AGPL-3.0, matching the repository.
If you combine sources with different licenses, use the strictest compatible
license for the combined dataset and document it in this card.
