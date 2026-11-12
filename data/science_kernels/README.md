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
- config_name: default
  data_files:
  - split: level_1
    path: data/level_1-*
  - split: level_2
    path: data/level_2-*
  - split: level_3
    path: data/level_3-*
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


## Versioning

Current release: **0.1.0** (`schema_version: kernelbench-v1`), compatible with
`robust-kernelbench >= 0.2.0`.

The dataset version is independent of the package version:

- `PATCH` — documentation, metadata or provenance fixes with no scoring change.
- `MINOR` — new tasks, domains or levels; existing tasks unchanged.
- `MAJOR` — removal/renumbering, reference-output changes, or benchmark-split
  changes.

Every task has a permanent `problem_id`, a stable `task_key`, a `task_version`
and a `reference_hash`. Removed tasks are marked `retired`; IDs are never reused.
Pin production runs to an HF tag or commit:

```bash
robust-kernelbench \
  --dataset ai-nikolai/ScienceKernelBench \
  --dataset_revision v0.1.0 \
  --dataset_split level_1 \
  ...
```

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
├── manifest.json              # versioned task list and provenance
├── CHANGELOG.md               # dataset release history
├── requirements.txt           # minimal HF/export dependencies
├── kernels/
│   ├── level_1/               # 8 tasks
│   ├── level_2/               # 7 tasks
│   └── level_3/               # 7 tasks
└── to_hf.py                   # per-level export / optional push_to_hub
```

## Minimal dependencies

```bash
pip install -r requirements.txt
```

`torch` is optional and only needed for `--execute` smoke tests; it is listed as
a comment in the requirements file so the dataset can be loaded and exported on
a lightweight machine.

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

# huggingface_hub >= 1 uses the `hf` command:
hf upload ai-nikolai/ScienceKernelBench /tmp/science_kernels_hf --repo-type dataset

# Works with any huggingface_hub version:
python data/science_kernels/upload_to_hf.py \
  --path /tmp/science_kernels_hf
```

Or export and push in one step:

```bash
python data/science_kernels/to_hf.py \
  --output /tmp/science_kernels_hf \
  --push
```

## Provenance and licensing

Only add kernels you have the right to redistribute.  Record the source
repository, commit, and license in each task's `source` block.  Original tasks
written for this dataset are released under AGPL-3.0, matching the repository.
If you combine sources with different licenses, use the strictest compatible
license for the combined dataset and document it in this card.


## Citation

If you use this dataset, please cite `robust-kernelbench`:

```bibtex
@misc{rozanov2026robustkernelbench,
  title        = {Robust KernelBench: A Robust Harness for GPU Kernel Generation and Evaluation},
  author       = {Rozanov, Nikolai},
  year         = {2026},
  howpublished = {\url{https://github.com/ai-nikolai/robust_kernelbench}},
  note         = {ScienceKernelBench dataset, version 0.1.0}
}
```

The dataset card and version should also be cited when reporting benchmark
results, together with the exact `dataset_revision` used.
