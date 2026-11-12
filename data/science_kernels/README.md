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
    path: science_kernels.parquet
---

# Science KernelBench

A science-focused task set for `robust-kernelbench`.  Each task provides a
PyTorch reference implementation of a numerical primitive; the candidate must
produce a faster, numerically equivalent GPU kernel.

This directory is the **source of truth**.  `to_hf.py` expands the manifest into
a Hugging Face dataset; the generated export is not committed.

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
| `level` | dataset split / difficulty grouping |
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
│   ├── 0001_pairwise_rbf.py
│   └── 0002_laplacian_2d.py
└── to_hf.py                   # local export / optional push_to_hub
```

## Validate locally

```bash
# Schema only
robust-kernelbench-dataset validate \
  --dataset_path data/science_kernels \
  --dataset_format directory

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
