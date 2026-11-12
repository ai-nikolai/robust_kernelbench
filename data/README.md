# Dataset sources

This directory holds dataset **sources** that are prepared locally and exported
to Hugging Face.  It is intentionally outside the Python package and is not
included in the `robust-kernelbench` wheel.

Why not call it `datasets/`?  A top-level directory with that name can shadow the
installed Hugging Face `datasets` package when running from the repository root.

Each dataset follows the same layout:

```text
data/<dataset-name>/
├── README.md          # Hugging Face dataset card
├── manifest.json      # task list, optional metadata/provenance
├── kernels/           # reference implementations
└── to_hf.py           # export / optional upload script
```

Validate and export:

```bash
robust-kernelbench-dataset validate \
  --dataset_path data/science_kernels \
  --dataset_format directory

python data/science_kernels/to_hf.py --output /tmp/science_kernels_hf
huggingface-cli upload ai-nikolai/ScienceKernelBench /tmp/science_kernels_hf \
  --repo-type dataset
```

See `data/science_kernels/README.md` for the science-focused starter set.
