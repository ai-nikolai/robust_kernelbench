#!/usr/bin/env python3
"""Build a Hugging Face dataset from the science-kernels manifest.

Usage::

    python data/science_kernels/to_hf.py --output /tmp/science_kernels_hf
    python data/science_kernels/to_hf.py --output /tmp/science_kernels_hf \
        --repo-id <your-org>/science-kernels
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_REPO_ID = "ai-nikolai/ScienceKernelBench"

from robust_kernelbench import tasks  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True, help="Directory for the HF export.")
    parser.add_argument("--dataset-name", default="ScienceKernelBench")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument("--push", action="store_true", help="Push the exported dataset to the Hub.")
    parser.add_argument("--private", action="store_true", help="Create a private Hub dataset.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    source = tasks.TaskSource(
        name=args.dataset_name,
        format="directory",
        path=str(root),
    )

    dataset = tasks.load_tasks(source)
    args.output.mkdir(parents=True, exist_ok=True)

    levels = sorted({int(level) for level in dataset["level"]})
    configs = {}
    data_dir = args.output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for level in levels:
        split = dataset.filter(lambda row: int(row["level"]) == level)
        configs[f"level_{level}"] = split
        split.to_parquet(data_dir / f"level_{level}-00000-of-00001.parquet")
        print(f"  level_{level}: {len(split)} task(s)")

    readme_src = root / "README.md"
    readme_dst = args.output / "README.md"
    if readme_src.exists():
        shutil.copyfile(readme_src, readme_dst)

    requirements_src = root / "requirements.txt"
    if requirements_src.exists():
        shutil.copyfile(requirements_src, args.output / "requirements.txt")

    print(f"exported {len(dataset)} task(s) to {args.output}")
    print(f"  card:   {readme_dst}")

    if args.push:
        from datasets import DatasetDict

        DatasetDict(configs).push_to_hub(args.repo_id, private=args.private)
        if readme_dst.exists():
            from huggingface_hub import HfApi

            HfApi().upload_file(
                path_or_fileobj=str(readme_dst),
                path_in_repo="README.md",
                repo_id=args.repo_id,
                repo_type="dataset",
            )
        print(f"pushed to https://huggingface.co/datasets/{args.repo_id}")
    else:
        print(
            "export ready; upload with: "
            f"huggingface-cli upload {args.repo_id} {args.output} --repo-type dataset"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
