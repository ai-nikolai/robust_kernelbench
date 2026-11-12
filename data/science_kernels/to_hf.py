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

from robust_kernelbench import tasks  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True, help="Directory for the HF export.")
    parser.add_argument("--dataset-name", default="science-kernels")
    parser.add_argument("--repo-id", default=None, help="If set, push the dataset to the Hub.")
    parser.add_argument("--private", action="store_true", help="Create a private Hub dataset.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    source = tasks.TaskSource(
        name=args.dataset_name,
        format="directory",
        path=str(root),
        split="level_1",
    )

    dataset = tasks.load_tasks(source)
    args.output.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output / "science_kernels.parquet"
    dataset.to_parquet(parquet_path)

    readme_src = root / "README.md"
    readme_dst = args.output / "README.md"
    if readme_src.exists():
        shutil.copyfile(readme_src, readme_dst)

    print(f"exported {len(dataset)} task(s) to {args.output}")
    print(f"  data:   {parquet_path}")
    print(f"  card:   {readme_dst}")

    if args.repo_id:
        dataset.push_to_hub(args.repo_id, private=args.private)
        if readme_dst.exists():
            from huggingface_hub import HfApi

            HfApi().upload_file(
                path_or_fileobj=str(readme_dst),
                path_in_repo="README.md",
                repo_id=args.repo_id,
                repo_type="dataset",
            )
        print(f"pushed to https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
