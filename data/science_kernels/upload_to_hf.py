#!/usr/bin/env python3
"""Upload an exported ScienceKernelBench folder with the Python API.

This avoids depending on a specific `hf` / `huggingface-cli` binary, which
changed between huggingface_hub major versions.

Usage::

    huggingface-cli login          # or: hf auth login
    python data/science_kernels/to_hf.py --output /tmp/science_kernels_hf
    python data/science_kernels/upload_to_hf.py \
        --path /tmp/science_kernels_hf

The token can also be supplied through ``HF_TOKEN``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_REPO_ID = "ai-nikolai/ScienceKernelBench"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True, help="Folder produced by to_hf.py.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true", help="Create a private dataset repo.")
    parser.add_argument("--token", default=None, help="HF token (default: cached login or HF_TOKEN).")
    parser.add_argument("--commit-message", default="Upload ScienceKernelBench")
    parser.add_argument("--tag", default=None, help="Optional release tag, e.g. v0.1.0.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    folder = args.path.expanduser().resolve()
    if not (folder / "README.md").exists():
        print(f"ERROR: {folder}/README.md not found; run to_hf.py first.", file=sys.stderr)
        return 1
    if not (folder / "data").is_dir():
        print(f"ERROR: {folder}/data not found; run to_hf.py first.", file=sys.stderr)
        return 1

    token = args.token or os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: install huggingface_hub (`pip install huggingface_hub`).", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
    )
    if args.tag:
        api.create_tag(args.repo_id, tag=args.tag, repo_type="dataset")
        print(f"tagged release {args.tag}")
    print(f"uploaded to https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
