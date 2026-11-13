#!/usr/bin/env python3
"""Prune old run artefacts under ``experiments/``.

``experiments/`` grows without bound (tens of GB per week on a busy machine, since every
trial keeps its ``code/``, ``kernel/`` build trees, CSVs and prompts).  This utility
decides what is stale -- by age and/or by count -- and **only deletes when told to**.

Dry-run is the default; nothing is removed unless ``--delete`` is passed::

    # show what would be removed by keeping the 5 newest experiment directories
    python scripts/prune_experiments.py --keep-last 5

    # remove experiments untouched for 30 days
    python scripts/prune_experiments.py --older-than-days 30 --delete

    # remove everything except the 10 newest, but ask first
    python scripts/prune_experiments.py --keep-last 10 --delete --interactive

Safety:
* only directories *inside* ``--experiments-dir`` are ever considered;
* the target must be a real sub-directory (no symlink escapes, no ``..``);
* ``--delete`` prints a per-directory summary and the total reclaimed space.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass


@dataclass
class Candidate:
    path: str
    name: str
    modified: float
    size_bytes: int

    @property
    def age_days(self) -> float:
        return (time.time() - self.modified) / 86400.0

    @property
    def size_human(self) -> str:
        return human_size(self.size_bytes)


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}PiB"


def directory_size(path: str) -> int:
    """Total size of the regular files under ``path`` (best effort)."""
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                if not os.path.islink(file_path):
                    total += os.path.getsize(file_path)
            except OSError:
                continue
    return total


def collect_candidates(experiments_dir: str, with_sizes: bool = True) -> list[Candidate]:
    """Every immediate sub-directory of ``experiments_dir``, newest first."""
    if not os.path.isdir(experiments_dir):
        raise SystemExit(f"ERROR: experiments directory not found: {experiments_dir}")

    root = os.path.realpath(experiments_dir)
    candidates: list[Candidate] = []

    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        real = os.path.realpath(path)
        if not os.path.isdir(real):
            continue
        # Never follow a symlink out of the experiments directory.
        if os.path.islink(path) or os.path.commonpath([root, real]) != root:
            print(f"skipping {name}: not a plain directory inside {root}", file=sys.stderr)
            continue
        candidates.append(
            Candidate(
                path=path,
                name=name,
                modified=os.path.getmtime(path),
                size_bytes=directory_size(path) if with_sizes else 0,
            )
        )

    candidates.sort(key=lambda c: c.modified, reverse=True)
    return candidates


def select_for_removal(candidates: list[Candidate], keep_last: int | None, older_than_days: float | None) -> list[Candidate]:
    """Candidates to delete.

    * ``keep_last`` keeps the N most recently modified directories.
    * ``older_than_days`` additionally requires the directory to be old enough.
    * With both set, a directory is removed only if it is both outside the newest N and
      old enough -- the conservative reading, so a typo cannot nuke fresh runs.
    """
    if keep_last is None and older_than_days is None:
        return []

    selected = list(candidates)

    if keep_last is not None:
        if keep_last < 0:
            raise SystemExit("ERROR: --keep-last must be >= 0")
        selected = selected[keep_last:]

    if older_than_days is not None:
        if older_than_days < 0:
            raise SystemExit("ERROR: --older-than-days must be >= 0")
        selected = [c for c in selected if c.age_days >= older_than_days]

    return selected


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Prune stale directories under experiments/ (dry-run unless --delete)."
    )
    parser.add_argument("--experiments-dir", default="experiments", help="Directory holding the experiments.")
    parser.add_argument("--keep-last", type=int, default=None, help="Keep the N most recently modified experiments.")
    parser.add_argument("--older-than-days", type=float, default=None, help="Only prune experiments older than this.")
    parser.add_argument("--delete", action="store_true", help="Actually delete (default: dry-run).")
    parser.add_argument("--interactive", action="store_true", help="Ask before deleting each experiment.")
    parser.add_argument("--no-sizes", action="store_true", help="Skip the (slow) size computation.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.keep_last is None and args.older_than_days is None:
        print(
            "Nothing to do: pass --keep-last N and/or --older-than-days D "
            "(nothing was selected, so nothing will be removed)."
        )
        return 0

    candidates = collect_candidates(args.experiments_dir, with_sizes=not args.no_sizes)
    to_remove = select_for_removal(candidates, args.keep_last, args.older_than_days)

    total = sum(c.size_bytes for c in to_remove)

    print(f"{args.experiments_dir}: {len(candidates)} experiment(s)")
    for candidate in candidates:
        marker = "DELETE" if candidate in to_remove else "keep  "
        print(
            f"  [{marker}] {candidate.name:<60} {candidate.age_days:7.1f}d "
            f"{candidate.size_human if not args.no_sizes else '':>10}"
        )

    if not to_remove:
        print("\nNothing selected for removal.")
        return 0

    print(f"\n{len(to_remove)} director{'y' if len(to_remove) == 1 else 'ies'} selected, "
          f"{human_size(total)} reclaimable.")

    if not args.delete:
        print("Dry-run: re-run with --delete to remove them.")
        return 0

    removed = 0
    for candidate in to_remove:
        if args.interactive:
            answer = input(f"Delete {candidate.name} ({candidate.size_human})? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print(f"  skipped {candidate.name}")
                continue
        try:
            shutil.rmtree(candidate.path)
            removed += 1
            print(f"  removed {candidate.name}")
        except OSError as e:
            print(f"  FAILED to remove {candidate.name}: {e}", file=sys.stderr)

    print(f"\nRemoved {removed}/{len(to_remove)} director{'y' if len(to_remove) == 1 else 'ies'}.")
    return 0 if removed == len(to_remove) else 1


if __name__ == "__main__":
    sys.exit(main())
