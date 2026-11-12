"""Task sources for KernelBench-style datasets.

The evaluator only needs three columns:

* ``problem_id`` -- unique integer
* ``name``        -- filesystem-safe task name
* ``code``        -- self-contained source defining ``Model``, ``get_inputs`` and
                     ``get_init_inputs``

The official KernelBench dataset is one source; a local directory, JSONL file or
Parquet file with the same columns is equally valid.  Keeping loading and
validation here means the agentic and evaluation entry points cannot disagree
about where a dataset comes from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from robust_kernelbench.cli_utils import positive_int
from robust_kernelbench.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_DATASET = "ai-nikolai/KernelBench"
SUPPORTED_FORMATS = ("hf", "directory", "jsonl", "parquet")
REQUIRED_COLUMNS = ("problem_id", "name", "code")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SCHEMA_VERSION = "kernelbench-v1"

# KernelBench's representative subset, kept here so a custom source can use the
# same `--filter` flag without importing the evaluation module (which would cycle).
REPRESENTATIVE_IDS = {
    1: {1, 3, 6, 18, 23, 26, 33, 36, 40, 42, 48, 54, 57, 65, 77, 82, 86, 87},
    2: {1, 2, 8, 18, 23, 28, 33, 43},
    3: {1, 5, 8, 11, 20, 33, 38, 43},
}


class TaskValidationError(ValueError):
    """Raised when a task source does not satisfy the KernelBench schema."""


@dataclass(frozen=True)
class TaskSource:
    """Where a task set comes from."""

    name: str = DEFAULT_DATASET
    label: str | None = None
    format: str = "hf"
    path: str | None = None
    split: str | None = None
    revision: str | None = None
    version: str | None = None
    schema: str = SCHEMA_VERSION
    domains: tuple[str, ...] = ()

    def for_level(self, level: int) -> "TaskSource":
        """Use ``level_<n>`` when the caller did not name a split."""
        if self.split:
            return self
        return replace(self, split=f"level_{level}")

    @property
    def display_name(self) -> str:
        if self.label:
            return self.label
        if self.path:
            return Path(self.path).name or self.name
        return self.name

    @property
    def slug(self) -> str:
        # Keep default experiment names unchanged; custom sources include a slug.
        raw = self.display_name or self.name
        return re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_")[:40] or "dataset"

    def metadata(self, dataset: Dataset | None = None) -> dict[str, Any]:
        payload = {
            "dataset_name": self.name,
            "dataset_label": self.label,
            "dataset_display_name": self.display_name,
            "dataset_format": self.format,
            "dataset_path": self.path,
            "dataset_split": self.split,
            "dataset_revision": self.revision,
            "dataset_version": self.version,
            "dataset_schema": self.schema,
            "dataset_domains": list(self.domains),
        }
        if dataset is not None:
            payload["dataset_fingerprint"] = fingerprint(dataset, self)
            payload["dataset_size"] = len(dataset)
        return payload


def add_dataset_args(parser: argparse.ArgumentParser, default_dataset: str = DEFAULT_DATASET) -> argparse.ArgumentParser:
    """Add the shared dataset-selection flags to a CLI parser."""
    parser.add_argument("--dataset", type=str, default=default_dataset, help="HF dataset id.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Local dataset directory/JSONL/Parquet path.")
    parser.add_argument(
        "--dataset_format",
        type=str,
        default=None,
        choices=SUPPORTED_FORMATS,
        help="Dataset format (default: directory when --dataset_path is set, otherwise hf).",
    )
    parser.add_argument("--dataset_split", type=str, default=None, help="Dataset split (e.g. level_1, train).")
    parser.add_argument("--dataset_revision", type=str, default=None, help="HF revision / commit SHA.")
    parser.add_argument(
        "--dataset_version",
        type=str,
        default=None,
        help="Dataset content version (e.g. 0.1.0), recorded in metadata.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Short dataset name used in metadata and the experiment signature.",
    )
    parser.add_argument(
        "--domain",
        action="append",
        default=None,
        metavar="NAME",
        help="Restrict tasks to a domain (repeatable), e.g. --domain physics.",
    )
    return parser


def source_from_args(args: argparse.Namespace, level: int | None = None) -> TaskSource:
    """Build a :class:`TaskSource` from parsed CLI arguments."""
    path = getattr(args, "dataset_path", None)
    fmt = getattr(args, "dataset_format", None)
    if fmt is None:
        fmt = "directory" if path else "hf"

    label = getattr(args, "dataset_name", None)
    if not label and path:
        label = Path(path).name

    source = TaskSource(
        name=getattr(args, "dataset", DEFAULT_DATASET),
        label=label,
        format=fmt,
        path=path,
        split=getattr(args, "dataset_split", None),
        revision=getattr(args, "dataset_revision", None),
        version=getattr(args, "dataset_version", None),
        domains=tuple(getattr(args, "domain", None) or ()),
    )
    return source.for_level(level) if level is not None else source


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _safe_join(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        if os.path.commonpath([str(root.resolve()), str(candidate)]) != str(root.resolve()):
            raise TaskValidationError(f"code_path escapes the dataset root: {relative!r}")
    except ValueError as e:
        raise TaskValidationError(f"invalid code_path: {relative!r}") from e
    return candidate


def _load_directory(source: TaskSource) -> list[dict[str, Any]]:
    if not source.path:
        raise TaskValidationError("directory datasets require --dataset_path")
    root = Path(source.path).expanduser().resolve()
    if not root.is_dir():
        raise TaskValidationError(f"dataset directory not found: {root}")

    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(_read_text(manifest_path))
        if isinstance(manifest, dict):
            raw_tasks = manifest.get("tasks", [])
        elif isinstance(manifest, list):
            raw_tasks = manifest
        else:
            raise TaskValidationError("manifest.json must contain an object or a list of tasks")
        tasks: list[dict[str, Any]] = []
        for item in raw_tasks:
            task = dict(item)
            code_path = task.pop("code_path", None)
            if code_path and not task.get("code"):
                task["code"] = _read_text(_safe_join(root, code_path))
            if not task.get("name") and code_path:
                task["name"] = Path(code_path).stem
            tasks.append(task)
        if tasks:
            return tasks

    # No manifest (or an empty one): treat every Python file as a task and derive stable ids.
    tasks = []
    for index, path in enumerate(sorted(root.rglob("*.py")), start=1):
        tasks.append(
            {
                "problem_id": index,
                "name": path.stem,
                "code": _read_text(path),
                "path": str(path.relative_to(root)),
            }
        )
    return tasks


def _load_tabular(source: TaskSource) -> list[dict[str, Any]]:
    if not source.path:
        raise TaskValidationError(f"{source.format} datasets require --dataset_path")
    path = Path(source.path).expanduser()
    if not path.exists():
        raise TaskValidationError(f"dataset file not found: {path}")
    if source.format == "jsonl":
        frame = pd.read_json(path, lines=True)
    else:
        frame = pd.read_parquet(path)
    return frame.to_dict(orient="records")


def _load_hf(source: TaskSource) -> Dataset:
    loaded = load_dataset(source.name, split=source.split, revision=source.revision)
    if isinstance(loaded, DatasetDict):
        # Returned when no split was requested.  Prefer `train`; otherwise ask
        # the caller to choose instead of silently picking a split.
        if "train" in loaded:
            loaded = loaded["train"]
        else:
            raise TaskValidationError(
                f"{source.name} has splits {list(loaded)}; pass --dataset_split"
            )
    return loaded


def _normalize_rows(rows: Iterable[dict[str, Any]]) -> Dataset:
    records = []
    for row in rows:
        record = dict(row)
        if "problem_id" not in record or "name" not in record or "code" not in record:
            missing = [column for column in REQUIRED_COLUMNS if column not in record]
            raise TaskValidationError(f"task {record.get('name', '?')}: missing column(s) {missing}")
        try:
            record["problem_id"] = int(record["problem_id"])
        except (TypeError, ValueError) as e:
            raise TaskValidationError(f"problem_id is not an integer: {record['problem_id']!r}") from e
        record["name"] = str(record["name"])
        code = record["code"]
        if code is None or (isinstance(code, float) and pd.isna(code)):
            raise TaskValidationError(f"task {record['name']}: code is empty")
        record["code"] = str(code)
        records.append(record)
    return Dataset.from_list(records)


def reference_hash(code: str) -> str:
    """Canonical content hash for a reference implementation."""
    return "sha256:" + hashlib.sha256(code.encode("utf-8")).hexdigest()


def validate_rows(rows: Iterable[dict[str, Any]]) -> list[str]:
    """Return human-readable schema errors for a list of task dictionaries."""
    errors: list[str] = []
    seen: set[int] = set()
    for index, row in enumerate(rows):
        label = row.get("name", f"row {index}")
        missing = [column for column in REQUIRED_COLUMNS if column not in row]
        if missing:
            errors.append(f"{label}: missing column(s) {missing}")
            continue
        try:
            problem_id = int(row["problem_id"])
        except (TypeError, ValueError):
            errors.append(f"{label}: problem_id is not an integer")
            continue
        if problem_id in seen:
            errors.append(f"{label}: duplicate problem_id {problem_id}")
        seen.add(problem_id)
        name = str(row.get("name", ""))
        if not SAFE_NAME.match(name):
            errors.append(f"{label}: name must be a safe file stem, got {name!r}")
        code = row.get("code", "")
        if code is None or (isinstance(code, float) and pd.isna(code)) or not str(code).strip():
            errors.append(f"{label}: code is empty")
            continue
        expected_hash = row.get("reference_hash")
        if expected_hash and expected_hash != reference_hash(str(code)):
            errors.append(
                f"{label}: reference_hash mismatch (manifest={expected_hash}, "
                f"actual={reference_hash(str(code))})"
            )
    return errors


def _filter_local_split(rows: list[dict[str, Any]], split: str | None) -> list[dict[str, Any]]:
    if not split:
        return rows
    level = split[6:] if split.startswith("level_") else None

    def matches(row):
        if "split" in row:
            return str(row["split"]) == split
        if level is not None and "level" in row:
            return str(row["level"]) == level
        return None

    decisions = [matches(row) for row in rows]
    if any(decision is not None for decision in decisions):
        return [row for row, decision in zip(rows, decisions) if decision]
    return rows


def load_tasks(
    source: TaskSource,
    *,
    representative: bool = False,
    representative_ids: set[int] | None = None,
    limit: int | None = None,
    validate: bool = True,
) -> Dataset:
    """Load a task source and return a normalized ``datasets.Dataset``.

    ``representative`` filters to the dataset's ``filter`` column when present,
    otherwise to ``representative_ids`` (the KernelBench subset).  Sources without
    either are returned unchanged with a warning.
    """
    if source.format not in SUPPORTED_FORMATS:
        raise TaskValidationError(f"unsupported dataset format {source.format!r}")

    if source.format == "hf":
        dataset = _load_hf(source)
    else:
        rows = _load_directory(source) if source.format == "directory" else _load_tabular(source)
        rows = _filter_local_split(rows, source.split)
        dataset = _normalize_rows(rows)

    if validate:
        errors = validate_rows(dataset)
        if errors:
            preview = "; ".join(errors[:8])
            raise TaskValidationError(f"{len(errors)} task schema error(s): {preview}")

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    if source.domains:
        if "domain" in dataset.column_names:
            allowed = set(source.domains)
            dataset = dataset.filter(lambda row: row["domain"] in allowed)
        else:
            logger.warning("domain filter requested but the dataset has no 'domain' column")

    if representative:
        if "filter" in dataset.column_names:
            dataset = dataset.filter(lambda row: bool(row["filter"]))
        elif representative_ids:
            dataset = dataset.filter(lambda row: int(row["problem_id"]) in representative_ids)
        else:
            logger.warning("representative filter requested but the dataset has no 'filter' column")

    logger.info(
        "loaded %s task(s) from %s (%s, split=%s)",
        len(dataset),
        source.display_name,
        source.format,
        source.split,
    )
    return dataset


def fingerprint(dataset: Dataset, source: TaskSource | None = None) -> str:
    """Stable content hash used to tie results to an exact task set."""
    digest = hashlib.sha256()
    if source is not None:
        digest.update(
            json.dumps(
                {
                    "name": source.name,
                    "format": source.format,
                    "split": source.split,
                    "revision": source.revision,
                },
                sort_keys=True,
            ).encode()
        )
    for row in sorted(dataset, key=lambda item: int(item["problem_id"])):
        digest.update(str(row["problem_id"]).encode())
        digest.update(b"\0")
        digest.update(str(row.get("name", "")).encode())
        digest.update(b"\0")
        digest.update(str(row.get("code", "")).encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def dataset_metadata(source: TaskSource, dataset: Dataset) -> dict[str, Any]:
    return source.metadata(dataset)


def _execute_task(row: dict[str, Any], device: str) -> None:
    """Best-effort smoke test for a reference task (used by ``validate --execute``)."""
    import torch

    namespace: dict[str, Any] = {}
    exec(compile(row["code"], f"<task {row['problem_id']}>", "exec"), namespace)
    for name in ("Model", "get_inputs", "get_init_inputs"):
        if name not in namespace:
            raise TaskValidationError(f"task {row['problem_id']}: missing {name}")

    model = namespace["Model"](*namespace["get_init_inputs"]())
    inputs = namespace["get_inputs"]()
    if device != "cpu":
        model = model.to(device)
        inputs = [item.to(device) if isinstance(item, torch.Tensor) else item for item in inputs]
    with torch.no_grad():
        output = model(*inputs)
    if isinstance(output, torch.Tensor) and not torch.isfinite(output).all():
        raise TaskValidationError(f"task {row['problem_id']}: output contains non-finite values")


def _dataset_summary(dataset: Dataset) -> dict[str, Any]:
    levels: dict[str, int] = {}
    domains: dict[str, int] = {}
    for row in dataset:
        if "level" in row:
            key = str(row["level"])
            levels[key] = levels.get(key, 0) + 1
        if "domain" in row:
            key = str(row["domain"])
            domains[key] = domains.get(key, 0) + 1
    return {
        "rows": len(dataset),
        "levels": dict(sorted(levels.items())),
        "domains": dict(sorted(domains.items())),
    }


def _dataset_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and validate KernelBench task sources.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate a task source.")
    add_dataset_args(validate)
    validate.add_argument("--limit", type=positive_int, default=None, help="Validate at most N tasks.")
    validate.add_argument("--execute", action="store_true", help="Run one forward pass per task.")
    validate.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device for --execute."
    )

    stats = subparsers.add_parser("stats", help="Print task counts by level and domain.")
    add_dataset_args(stats)

    listing = subparsers.add_parser("list", help="List tasks in a source.")
    add_dataset_args(listing)
    listing.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    return parser


def cli(argv=None) -> None:
    args = _dataset_cli().parse_args(argv)
    source = source_from_args(args)
    try:
        dataset = load_tasks(source, limit=getattr(args, "limit", None))
    except Exception as e:  # noqa: BLE001 - CLI should report, not traceback
        print(json.dumps({"ok": False, "error": str(e)}, indent=2))
        sys.exit(1)

    if args.command == "stats":
        print(json.dumps({"ok": True, "dataset": source.metadata(dataset), **_dataset_summary(dataset)}, indent=2))
        sys.exit(0)

    if args.command == "list":
        tasks_list = [
            {
                "problem_id": int(row["problem_id"]),
                "name": row["name"],
                "level": row.get("level"),
                "domain": row.get("domain"),
                "subdomain": row.get("subdomain"),
            }
            for row in dataset
        ]
        if args.json:
            print(json.dumps(tasks_list, indent=2))
        else:
            print(f"{'id':>6}  {'level':>5}  {'domain':<12}  name")
            for task in tasks_list:
                print(
                    f"{task['problem_id']:>6}  {str(task['level']):>5}  "
                    f"{str(task['domain']):<12}  {task['name']}"
                )
        sys.exit(0)

    errors: list[str] = []
    if args.execute:
        device = args.device
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        for row in dataset:
            try:
                _execute_task(row, device)
            except BaseException as e:  # noqa: BLE001 - generated/reference code may sys.exit
                errors.append(f"task {row['problem_id']}: {e}")

    summary = {
        "ok": not errors,
        "dataset": source.metadata(dataset),
        "rows": len(dataset),
        "errors": errors,
    }
    print(json.dumps(summary, indent=2))
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    cli()
