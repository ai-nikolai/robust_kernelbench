"""Tests for configurable task sources (HF, directory, JSONL, Parquet)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from robust_kernelbench import tasks
from robust_kernelbench.agent.orchestrator import (
    JobSpec,
    PipelineConfig,
    build_evaluate_command,
    build_inference_command,
)


def _write_task(root: Path, name: str, problem_id: int, level: int = 1) -> None:
    kernels = root / "kernels"
    kernels.mkdir(parents=True, exist_ok=True)
    (kernels / f"{name}.py").write_text(
        "import torch\n"
        "from torch import nn\n\n"
        "class Model(nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x + 1\n\n"
        "def get_inputs():\n"
        "    return [torch.ones(4)]\n\n"
        "def get_init_inputs():\n"
        "    return []\n"
    )
    manifest = {"schema": "kernelbench-v1", "tasks": []}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest["tasks"].append(
        {
            "problem_id": problem_id,
            "name": name,
            "code_path": f"kernels/{name}.py",
            "level": level,
        }
    )
    manifest_path.write_text(json.dumps(manifest))


def test_directory_task_source_loads_manifest(tmp_path):
    _write_task(tmp_path, "spectral_filter", 1)

    dataset = tasks.load_tasks(
        tasks.TaskSource(name="science", format="directory", path=str(tmp_path), split="level_1")
    )

    assert len(dataset) == 1
    assert dataset[0]["problem_id"] == 1
    assert dataset[0]["name"] == "spectral_filter"
    assert "class Model" in dataset[0]["code"]


def test_directory_task_source_filters_by_level(tmp_path):
    _write_task(tmp_path, "level_one", 1, level=1)
    _write_task(tmp_path, "level_two", 2, level=2)

    dataset = tasks.load_tasks(
        tasks.TaskSource(name="science", format="directory", path=str(tmp_path), split="level_2")
    )

    assert [row["name"] for row in dataset] == ["level_two"]


def test_representative_filter_uses_filter_column(tmp_path):
    _write_task(tmp_path, "keep", 1)
    _write_task(tmp_path, "drop", 2)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tasks"][0]["filter"] = True
    manifest["tasks"][1]["filter"] = False
    manifest_path.write_text(json.dumps(manifest))

    dataset = tasks.load_tasks(
        tasks.TaskSource(name="science", format="directory", path=str(tmp_path)),
        representative=True,
    )

    assert [row["name"] for row in dataset] == ["keep"]


def test_jsonl_task_source(tmp_path):
    path = tmp_path / "tasks.jsonl"
    pd.DataFrame(
        [{"problem_id": 1, "name": "json_task", "code": "class Model: pass"}]
    ).to_json(path, orient="records", lines=True)

    dataset = tasks.load_tasks(tasks.TaskSource(name="json", format="jsonl", path=str(path)))

    assert len(dataset) == 1
    assert dataset[0]["name"] == "json_task"


def test_invalid_task_source_reports_the_problem(tmp_path):
    path = tmp_path / "tasks.jsonl"
    pd.DataFrame(
        [
            {"problem_id": 1, "name": "bad/name", "code": ""},
            {"problem_id": 1, "name": "duplicate", "code": "class Model: pass"},
        ]
    ).to_json(path, orient="records", lines=True)

    with pytest.raises(tasks.TaskValidationError) as error:
        tasks.load_tasks(tasks.TaskSource(name="json", format="jsonl", path=str(path)))

    message = str(error.value)
    assert "safe file stem" in message
    assert "duplicate problem_id" in message
    assert "code is empty" in message


def test_fingerprint_is_stable_and_source_sensitive(tmp_path):
    _write_task(tmp_path, "task", 1)
    source = tasks.TaskSource(name="science", format="directory", path=str(tmp_path))
    first = tasks.load_tasks(source)
    second = tasks.load_tasks(source)
    renamed = tasks.TaskSource(name="other", format="directory", path=str(tmp_path))

    assert tasks.fingerprint(first, source) == tasks.fingerprint(second, source)
    assert tasks.fingerprint(first, source) != tasks.fingerprint(second, renamed)


def test_bundled_science_dataset_has_all_levels():
    root = Path(__file__).resolve().parents[1] / "data" / "science_kernels"
    dataset = tasks.load_tasks(
        tasks.TaskSource(name="science", format="directory", path=str(root))
    )

    assert len(dataset) >= 15
    assert {int(level) for level in dataset["level"]} == {1, 2, 3}
    assert {"physics", "chemistry", "biology", "statistics", "mathematics", "economics"} <= set(
        dataset["domain"]
    )


def test_custom_dataset_reaches_stage_commands_and_experiment_name():
    config = PipelineConfig(
        dataset_path="./data/science_kernels",
        dataset_format="directory",
        dataset_split="level_1",
        dataset_name="science-kernels",
        version="v1",
        model_name="org/model",
    )

    inference = build_inference_command(
        config, JobSpec(1, 1, "kernelbench"), "exp"
    )
    evaluation = build_evaluate_command(
        config, JobSpec(1, 1, "kernelbench"), "exp"
    )

    assert inference[inference.index("--dataset_path") + 1] == "./data/science_kernels"
    assert evaluation[evaluation.index("--dataset_format") + 1] == "directory"
    assert "science_kernels" in config.signature()
    assert config.dataset_source().name == tasks.DEFAULT_DATASET
    assert config.dataset_source().display_name == "science-kernels"
    assert config.resolved_experiment_name().startswith("exp_local_")
