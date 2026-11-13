"""Regression tests for stage completeness and lost evaluation rows."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from robust_kernelbench import artifacts
from robust_kernelbench.agent import orchestrator
from robust_kernelbench.agent.orchestrator import JobSpec, PipelineConfig, StageError
from robust_kernelbench.evaluate_all import write_failure_result
from robust_kernelbench.utils.utils import get_folder_path


def _code(experiment_name, trial, *keys):
    folder = get_folder_path(experiment_name, trial, postfix="code")
    for problem_id, sample_id in keys:
        with open(os.path.join(folder, f"{problem_id}__sample_{sample_id}.py"), "w") as handle:
            handle.write("# kernel\n")


def _compilations(experiment_name, trial, rows):
    path = artifacts.csv_path(experiment_name, trial, artifacts.COMPILATIONS_CSV)
    pd.DataFrame(rows, columns=["problem_id", "sample_id", "model_new_available"]).to_csv(
        path, index=False
    )


def _evaluations(experiment_name, trial, rows):
    path = artifacts.csv_path(experiment_name, trial, artifacts.EVALUATIONS_CSV)
    pd.DataFrame(rows, columns=["problem_id", "sample_id", "correctness"]).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Stage completeness: a kernel with no row is not a model failure.
# ---------------------------------------------------------------------------
def test_compile_completeness_ignores_failed_but_requires_recorded(experiment_dir):
    trial = 1
    _code("exp", trial, (1, 0), (2, 0))
    _compilations(
        "exp",
        trial,
        [
            {"problem_id": 1, "sample_id": 0, "model_new_available": False},
            {"problem_id": 2, "sample_id": 0, "model_new_available": True},
        ],
    )
    orchestrator._verify_compile_complete(JobSpec(1, trial, "kernelbench"), "exp")


def test_compile_completeness_detects_missing_row(experiment_dir):
    trial = 1
    _code("exp", trial, (1, 0), (2, 0))
    _compilations("exp", trial, [{"problem_id": 1, "sample_id": 0, "model_new_available": False}])

    with pytest.raises(StageError, match="no compilation row"):
        orchestrator._verify_compile_complete(
            JobSpec(1, trial, "kernelbench"), "exp"
        )


def test_evaluation_completeness_only_requires_successful_compiles(experiment_dir):
    trial = 1
    _compilations(
        "exp",
        trial,
        [
            {"problem_id": 1, "sample_id": 0, "model_new_available": True},
            {"problem_id": 2, "sample_id": 0, "model_new_available": False},
        ],
    )
    _evaluations("exp", trial, [{"problem_id": 1, "sample_id": 0, "correctness": False}])

    orchestrator._verify_evaluation_complete(JobSpec(1, trial, "kernelbench"), "exp")


def test_evaluation_completeness_detects_missing_row(experiment_dir):
    trial = 1
    _compilations(
        "exp", trial, [{"problem_id": 1, "sample_id": 0, "model_new_available": True}]
    )
    _evaluations("exp", trial, [])

    with pytest.raises(StageError, match="no evaluation row"):
        orchestrator._verify_evaluation_complete(
            JobSpec(1, trial, "kernelbench"), "exp"
        )


# ---------------------------------------------------------------------------
# A crashed evaluation child must leave a row, not disappear.
# ---------------------------------------------------------------------------
def test_run_job_fails_when_a_compiled_kernel_has_no_eval_row(monkeypatch, experiment_dir):
    trial = 1
    _code("exp", trial, (1, 0))
    _compilations("exp", trial, [{"problem_id": 1, "sample_id": 0, "model_new_available": True}])
    monkeypatch.setattr(orchestrator, "run", lambda *args, **kwargs: None)

    config = PipelineConfig(inference=0, compile=1, eval=1, use_online=0)
    with pytest.raises(StageError, match="no evaluation row"):
        orchestrator.run_job(config, JobSpec(1, trial, "kernelbench"), "exp")


def test_run_job_accepts_complete_stage_records(monkeypatch, experiment_dir):
    trial = 1
    _code("exp", trial, (1, 0))
    _compilations("exp", trial, [{"problem_id": 1, "sample_id": 0, "model_new_available": True}])
    _evaluations("exp", trial, [{"problem_id": 1, "sample_id": 0, "correctness": True}])
    monkeypatch.setattr(orchestrator, "run", lambda *args, **kwargs: None)

    config = PipelineConfig(inference=0, compile=1, eval=1, use_online=0)
    orchestrator.run_job(config, JobSpec(1, trial, "kernelbench"), "exp")


def test_write_failure_result_records_a_row(experiment_dir):
    write_failure_result(
        "exp", 1, problem_id=7, sample_id=0, reason="rc=-9", log_tail="killed", compiled=True
    )

    path = artifacts.csv_path("exp", 1, artifacts.EVALUATIONS_CSV)
    row = pd.read_csv(path).iloc[0]
    assert row["problem_id"] == 7
    assert row["sample_id"] == 0
    assert bool(row["compiled"]) is True
    assert bool(row["cuda_success"]) is False
    assert bool(row["correctness_success"]) is False
    assert row["runtime"] == -1.0
    assert "rc=-9" in row["main_error"]
