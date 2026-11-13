"""Tests for joining the previous round onto the KernelBench dataset."""

from __future__ import annotations

import pandas as pd
from datasets import Dataset

from robust_kernelbench.utils.utils_inference import create_joined_dataset


def _dataset():
    return Dataset.from_list(
        [
            {"problem_id": 1, "name": "one", "code": "class Model: pass"},
            {"problem_id": 2, "name": "two", "code": "class Model: pass"},
        ]
    )


def test_join_expands_samples_and_copies_previous_fields():
    generations = pd.DataFrame(
        [
            {"problem_id": 1, "generations": "['gen0', 'gen1']"},
            {"problem_id": 2, "generations": "['only']"},
        ]
    )
    compilations = pd.DataFrame(
        [
            {"problem_id": 1, "sample_id": 0, "model_new_available": True, "main_output": "ok"},
            {"problem_id": 2, "sample_id": 0, "model_new_available": False, "main_output": "bad"},
        ]
    )
    evaluations = pd.DataFrame(
        [
            {
                "problem_id": 1,
                "sample_id": 0,
                "compiled": True,
                "correctness_success": True,
                "correctness": True,
                "runtime": 1.0,
                "runtime_original": 2.0,
            }
        ]
    )

    joined = create_joined_dataset(
        _dataset(),
        previous_generations=generations,
        previous_compilations=compilations,
        previous_evaluations=evaluations,
    )

    rows = {(row["problem_id"], row["sample_id"]): row for row in joined}
    assert len(joined) == 3
    assert rows[(1, 0)]["generation"] == "gen0"
    assert rows[(1, 1)]["generation"] == "gen1"
    assert rows[(1, 0)]["model_new_available"] is True
    assert rows[(1, 0)]["correctness"] is True
    assert rows[(2, 0)]["model_new_available"] is False


def test_join_without_previous_data_is_a_noop():
    dataset = _dataset()
    assert create_joined_dataset(dataset) is dataset
