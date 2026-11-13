"""Smoke test for the ScienceKernelBench report generator."""

from __future__ import annotations

import json

import pandas as pd

from analysis.run_science_benchmark_report import main


def _write_run(root, level, mode, trial, correct):
    folder = root / "experiments" / f"exp_science_v0_level_{level}_{mode}" / f"trial_{trial}"
    folder.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "problem_id": 1,
            "sample_id": 0,
            "cuda_success": True,
            "compiled": True,
            "runtime_success": True,
            "loading_success": True,
            "correctness_success": True,
            "timing_success": True,
            "correctness": correct,
            "runtime": 10.0,
            "runtime_original": 20.0,
            "metadata": {},
        }
    ]
    pd.DataFrame(rows).to_csv(folder / "evaluations.csv", index=False)
    (folder / "metadata.json").write_text(
        json.dumps({"dataset_version": "0.1.0", "dataset_revision": "v0.1.0"})
    )


def test_report_writes_table_and_plots(tmp_path):
    for level in (1, 2):
        for mode in ("kernelbench", "kb_multi_stage"):
            for trial in range(1, 7):
                _write_run(tmp_path, level, mode, trial, correct=trial > 1)

    output = tmp_path / "out"
    assert (
        main(
            [
                "--experiments-dir",
                str(tmp_path / "experiments"),
                "--output-dir",
                str(output),
                "--levels",
                "1",
                "2",
                "--rounds",
                "6",
            ]
        )
        == 0
    )

    assert (output / "science_v0_rounds.csv").exists()
    assert (output / "science_v0_table.md").exists()
    assert (output / "science_v0_correctness.png").exists()
    assert (output / "science_v0_final_comparison.png").exists()
