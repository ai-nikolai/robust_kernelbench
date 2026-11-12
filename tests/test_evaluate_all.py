"""Tests for the evaluation driver: process-group kill, timeout rows, append safety.

These cover the fixes made after a kernel hung inside an uninterruptible CUDA call
and survived its per-kernel timeout (problem 45).  Everything here is CPU-only.
"""

from __future__ import annotations

import csv
import os
import signal
import subprocess
import time

import pytest

from robust_kernelbench.evaluate_all import (
    _ensure_trailing_newline,
    _kill_subprocess_group,
    write_timeout_result,
)
from robust_kernelbench.utils.utils import get_folder_path

csv.field_size_limit(1 << 30)


def _group_exists(pgid: int, retries: int = 40) -> bool:
    """Whether the process group still has live members (with a short grace period)."""
    for _ in range(retries):
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        time.sleep(0.05)
    return True


# ---------------------------------------------------------------------------
# The hang fix: killing only the direct child used to leave grandchildren
# alive, and because they inherited the pipes the parent blocked past its
# timeout.  The group kill must reap everything.
# ---------------------------------------------------------------------------
def test_kill_subprocess_group_reaps_child_and_grandchild():
    proc = subprocess.Popen(
        ["bash", "-c", "sleep 60 & sleep 60"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.3)
        pgid = os.getpgid(proc.pid)
        assert _group_exists(pgid, retries=1), "process group should be alive before the kill"

        _kill_subprocess_group(proc)

        assert not _group_exists(pgid), "process group still alive after killpg"
        assert proc.returncode == -signal.SIGKILL
    finally:
        if proc.poll() is None:
            proc.kill()


# ---------------------------------------------------------------------------
# Timeout rows must record only what was observed: the compile stage succeeded,
# but the process was killed before it could load, run or time anything.
# ---------------------------------------------------------------------------
def test_write_timeout_result_records_observed_state_only(experiment_dir):
    os.makedirs(get_folder_path("exp_timeout", trial=1), exist_ok=True)

    write_timeout_result("exp_timeout", 1, problem_id=45, sample_id=0, timeout=1800, log_tail="stuck")

    path = os.path.join(get_folder_path("exp_timeout", trial=1), "evaluations.csv")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["problem_id"] == "45"
    assert row["sample_id"] == "0"
    # Observed: the kernel compiled.
    assert row["compiled"] == "True"
    # Never observed: the process was killed.
    for flag in ("cuda_success", "runtime_success", "loading_success", "correctness",
                 "correctness_success", "timing_success"):
        assert row[flag] == "False", flag
    assert row["runtime"] == "-1.0"
    assert row["runtime_stats"] == "{}"
    assert "timed out after 1800s" in row["main_error"]


def test_write_timeout_result_appends_after_existing_rows(experiment_dir):
    os.makedirs(get_folder_path("exp_two", trial=1), exist_ok=True)
    write_timeout_result("exp_two", 1, problem_id=1, sample_id=0, timeout=60)
    write_timeout_result("exp_two", 1, problem_id=2, sample_id=0, timeout=60)

    path = os.path.join(get_folder_path("exp_two", trial=1), "evaluations.csv")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["problem_id"] for r in rows] == ["1", "2"]
    assert len(open(path).readline().strip().split(",")) == len(rows[0])


# ---------------------------------------------------------------------------
# A child killed mid-append can leave a truncated last line; the next append
# must not merge with it.
# ---------------------------------------------------------------------------
def test_ensure_trailing_newline_repairs_truncated_last_line(tmp_path):
    path = tmp_path / "evaluations.csv"
    path.write_bytes(b"a,b\n1,2")  # no trailing newline

    _ensure_trailing_newline(str(path))
    assert path.read_bytes() == b"a,b\n1,2\n"

    # idempotent: a file that already ends with a newline is untouched
    _ensure_trailing_newline(str(path))
    assert path.read_bytes() == b"a,b\n1,2\n"


def test_ensure_trailing_newline_ignores_missing_file(tmp_path):
    _ensure_trailing_newline(str(tmp_path / "absent.csv"))  # must not raise
