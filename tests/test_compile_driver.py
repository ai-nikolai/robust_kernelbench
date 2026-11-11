"""Tests for the shared compile driver: job building, argument construction, modes."""

from __future__ import annotations

import pytest

from robust_kernelbench import compile_driver
from robust_kernelbench.compile_driver import (
    CompileJob,
    add_common_compile_args,
    build_compile_jobs,
    build_single_compile_command,
    compile_all,
    resolve_experiment_name,
)
from robust_kernelbench.utils.utils import construct_generation_dict


@pytest.fixture
def code_dir(tmp_path):
    """A fake `<experiment>/trial_1/code` directory with two generated kernels."""
    code = tmp_path / "experiments" / "exp" / "trial_1" / "code"
    code.mkdir(parents=True)
    (code / "7__sample_0.py").write_text("# kernel for problem 7 sample 0\n")
    (code / "7__sample_1.py").write_text("# kernel for problem 7 sample 1\n")
    (code / "9__sample_0.py").write_text("# kernel for problem 9 sample 0\n")
    return code


# ---------------------------------------------------------------------------
# Grouping helper (moved out of evaluate_all, which created an import cycle)
# ---------------------------------------------------------------------------
def test_construct_generation_dict_groups_by_problem(code_dir):
    grouped = construct_generation_dict(str(code_dir))
    assert set(grouped) == {"7__sample_", "9__sample_"}
    assert len(grouped["7__sample_"]) == 2


# ---------------------------------------------------------------------------
# Job building
# ---------------------------------------------------------------------------
def test_build_compile_jobs_produces_one_job_per_solution(code_dir, tmp_path):
    kernel = tmp_path / "experiments" / "exp" / "trial_1" / "kernel"
    kernel.mkdir(parents=True)

    jobs = build_compile_jobs(
        folder_path_code=str(code_dir),
        folder_path_kernel=str(kernel),
        trial=1,
        experiment_name="exp",
        device=None,
    )

    assert len(jobs) == 3
    assert all(isinstance(job, CompileJob) for job in jobs)
    assert {job.sample_id for job in jobs if job.problem_id == 7} == {0, 1}
    # The build dir must exist and follow the documented naming convention.
    for job in jobs:
        assert job.build_dir_new.endswith("_solution_kernel")
        assert job.custom_model_src.startswith("# kernel for problem")
        assert job.code_path.endswith(".py")


def test_build_compile_jobs_skips_already_compiled_pairs(code_dir, tmp_path):
    kernel = tmp_path / "experiments" / "exp" / "trial_1" / "kernel"
    kernel.mkdir(parents=True)

    jobs = build_compile_jobs(
        folder_path_code=str(code_dir),
        folder_path_kernel=str(kernel),
        trial=1,
        experiment_name="exp",
        device=None,
        pid_sid_set={(7, 0), (9, 0)},
    )

    assert [(j.problem_id, j.sample_id) for j in jobs] == [(7, 1)]


def test_build_compile_jobs_respects_target_problem_id(code_dir, tmp_path):
    kernel = tmp_path / "experiments" / "exp" / "trial_1" / "kernel"
    kernel.mkdir(parents=True)

    jobs = build_compile_jobs(
        folder_path_code=str(code_dir),
        folder_path_kernel=str(kernel),
        trial=1,
        experiment_name="exp",
        device=None,
        target_problem_id=9,
    )

    assert [(j.problem_id, j.sample_id) for j in jobs] == [(9, 0)]


def test_build_compile_jobs_honours_limit(code_dir, tmp_path):
    kernel = tmp_path / "experiments" / "exp" / "trial_1" / "kernel"
    kernel.mkdir(parents=True)

    jobs = build_compile_jobs(
        folder_path_code=str(code_dir),
        folder_path_kernel=str(kernel),
        trial=1,
        experiment_name="exp",
        device=None,
        limit=2,
    )
    assert len(jobs) == 2


# ---------------------------------------------------------------------------
# Sequential-mode command construction (used to be built inline, with an
# undefined `eval_file_path`/`level`, so the module could never have run)
# ---------------------------------------------------------------------------
def test_single_compile_command_is_module_invocation(tmp_path):
    job = CompileJob(
        device=None,
        custom_model_src="src",
        build_dir_new=str(tmp_path / "build"),
        trial=3,
        problem_id=7,
        sample_id=1,
        experiment_name="exp",
        len_list_of_solutions=2,
        code_path=str(tmp_path / "7__sample_1.py"),
    )

    command = build_single_compile_command(job)

    assert command[0].endswith(("python", "python3")) or "python" in command[0]
    assert command[1:3] == ["-m", "robust_kernelbench.compile_single"]
    assert command[command.index("--code_path") + 1] == job.code_path
    assert command[command.index("--build_dir_new") + 1] == job.build_dir_new
    assert command[command.index("--trial") + 1] == "3"
    assert command[command.index("--problem_id") + 1] == "7"
    assert command[command.index("--sample_id") + 1] == "1"
    assert command[command.index("--experiment") + 1] == "exp"


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------
def test_common_compile_args_defaults():
    parser = pytest.importorskip("argparse").ArgumentParser()
    add_common_compile_args(parser)

    args = parser.parse_args(["--experiment_name", "exp", "--trial", "4"])
    assert args.trial == 4
    assert args.mode is None
    assert resolve_experiment_name(args) == "exp"


def test_experiment_alias_is_accepted():
    parser = pytest.importorskip("argparse").ArgumentParser()
    add_common_compile_args(parser)

    args = parser.parse_args(["--experiment", "legacy-name"])
    assert resolve_experiment_name(args) == "legacy-name"


def test_missing_experiment_name_is_a_usage_error():
    parser = pytest.importorskip("argparse").ArgumentParser()
    add_common_compile_args(parser)

    args = parser.parse_args([])
    with pytest.raises(SystemExit):
        resolve_experiment_name(args)


def test_compile_all_rejects_unknown_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "experiments").mkdir()
    with pytest.raises(ValueError):
        compile_all(experiment_name="exp", trial=1, mode="bogus")


def test_compile_all_with_no_jobs_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "experiments").mkdir()
    assert compile_all(experiment_name="exp", trial=1, mode="parallel") == 0
