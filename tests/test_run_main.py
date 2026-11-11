"""Tests for the pipeline orchestrator: CLI parsing, job handling, command building."""

from __future__ import annotations

import pytest

from robust_kernelbench import run_main
from robust_kernelbench.run_main import (
    EXIT_FAILURE,
    EXIT_OK,
    EXIT_USAGE,
    JobSpec,
    PipelineConfig,
    build_compile_command,
    build_evaluate_command,
    build_inference_command,
    config_from_args,
    main,
    parse_args,
    parse_jobs,
    server_control,
)


# ---------------------------------------------------------------------------
# The bug this used to have: with no jobs, `run_config` was never assigned and
# `main_loop` raised NameError.  It must now be a clean usage error.
# ---------------------------------------------------------------------------
def test_no_jobs_exits_cleanly_with_usage_error(capsys):
    exit_code = main([])
    assert exit_code == EXIT_USAGE
    assert "no jobs supplied" in capsys.readouterr().err


def test_no_jobs_does_not_raise(monkeypatch):
    monkeypatch.setattr(run_main, "run_job", lambda *a, **k: pytest.fail("stages must not run"))
    assert main([]) == EXIT_USAGE


# ---------------------------------------------------------------------------
# Job parsing
# ---------------------------------------------------------------------------
def test_parse_jobs_ok():
    jobs = parse_jobs(["1 1 kernelbench", "4  204  kb_multi_stage "])
    assert jobs == [JobSpec(1, 1, "kernelbench"), JobSpec(4, 204, "kb_multi_stage")]


@pytest.mark.parametrize(
    "bad",
    [
        "1 1",  # too few fields
        "1 1 kernelbench extra",  # too many fields
        "a 1 kernelbench",  # non-integer trial
        "1 0 kernelbench",  # non-positive new trial
    ],
)
def test_parse_jobs_rejects_malformed_jobs(bad):
    with pytest.raises(ValueError):
        parse_jobs([bad])


def test_malformed_job_is_reported_as_usage_error(capsys):
    assert main(["1 1"]) == EXIT_USAGE
    assert "malformed job" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_config_signature_and_experiment_name():
    config = PipelineConfig(model_name="Qwen/Qwen3-Coder-Next", version="v9_7", level=1, use_online=1)
    assert config.short_model_name() == "Qwen3_Coder_Next"
    assert config.signature() == "L1_v9_7_Qwen3_Coder_Next_kernelbench_API1"
    assert config.resolved_experiment_name() == "exp_local_L1_v9_7_Qwen3_Coder_Next_kernelbench_API1"


def test_config_is_immutable():
    config = PipelineConfig()
    with pytest.raises(Exception):
        config.model_name = "other"  # type: ignore[misc]


def test_parse_args_does_not_run_at_import():
    # Importing the module must not parse sys.argv (it used to).
    assert callable(parse_args)


# ---------------------------------------------------------------------------
# Command construction: tensor-parallel size used to be hardcoded to 1.
# ---------------------------------------------------------------------------
def test_inference_command_uses_configured_tensor_parallel_size():
    config = PipelineConfig(tensor_parallel_size=2, num_samples=3)
    command = build_inference_command(config, JobSpec(1, 1, "kernelbench"), "exp_test")
    assert command[command.index("--tensor_parallel_size") + 1] == "2"
    assert command[command.index("--num_samples") + 1] == "3"
    assert "1" not in command[command.index("--tensor_parallel_size") + 1 :][:1]


def test_inference_command_default_tp_is_one():
    config = PipelineConfig()
    command = build_inference_command(config, JobSpec(1, 1, "kernelbench"), "exp_test")
    assert command[command.index("--tensor_parallel_size") + 1] == "1"


def test_inference_command_passes_online_url_and_backend_when_online():
    config = PipelineConfig(use_online=1, serving_backend="sglang", online_service_url="http://h:1234/v1")
    command = build_inference_command(config, JobSpec(1, 1, "kernelbench"), "exp_test")
    assert command[command.index("--online_service_url") + 1] == "http://h:1234/v1"
    assert command[command.index("--backend") + 1] == "sglang"
    assert "robust_kernelbench.run_inference_test_time_scaling_v2" in command


def test_inference_command_omits_online_url_when_offline():
    config = PipelineConfig(use_online=0, serving_backend="vllm")
    command = build_inference_command(config, JobSpec(1, 1, "kernelbench"), "exp_test")
    assert "--online_service_url" not in command
    assert "robust_kernelbench.run_inference_test_time_scaling" in command


def test_compile_command_modes():
    parallel = build_compile_command(PipelineConfig(compile=1, num_procs=8), JobSpec(1, 1, "kb"), "e")
    sequential = build_compile_command(PipelineConfig(compile=2), JobSpec(1, 1, "kb"), "e")
    assert "robust_kernelbench.compile_all" in parallel
    assert parallel[parallel.index("--num_processes") + 1] == "8"
    assert "robust_kernelbench.compile_all_sequential" in sequential


def test_compile_command_rejects_disabled_stage():
    with pytest.raises(ValueError):
        build_compile_command(PipelineConfig(compile=0), JobSpec(1, 1, "kb"), "e")


def test_evaluate_command_forwards_trial_and_tolerance_policy():
    command = build_evaluate_command(PipelineConfig(num_correct_trials=7, num_perf_trials=4), JobSpec(1, 5, "kb"), "e")
    assert command[command.index("--num_correct_trials") + 1] == "7"
    assert command[command.index("--num_perf_trials") + 1] == "4"
    assert command[command.index("--trial") + 1] == "5"
    # No explicit tolerance => evaluate_single applies its dtype policy.
    assert "--atol" not in command and "--rtol" not in command
    assert "--flush_l2" in command


def test_evaluate_command_forwards_explicit_tolerances():
    command = build_evaluate_command(PipelineConfig(atol=1e-2, rtol=1e-2, flush_l2=False), JobSpec(1, 1, "kb"), "e")
    assert command[command.index("--atol") + 1] == "0.01"
    assert command[command.index("--rtol") + 1] == "0.01"
    assert "--no-flush_l2" in command


def test_default_num_correct_trials_is_five():
    assert PipelineConfig().num_correct_trials >= 5


def test_evaluate_command_forwards_eval_memory_cap():
    """--eval_mem_gb must reach the evaluate stage (it bounds per-kernel host memory)."""
    command = build_evaluate_command(PipelineConfig(eval_mem_gb=12.5), JobSpec(1, 1, "kb"), "e")
    assert command[command.index("--eval_mem_gb") + 1] == "12.5"


def test_evaluate_command_forwards_eval_timeout():
    command = build_evaluate_command(PipelineConfig(eval_timeout=42), JobSpec(1, 1, "kb"), "e")
    assert command[command.index("--timeout") + 1] == "42"


# ---------------------------------------------------------------------------
# Serving backend: both SGLang and vLLM endpoints used to be called every time.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "backend,expected_path,enable_sleep",
    [
        ("sglang", "/release_memory_occupation", False),
        # vLLM only exposes /sleep when the server was started with sleep mode.
        ("vllm", "/sleep?level=1", True),
    ],
)
def test_pause_contacts_only_the_selected_backend(monkeypatch, backend, expected_path, enable_sleep):
    calls = []
    monkeypatch.setattr(run_main, "_post_json", lambda url, timeout=60: calls.append(url) or {})

    server_control(PipelineConfig(serving_backend=backend, use_online=1, enable_sleep=enable_sleep), "pause")

    assert len(calls) == 1
    assert calls[0].endswith(expected_path)


def test_vllm_without_sleep_mode_contacts_nothing(monkeypatch):
    """vLLM started without --enable-sleep-mode has no /sleep endpoint to call."""
    calls = []
    monkeypatch.setattr(run_main, "_post_json", lambda url, timeout=60: calls.append(url) or {})

    server_control(PipelineConfig(serving_backend="vllm", use_online=1), "pause")
    server_control(PipelineConfig(serving_backend="vllm", use_online=1), "resume")

    assert calls == []


def test_serving_backend_none_contacts_nothing(monkeypatch):
    calls = []
    monkeypatch.setattr(run_main, "_post_json", lambda url, timeout=60: calls.append(url) or {})

    server_control(PipelineConfig(serving_backend="none"), "pause")
    server_control(PipelineConfig(serving_backend="none"), "resume")

    assert calls == []


def test_server_host_port_parsing():
    config = PipelineConfig(online_service_url="http://example.host:1234/v1")
    assert config.server_host_port() == ("example.host", 1234)


# ---------------------------------------------------------------------------
# Failure handling: a failing stage used to be swallowed with `continue`.
# ---------------------------------------------------------------------------
def test_failing_job_returns_non_zero_and_stops_by_default(monkeypatch):
    attempted = []

    def boom(config, job, experiment_name):
        attempted.append(job)

    monkeypatch.setattr(run_main, "run_job", boom)
    assert main(["1 1 kernelbench", "1 2 kernelbench"], config=PipelineConfig()) == EXIT_OK
    assert len(attempted) == 2


def test_stage_error_stops_remaining_jobs(monkeypatch):
    attempted = []

    def boom(config, job, experiment_name):
        attempted.append(job)
        raise run_main.StageError("inference", job, returncode=1)

    monkeypatch.setattr(run_main, "run_job", boom)
    exit_code = main(["1 1 kb", "1 2 kb", "1 3 kb"], config=PipelineConfig())
    assert exit_code == EXIT_FAILURE
    assert len(attempted) == 1  # stopped after the first failure


def test_continue_on_error_runs_every_job(monkeypatch):
    attempted = []

    def boom(config, job, experiment_name):
        attempted.append(job)
        raise run_main.StageError("compile", job, returncode=2)

    monkeypatch.setattr(run_main, "run_job", boom)
    exit_code = main(["1 1 kb", "1 2 kb"], config=PipelineConfig(continue_on_error=True))
    assert exit_code == EXIT_FAILURE
    assert len(attempted) == 2


def test_unexpected_exception_is_contained(monkeypatch):
    def boom(config, job, experiment_name):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(run_main, "run_job", boom)
    assert main(["1 1 kb"], config=PipelineConfig()) == EXIT_FAILURE


def test_config_from_args_roundtrip():
    args = parse_args(
        [
            "--model_name", "org/model",
            "--tensor_parallel_size", "2",
            "--serving_backend", "vllm",
            "--atol", "0.5",
            "--num_correct_trials", "9",
            "--reset_experiments",
            "1 4 kernelbench",
        ]
    )
    config = config_from_args(args)
    assert config.model_name == "org/model"
    assert config.tensor_parallel_size == 2
    assert config.serving_backend == "vllm"
    assert config.atol == 0.5
    assert config.num_correct_trials == 9
    assert config.reset_experiments is True
