"""End-to-end evaluation tests.

Two layers:

* ``build_evaluate_single_command`` is pure, so the correctness/timing policy that crosses
  the subprocess boundary is asserted without any hardware.
* ``eval_kernel_against_ref`` is exercised with a *mock kernel*: ``ModelNew`` is a plain
  torch module, so no CUDA compiler is needed, but the full load -> correctness -> timing
  path runs on a real device.  Skipped when CUDA is unavailable.
"""

from __future__ import annotations

import pytest

from robust_kernelbench.evaluate_all import build_evaluate_single_command


# ---------------------------------------------------------------------------
# Subprocess boundary (CPU-only)
# ---------------------------------------------------------------------------
def test_evaluate_command_is_a_module_invocation():
    command = build_evaluate_single_command(
        original_path="/a/ref.py",
        new_path="/a/new.py",
        build_dir="/a/build",
        build_dir_new="/a/build_new",
        experiment_name="exp",
        problem_id=7,
        sample_id=1,
        level=1,
        trial=2,
    )
    assert command[1:3] == ["-m", "robust_kernelbench.evaluate_single"]
    assert command[command.index("--original_path") + 1] == "/a/ref.py"
    assert command[command.index("--new_path") + 1] == "/a/new.py"
    assert command[command.index("--experiment") + 1] == "exp"
    assert command[command.index("--problem_id") + 1] == "7"
    assert command[command.index("--sample_id") + 1] == "1"
    assert command[command.index("--trial") + 1] == "2"


def test_evaluate_command_forwards_trials_and_tolerances():
    command = build_evaluate_single_command(
        original_path="a",
        new_path="b",
        build_dir="c",
        build_dir_new="d",
        experiment_name="exp",
        problem_id=1,
        sample_id=0,
        level=1,
        trial=1,
        num_correct_trials=6,
        num_perf_trials=9,
        atol=1e-3,
        rtol=1e-4,
        flush_l2=False,
    )
    assert command[command.index("--num_correct_trials") + 1] == "6"
    assert command[command.index("--num_perf_trials") + 1] == "9"
    assert command[command.index("--atol") + 1] == "0.001"
    assert command[command.index("--rtol") + 1] == "0.0001"
    assert "--no-flush_l2" in command


def test_evaluate_command_omits_tolerances_when_unset():
    command = build_evaluate_single_command(
        original_path="a",
        new_path="b",
        build_dir="c",
        build_dir_new="d",
        experiment_name="exp",
        problem_id=1,
        sample_id=0,
        level=1,
        trial=1,
    )
    assert "--atol" not in command
    assert "--rtol" not in command
    assert "--flush_l2" in command


# ---------------------------------------------------------------------------
# Full evaluation path with a mock (pure-torch) "kernel"
# ---------------------------------------------------------------------------
REFERENCE_SRC = '''
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, a, b):
        return a + b


def get_inputs():
    return [torch.randn(16, 16), torch.randn(16, 16)]


def get_init_inputs():
    return []
'''

MOCK_KERNEL_SRC = '''
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, a, b):
        return a + b
'''

WRONG_KERNEL_SRC = '''
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, a, b):
        return a - b
'''


@pytest.fixture
def cuda_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    return torch.cuda.current_device()


def _evaluate(custom_src, device, tmp_path, **kwargs):
    from robust_kernelbench.evaluate_single import eval_kernel_against_ref

    build_dir = tmp_path / "ref_build"
    build_dir_new = tmp_path / "new_build"
    build_dir.mkdir(exist_ok=True)
    build_dir_new.mkdir(exist_ok=True)

    return eval_kernel_against_ref(
        original_model_src=REFERENCE_SRC,
        custom_model_src=custom_src,
        seed_num=42,
        num_correct_trials=kwargs.pop("num_correct_trials", 2),
        num_perf_trials=kwargs.pop("num_perf_trials", 2),
        verbose=False,
        measure_performance=kwargs.pop("measure_performance", False),
        build_dir=str(build_dir),
        build_dir_new=str(build_dir_new),
        device=device,
        **kwargs,
    )


@pytest.mark.gpu
def test_correct_mock_kernel_passes(cuda_device, tmp_path):
    result = _evaluate(MOCK_KERNEL_SRC, cuda_device, tmp_path)

    assert result.compiled is True
    assert result.runtime_success is True
    assert result.loading_success is True
    assert result.correctness_success is True
    assert result.correctness is True
    assert result.metadata["correctness_trials"] == "(2 / 2)"


@pytest.mark.gpu
def test_incorrect_mock_kernel_fails(cuda_device, tmp_path):
    result = _evaluate(WRONG_KERNEL_SRC, cuda_device, tmp_path)

    assert result.correctness_success is True  # the check itself ran
    assert result.correctness is False  # ...and reported a mismatch
    assert "Output mismatch" in (result.main_error or "")


@pytest.mark.gpu
def test_tolerance_is_resolved_from_dtype_and_recorded(cuda_device, tmp_path):
    result = _evaluate(MOCK_KERNEL_SRC, cuda_device, tmp_path)

    assert result.metadata["tolerance_dtype"] == "torch.float32"
    assert result.metadata["atol"] == 1e-4
    assert result.metadata["rtol"] == 1e-4


@pytest.mark.gpu
def test_explicit_tolerance_override_is_recorded(cuda_device, tmp_path):
    result = _evaluate(MOCK_KERNEL_SRC, cuda_device, tmp_path, atol=1e-2, rtol=1e-2)

    assert result.metadata["atol"] == 1e-2
    assert result.metadata["rtol"] == 1e-2


@pytest.mark.gpu
def test_performance_is_measured_with_l2_flush(cuda_device, tmp_path):
    result = _evaluate(MOCK_KERNEL_SRC, cuda_device, tmp_path, measure_performance=True, flush_l2=True)

    assert result.correctness is True
    assert result.timing_success is True
    assert result.runtime > 0
    assert result.runtime_original > 0
    assert result.runtime_stats["num_trials"] == 2
    assert result.metadata["flush_l2"] is True


@pytest.mark.gpu
def test_performance_without_flush_is_also_supported(cuda_device, tmp_path):
    result = _evaluate(MOCK_KERNEL_SRC, cuda_device, tmp_path, measure_performance=True, flush_l2=False)

    assert result.timing_success is True
    assert result.metadata["flush_l2"] is False
