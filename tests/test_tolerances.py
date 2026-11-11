"""Tests for dtype-aware correctness tolerances and the L2-flush helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from robust_kernelbench.evaluate_single import (  # noqa: E402
    DTYPE_TOLERANCES,
    FALLBACK_TOLERANCE,
    dtype_key,
    flush_l2_cache,
    make_l2_flush_buffer,
    resolve_tolerances,
    tolerance_for_dtype,
)


# ---------------------------------------------------------------------------
# Dtype policy
# ---------------------------------------------------------------------------
def test_float64_is_stricter_than_float32():
    atol64, _ = tolerance_for_dtype(torch.float64)
    atol32, _ = tolerance_for_dtype(torch.float32)
    assert atol64 < atol32


def test_low_precision_dtypes_are_looser_than_float32():
    atol32, _ = tolerance_for_dtype(torch.float32)
    atol16, _ = tolerance_for_dtype(torch.float16)
    atolbf16, _ = tolerance_for_dtype(torch.bfloat16)
    assert atol16 >= atol32
    assert atolbf16 >= atol16


def test_integer_and_bool_are_exact():
    assert tolerance_for_dtype(torch.int32) == (0.0, 0.0)
    assert tolerance_for_dtype(torch.int64) == (0.0, 0.0)
    assert tolerance_for_dtype(torch.bool) == (0.0, 0.0)


def test_unknown_dtype_falls_back_to_legacy_tolerance():
    assert tolerance_for_dtype(None) == FALLBACK_TOLERANCE


def test_every_known_dtype_is_ordered_sensibly():
    for key, (atol, rtol) in DTYPE_TOLERANCES.items():
        assert atol >= 0 and rtol >= 0
        assert key.startswith("torch.")


def test_dtype_key_is_stable():
    assert dtype_key(torch.float32) == "torch.float32"


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------
def test_explicit_tolerances_win_over_dtype_policy():
    assert resolve_tolerances(torch.float32, atol=1e-2, rtol=1e-2) == (1e-2, 1e-2)


def test_legacy_values_reproduce_old_behaviour():
    # `--atol 1e-2 --rtol 1e-2` must reproduce the historical comparison exactly.
    assert resolve_tolerances(torch.bfloat16, atol=1e-2, rtol=1e-2) == (1e-2, 1e-2)


def test_partial_override_uses_dtype_for_the_other_term():
    dtype_atol, dtype_rtol = tolerance_for_dtype(torch.float64)
    atol, rtol = resolve_tolerances(torch.float64, atol=0.5, rtol=None)
    assert atol == 0.5
    assert rtol == dtype_rtol


def test_no_override_uses_dtype_policy():
    assert resolve_tolerances(torch.float16) == tolerance_for_dtype(torch.float16)


# ---------------------------------------------------------------------------
# L2 flush
# ---------------------------------------------------------------------------
def test_flush_buffer_can_be_allocated_and_zeroed_on_cpu():
    buffer = make_l2_flush_buffer(device="cpu", size_bytes=1024)
    assert buffer is not None
    buffer.fill_(1)
    flush_l2_cache(buffer)
    assert int(buffer.abs().sum()) == 0


def test_flush_of_none_buffer_is_a_noop():
    flush_l2_cache(None)


def test_unallocatable_buffer_returns_none_instead_of_raising(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(torch, "empty", boom)
    assert make_l2_flush_buffer(device="cpu", size_bytes=1024) is None
