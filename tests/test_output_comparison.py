"""Tests for tensor and tuple output comparison."""

from __future__ import annotations

import torch

from robust_kernelbench.evaluate_single import _compare_output_sequences


def test_single_tensor_match_records_tolerance():
    metadata = {}
    matched, message = _compare_output_sequences(
        torch.ones(2, 2), torch.ones(2, 2), atol=None, rtol=None, metadata=metadata
    )

    assert matched
    assert message == ""
    assert metadata["tolerance_dtype"] == "torch.float32"
    assert metadata["atol"] == 1e-4


def test_tuple_outputs_are_compared_elementwise():
    expected = (torch.ones(2), torch.zeros(1))
    actual = (torch.ones(2), torch.zeros(1))

    matched, _ = _compare_output_sequences(expected, actual, atol=None, rtol=None, metadata={})

    assert matched


def test_tuple_shape_mismatch_is_reported_as_incorrect_not_a_crash():
    expected = (torch.ones(2), torch.ones(3))
    actual = (torch.ones(2), torch.ones(4))

    matched, message = _compare_output_sequences(expected, actual, atol=None, rtol=None, metadata={})

    assert not matched
    assert "shape mismatch" in message
