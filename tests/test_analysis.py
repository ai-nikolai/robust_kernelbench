"""Tests for the post-hoc analysis aggregation.

The reported accuracy must be `correctness` (outputs matched), not
`correctness_success` (the checker executed), and an empty evaluations.csv must
produce zeros rather than a table full of NaN.
"""

from __future__ import annotations

import math
import warnings

import pandas as pd

from analysis.run_analysis_summary_clean import output_analysis

COLUMNS = [
    "compiled",
    "correctness",
    "runtime_success",
    "loading_success",
    "correctness_success",
    "timing_success",
    "runtime",
    "runtime_original",
]


def _frame(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def test_empty_frame_returns_zeros_without_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = output_analysis(_frame([]))

    assert not caught, [str(w.message) for w in caught]
    for key in ("compiled_global", "correctness_global", "check_ran_global", "coverage_global"):
        assert out[key] == 0
    assert not any(isinstance(v, float) and math.isnan(v) for v in out.values())


def test_accuracy_is_correctness_not_check_ran():
    """`correctness_success` only means the checker ran; it must not be the accuracy."""
    # 2 rows, both with the checker executed, only one numerically correct.
    df = _frame([
        dict(compiled=True, correctness=True, runtime_success=True, loading_success=True,
             correctness_success=True, timing_success=True, runtime=10.0, runtime_original=20.0),
        dict(compiled=True, correctness=False, runtime_success=True, loading_success=True,
             correctness_success=True, timing_success=False, runtime=-1.0, runtime_original=20.0),
    ])
    out = output_analysis(df)

    assert out["correctness_global"] == 0.5   # 1 of 2 actually correct
    assert out["check_ran_global"] == 1.0     # the checker executed for both
    assert out["correctness_global"] < out["check_ran_global"]
    assert "correctness_success_global" not in out


def test_global_denominator_is_the_row_count_not_100():
    """5 rows, all correct -> 1.0, not 0.05 (the old hardcoded /100)."""
    row = dict(compiled=True, correctness=True, runtime_success=True, loading_success=True,
               correctness_success=True, timing_success=True, runtime=10.0, runtime_original=10.0)
    out = output_analysis(_frame([row] * 5))
    assert out["correctness_global"] == 1.0
    assert out["compiled_global"] == 1.0
    assert out["coverage_global"] == 1.0
