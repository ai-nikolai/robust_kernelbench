"""Regression tests for prompt rendering edge cases."""

from __future__ import annotations

from robust_kernelbench.agent.prompts_v2 import generate_kb_tts_prompt


def _sample(**extra):
    payload = {"code": "class Model: pass", "generation": "```python\nclass ModelNew: pass\n```"}
    payload.update(extra)
    return payload


def test_improve_prompt_handles_missing_timing_score():
    prompt = generate_kb_tts_prompt(_sample(), case="improve")

    assert "No baseline timing was available" in prompt


def test_improve_prompt_handles_zero_timing_score():
    prompt = generate_kb_tts_prompt(_sample(runtime=0.0, runtime_original=2.0), case="improve")

    assert "No baseline timing was available" in prompt


def test_improve_prompt_formats_a_positive_speedup():
    prompt = generate_kb_tts_prompt(_sample(runtime=1.0, runtime_original=4.0), case="improve")

    assert "relative to the baseline: 3.999" in prompt
