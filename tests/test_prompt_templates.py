"""Tests for the multi-stage KernelBench prompt plumbing."""

from __future__ import annotations

from robust_kernelbench.agent import prompts_kb
from robust_kernelbench.agent.prompts_kb import get_prompt_kb_tts_multi_stage


def test_multi_stage_passes_custom_instruction_as_a_keyword(monkeypatch):
    captured = {}

    def fake_base(code_src, **kwargs):
        captured.update(kwargs)
        return ""

    monkeypatch.setattr(prompts_kb, "kb_tts_base", fake_base)

    get_prompt_kb_tts_multi_stage(
        "class Model: pass", "previous", case="improve", score=2.0, version="v2"
    )

    assert "complete freedom" in captured["custom_instruction"]


def test_multi_stage_improve_prompt_handles_missing_score():
    prompt = get_prompt_kb_tts_multi_stage(
        "class Model: pass", "previous", case="improve", score=None, version="v1"
    )

    assert "No baseline timing was available" in prompt
