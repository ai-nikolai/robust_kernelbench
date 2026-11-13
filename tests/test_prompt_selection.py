"""Tests for the prompt decision table in :mod:`robust_kernelbench.agent.prompt_common`."""

from __future__ import annotations

import pytest

from robust_kernelbench.agent.prompt_common import (
    PromptSpec,
    failure_stage,
    get_compile_error_info,
    get_correctness_error_info,
    get_runtime_error_info,
    select_prompt_spec,
)

BASE = {"code": "class Model: pass", "generation": "```python\nx=1\n```"}


def sample(**overrides):
    payload = dict(BASE)
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"generation": ""}, "empty_generation"),
        ({"model_new_available": False}, "compile_error"),
        ({"model_new_available": True, "second_compile_success": False}, "compile_error_v2"),
        ({"model_new_available": True, "second_compile_success": True, "correctness_success": False}, "runtime_error"),
        (
            {
                "model_new_available": True,
                "second_compile_success": True,
                "correctness_success": True,
                "correctness": False,
            },
            "incorrect",
        ),
        (
            {
                "model_new_available": True,
                "second_compile_success": True,
                "correctness_success": True,
                "correctness": True,
            },
            "correct",
        ),
    ],
)
def test_failure_stage(overrides, expected):
    payload = {
        "model_new_available": False,
        "second_compile_success": False,
        "correctness_success": False,
        "correctness": False,
    }
    payload.update(overrides)
    assert failure_stage(payload) == expected


# ---------------------------------------------------------------------------
# First round: always a fresh generation from the reference code
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prompt_type,expected_func",
    [
        ("kernelbench", "generate_kb_prompt"),
        ("normal", "generate_prompt"),
        ("single_stage", "generate_prompt"),
        ("multi_stage", "generate_prompt"),
        ("kb_multi_stage", "generate_kb_prompt"),
        ("something_new", "generate_kb_prompt"),
    ],
)
def test_first_round_uses_clean_generation(prompt_type, expected_func):
    spec = select_prompt_spec(sample(), trial=1, prompt_type=prompt_type)
    assert spec.func_name == expected_func


def test_run_clean_trial_forces_first_round_behaviour():
    spec = select_prompt_spec(sample(model_new_available=False), trial=5, prompt_type="kernelbench", run_clean_trial=True)
    assert spec.func_name == "generate_kb_prompt"


# ---------------------------------------------------------------------------
# Later rounds: repair prompt chosen by the first thing that broke
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides,expected_category,expected_case,expected_func",
    [
        # An empty generation is not repaired: it is simply retried from scratch.
        ({"generation": ""}, "empty_generation", None, "generate_kb_prompt"),
        ({"model_new_available": False}, "kb_debug_error", "compilation_error", "generate_kb_tts_prompt"),
        (
            {"model_new_available": True, "second_compile_success": False},
            "kb_debug2_error",
            "compilation_error",
            "generate_kb_tts_prompt",
        ),
        (
            {"model_new_available": True, "second_compile_success": True, "correctness_success": False},
            "kb_runtime_error",
            "runtime_error",
            "generate_kb_tts_prompt",
        ),
        (
            {
                "model_new_available": True,
                "second_compile_success": True,
                "correctness_success": True,
                "correctness": False,
            },
            "kb_correctness_error",
            "incorrect",
            "generate_kb_tts_prompt",
        ),
        (
            {
                "model_new_available": True,
                "second_compile_success": True,
                "correctness_success": True,
                "correctness": True,
            },
            "kb_improve",
            "improve",
            "generate_kb_tts_prompt",
        ),
    ],
)
def test_kernelbench_round_two_selection(overrides, expected_category, expected_case, expected_func):
    spec = select_prompt_spec(sample(**overrides), trial=2, prompt_type="kernelbench")
    assert isinstance(spec, PromptSpec)
    assert spec.category == expected_category
    assert spec.func_name == expected_func
    assert spec.kwargs.get("case") == expected_case


def test_single_stage_repair_prompts():
    spec = select_prompt_spec(sample(model_new_available=False), trial=2, prompt_type="single_stage")
    assert spec.func_name == "generate_improvement_prompt_single_stage"
    assert spec.kwargs["case"] == "compilation_error"
    assert spec.category == "improve_compilation"


def test_multi_stage_repair_prompts():
    payload = sample(model_new_available=True, second_compile_success=True, correctness_success=True, correctness=False)
    spec = select_prompt_spec(payload, trial=3, prompt_type="multi_stage")
    assert spec.func_name == "generate_correctness_prompt"


@pytest.mark.parametrize("prompt_type,expected_version", [("kb_multi_stage", "v1"), ("kb_multi_stage_v2", "v2")])
def test_kb_multi_stage_carries_template_version(prompt_type, expected_version):
    spec = select_prompt_spec(sample(model_new_available=False), trial=2, prompt_type=prompt_type)
    assert spec.kwargs["multi_stage"] is True
    assert spec.kwargs["version"] == expected_version


def test_normal_prompt_never_repairs():
    spec = select_prompt_spec(sample(model_new_available=False), trial=2, prompt_type="normal")
    assert spec.func_name == "generate_prompt"
    assert spec.kwargs == {}


def test_invalid_trial_rejected():
    with pytest.raises(ValueError):
        select_prompt_spec(sample(), trial=0, prompt_type="kernelbench")


# ---------------------------------------------------------------------------
# Error extraction
# ---------------------------------------------------------------------------
def test_compile_error_prefers_runtime_output():
    payload = {"compilation_output_runtime": "first", "compilation_output_error": "second"}
    assert get_compile_error_info(payload) == "first"


def test_error_info_skips_nan_and_empty():
    payload = {"correctness_output_error": "nan", "correctness_output_traceback": "", "correctness_output": "real"}
    assert get_runtime_error_info(payload) == "real"
    assert get_correctness_error_info(payload) == "real"


def test_error_info_all_empty_returns_empty_string():
    assert get_compile_error_info({}) == ""
