"""Shared prompt helpers: failure classification, error extraction, prompt selection.

The multi-round loop works by looking at the *previous* round's ``compilations.csv`` /
``evaluations.csv`` and choosing a repair prompt based on the first thing that broke.
That decision table used to live inline in ``prompts_v2.choose_prompt`` (a ~120-line
``if/elif`` ladder) mixed together with the prompt templates themselves.

It now lives here as a pure function (:func:`select_prompt_spec`) that returns a
:class:`PromptSpec` describing *which* generator to call and *with what* arguments.
``prompts_v2`` resolves the spec against its own template functions, which keeps
``prompts_v2`` a thin facade and makes the decision table unit-testable without
rendering a single prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from robust_kernelbench.logging_utils import get_logger

logger = get_logger(__name__)

NAN_STRINGS = {"nan", "None", ""}


# ============================================================================
#  Error-information extraction
# ============================================================================
def _first_meaningful(sample: Mapping[str, Any], keys) -> str:
    """Return the first non-empty, non-NaN value among ``keys``."""
    for key in keys:
        value = sample.get(key, "")
        if isinstance(value, float) and value != value:  # NaN
            continue
        if value is None:
            continue
        text = str(value)
        if text.strip() and text.strip() not in NAN_STRINGS:
            return text
    return ""


def get_compile_error_info(sample) -> str:
    """Best available message for a *compilation* failure."""
    return _first_meaningful(
        sample,
        ("compilation_output_runtime", "compilation_output_error", "compilation_output", "main_output"),
    )


def get_runtime_error_info(sample) -> str:
    """Best available message for a *runtime/execution* failure."""
    return _first_meaningful(
        sample,
        ("correctness_output_error", "correctness_output_traceback", "correctness_output", "main_output"),
    )


def get_correctness_error_info(sample) -> str:
    """Best available message for an *incorrect output* failure."""
    return get_runtime_error_info(sample)


def get_generated_code(sample) -> str:
    """Extract the previous round's kernel source from a joined dataset row."""
    generated_code = sample.get("generation")
    if not generated_code:
        logger.debug("no previous generation available for this sample")
        return ""

    # Imported lazily: `utils_inference` pulls in `datasets`, which this pure decision
    # module should not require.
    from robust_kernelbench.utils.utils_inference import extract_code

    try:
        extracted = extract_code(generated_code)
    except Exception as e:  # noqa: BLE001 - extraction is heuristic, never fatal
        logger.warning("code extraction failed: %s", e)
        return ""

    if not extracted:
        logger.debug("code extraction produced empty output")
    return extracted


# ============================================================================
#  Prompt selection
# ============================================================================
@dataclass(frozen=True)
class PromptSpec:
    """Which prompt generator to use, and with what arguments.

    ``func_name`` is the name of a generator function in :mod:`robust_kernelbench.agent.prompts_v2`;
    ``kwargs`` are bound with :func:`functools.partial`.
    """

    category: str
    func_name: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)


# The first round always starts from the reference code, whatever the prompt type.
FIRST_ROUND_FALLBACK = "generate_kb_prompt"
FIRST_ROUND_SIMPLE = "generate_prompt"


def _first_round_spec(prompt_type: str) -> PromptSpec:
    if prompt_type in ("normal", "single_stage", "multi_stage"):
        return PromptSpec(category=prompt_type, func_name=FIRST_ROUND_SIMPLE)
    if prompt_type == "kernelbench":
        return PromptSpec(category="kernelbench", func_name="generate_kb_prompt")
    logger.warning("unknown prompt_type %r on the first round; falling back to kernelbench", prompt_type)
    return PromptSpec(category="kernelbench", func_name=FIRST_ROUND_FALLBACK)


def failure_stage(sample: Mapping[str, Any]) -> str:
    """Classify how the previous round failed, most-fatal first.

    Returns one of ``empty_generation``, ``compile_error``, ``compile_error_v2``,
    ``runtime_error``, ``incorrect`` or ``correct``.
    """
    if "generation" in sample and not sample.get("generation"):
        return "empty_generation"
    if not sample.get("model_new_available", False):
        return "compile_error"
    if not sample.get("second_compile_success", False):
        return "compile_error_v2"
    if not sample.get("correctness_success", False):
        return "runtime_error"
    if not sample.get("correctness", False):
        return "incorrect"
    return "correct"


# For each prompt_type, map (failure stage) -> (category, generator, extra kwargs).
# `improve_*` is kept for backwards compatibility with the recorded prompt_category
# column in existing experiments.
_STAGE_TABLE = {
    "single_stage": {
        "compile_error": ("improve_compilation", "generate_improvement_prompt_single_stage", {"case": "compilation_error"}),
        "compile_error_v2": ("improve_compilation_v2", "generate_improvement_prompt_single_stage", {"case": "runtime_error"}),
        "runtime_error": ("improve_debug", "generate_improvement_prompt_single_stage", {"case": "runtime_error"}),
        "incorrect": ("improve_correctness", "generate_improvement_prompt_single_stage", {"case": "incorrect"}),
        "correct": ("improve_runtime", "generate_improvement_prompt_single_stage", {"case": "improve"}),
    },
    "multi_stage": {
        "compile_error": ("improve_compilation", "generate_debug_compilation_prompt", {}),
        "compile_error_v2": ("improve_compilation_v2", "generate_debug_compilation_prompt", {}),
        "runtime_error": ("improve_debug", "generate_debug_runtime_prompt", {}),
        "incorrect": ("improve_correctness", "generate_correctness_prompt", {}),
        "correct": ("improve_runtime", "generate_improvement_prompt", {}),
    },
    "kernelbench": {
        "empty_generation": ("empty_generation", "generate_kb_prompt", {}),
        "compile_error": ("kb_debug_error", "generate_kb_tts_prompt", {"case": "compilation_error"}),
        "compile_error_v2": ("kb_debug2_error", "generate_kb_tts_prompt", {"case": "compilation_error"}),
        "runtime_error": ("kb_runtime_error", "generate_kb_tts_prompt", {"case": "runtime_error"}),
        "incorrect": ("kb_correctness_error", "generate_kb_tts_prompt", {"case": "incorrect"}),
        "correct": ("kb_improve", "generate_kb_tts_prompt", {"case": "improve"}),
    },
    "kb_multi_stage": {
        "empty_generation": ("empty_generation", "generate_kb_tts_prompt", {"case": "empty_generation", "multi_stage": True}),
        "compile_error": ("kb_debug_error", "generate_kb_tts_prompt", {"case": "compilation_error", "multi_stage": True}),
        "compile_error_v2": ("kb_debug2_error", "generate_kb_tts_prompt", {"case": "compilation_error", "multi_stage": True}),
        "runtime_error": ("kb_runtime_error", "generate_kb_tts_prompt", {"case": "runtime_error", "multi_stage": True}),
        "incorrect": ("kb_correctness_error", "generate_kb_tts_prompt", {"case": "incorrect", "multi_stage": True}),
        "correct": ("kb_improve", "generate_kb_tts_prompt", {"case": "improve", "multi_stage": True}),
    },
}


def select_prompt_spec(sample, trial: int, prompt_type: str = "normal", run_clean_trial: bool = False) -> PromptSpec:
    """Decide which prompt to send for this (sample, round).

    ``trial == 1`` (or ``run_clean_trial``) always starts from the reference code;
    later rounds repair whatever broke, according to :func:`failure_stage`.
    """
    if trial < 1:
        raise ValueError(f"Trial needs to be >= 1. Actually it is: {trial}")

    if trial == 1 or run_clean_trial:
        return _first_round_spec(prompt_type)

    stage = failure_stage(sample)

    if prompt_type == "normal":
        # The plain "normal" prompt is a pure re-generation, never a repair.
        return PromptSpec(category="normal", func_name=FIRST_ROUND_SIMPLE)

    table = _STAGE_TABLE.get(prompt_type)
    if prompt_type.startswith("kb_multi_stage"):
        # "kb_multi_stage" and "kb_multi_stage_v2" share the same decision table; the
        # suffix only selects which template variant is rendered.
        kb_tts_version = "v2" if prompt_type.endswith("v2") else "v1"
        table = {
            stage_name: (category, func, dict(kwargs, version=kb_tts_version))
            for stage_name, (category, func, kwargs) in _STAGE_TABLE["kb_multi_stage"].items()
        }

    if table is None:
        logger.warning("unknown prompt_type %r after round 1; falling back to normal", prompt_type)
        return PromptSpec(category="normal", func_name=FIRST_ROUND_SIMPLE)

    if stage not in table:
        raise ValueError(f"Unhandled failure stage {stage!r} for prompt_type {prompt_type!r}")

    category, func_name, kwargs = table[stage]
    return PromptSpec(category=category, func_name=func_name, kwargs=kwargs)


def resolve_prompt_func(spec: PromptSpec, module=None) -> Callable:
    """Look ``spec.func_name`` up in ``module`` (default: :mod:`prompts_v2`).

    ``module`` is imported lazily so that ``prompt_common`` does not import the prompt
    templates at module load (they pull in the ``datasets``/``toml`` stack).
    """
    if module is None:
        from robust_kernelbench.agent import prompts_v2 as module  # local import: avoids a cycle

    try:
        func = getattr(module, spec.func_name)
    except AttributeError as e:
        raise AttributeError(f"prompt generator {spec.func_name!r} not found in {module.__name__}") from e

    if spec.kwargs:
        from functools import partial

        return partial(func, **dict(spec.kwargs))
    return func


def choose_prompt(sample, trial, prompt_type="normal", run_clean_trial=False, module=None):
    """Return ``(prompt_func, prompt_category)`` for this round.

    Drop-in replacement for the old ``prompts_v2.choose_prompt``.
    """
    spec = select_prompt_spec(sample, trial=trial, prompt_type=prompt_type, run_clean_trial=run_clean_trial)
    logger.info("prompt category=%s (generator=%s) for trial=%s", spec.category, spec.func_name, trial)
    return resolve_prompt_func(spec, module=module), spec.category
