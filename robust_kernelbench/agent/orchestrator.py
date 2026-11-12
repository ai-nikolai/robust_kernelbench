#!/usr/bin/env python3
"""End-to-end pipeline orchestrator.

For each ``(previous_trial, new_trial, prompt_type)`` job this runs the three stages:

1. **inference** -- generate kernels (``run_inference_test_time_scaling*``)
2. **compile**    -- compile them (``compile_all`` or ``compile_all_sequential``)
3. **evaluate**   -- verify correctness and time them (``evaluate_all``)

Usage::

    python -m robust_kernelbench.run_main \
        --version v9_4 --model_name Qwen/Qwen3-Coder-30B-A3B-Instruct \
        --parent_prompt_type kernelbench --num_samples 1 \
        "1 1 kernelbench"

Design notes
------------
* Configuration is a frozen :class:`PipelineConfig` dataclass, not module-level globals:
  the module can be imported, the stages can be driven programmatically, and the flags are
  documented in one place.
* ``parse_args`` is only called from ``main``/``cli``.  Previously it ran at *import* time,
  which made the module unimportable and untestable.
* With no jobs the pipeline now exits with a clear error instead of raising ``NameError``.
* Stage failures are logged with a traceback and recorded; ``--continue_on_error`` decides
  whether the remaining jobs still run, and the process exits non-zero if any job failed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from robust_kernelbench import artifacts, tasks
from robust_kernelbench.cli_utils import non_negative_float, positive_int
from robust_kernelbench.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_ONLINE_SERVICE_URL = "http://localhost:30000/v1"
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2

DEFAULT_INFERENCE_SCRIPTS = {
    "vllm": "robust_kernelbench.run_inference_test_time_scaling",
    "sglang": "robust_kernelbench.run_inference_test_time_scaling_v2",
}


# ============================================================================
#  Configuration
# ============================================================================
@dataclass(frozen=True)
class JobSpec:
    """One round-triplet: which previous round to learn from, which round to write."""

    previous_trial: int
    new_trial: int
    prompt_type: str

    def __str__(self) -> str:
        return f"prev={self.previous_trial} new={self.new_trial} prompt={self.prompt_type}"


@dataclass(frozen=True)
class PipelineConfig:
    """Everything one pipeline invocation needs.  Immutable: derive new configs with ``replace``."""

    # Experiment identity
    experiment_name: str = ""
    model_name: str = "default"
    version: str = "V8"
    parent_prompt_type: str = "kernelbench"
    level: int = 1

    # Dataset identity.  Defaults to the official KernelBench source so the
    # generated experiment names for existing runs stay unchanged.
    dataset: str = tasks.DEFAULT_DATASET
    dataset_path: str | None = None
    dataset_format: str | None = None
    dataset_split: str | None = None
    dataset_revision: str | None = None
    dataset_name: str | None = None

    # Stage selection
    inference: int = 1
    compile: int = 1  # 0=skip, 1=parallel, 2=sequential
    eval: int = 1

    # Generation
    prompt_type: str = "kernelbench"
    num_samples: int = 1
    seed: int | None = None
    num_items: int | None = None
    max_tokens: int = 16384
    max_model_len: int = 32768
    max_mem_util: float = 0.90
    filter: bool = False
    use_online: int = 1
    online_service_url: str = DEFAULT_ONLINE_SERVICE_URL
    tensor_parallel_size: int = 1
    inference_script: str = ""

    # Serving backend used by the pause/resume hooks between stages
    serving_backend: str = "vllm"  # vllm | sglang | none

    # vLLM only frees GPU memory when the server was started with --enable-sleep-mode.
    # Without it the server cannot be paused, so the hooks must be skipped (they would
    # otherwise log an error on every stage boundary).
    enable_sleep: bool = False

    # Compilation
    num_procs: int = 4

    # Evaluation policy
    num_correct_trials: int = 5
    num_perf_trials: int = 3
    atol: float | None = None
    rtol: float | None = None
    flush_l2: bool = True
    eval_timeout: int = 1800

    # Address-space cap for a single eval child (GB).  0 disables the cap.
    # Stops one pathological kernel from exhausting the job's host memory.
    eval_mem_gb: float = 24.0

    # Orchestration
    reset_experiments: bool = False
    continue_on_error: bool = False
    log_level: str = "INFO"
    python_executable: str = field(default_factory=lambda: sys.executable)

    # ------------------------------------------------------------------
    def short_model_name(self) -> str:
        """Filesystem-safe, truncated model slug used in the experiment name."""
        return re.sub(r"[^a-zA-Z0-9]", "_", self.model_name.split("/")[-1])[:32]

    def resolved_dataset_format(self) -> str:
        return self.dataset_format or ("directory" if self.dataset_path else "hf")

    def dataset_source(self) -> tasks.TaskSource:
        source = tasks.TaskSource(
            name=self.dataset,
            label=self.dataset_name or (Path(self.dataset_path).name if self.dataset_path else None),
            format=self.resolved_dataset_format(),
            path=self.dataset_path,
            split=self.dataset_split,
            revision=self.dataset_revision,
        )
        return source.for_level(self.level)

    def is_default_dataset(self) -> bool:
        return (
            not self.dataset_path
            and self.resolved_dataset_format() == "hf"
            and self.dataset == tasks.DEFAULT_DATASET
            and self.dataset_split is None
            and self.dataset_revision is None
            and self.dataset_name is None
        )

    def signature(self) -> str:
        base = f"L{self.level}_{self.version}_{self.short_model_name()}_{self.parent_prompt_type}_API{self.use_online}"
        if not self.is_default_dataset():
            base += f"_{self.dataset_source().slug}"
        return base

    def resolved_experiment_name(self) -> str:
        return self.experiment_name or f"exp_local_{self.signature()}"

    def resolve_inference_script(self) -> str:
        """Which generation entry point to run.

        The wrapper mainly matters for the environment variables it sets before the serving
        stack is imported, so it follows ``--serving_backend``; ``--inference_script``
        overrides it entirely.
        """
        if self.inference_script:
            return self.inference_script
        return DEFAULT_INFERENCE_SCRIPTS.get(self.serving_backend, DEFAULT_INFERENCE_SCRIPTS["vllm"])

    def server_host_port(self) -> tuple[str, int]:
        """Parse ``online_service_url`` into ``(host, port)``."""
        parsed = urllib.parse.urlparse(self.online_service_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 30000)
        return host, port


# ============================================================================
#  Command building (pure, so it can be unit tested without a GPU)
# ============================================================================
def _dataset_command_args(config: PipelineConfig) -> list[str]:
    """Forward the dataset selection to a stage subprocess."""
    source = config.dataset_source()
    args = [
        "--dataset",
        config.dataset,
        "--dataset_format",
        config.resolved_dataset_format(),
        "--dataset_name",
        source.display_name,
    ]
    if source.split:
        args += ["--dataset_split", source.split]
    if config.dataset_path:
        args += ["--dataset_path", config.dataset_path]
    if config.dataset_revision:
        args += ["--dataset_revision", config.dataset_revision]
    return args


def build_inference_command(config: PipelineConfig, job: JobSpec, experiment_name: str) -> list[str]:
    script = config.resolve_inference_script()
    command = [
        config.python_executable,
        "-m",
        script,
        "--experiment_name",
        experiment_name,
        "--model_name",
        config.model_name,
        "--prompt_type",
        job.prompt_type,
        "--trial",
        str(job.new_trial),
        "--previous_trial",
        str(job.previous_trial),
        "--level",
        str(config.level),
        *_dataset_command_args(config),
        "--max_model_len",
        str(config.max_model_len),
        "--max_tokens",
        str(config.max_tokens),
        "--max_mem_util",
        str(config.max_mem_util),
        "--num_samples",
        str(config.num_samples),
        "--tensor_parallel_size",
        str(config.tensor_parallel_size),
        "--log-level",
        config.log_level,
    ]

    if config.seed is not None:
        command += ["--seed", str(config.seed)]

    if config.use_online == 1:
        command += ["--online_service_url", config.online_service_url]
        if config.serving_backend in DEFAULT_INFERENCE_SCRIPTS:
            command += ["--backend", config.serving_backend]
    if config.filter:
        command.append("--filter")
    if config.num_items is not None:
        command += ["--num_items", str(config.num_items)]

    return command


def build_compile_command(config: PipelineConfig, job: JobSpec, experiment_name: str) -> list[str]:
    if config.compile == 2:
        script = "robust_kernelbench.compile_all_sequential"
    elif config.compile == 1:
        script = "robust_kernelbench.compile_all"
    else:
        raise ValueError("build_compile_command called with compile stage disabled")

    return [
        config.python_executable,
        "-m",
        script,
        "--experiment_name",
        experiment_name,
        "--trial",
        str(job.new_trial),
        "--level",
        str(config.level),
        "--num_processes",
        str(config.num_procs),
        "--log-level",
        config.log_level,
    ]


def build_evaluate_command(config: PipelineConfig, job: JobSpec, experiment_name: str) -> list[str]:
    command = [
        config.python_executable,
        "-m",
        "robust_kernelbench.evaluate_all",
        "--experiment_name",
        experiment_name,
        "--trial",
        str(job.new_trial),
        "--level",
        str(config.level),
        *_dataset_command_args(config),
        "--num_correct_trials",
        str(config.num_correct_trials),
        "--num_perf_trials",
        str(config.num_perf_trials),
        "--timeout",
        str(config.eval_timeout),
        "--eval_mem_gb",
        str(config.eval_mem_gb),
        "--log-level",
        config.log_level,
        "--flush_l2" if config.flush_l2 else "--no-flush_l2",
    ]

    if config.atol is not None:
        command += ["--atol", str(config.atol)]
    if config.rtol is not None:
        command += ["--rtol", str(config.rtol)]

    return command


def build_clean_command(config: PipelineConfig, job: JobSpec, experiment_name: str) -> list[str]:
    return [
        config.python_executable,
        "-m",
        "robust_kernelbench.clean_cuda_cache",
        "--experiment",
        experiment_name,
        "--trial",
        str(job.new_trial),
        "--remove",
    ]


# ============================================================================
#  Serving-backend pause/resume
# ============================================================================
def _post_json(url: str, timeout: int = 60) -> dict | None:
    request = urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
        return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        logger.error("POST %s failed with HTTP %s: %s", url, e.code, body)
    except urllib.error.URLError as e:
        logger.error("POST %s failed: %s", url, e.reason)
    except json.JSONDecodeError:
        logger.warning("POST %s returned non-JSON body", url)
        return {}
    except Exception as e:  # noqa: BLE001 - never let server housekeeping abort a run
        logger.error("POST %s failed: %s", url, e)
    return None


SERVING_ENDPOINTS = {
    "sglang": {"pause": "/release_memory_occupation", "resume": "/resume_memory_occupation"},
    "vllm": {"pause": "/sleep?level=1", "resume": "/wake_up"},
}


def server_control(config: PipelineConfig, action: str) -> None:
    """Ask the serving backend to release or reclaim GPU memory.

    Only the selected backend is contacted, and ``--serving_backend none`` skips both.
    For vLLM the endpoints only exist when the server was launched with
    ``--enable-sleep-mode``, so they are skipped unless ``--enable_sleep`` is given.
    """
    if config.serving_backend == "none" or config.use_online != 1:
        logger.debug("skipping '%s': serving_backend=%s use_online=%s", action, config.serving_backend, config.use_online)
        return

    if config.serving_backend == "vllm" and not config.enable_sleep:
        logger.debug("skipping '%s': vLLM was not started with --enable-sleep-mode", action)
        return

    endpoints = SERVING_ENDPOINTS.get(config.serving_backend)
    if endpoints is None:
        logger.warning("unknown serving_backend %r; not contacting any server", config.serving_backend)
        return

    host, port = config.server_host_port()
    url = f"http://{host}:{port}{endpoints[action]}"
    logger.info("serving backend %s: %s -> %s", config.serving_backend, action, url)
    _post_json(url)


def pause_server(config: PipelineConfig) -> None:
    server_control(config, "pause")


def resume_server(config: PipelineConfig) -> None:
    server_control(config, "resume")


# ============================================================================
#  Stage execution
# ============================================================================
class StageError(RuntimeError):
    """Raised when a pipeline stage fails."""

    def __init__(self, stage: str, job: JobSpec | None = None, returncode: int | None = None, reason: str = ""):
        self.stage = stage
        self.job = job
        self.returncode = returncode
        message = f"stage {stage!r} failed"
        if job is not None:
            message += f" for job ({job})"
        if returncode is not None:
            message += f" with exit code {returncode}"
        if reason:
            message += f": {reason}"
        super().__init__(message)


def run(command: Sequence[str], description: str, job: JobSpec | None = None, timeout: int | None = None) -> None:
    """Run one stage, raising :class:`StageError` on failure."""
    printable = " ".join(str(part) for part in command)
    logger.info("[%s] %s", description, printable)

    try:
        completed = subprocess.run([str(part) for part in command], timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise StageError(description, job, reason=f"timed out after {timeout}s") from e
    except FileNotFoundError as e:
        raise StageError(description, job, reason=f"command not found: {e}") from e
    except Exception as e:  # noqa: BLE001 - reported by the caller with context
        raise StageError(description, job, reason=str(e)) from e

    if completed.returncode != 0:
        raise StageError(description, job, returncode=completed.returncode)


def count_generation_rows(experiment_name: str, trial: int) -> int:
    """Number of rows in ``experiments/<exp>/trial_<trial>/generations.csv`` (0 if absent)."""
    return artifacts.count_rows(
        artifacts.csv_path(experiment_name, trial, artifacts.GENERATIONS_CSV)
    )


def _verify_compile_complete(job: JobSpec, experiment_name: str) -> None:
    """Every generated kernel must have a recorded compile outcome.

    A kernel that fails to compile is a valid measurement; a kernel with **no
    row at all** means a worker crashed or the CSV append was lost, which would
    silently shrink every downstream denominator.
    """
    generated = artifacts.generated_kernel_keys(experiment_name, job.new_trial)
    recorded = artifacts.compilation_keys(experiment_name, job.new_trial)
    missing = generated - recorded
    if missing:
        raise StageError(
            "compile",
            job,
            reason=(
                f"{len(missing)} generated kernel(s) have no compilation row: "
                f"{artifacts.format_keys(missing)}"
            ),
        )
    logger.info("compile stage recorded outcomes for all %s generated kernel(s)", len(generated))


def _verify_evaluation_complete(job: JobSpec, experiment_name: str) -> None:
    """Every successfully compiled kernel must have an evaluation outcome.

    Failed compiles are not expected in evaluations.csv.  A successful compile
    that vanishes from evaluations.csv is a lost measurement, not a model error.
    """
    compiled = artifacts.compilation_keys(experiment_name, job.new_trial, successful_only=True)
    evaluated = artifacts.evaluation_keys(experiment_name, job.new_trial)
    missing = compiled - evaluated
    if missing:
        raise StageError(
            "evaluate",
            job,
            reason=(
                f"{len(missing)} successfully compiled kernel(s) have no evaluation row: "
                f"{artifacts.format_keys(missing)}"
            ),
        )
    logger.info("evaluation stage recorded outcomes for all %s compiled kernel(s)", len(compiled))


def run_job(config: PipelineConfig, job: JobSpec, experiment_name: str) -> None:
    """Run all enabled stages for one job.  Raises :class:`StageError` on the first failure."""
    # ---- 1. Inference
    if config.inference == 1:
        try:
            resume_server(config)
            run(build_inference_command(config, job, experiment_name), "inference", job=job)
        finally:
            pause_server(config)

        # The inference entry point used to exit 0 even when every request failed (e.g. the
        # model server had crashed), which let the pipeline report "jobs succeeded" while
        # producing an empty generations.csv and no evaluations at all.
        produced = count_generation_rows(experiment_name, job.new_trial)
        generated = artifacts.generated_kernel_keys(experiment_name, job.new_trial)
        if produced == 0 or not generated:
            raise StageError(
                "inference",
                job,
                reason=(
                    "no generations were written to "
                    f"experiments/{experiment_name}/trial_{job.new_trial}/generations.csv; "
                    f"the model server at {config.online_service_url} is probably down or crashed "
                    "(check its log for APIConnectionError / startup errors)"
                ),
            )
        logger.info("inference produced %s generation row(s) and %s kernel file(s)", produced, len(generated))

    # ---- 2. Compilation
    if config.compile in (1, 2):
        try:
            run(build_clean_command(config, job, experiment_name), "clean-cuda-cache (pre-compile)", job=job)
        except StageError as e:
            logger.warning("pre-compile cache clean-up failed (continuing): %s", e)

        run(build_compile_command(config, job, experiment_name), "compile", job=job)
        _verify_compile_complete(job, experiment_name)

    # ---- 3. Evaluation
    if config.eval == 1:
        try:
            run(build_clean_command(config, job, experiment_name), "clean-cuda-cache (pre-eval)", job=job)
        except StageError as e:
            logger.warning("pre-eval cache clean-up failed (continuing): %s", e)

        run(build_evaluate_command(config, job, experiment_name), "evaluate", job=job)
        _verify_evaluation_complete(job, experiment_name)


# ============================================================================
#  Job parsing / experiment reset
# ============================================================================
def parse_jobs(jobs: Sequence[str]) -> list[JobSpec]:
    """Parse ``"<previous_trial> <new_trial> <prompt_type>"`` strings.

    Raises ``ValueError`` with a helpful message on malformed input (previously this
    exploded with an unpacking error).
    """
    parsed: list[JobSpec] = []
    for raw in jobs:
        parts = str(raw).split()
        if len(parts) != 3:
            raise ValueError(
                f"malformed job {raw!r}: expected '<previous_trial> <new_trial> <prompt_type>' "
                f"(3 whitespace-separated fields), got {len(parts)}"
            )
        previous_trial, new_trial, prompt_type = parts
        try:
            previous_trial_int, new_trial_int = int(previous_trial), int(new_trial)
        except ValueError as e:
            raise ValueError(f"malformed job {raw!r}: trial numbers must be integers") from e
        if new_trial_int < 1:
            raise ValueError(f"malformed job {raw!r}: new trial must be >= 1")
        parsed.append(JobSpec(previous_trial_int, new_trial_int, prompt_type))
    return parsed


def reset_experiment(experiment_name: str) -> None:
    """Delete ``./experiments/<experiment_name>`` if it exists."""
    target_dir = os.path.join(".", "experiments", experiment_name)
    if os.path.isdir(target_dir):
        logger.warning("--reset_experiments: removing %s", target_dir)
        shutil.rmtree(target_dir)


# ============================================================================
#  CLI
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the robust_kernelbench pipeline (inference -> compile -> evaluate).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment settings
    parser.add_argument("--previous_trial", type=int, default=1, help="Previous trial number")
    parser.add_argument("--new_trial", type=int, default=1, help="New trial number")
    parser.add_argument("--prompt_type", type=str, default="kernelbench", help="Type of prompt")
    parser.add_argument("--experiment_name", type=str, default="", help="Experiment name (auto-generated if empty)")
    parser.add_argument("--model_name", "--model", dest="model_name", type=str, default="default", help="Model name")
    parser.add_argument("--parent_prompt_type", type=str, default="kernelbench", help="Parent prompt type")

    # Pipeline stages
    parser.add_argument("--inference", type=int, default=1, choices=[0, 1], help="Run inference (1) or skip (0)")
    parser.add_argument(
        "--compile", type=int, default=1, choices=[0, 1, 2], help="0=skip, 1=parallel, 2=sequential"
    )
    parser.add_argument("--eval", type=int, default=1, choices=[0, 1], help="Run evaluation (1) or skip (0)")

    # Configuration
    parser.add_argument("--level", type=int, default=1, help="KernelBench level")
    tasks.add_dataset_args(parser)
    parser.add_argument("--version", type=str, default="V8", help="Version string")
    parser.add_argument("--use_online", type=int, default=1, choices=[0, 1], help="Use online service (1) or offline (0)")
    parser.add_argument("--num_items", type=int, default=None, help="Limit number of items (None for all)")
    parser.add_argument("--max_mem_util", type=float, default=0.90, help="Maximum memory utilization")
    parser.add_argument("--max_model_len", type=int, default=32768, help="Maximum model length")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Maximum tokens")
    parser.add_argument("--filter", action="store_true", help="Enable the representative-subset filter")
    parser.add_argument("--num_samples", type=positive_int, default=1, help="Number of samples per problem")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed (None = engine default)")
    parser.add_argument("--num_procs", type=int, default=4, help="Processes for parallel compilation")
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="Tensor-parallel size for the local engine"
    )
    parser.add_argument(
        "--online_service_url",
        type=str,
        default=DEFAULT_ONLINE_SERVICE_URL,
        help="Base URL of the OpenAI-compatible serving endpoint",
    )
    parser.add_argument(
        "--serving_backend",
        type=str,
        default="vllm",
        choices=["vllm", "sglang", "none"],
        help="Which serving backend to pause/resume around stages ('none' to skip).",
    )
    parser.add_argument(
        "--enable_sleep",
        action="store_true",
        default=False,
        help="The vLLM server was started with --enable-sleep-mode, so pause/resume may be used.",
    )
    parser.add_argument(
        "--inference_script",
        type=str,
        default="",
        help="Override the generation entry point (default: derived from --serving_backend/--use_online).",
    )

    # Evaluation policy
    parser.add_argument(
        "--num_correct_trials",
        type=positive_int,
        default=5,
        help="Independently seeded correctness trials per kernel; all must match.",
    )
    parser.add_argument(
        "--num_perf_trials", type=positive_int, default=3, help="Performance trials per kernel"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Absolute tolerance override (default: chosen from the output dtype).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Relative tolerance override (default: chosen from the output dtype).",
    )
    parser.add_argument(
        "--flush_l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evict L2 between timing trials (cache-cold measurements).",
    )
    parser.add_argument(
        "--eval_timeout",
        type=positive_int,
        default=1800,
        help="Per-kernel evaluation timeout (s); generous so slow-but-correct kernels are not cut off.",
    )
    parser.add_argument(
        "--eval_mem_gb",
        type=non_negative_float,
        default=24.0,
        help="Address-space cap for a single eval child in GB (0 disables).",
    )

    # Orchestration
    parser.add_argument("--reset_experiments", action="store_true", default=False, help="Delete the experiment dir")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        default=False,
        help="Keep running the remaining jobs when one fails (exit code still reflects the failure).",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG/INFO/WARNING/...)")

    parser.add_argument("jobs", nargs="*", help="Job strings: '<previous_trial> <new_trial> <prompt_type>'")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Build an immutable :class:`PipelineConfig` from parsed CLI arguments."""
    return PipelineConfig(
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        version=args.version,
        parent_prompt_type=args.parent_prompt_type,
        level=args.level,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        dataset_format=args.dataset_format,
        dataset_split=args.dataset_split,
        dataset_revision=args.dataset_revision,
        dataset_name=args.dataset_name,
        inference=args.inference,
        compile=args.compile,
        eval=args.eval,
        prompt_type=args.prompt_type,
        num_samples=args.num_samples,
        seed=args.seed,
        num_items=args.num_items,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        max_mem_util=args.max_mem_util,
        filter=args.filter,
        use_online=args.use_online,
        online_service_url=args.online_service_url,
        tensor_parallel_size=args.tensor_parallel_size,
        inference_script=args.inference_script,
        serving_backend=args.serving_backend,
        enable_sleep=args.enable_sleep,
        num_procs=args.num_procs,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        atol=args.atol,
        rtol=args.rtol,
        flush_l2=args.flush_l2,
        eval_timeout=args.eval_timeout,
        eval_mem_gb=args.eval_mem_gb,
        reset_experiments=args.reset_experiments,
        continue_on_error=args.continue_on_error,
        log_level=args.log_level,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.  Never called at import time."""
    return build_parser().parse_args(argv)


# ============================================================================
#  Entry point
# ============================================================================
def main(argv: Sequence[str] | None = None, config: PipelineConfig | None = None) -> int:
    """Run the pipeline.  Returns a process exit code (0 = success)."""
    if config is None:
        args = parse_args(argv)
        config = config_from_args(args)
        raw_jobs = args.jobs
    else:
        raw_jobs = list(argv or [])

    setup_logging(config.log_level, force=True)

    if config.num_correct_trials < 1 or config.num_perf_trials < 1:
        logger.error("--num_correct_trials and --num_perf_trials must both be >= 1")
        return EXIT_USAGE
    if config.eval_mem_gb < 0:
        logger.error("--eval_mem_gb must be >= 0")
        return EXIT_USAGE

    try:
        jobs = parse_jobs(raw_jobs)
    except ValueError as e:
        logger.error("%s", e)
        return EXIT_USAGE

    if not jobs:
        logger.error(
            "no jobs supplied. Pass one or more '<previous_trial> <new_trial> <prompt_type>' strings, "
            "e.g. '1 1 kernelbench'. Use --help for the full option list."
        )
        return EXIT_USAGE

    experiment_name = config.resolved_experiment_name()
    logger.info("experiment=%s signature=%s", experiment_name, config.signature())

    if config.reset_experiments:
        reset_experiment(experiment_name)

    failures: list[tuple[JobSpec, str]] = []
    total = len(jobs)

    for index, job in enumerate(jobs, start=1):
        logger.info("=== job %s/%s: %s ===", index, total, job)
        try:
            run_job(config, job, experiment_name)
        except StageError as e:
            logger.error("%s", e)
            logger.debug("traceback:\n%s", traceback.format_exc())
            failures.append((job, str(e)))
            if not config.continue_on_error:
                logger.error("aborting: use --continue_on_error to run the remaining jobs")
                break
        except Exception as e:  # noqa: BLE001 - report, then honour --continue_on_error
            logger.error("unexpected error for job (%s): %s", job, e)
            logger.debug("traceback:\n%s", traceback.format_exc())
            failures.append((job, f"unexpected error: {e}"))
            if not config.continue_on_error:
                break

    succeeded = total - len(failures)
    if failures:
        logger.error("%s/%s jobs failed:", len(failures), total)
        for job, reason in failures:
            logger.error("  - (%s): %s", job, reason)
        return EXIT_FAILURE

    logger.info("finished: %s/%s jobs succeeded", succeeded, total)
    return EXIT_OK


def cli() -> None:
    """Console-script entry point (``robust-kernelbench``)."""
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
