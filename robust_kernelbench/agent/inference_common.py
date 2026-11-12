"""Shared inference (kernel-generation) logic.

``run_inference_test_time_scaling.py`` (vLLM) and ``run_inference_test_time_scaling_v2.py``
(SGLang) were ~95% identical: same dataset handling, same prompt writing, same CSV bookkeeping,
same token counting.  The only real differences were the engine construction and the shape of
the ``generate`` call.  That difference is now isolated behind two small backend classes here,
and the two entry-point scripts are thin wrappers that pick one.

The backend classes import their engine lazily, so this module can be imported (and unit
tested) without vLLM or SGLang installed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

from robust_kernelbench import artifacts
from robust_kernelbench.logging_utils import get_logger, setup_logging
from robust_kernelbench.agent.prompt_common import choose_prompt
from robust_kernelbench.utils.utils import get_file_name, get_folder_path
from robust_kernelbench.utils.utils_inference import create_joined_dataset, extract_code

# Generated kernel sources land in CSV cells; the 128 KiB default is too small.
csv.field_size_limit(sys.maxsize)

logger = get_logger(__name__)

GenerationResult = NamedTuple(
    "GenerationResult",
    [("generations", list[str]), ("in_tokens", int), ("out_tokens_raw", int | None)],
)


# ============================================================================
#  Backends
# ============================================================================
class LocalBackend:
    """Base class for a locally hosted engine.  Subclasses implement ``generate``."""

    name = "local"

    def __init__(self, engine, tokenizer, model_name=None, lora_name=None):
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.lora_name = lora_name

    def generate(self, messages, temperature=0.6, max_tokens=1000, num_samples=1, seed=None) -> GenerationResult:
        raise NotImplementedError

    def _count_prompt_tokens(self, messages) -> int:
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return len(token_ids)


class VLLMBackend(LocalBackend):
    """In-process vLLM engine (``vllm.LLM``)."""

    name = "vllm"

    def __init__(self, engine, tokenizer, model_name=None, lora_name=None):
        super().__init__(engine, tokenizer, model_name=model_name, lora_name=lora_name)
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        self._SamplingParams = SamplingParams

        self.lora_request = None
        if lora_name:
            try:
                self.lora_request = LoRARequest("vllm_lora_request", 1, lora_name)
            except Exception as e:  # noqa: BLE001 - LoRA is optional
                logger.warning("could not build LoRA request: %s", e)
                self.lora_request = None

    def generate(self, messages, temperature=0.6, max_tokens=1000, num_samples=1, seed=None) -> GenerationResult:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        in_tokens = self._count_prompt_tokens(messages)

        sampling_params = self._SamplingParams(
            n=num_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

        outputs = self.engine.generate(
            [prompt], sampling_params=sampling_params, lora_request=self.lora_request, use_tqdm=False
        )

        # vLLM returns one RequestOutput per prompt, each holding ``n`` completions.
        generations = [
            completion.text.strip()
            for request_output in outputs
            for completion in request_output.outputs
        ]
        return GenerationResult(generations=generations, in_tokens=in_tokens, out_tokens_raw=None)


class SGLangBackend(LocalBackend):
    """In-process SGLang engine (``sglang.Engine``)."""

    name = "sglang"

    def generate(self, messages, temperature=0.6, max_tokens=1000, num_samples=1, seed=None) -> GenerationResult:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        in_tokens = self._count_prompt_tokens(messages)

        sampling_params = {
            "n": num_samples,
            "temperature": temperature,
            "max_new_tokens": max_tokens,
        }
        if seed is not None:
            sampling_params["seed"] = seed

        outputs = self.engine.generate([prompt], sampling_params=sampling_params, lora_path=self.lora_name)

        generations = [output["text"].strip() for output in outputs]
        return GenerationResult(generations=generations, in_tokens=in_tokens, out_tokens_raw=None)


class APIBackend:
    """OpenAI-compatible remote endpoint (vLLM server, SGLang server, OpenRouter, ...)."""

    name = "api"

    def __init__(self, client, model_name, use_tts_service=False, tts_args=None):
        self.client = client
        self.model_name = model_name
        self.use_tts_service = use_tts_service
        self.tts_args = tts_args or {}

    def _extra_body(self) -> dict:
        if not self.use_tts_service:
            return {}
        args = self.tts_args
        return {
            "tts_strategy": args.get("tts_strategy", "offline_bon"),
            "tts_num_trajectories": args.get("tts_num_trajectories", 8),
            "tts_scorer": args.get("tts_scorer", "entropy"),
            "tts_candidates_per_step": args.get("tts_candidates_per_step", 4),
            "tts_beam_size": args.get("tts_beam_size", 4),
            "tts_max_steps": args.get("tts_max_steps", 100),
            "tts_score_aggregation": args.get("tts_score_aggregation", "min"),
            "tts_window_size": args.get("tts_window_size", None),
        }

    def generate(self, messages, temperature=0.6, max_tokens=1000, num_samples=1, seed=None) -> GenerationResult:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=num_samples,
            seed=seed,
            extra_body=self._extra_body(),
        )
        generations = [(choice.message.content or "").strip() for choice in response.choices]
        in_tokens = response.usage.prompt_tokens if response.usage else 0
        out_tokens = response.usage.completion_tokens if response.usage else 0
        return GenerationResult(generations=generations, in_tokens=in_tokens, out_tokens_raw=out_tokens)


def build_local_backend(
    backend_kind: str,
    model_name: str,
    tokenizer_name: str | None = None,
    gpu_memory_util: float = 0.9,
    max_model_len: int = 32000,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    lora_name: str | None = None,
    enable_lora: bool = True,
    max_lora_rank: int = 64,
) -> LocalBackend:
    """Construct the requested local engine.  Imports the engine lazily (CUDA init)."""
    if not tokenizer_name:
        tokenizer_name = model_name

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    if backend_kind == "vllm":
        from vllm import LLM

        logger.info("setting up vLLM engine for %s (tp=%s)", model_name, tensor_parallel_size)
        engine = LLM(
            model=model_name,
            tokenizer=tokenizer_name,
            trust_remote_code=True,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_util,
            max_model_len=max_model_len,
            enable_lora=enable_lora,
            max_lora_rank=max_lora_rank,
            tensor_parallel_size=tensor_parallel_size,
            disable_custom_all_reduce=True,
        )
        return VLLMBackend(engine, tokenizer, model_name=model_name, lora_name=lora_name)

    if backend_kind == "sglang":
        import sglang as sgl

        logger.info("setting up SGLang engine for %s (tp=%s)", model_name, tensor_parallel_size)
        engine = sgl.Engine(
            model_path=model_name,
            tokenizer_path=tokenizer_name,
            trust_remote_code=True,
            dtype=dtype,
            mem_fraction_static=gpu_memory_util,
            context_length=max_model_len,
            tp_size=tensor_parallel_size,
        )
        return SGLangBackend(engine, tokenizer, model_name=model_name, lora_name=lora_name)

    raise ValueError(f"Unknown local backend {backend_kind!r} (expected 'vllm' or 'sglang')")


# Backwards-compatible alias: the old modules exposed `load_model_and_tokenizer`.
def load_model_and_tokenizer(backend_kind="vllm", **kwargs):
    backend = build_local_backend(backend_kind, **kwargs)
    return backend.engine, backend.tokenizer


# ============================================================================
#  File bookkeeping
# ============================================================================
def write_metadata_json(experiment_name, trial=1, metadata=None):
    """Writes metadata to a json file in experiment folder"""
    folder_path = get_folder_path(experiment_name, trial)
    metadata_file = os.path.join(folder_path, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata or {}, f, indent=4)


def sample_file_exists(experiment_name, trial, problem_id, sample_id=None):
    """Whether any generated code file already exists for this (problem_id, sample_id).

    Used to make inference resumable.  ``write_file_with_idx`` always starts at index 0,
    so checking index 0 is equivalent to the old (buggy) loop that returned on the first
    iteration anyway.
    """
    folder_path_code = get_folder_path(experiment_name, trial, postfix="code")
    file_path = os.path.join(folder_path_code, get_file_name(problem_id, sample_id)) + "0.py"
    return os.path.exists(file_path)


def find_file_idx(file_path_base, ending=".py", starting_idx=0):
    """Finds the first file index that does not exist yet."""
    idx = starting_idx
    file_path = file_path_base + str(idx) + ending

    while os.path.exists(file_path):
        logger.debug("found existing file %s at idx=%s, increasing idx", file_path, idx)
        idx += 1
        file_path = file_path_base + str(idx) + ending

    return file_path, idx


def write_file_with_idx(
    problem_name,
    problem_id,
    generations,
    experiment_name=None,
    extract_code_flag=True,
    trial=None,
    model_name=None,
    suffix=None,
    ending=".py",
):
    """Write generated code files, returning the last index used."""
    file_name = get_file_name(problem_id, suffix=suffix)

    if not experiment_name:
        experiment_name = "temp_solutions"

    folder_path_code = get_folder_path(experiment_name, trial, postfix="code")
    file_path_base = os.path.join(folder_path_code, file_name)

    idx = 0
    logger.info("saving %s generation(s) under %s", len(generations), file_path_base)
    for generation in generations:
        file_path, idx = find_file_idx(file_path_base, ending, starting_idx=idx)

        if extract_code_flag:
            try:
                extracted = extract_code(generation, model_name=model_name)
                if extracted:
                    generation = extracted
            except Exception as e:  # noqa: BLE001 - extraction is heuristic
                logger.warning("code extraction failed: %s", e)

        if generation:
            with open(file_path, "w") as file:
                file.write(generation)
            logger.debug("wrote %s", file_path)
        else:
            logger.warning("empty generation for pid=%s; skipping write", problem_id)

    return idx


def count_output_tokens(tokenizer, generations):
    """Counts tokens based on generations"""
    out_tokens = {
        "num_tokens": [],
        "mean": None,
        "median": None,
        "stdev": None,
        "max": None,
        "min": None,
    }
    for generated_text in generations:
        out_token_ids = tokenizer(generated_text)
        out_tokens["num_tokens"].append(len(out_token_ids["input_ids"]))

    if not out_tokens["num_tokens"]:
        return out_tokens

    out_tokens["mean"] = statistics.mean(out_tokens["num_tokens"])
    out_tokens["median"] = statistics.median(out_tokens["num_tokens"])
    try:
        out_tokens["stdev"] = statistics.stdev(out_tokens["num_tokens"])
    except statistics.StatisticsError:
        pass

    out_tokens["max"] = max(out_tokens["num_tokens"])
    out_tokens["min"] = min(out_tokens["num_tokens"])

    return out_tokens


# ============================================================================
#  The generation pipeline
# ============================================================================
def run_pipeline(
    dataset,
    tokenizer=None,
    engine=None,
    backend=None,
    run_clean_trial=False,
    num_items=None,
    lora_name=None,
    temperature=0.6,
    max_tokens=1000,
    num_samples=1,
    seed=None,
    experiment_name=None,
    prompt_type="kernelbench",
    trial=None,
    online_service_url=None,
    online_api_key="<NO_KEY>",
    use_tts_service=False,
    tts_args=None,
    model_name=None,
    **kwargs,
):
    """Generate one kernel per dataset row and persist prompt/code/CSV artefacts.

    ``backend`` is any object with a ``generate(messages, temperature, max_tokens,
    num_samples) -> GenerationResult`` method (see the classes above).  When omitted, one
    is built from ``online_service_url`` (remote API) or from ``engine`` (assumed vLLM,
    for backwards compatibility).
    """
    logger.info(
        "starting inference: %s rows, trial=%s, prompt_type=%s, experiment=%s, samples=%s",
        len(dataset),
        trial,
        prompt_type,
        experiment_name,
        num_samples,
    )
    if len(dataset) <= 0:
        return None

    if not experiment_name:
        experiment_name = "temp_solutions"

    tokenizer = tokenizer if tokenizer is not None else getattr(backend, "tokenizer", None)

    if backend is None:
        if online_service_url:
            from openai import OpenAI

            client = OpenAI(base_url=online_service_url, api_key=online_api_key, timeout=3600)
            backend = APIBackend(client, model_name, use_tts_service=use_tts_service, tts_args=tts_args)
        elif engine is not None:
            backend = VLLMBackend(engine, tokenizer, model_name=model_name, lora_name=lora_name)
        else:
            raise ValueError("run_pipeline needs either `backend`, `engine` or `online_service_url`")

    results = {
        "problem_name": [],
        "problem_id": [],
        "in_tokens": [],
        "out_tokens": [],
        "temperature": [],
        "time_taken": [],
        "generations": [],
        "file_index": [],
        "prompt_category": [],
    }

    dataset = dataset.select(range(num_items)) if num_items else dataset

    from tqdm import tqdm

    pbar = tqdm(dataset)
    failures = 0
    for sample in pbar:
        problem_name = sample.get("name", "<unnamed>")
        if not problem_name.endswith("_"):
            problem_name = problem_name + "_"
        problem_id = sample["problem_id"]
        sample_id = sample.get("sample_id", None)

        try:
            pbar.set_description(f"[inference] p={problem_id} s={sample_id}")

            # Resumability: skip rows whose code file already exists.
            if sample_file_exists(experiment_name, trial, problem_id, sample_id):
                logger.info("skipping pid=%s sid=%s (code file already exists)", problem_id, sample_id)
                continue

            prompt_func, prompt_category = choose_prompt(sample, trial, prompt_type, run_clean_trial)

            try:
                prompt = prompt_func(sample, **kwargs)
            except Exception:
                logger.error("prompt generation failed for pid=%s sid=%s", problem_id, sample_id)
                raise

            if not prompt:
                logger.warning("empty prompt for pid=%s sid=%s", problem_id, sample_id)

            write_file_with_idx(
                problem_name,
                problem_id,
                [prompt],
                experiment_name=experiment_name,
                extract_code_flag=False,
                trial=trial,
                model_name=model_name,
                suffix="prompt",
                ending=".txt",
            )

            messages = [{"content": prompt, "role": "user"}]

            start = time.time()
            generation_result = backend.generate(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                num_samples=num_samples,
                seed=seed,
            )
            end = time.time()

            generations = generation_result.generations
            in_num_tokens = generation_result.in_tokens
            logger.info(
                "generated %s sample(s) for pid=%s sid=%s in %.1fs",
                len(generations),
                problem_id,
                sample_id,
                end - start,
            )

            file_idx = write_file_with_idx(
                problem_name,
                problem_id,
                generations,
                experiment_name=experiment_name,
                extract_code_flag=True,
                trial=trial,
                model_name=model_name,
            )

            if generation_result.out_tokens_raw is not None:
                raw = generation_result.out_tokens_raw
                out_num_tokens = {
                    "num_tokens": [raw],
                    "mean": raw,
                    "median": raw,
                    "stdev": None,
                    "max": raw,
                    "min": raw,
                }
            else:
                out_num_tokens = count_output_tokens(tokenizer, generations)

            results["problem_name"].append(problem_name)
            results["problem_id"].append(problem_id)
            results["generations"].append(generations)
            results["in_tokens"].append(in_num_tokens)
            results["out_tokens"].append(out_num_tokens)
            results["temperature"].append(temperature)
            results["time_taken"].append(end - start)
            results["file_index"].append(file_idx)
            results["prompt_category"].append(prompt_category)
        except Exception as e:  # noqa: BLE001 - one bad row must not stop the run
            failures += 1
            logger.error(
                "inference failed for pid=%s sid=%s: %s", problem_id, sample_id, e, exc_info=True
            )

    if failures:
        logger.error("%s problem(s) failed during inference", failures)

    import pandas as pd

    return pd.DataFrame.from_dict(results)


# ============================================================================
#  CLI plumbing shared by both wrappers
# ============================================================================
def build_arg_parser(description: str, backend_default: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--num_samples", type=int, default=1, help="Num of Parallel responses")
    parser.add_argument("--prompt_type", type=str, default="kernelbench", help="Which prompt to use...")
    parser.add_argument("--trial", type=int, required=True, help="Which trial you are currently running...")
    parser.add_argument("--previous_trial", type=int, help="Which previous trial to use...")
    parser.add_argument(
        "--run_clean_trial", action="store_true", help="Ignore previous trials (behave like trial 1)"
    )

    parser.add_argument("--experiment_name", type=str, default="exp_test_run", help="experiment names")
    parser.add_argument("--temperature", type=float, default=0.6, help="temp for LLM")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed (None = engine default).")
    parser.add_argument("--max_mem_util", type=float, default=0.8, help="GPU memory fraction for the engine")
    parser.add_argument("--num_items", type=int, help="How many data points to run...")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Model Name")
    parser.add_argument("--level", type=int, default=1, help="Which level to run...")
    parser.add_argument("--max_model_len", type=int, default=32000, help="Max input and output tokens...")
    parser.add_argument("--max_tokens", type=int, default=8000, help="max tokens to generate")
    parser.add_argument("--tokenizer_name", type=str, help="tokenizer name if different from model name.")
    parser.add_argument("--filter", action="store_true", help="Whether to run only on the filtered examples...")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="TP size for multi-gpu, single node...")
    parser.add_argument(
        "--backend",
        type=str,
        default=backend_default,
        choices=["vllm", "sglang"],
        help="Local engine implementation to use.",
    )
    # TODO: add lora support

    # TTS service args
    parser.add_argument(
        "--online_service_url",
        type=str,
        default=None,
        help="URL of API LLM, e.g. http://localhost:8001/v1 or https://openrouter.ai/api/v1",
    )
    parser.add_argument(
        "--api_key_env_var",
        type=str,
        default="LLM_API_KEY",
        help="Env variable holding the LLM API key (default: LLM_API_KEY)",
    )
    parser.add_argument(
        "--use_tts_service",
        action="store_true",
        help="Whether to run the TTS service or just the online LLM endpoint.",
    )

    # BOOSTER LLM IMPLEMENTATION (Citation Later)
    parser.add_argument("--tts_strategy", type=str, default="offline_bon")
    parser.add_argument("--tts_num_trajectories", type=int, default=8)
    parser.add_argument("--tts_scorer", type=str, default="entropy")
    parser.add_argument("--tts_candidates_per_step", type=int, default=4)
    parser.add_argument("--tts_beam_size", type=int, default=4)
    parser.add_argument("--tts_max_steps", type=int, default=100)
    parser.add_argument("--tts_score_aggregation", type=str, default="min")
    parser.add_argument("--tts_window_size", type=int, default=5)
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG/INFO/WARNING/...).")

    return parser


def load_dataset_for_level(level: int, use_filter: bool = False):
    """Load one KernelBench level from the HuggingFace dataset."""
    from datasets import load_dataset

    data_repo = "ai-nikolai/KernelBench"
    dataset = load_dataset(data_repo)
    dataset = dataset[f"level_{level}"]

    if use_filter:
        # KernelBench's "representative subset" for level 1.
        from robust_kernelbench.evaluate_single import level1_representative_subset_problem_ids

        logger.info("filtering to the representative subset")
        dataset = dataset.filter(lambda x: x["problem_id"] in level1_representative_subset_problem_ids)

    logger.info("dataset:\n%s", dataset)
    return dataset


def attach_previous_trial(dataset, experiment_name, trial, previous_trial=None, run_clean_trial=False):
    """Join the previous round's artefacts onto the dataset (no-op on round 1)."""
    import pandas as pd

    if run_clean_trial or trial <= 1:
        return dataset

    if not previous_trial:
        previous_trial = trial - 1

    previous_folder_path = get_folder_path(experiment_name, previous_trial)
    generations_path = os.path.join(previous_folder_path, "generations.csv")
    previous_generations = pd.read_csv(generations_path)

    def _try_read(name):
        path = os.path.join(previous_folder_path, name)
        try:
            return pd.read_csv(path)
        except Exception as e:  # noqa: BLE001 - previous round may be partial
            logger.warning("could not read previous %s: %s", name, e)
            return None

    return create_joined_dataset(
        dataset,
        previous_generations=previous_generations,
        previous_compilations=_try_read("compilations.csv"),
        previous_evaluations=_try_read("evaluations.csv"),
    )


def run_main(argv=None, backend_kind="vllm", description="Kernel generation"):
    """Entry point shared by both inference wrappers."""
    parser = build_arg_parser(description, backend_default=backend_kind)
    args = parser.parse_args(argv)
    setup_logging(args.log_level, force=True)

    # `--backend` wins over the wrapper default so one wrapper can drive either engine.
    backend_kind = getattr(args, "backend", backend_kind) or backend_kind

    dataset = load_dataset_for_level(args.level, use_filter=args.filter)
    dataset = attach_previous_trial(
        dataset, args.experiment_name, args.trial, previous_trial=args.previous_trial, run_clean_trial=args.run_clean_trial
    )

    tts_args_dict = None
    online_api_key = None
    backend = None

    if args.online_service_url:
        online_api_key = os.environ.get(args.api_key_env_var, "<NO_KEY>")
        from openai import OpenAI

        client = OpenAI(base_url=args.online_service_url, api_key=online_api_key, timeout=3600)
        backend = APIBackend(client, args.model_name, use_tts_service=args.use_tts_service, tts_args=None)

        if args.use_tts_service:
            tts_args_dict = {
                "model_name": args.model_name,
                "tts_strategy": args.tts_strategy,
                "tts_num_trajectories": args.tts_num_trajectories,
                "tts_scorer": args.tts_scorer,
                "tts_candidates_per_step": args.tts_candidates_per_step,
                "tts_beam_size": args.tts_beam_size,
                "tts_max_steps": args.tts_max_steps,
                "tts_score_aggregation": args.tts_score_aggregation,
                "tts_window_size": args.tts_window_size,
            }
            backend.tts_args = tts_args_dict
    else:
        backend = build_local_backend(
            backend_kind,
            model_name=args.model_name,
            tokenizer_name=args.tokenizer_name,
            gpu_memory_util=args.max_mem_util,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
        )

    logger.info("all init finished; running the pipeline")

    import pandas as pd

    results_df = run_pipeline(
        dataset,
        tokenizer=getattr(backend, "tokenizer", None),
        engine=getattr(backend, "engine", None),
        backend=backend,
        num_items=args.num_items,
        run_clean_trial=args.run_clean_trial,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_samples=args.num_samples,
        seed=args.seed,
        experiment_name=args.experiment_name,
        prompt_type=args.prompt_type,
        trial=args.trial,
        online_service_url=args.online_service_url,
        online_api_key=online_api_key,
        tts_args=tts_args_dict,
        use_tts_service=args.use_tts_service,
        model_name=args.model_name,
    )

    if results_df is None or results_df.empty:
        logger.error(
            "no generations were produced (0 rows); the serving endpoint %r did not answer. "
            "Check the model-server log for startup errors or APIConnectionError.",
            args.online_service_url or "<local engine>",
        )
        return 1

    logger.info("results head:\n%s", results_df.head())
    Path("out").mkdir(parents=True, exist_ok=True)
    results_df.to_csv(
        f'out/generations_{args.experiment_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False
    )

    generations_csv_path = artifacts.csv_path(
        args.experiment_name, args.trial, artifacts.GENERATIONS_CSV
    )
    artifacts.append_rows(results_df, generations_csv_path)

    metadata = {
        "model_name": args.model_name,
        "seed": args.seed,
        "git_commit": _git_commit(),
        "torch_version": _module_version("torch"),
        "python_version": sys.version.split()[0],
        "backend": backend_kind,
        "max_model_len": args.max_model_len,
        "max_mem_util": args.max_mem_util,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "num_samples": args.num_samples,
        "prompt_type": args.prompt_type,
        "trial": args.trial,
        "num_items": args.num_items,
        "level": args.level,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    if args.online_service_url:
        metadata["online_service_url"] = args.online_service_url
        metadata["tts_strategy"] = args.tts_strategy
        metadata["tts_num_trajectories"] = args.tts_num_trajectories
        metadata["tts_scorer"] = args.tts_scorer
        metadata["use_tts_service"] = args.use_tts_service

    write_metadata_json(args.experiment_name, args.trial, metadata)

    _teardown(backend)
    logger.info("inference finished")
    return 0


def _git_commit() -> str | None:
    """Best-effort commit hash for reproducibility (None outside a checkout)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - optional metadata
        return None


def _module_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 - package may not be installed
        return None


def _teardown(backend) -> None:
    """Release the engine and empty the CUDA cache before the compile/eval stages."""
    try:
        if backend is not None:
            backend.engine = None
            del backend
    except Exception as e:  # noqa: BLE001
        logger.warning("could not release the engine: %s", e)

    try:
        gc.collect()
    except Exception as e:  # noqa: BLE001
        logger.warning("garbage collection failed: %s", e)

    try:
        import torch

        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        logger.warning("torch.cuda.empty_cache() failed: %s", e)
