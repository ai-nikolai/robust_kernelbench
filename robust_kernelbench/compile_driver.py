"""Shared compile driver.

``compile_all.py``, ``compile_all_sequential.py`` and ``compile_single.py`` used to be
three near-identical copies of this logic (each with ~70 duplicated imports and its own
copy of the same bugs — ``compile_all_sequential`` referenced an undefined
``eval_file_path`` and ``level`` and could never have run).  They are now thin wrappers
over this module, selecting a *mode*:

``parallel``
    A multiprocessing ``Pool`` compiles many kernels concurrently.  Fastest, but a hard
    CUDA fault can take down the pool.
``sequential``
    One ``python -m robust_kernelbench.compile_single`` subprocess per kernel.  Slower,
    but each kernel is fully crash- and timeout-isolated.
``single``
    Compile exactly one ``(problem_id, sample_id)`` pair in the current process.

Torch is imported lazily so that argument parsing and job construction stay importable
(and testable) without a GPU.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import traceback
from collections import namedtuple
from datetime import datetime
from pathlib import Path

from robust_kernelbench import artifacts
from robust_kernelbench.logging_utils import get_logger
from robust_kernelbench.utils.utils import (
    construct_generation_dict,
    get_file_name,
    get_folder_path,
    get_problem_id_from_file_name,
    read_file,
)

logger = get_logger(__name__)

COMPILATIONS_CSV = artifacts.COMPILATIONS_CSV

# One unit of compilation work.  The first eight fields are exactly the positional
# arguments of ``single_compile_job`` (so ``pool.starmap(single_compile_job, [j[:8] ...])``
# works); ``code_path`` is kept alongside so the sequential mode can spawn a subprocess
# without re-deriving the source filename.
CompileJob = namedtuple(
    "CompileJob",
    [
        "device",
        "custom_model_src",
        "build_dir_new",
        "trial",
        "problem_id",
        "sample_id",
        "experiment_name",
        "len_list_of_solutions",
        "code_path",
    ],
)

# Multiprocessing synchronisation primitives for the `parallel` mode.  They are set by
# `job_init` in pool workers and by `_ensure_inprocess_locks` when compiling in-process.
global_lock = None
global_counter = None


def configure_multiprocessing() -> None:
    """Force the ``spawn`` start method (CUDA cannot be forked safely).

    ``set_start_method`` may only be called once per interpreter, so a second call from
    e.g. a test harness must not explode.
    """
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:  # pragma: no cover - already set by the embedding process
        pass


def job_init(local_lock, local_counter):
    """Pool initializer: hand the shared lock/counter to the worker process."""
    global global_lock, global_counter
    global_lock = local_lock
    global_counter = local_counter


def _ensure_inprocess_locks():
    """Provide a lock/counter when ``single_compile_job`` runs outside a Pool."""
    global global_lock, global_counter
    if global_lock is None:
        global_lock = mp.Lock()
    if global_counter is None:
        global_counter = mp.Value("i", 0)


def _record_result(experiment_name, trial, out_results, use_lock=True) -> bool:
    """Append one compile result row to ``<experiment>/trial_N/compilations.csv``.

    Returns ``False`` when the row could not be persisted; callers use that to
    distinguish a *kernel* failure (recorded, exit 0) from an infrastructure
    failure (not recorded, exit non-zero).
    """
    from robust_kernelbench.evaluate_single import get_dataframe_results

    eval_filename = artifacts.csv_path(experiment_name, trial, COMPILATIONS_CSV)

    try:
        results_df = get_dataframe_results(out_results)
        lock = global_lock
        if use_lock and lock is not None:
            with lock:
                artifacts.append_rows(results_df, eval_filename)
        else:
            artifacts.append_rows(results_df, eval_filename)
    except Exception as e:  # noqa: BLE001 - persistence failure is reported to the caller
        logger.error("could not append to %s: %s", eval_filename, e)
        return False
    return True


def record_compile_failure(job: CompileJob, reason: str) -> bool:
    """Record a compile row for a kernel whose subprocess died without one.

    A timeout, SIGKILL or interpreter abort means the normal ``compile_single``
    path never got to write its row.  Without this the problem simply disappears
    from ``compilations.csv`` and the stage-completeness check cannot tell a
    crashed kernel from a clean compile failure.
    """
    from robust_kernelbench.utils.utils_data import CompileResult

    result = CompileResult(main_error=reason, main_traceback=reason[-4000:])
    row = dict(result.__dict__)
    row["problem_id"] = job.problem_id
    row["sample_id"] = job.sample_id
    return _record_result(job.experiment_name, job.trial, [row])


def single_compile_job(
    device,
    custom_model_src,
    build_dir_new,
    trial,
    problem_id,
    sample_id,
    experiment_name,
    len_list_of_solutions,
):
    """Compile one generated kernel and persist the outcome.

    Never raises: a compile failure is itself the measurement, so it is recorded as a
    default ``CompileResult`` (all flags false) rather than aborting the batch.
    """
    from robust_kernelbench.utils.utils_compile import run_compilation
    from robust_kernelbench.utils.utils_data import CompileResult

    _ensure_inprocess_locks()

    logger.info(
        "compiling problem_id=%s sample_id=%s (%s/%s) -> %s",
        problem_id,
        sample_id,
        sample_id + 1,
        len_list_of_solutions,
        build_dir_new,
    )

    start = datetime.now()
    try:
        with global_counter.get_lock():
            global_counter.value += 1
        parallel = global_counter.value

        logger.debug("parallel jobs in flight: %s", parallel)

        context = {}
        compile_results = run_compilation(device, custom_model_src, context, build_dir_new)
        output = dict(compile_results.__dict__)
    except (Exception, SystemExit) as e:  # noqa: BLE001 - one failed kernel must not stop the batch
        logger.warning(
            "compilation failed for trial=%s problem_id=%s sample_id=%s: %s",
            trial,
            problem_id,
            sample_id,
            e,
        )
        logger.debug("traceback:\n%s", traceback.format_exc())
        # `start` may not be defined if the failure happened before it; be defensive.
        start = locals().get("start", datetime.now())
        output = dict(CompileResult().__dict__)
    finally:
        if global_counter is not None:
            with global_counter.get_lock():
                global_counter.value -= 1

    output["problem_id"] = problem_id
    output["sample_id"] = sample_id

    recorded = _record_result(experiment_name, trial, [output])
    if not recorded:
        logger.error("could not write compilation result for pid=%s sid=%s", problem_id, sample_id)

    logger.info(
        "compiled problem_id=%s sample_id=%s in %s (model_new_available=%s)",
        problem_id,
        sample_id,
        datetime.now() - start,
        output.get("model_new_available"),
    )
    return recorded


def build_compile_jobs(
    folder_path_code,
    folder_path_kernel,
    trial,
    experiment_name,
    device,
    target_problem_id=None,
    pid_sid_set=frozenset(),
    limit=None,
):
    """Build the list of ``single_compile_job`` positional-argument tuples.

    Pure filesystem work (no torch), which keeps this function easy to unit test.
    """
    dict_of_files_to_compile = construct_generation_dict(folder_path_code)
    logger.info("found %s problems to compile", len(dict_of_files_to_compile))

    jobs = []
    for sample_file_name, list_of_solutions in dict_of_files_to_compile.items():
        problem_id = get_problem_id_from_file_name(sample_file_name)

        if target_problem_id is not None and target_problem_id != problem_id:
            logger.debug("skipping problem_id=%s (not targeted)", problem_id)
            continue

        len_list_of_solutions = len(list_of_solutions)

        for sample_id, solution in enumerate(list_of_solutions):
            if (problem_id, sample_id) in pid_sid_set:
                logger.info("skipping already compiled pid=%s sid=%s", problem_id, sample_id)
                continue

            file_name_solution = get_file_name(problem_id, sample_id)
            solution_kernel_folder_name = file_name_solution + "_solution_kernel"

            build_dir_new = os.path.join(folder_path_kernel, solution_kernel_folder_name)
            Path(build_dir_new).mkdir(parents=True, exist_ok=True)

            code_path = os.path.join(folder_path_code, solution)
            custom_model_src = read_file(code_path)

            jobs.append(
                CompileJob(
                    device,
                    custom_model_src,
                    build_dir_new,
                    trial,
                    problem_id,
                    sample_id,
                    experiment_name,
                    len_list_of_solutions,
                    code_path,
                )
            )

            if limit is not None and len(jobs) >= limit:
                logger.info("reached job limit (%s)", limit)
                return jobs

    return jobs


def _compile_jobs_parallel(jobs, num_processes):
    lock = mp.Lock()
    counter = mp.Value("i", 0)

    with mp.Pool(processes=num_processes, initializer=job_init, initargs=(lock, counter)) as pool:
        logger.info("starting pool with %s processes for %s jobs", num_processes, len(jobs))
        results = pool.starmap(single_compile_job, [tuple(job[:8]) for job in jobs])
    logger.info("pool finished")
    return sum(1 for recorded in results if not recorded)


def build_single_compile_command(job):
    """Translate a ``CompileJob`` into a ``python -m robust_kernelbench.compile_single`` argv.

    Used by the ``sequential`` mode so each kernel gets its own interpreter (and thus
    its own CUDA context, so a hard fault cannot take down the batch).
    """
    return [
        sys.executable,
        "-m",
        "robust_kernelbench.compile_single",
        "--code_path",
        str(job.code_path),
        "--build_dir_new",
        str(job.build_dir_new),
        "--experiment",
        str(job.experiment_name),
        "--trial",
        str(job.trial),
        "--problem_id",
        str(job.problem_id),
        "--sample_id",
        str(job.sample_id),
    ]


def _kill_process_group(proc, grace=10):
    """SIGKILL ``proc``'s process group so nvcc/cc1plus grandchildren die too.

    Killing only the direct child leaves compilers running and, because they
    inherit the output pipes, the parent can block past its timeout.
    """
    import signal

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = None
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        logger.error("compile subprocess %s did not die after SIGKILL", proc.pid)


def _compile_jobs_sequential(jobs, timeout=600, verbose=False):
    """Compile each job in its own subprocess, so a hard crash cannot kill the run.

    Returns the number of kernels whose outcome could **not** be recorded.  A
    kernel that genuinely fails to compile is still a successful measurement and
    is recorded by ``compile_single``; only a lost row is an infrastructure
    failure.
    """
    failures = 0
    recorded: set[artifacts.KernelKey] = set()

    for job in jobs:
        command = build_single_compile_command(job)

        logger.info("sequential compile pid=%s sid=%s", job.problem_id, job.sample_id)
        logger.debug("command: %s", " ".join(command))
        # Files instead of pipes: a grandchild holding a pipe would block the
        # parent past the timeout (the same deadlock seen in evaluate_all).
        with tempfile.TemporaryFile(mode="w+") as out_fh, tempfile.TemporaryFile(mode="w+") as err_fh:
            proc = subprocess.Popen(command, stdout=out_fh, stderr=err_fh, start_new_session=True)
            try:
                returncode = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_process_group(proc)
                reason = f"compile subprocess timed out after {timeout}s (process group killed)"
                logger.error("%s for pid=%s sid=%s", reason, job.problem_id, job.sample_id)
                key = (job.problem_id, job.sample_id)
                recorded = artifacts.compilation_keys(job.experiment_name, job.trial)
                if key not in recorded:
                    if not record_compile_failure(job, reason):
                        failures += 1
                    recorded.add(key)
                continue
            out_fh.seek(0)
            stdout = out_fh.read()
            err_fh.seek(0)
            stderr = err_fh.read()

        key = (job.problem_id, job.sample_id)
        if returncode == 0:
            # compile_single returns 0 only after the row was written.
            recorded.add(key)
            if verbose:
                logger.debug("compile subprocess output:\n%s", stdout)
            continue

        logger.error(
            "compile subprocess failed (rc=%s) for pid=%s sid=%s\n--- stdout ---\n%s\n--- stderr ---\n%s",
            returncode,
            job.problem_id,
            job.sample_id,
            stdout,
            stderr,
        )
        # The child may have recorded a compile failure before dying.  Only a
        # missing row needs to be repaired here.
        recorded = artifacts.compilation_keys(job.experiment_name, job.trial)
        if key not in recorded:
            reason = f"compile subprocess exited with rc={returncode}"
            if not record_compile_failure(job, reason):
                failures += 1
            recorded.add(key)

    return failures


def compile_all(
    experiment_name,
    clean_build_dir=True,
    trial=None,
    verbose=True,
    num_processes=8,
    target_problem_id=None,
    pid_sid_set=frozenset(),
    mode="parallel",
    timeout=600,
    limit=None,
):
    """Compile every generated kernel for one trial.

    ``mode`` is one of ``parallel``, ``sequential`` or ``single`` (see module docstring).
    Returns the number of kernels whose row could not be persisted.  A kernel that
    fails to compile is a recorded measurement, not an infrastructure failure.
    """
    import torch

    valid_modes = ("parallel", "sequential", "single")
    if mode not in valid_modes:
        raise ValueError(f"Unknown compile mode: {mode!r} (expected one of {valid_modes})")

    # Surface the build knobs: MAX_JOBS/OMP_NUM_THREADS decide how many cudafe++
    # processes an nvcc build can spawn, which is what drives host-memory pressure.
    logger.info(
        "compile limits: mode=%s, processes=%s, per-compile timeout=%ss, MAX_JOBS=%s, OMP_NUM_THREADS=%s",
        mode,
        num_processes,
        timeout,
        os.environ.get("MAX_JOBS", "<unset>"),
        os.environ.get("OMP_NUM_THREADS", "<unset>"),
    )

    folder_path_code = get_folder_path(experiment_name, trial, postfix="code")
    folder_path_kernel = get_folder_path(experiment_name, trial, postfix="kernel")

    device = torch.cuda.current_device() if torch.cuda.is_available() else None

    jobs = build_compile_jobs(
        folder_path_code=folder_path_code,
        folder_path_kernel=folder_path_kernel,
        trial=trial,
        experiment_name=experiment_name,
        device=device,
        target_problem_id=target_problem_id,
        pid_sid_set=pid_sid_set,
        limit=limit,
    )

    if not jobs:
        logger.info("nothing to compile for %s trial_%s", experiment_name, trial)
        return 0

    logger.info("compiling %s jobs in %s mode", len(jobs), mode)
    start = datetime.now()

    if mode == "parallel":
        num_processes = max(1, min(int(num_processes or 1), len(jobs)))
        failures = _compile_jobs_parallel(jobs, num_processes)
    elif mode == "sequential":
        failures = _compile_jobs_sequential(jobs, timeout=timeout, verbose=verbose)
    elif mode == "single":
        _ensure_inprocess_locks()
        failures = sum(1 for job in jobs if not single_compile_job(*tuple(job[:8])))

    logger.info("finished compiling %s jobs in %s (failures=%s)", len(jobs), datetime.now() - start, failures)
    return failures


def get_compiled_pid_sid(experiment_name, trial):
    """Set of ``(problem_id, sample_id)`` pairs already present in compilations.csv."""
    try:
        return artifacts.compilation_keys(experiment_name, trial)
    except Exception as e:  # noqa: BLE001 - resumability is best-effort
        logger.warning("reading existing compilations failed: %s", e)
        return set()


def add_common_compile_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Flags shared by every compile entry point."""
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name.")
    parser.add_argument("--experiment", type=str, default=None, help="Alias for --experiment_name.")
    parser.add_argument("--trial", type=int, default=1, help="Which trial you are currently running.")
    parser.add_argument("--level", type=int, default=1, help="Which KernelBench level to run.")
    parser.add_argument("--num_processes", type=int, default=8, help="Parallel compilations to run.")
    parser.add_argument("--problem_id", type=int, default=None, help="Only compile this problem id.")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["parallel", "sequential", "single"],
        help="Compile mode; the entry points default to their own name.",
    )
    parser.add_argument("--timeout", type=int, default=600, help="Per-kernel timeout (s) in sequential mode.")
    parser.add_argument("--limit", type=int, default=None, help="Compile at most this many kernels.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG/INFO/WARNING/...).")
    return parser


def resolve_experiment_name(args) -> str:
    """``--experiment`` is the historical alias of ``--experiment_name``."""
    name = getattr(args, "experiment_name", None) or getattr(args, "experiment", None)
    if not name:
        raise SystemExit("ERROR: one of --experiment_name/--experiment is required")
    return name
