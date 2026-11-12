import multiprocessing as mp
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from robust_kernelbench import artifacts, tasks
from robust_kernelbench.cli_utils import non_negative_float, positive_int
from robust_kernelbench.evaluate_single import get_dataframe_results
from robust_kernelbench.logging_utils import get_logger, setup_logging
from robust_kernelbench.utils.utils import (
    construct_generation_dict,
    find_ending_digit_and_string,
    get_file_name,
    get_folder_path,
    get_problem_id_from_file_name,
    write_code,
)

logger = get_logger(__name__)

# Re-exported for backwards compatibility: these used to be defined here and were
# imported from this module by the compile drivers (creating an import cycle).
__all__ = [
    "find_ending_digit_and_string",
    "construct_generation_dict",
    "write_code",
    "eval_main_loop",
    "main",
]

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)  # Ensure spawn is used


# BASIC DEFINITIONS
level1_representative_subset_problem_ids = [1, 3, 6, 18, 23, 26, 33, 36, 40, 42, 48, 54, 57, 65, 77, 82, 86, 87]
level2_representative_subset_problem_ids = [1, 2, 8, 18, 23, 28, 33, 43]
level3_representative_subset_problem_ids = [1, 5, 8, 11, 20, 33, 38, 43]

CURRENT_DATETIME=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def build_evaluate_single_command(
    original_path,
    new_path,
    build_dir,
    build_dir_new,
    experiment_name,
    problem_id,
    sample_id,
    level,
    trial,
    num_correct_trials=5,
    num_perf_trials=3,
    atol=None,
    rtol=None,
    flush_l2=True,
    log_level=None,
):
    """Build the argv for one isolated ``python -m robust_kernelbench.evaluate_single``.

    Kept separate (and pure) so the correctness/timing settings that cross the process
    boundary can be asserted in a unit test without touching a GPU.
    """
    command = [
        sys.executable,
        "-m",
        "robust_kernelbench.evaluate_single",
        # Paths
        "--original_path", f"{original_path}",
        "--new_path", f"{new_path}",
        "--build_dir", f"{build_dir}",
        "--build_dir_new", f"{build_dir_new}",
        # Settings
        "--experiment", f"{experiment_name}",
        "--problem_id", f"{problem_id}",
        "--sample_id", f"{sample_id}",
        "--level", f"{level}",
        "--trial", f"{trial}",
        # Correctness / timing policy
        "--num_correct_trials", f"{num_correct_trials}",
        "--num_perf_trials", f"{num_perf_trials}",
    ]

    # Only forward explicit tolerances; otherwise evaluate_single applies its dtype policy.
    if atol is not None:
        command += ["--atol", f"{atol}"]
    if rtol is not None:
        command += ["--rtol", f"{rtol}"]

    command += ["--flush_l2" if flush_l2 else "--no-flush_l2"]

    if log_level:
        command += ["--log-level", str(log_level)]

    return command


def _eval_mem_limit_bytes(eval_mem_gb):
    """Address-space cap for one eval child, in bytes (None = unlimited)."""
    if not eval_mem_gb or eval_mem_gb <= 0:
        return None
    return int(eval_mem_gb * 1024 ** 3)


def _make_preexec_fn(eval_mem_gb):
    """Return a ``preexec_fn`` that caps the child's address space (RLIMIT_AS).

    A single evaluated kernel can otherwise allocate tens of GB of host memory
    (one conv3d case reached 16 GB), which on a small cluster cgroup triggers the
    kernel OOM-killer.  Returns ``None`` when the cap is disabled or unsupported.
    """
    limit = _eval_mem_limit_bytes(eval_mem_gb)
    if limit is None:
        return None
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX platforms
        logger.warning("--eval_mem_gb is not supported on this platform; ignoring")
        return None

    def _apply_limit():
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    return _apply_limit


def _kill_subprocess_group(proc, grace=10):
    """SIGKILL ``proc``'s whole process group, then reap it.

    Killing only the direct child leaves compiler/worker grandchildren alive, and
    because they inherit the stdout/stderr pipes the parent then blocks forever in
    ``communicate()`` -- which is exactly how one kernel hung past its timeout.
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
        except PermissionError:
            logger.warning("no permission to kill process group %s", pgid)

    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        logger.error("eval subprocess %s did not die after SIGKILL", proc.pid)


# Backwards-compatible alias for callers/tests that imported the helper here.
_ensure_trailing_newline = artifacts.ensure_trailing_newline


def _compiled_successfully(experiment_name, trial, problem_id, sample_id) -> bool:
    """Whether compilations.csv says this exact sample built."""
    try:
        return (problem_id, sample_id) in artifacts.compilation_keys(
            experiment_name, trial, successful_only=True
        )
    except Exception:  # noqa: BLE001 - the failure row is more important than this flag
        return False


def write_failure_result(
    experiment_name,
    trial,
    problem_id,
    sample_id,
    reason,
    log_tail="",
    compiled=None,
):
    """Append an evaluation row for a kernel whose child produced none.

    Only observed state is recorded.  ``compiled`` is looked up in
    compilations.csv when not supplied; the child was killed before it could
    load, execute or time anything, so every runtime/correctness flag is False
    and ``runtime`` is -1.0.
    """
    from robust_kernelbench.utils.utils_data import KernelExecResult

    key = (problem_id, sample_id)
    try:
        already_recorded = key in artifacts.evaluation_keys(experiment_name, trial)
    except Exception:  # noqa: BLE001 - a damaged CSV must not block the repair row
        already_recorded = False
    if already_recorded:
        logger.warning("not recording a duplicate failure row for pid=%s sid=%s", problem_id, sample_id)
        return

    if compiled is None:
        compiled = _compiled_successfully(experiment_name, trial, problem_id, sample_id)

    result = KernelExecResult(
        cuda_success=False,
        compiled=bool(compiled),
        runtime_success=False,
        loading_success=False,
        correctness_success=False,
        timing_success=False,
        correctness=False,
        runtime=-1.0,
        runtime_stats={},
        runtime_original=-1.0,
        runtime_original_stats={},
        main_error=reason,
        main_traceback=(log_tail or "")[-4000:],
    )

    row = result.__dict__.copy()
    row["problem_id"] = problem_id
    row["sample_id"] = sample_id

    eval_filename = artifacts.csv_path(experiment_name, trial, artifacts.EVALUATIONS_CSV)
    artifacts.append_rows(get_dataframe_results([row]), eval_filename)
    logger.error("recorded failure row for pid=%s sid=%s: %s", problem_id, sample_id, reason)


def write_timeout_result(experiment_name, trial, problem_id, sample_id, timeout, log_tail=""):
    """Append an evaluation row for a kernel that had to be killed."""
    return write_failure_result(
        experiment_name,
        trial,
        problem_id,
        sample_id,
        reason=f"eval subprocess timed out after {timeout}s (process group killed)",
        log_tail=log_tail,
        compiled=True,
    )


def _record_or_count_failure(writer, *args, **kwargs) -> int:
    """Call a failure-row writer, returning 1 if the row could not be stored."""
    try:
        writer(*args, **kwargs)
    except Exception:  # noqa: BLE001 - losing one row must not abort the batch
        logger.exception("could not record an evaluation failure row")
        return 1
    return 0


def eval_main_loop(
    dataset, 
    experiment_name, 
    clean_build_dir=True, 
    trial=None, 
    verbose=True, 
    compiled_problem_ids=None, 
    level=None, 
    already_evaluated=None,
    pid_sid_set=frozenset(),
    num_correct_trials=5,
    num_perf_trials=3,
    atol=None,
    rtol=None,
    flush_l2=True,
    timeout=600,
    eval_mem_gb=24,
    log_level=None,
    ):
    """Main Eval Loop"""
    print("[Main Evaluate] Entered `eval_main_loop()`")
    logger.info(
        "eval limits: per-kernel timeout=%ss, per-child address space=%s, MAX_JOBS=%s, OMP_NUM_THREADS=%s",
        timeout,
        f"{eval_mem_gb:.0f} GB" if _eval_mem_limit_bytes(eval_mem_gb) else "unlimited",
        os.environ.get("MAX_JOBS", "<unset>"),
        os.environ.get("OMP_NUM_THREADS", "<unset>"),
    )
    folder_path_code = get_folder_path(experiment_name, trial, postfix="code")
    folder_path_kernel = get_folder_path(experiment_name, trial, postfix="kernel")

    dict_of_files_to_evaluate = construct_generation_dict(folder_path_code)

    # One pass over the dataset instead of a full filter scan per problem.
    samples_by_problem_id = {}
    for sample in dataset:
        samples_by_problem_id.setdefault(int(sample["problem_id"]), sample)

    record_failures = 0

    for sample_file_name, list_of_solutions in tqdm(dict_of_files_to_evaluate.items()):
        problem_id = get_problem_id_from_file_name(sample_file_name)
        if compiled_problem_ids and problem_id not in compiled_problem_ids:
            logger.debug("skipping problem_id %s (not in the compiled list)", problem_id)
            continue

        logger.info("evaluating problem_id=%s (%s solution(s))", problem_id, len(list_of_solutions))

        build_dir = os.path.join(folder_path_kernel, get_file_name(problem_id) + "_ref_kernel")
        Path(build_dir).mkdir(parents=True, exist_ok=True)

        sample = samples_by_problem_id.get(problem_id)
        if sample is None:
            logger.warning("problem_id=%s is not in the dataset; skipping", problem_id)
            continue

        reference_code_path = os.path.join(build_dir, f"{sample['name']}.py")
        write_code(reference_code_path, sample["code"])

        for sample_id, solution in enumerate(list_of_solutions):
            if (problem_id, sample_id) in pid_sid_set:
                logger.info("skipping already-evaluated pid=%s sid=%s", problem_id, sample_id)
                continue

            logger.info("evaluating pid=%s sid=%s (%s/%s)", problem_id, sample_id, sample_id + 1, len(list_of_solutions))
            try:
                file_name_solution = get_file_name(problem_id, sample_id)
                solution_kernel_folder_name = file_name_solution+"_solution_kernel"

                build_dir_new = os.path.join(folder_path_kernel, solution_kernel_folder_name)               
                Path(build_dir_new).mkdir(parents=True, exist_ok=True)

                code_path = os.path.join(folder_path_code, solution)

                ##################################
                # WITH SUBPROCESS

                python_command_to_run = build_evaluate_single_command(
                    original_path=reference_code_path,
                    new_path=code_path,
                    build_dir=build_dir,
                    build_dir_new=build_dir_new,
                    experiment_name=experiment_name,
                    problem_id=problem_id,
                    sample_id=sample_id,
                    level=level,
                    trial=trial,
                    num_correct_trials=num_correct_trials,
                    num_perf_trials=num_perf_trials,
                    atol=atol,
                    rtol=rtol,
                    flush_l2=flush_l2,
                    log_level=log_level,
                )
                # kwargs = {
                #     "original_path":reference_code_path, 
                #     "new_path":code_path, 
                #     "build_dir":build_dir, 
                #     "build_dir_new":build_dir_new, 
                #     # other
                #     "device":device, # have to run on GPU
                #     # Admin:
                #     "experiment_folder":experiment_name,
                #     "level":level,
                #     "trial":trial,
                #     "problem_id":problem_id,
                #     "sample_id":sample_id,
                # }

                python_command_string = " ".join(str(x) for x in python_command_to_run)

                logger.debug("running eval subprocess: %s", python_command_string)

                start = datetime.now()
                # Redirect to *files*, not pipes: a grandchild that inherits a pipe
                # keeps it open, so a pipe-based timeout can block forever in
                # communicate() even after the direct child died.  start_new_session
                # gives the child its own process group so killpg reaches every
                # descendant, and preexec_fn caps the child's address space.
                stdout_path = os.path.join(build_dir_new, "eval_stdout.log")
                stderr_path = os.path.join(build_dir_new, "eval_stderr.log")
                timed_out = False
                returncode = None
                proc = None
                with open(stdout_path, "w") as out_fh, open(stderr_path, "w") as err_fh:
                    try:
                        proc = subprocess.Popen(
                            python_command_to_run,
                            stdout=out_fh,
                            stderr=err_fh,
                            start_new_session=True,
                            preexec_fn=_make_preexec_fn(eval_mem_gb),
                        )
                        returncode = proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        logger.error(
                            "eval subprocess timed out after %ss for pid=%s sid=%s; killing process group",
                            timeout, problem_id, sample_id,
                        )
                        if proc is not None:
                            _kill_subprocess_group(proc)

                captured_text = ""
                for _path in (stdout_path, stderr_path):
                    try:
                        with open(_path, "r", errors="replace") as _fh:
                            captured_text += _fh.read()
                    except OSError:
                        pass

                if timed_out:
                    record_failures += _record_or_count_failure(
                        write_timeout_result,
                        experiment_name, trial, problem_id, sample_id, timeout,
                        log_tail=captured_text,
                    )
                    continue

                if returncode != 0:
                    record_failures += _record_or_count_failure(
                        write_failure_result,
                        experiment_name, trial, problem_id, sample_id,
                        reason=f"evaluation subprocess exited with rc={returncode}",
                        log_tail=captured_text,
                    )
                    # Keep logs for every failed child, not just timeouts.
                    continue

                # Exit 0 means evaluate_single wrote its row; drop the per-problem
                # logs so they do not accumulate in every build directory.
                for _path in (stdout_path, stderr_path):
                    try:
                        os.remove(_path)
                    except OSError:
                        pass

                end = datetime.now()
                duration = end - start
                logger.info("eval subprocess finished in %s (pid=%s sid=%s)", duration, problem_id, sample_id)
                if verbose:
                    logger.debug(
                        "captured_text:\n----Start----\n\n%s\n\n----End----", captured_text
                    )

            except Exception as e:  # noqa: BLE001 - one kernel must not stop the batch
                logger.exception("evaluation setup failed for pid=%s sid=%s", problem_id, sample_id)
                record_failures += _record_or_count_failure(
                    write_failure_result,
                    experiment_name, trial, problem_id, sample_id,
                    reason=f"evaluation setup failed: {e}",
                )

    logger.info("done with eval loop")
    return record_failures



def get_successful_compilation(experiment_name, trial):
    """``(problem_id, sample_id)`` pairs that built a usable ``ModelNew``."""
    path = artifacts.csv_path(experiment_name, trial, artifacts.COMPILATIONS_CSV)
    if not os.path.exists(path):
        logger.warning("compilations.csv not found at %s; no kernels to evaluate", path)
        return []
    pairs = sorted(artifacts.compilation_keys(experiment_name, trial, successful_only=True))
    logger.info("%s successful compilation(s) with model_new_available=True", len(pairs))
    return pairs


def get_evaluation_pid_sid(experiment_name, trial):
    """``(problem_id, sample_id)`` pairs already present in evaluations.csv."""
    try:
        return artifacts.evaluation_keys(experiment_name, trial)
    except Exception as e:  # noqa: BLE001 - resumability is best-effort
        logger.warning("reading existing evaluations failed: %s", e)
        return set()

def get_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate every compiled kernel for one trial.")
    parser.add_argument("--trial", type=int, default=1, help="Which trial you are currently running...")
    parser.add_argument("--experiment_name", type=str, default="exp_test_run", help="experiment names")
    parser.add_argument("--experiment", type=str, default=None, help="Alias for --experiment_name.")
    parser.add_argument("--level", type=int, default=1, help="Which level to run...")
    tasks.add_dataset_args(parser)

    # Correctness / timing policy (forwarded to every evaluate_single subprocess)
    parser.add_argument(
        "--num_correct_trials",
        type=positive_int,
        default=5,
        help="Correctness trials per kernel; all must match the reference.",
    )
    parser.add_argument(
        "--num_perf_trials", type=positive_int, default=3, help="Performance trials per kernel."
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Absolute tolerance. Default: chosen from the output dtype.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Relative tolerance. Default: chosen from the output dtype.",
    )
    parser.add_argument(
        "--flush_l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evict L2 before each timed trial (default: on).",
    )
    parser.add_argument("--timeout", type=positive_int, default=600, help="Per-kernel eval timeout (s).")
    parser.add_argument(
        "--eval_mem_gb",
        type=non_negative_float,
        default=24.0,
        help="Address-space cap for one eval child in GB (0 disables the cap).",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG/INFO/WARNING/...).")

    args = parser.parse_args(argv)
    if not args.experiment_name or args.experiment_name == "exp_test_run":
        if args.experiment:
            args.experiment_name = args.experiment
    return args

def _ensure_evaluations_csv(experiment_name: str, trial: int) -> None:
    """Create an empty evaluations.csv when a trial has compilations but no eval rows.

    Analysis and multi-round prompting both expect the file to exist.
    """
    compilations_csv = artifacts.csv_path(experiment_name, trial, artifacts.COMPILATIONS_CSV)
    evaluations_csv = artifacts.csv_path(experiment_name, trial, artifacts.EVALUATIONS_CSV)
    if not os.path.exists(compilations_csv) or os.path.exists(evaluations_csv):
        return
    try:
        from robust_kernelbench.utils.utils_data import KernelExecResult

        columns = list(KernelExecResult().model_dump().keys()) + ["problem_id", "sample_id"]
        pd.DataFrame(columns=columns).to_csv(evaluations_csv, index=False)
        logger.warning("created empty evaluations.csv at %s", evaluations_csv)
    except Exception as e:  # noqa: BLE001 - analysis can still report a missing file
        logger.error("could not create %s: %s", evaluations_csv, e)


def main(argv=None) -> int:
    args = get_args(argv)
    setup_logging(args.log_level, force=True)

    source = tasks.source_from_args(args, level=args.level)
    actual_ds = tasks.load_tasks(source)
    logger.info("dataset: %s", source.metadata(actual_ds))

    experiment_name = args.experiment_name

    successful_pairs = get_successful_compilation(experiment_name, args.trial)
    record_failures = 0

    if successful_pairs:
        successful_problem_ids = {pid for pid, _ in successful_pairs}
        actual_ds = actual_ds.filter(lambda x: x["problem_id"] in successful_problem_ids)
        logger.info("filtered dataset to %s problems with successful compilations", len(actual_ds))

        pid_sid_set = get_evaluation_pid_sid(experiment_name, args.trial)
        logger.info("found %s existing evaluations", len(pid_sid_set))
        record_failures = eval_main_loop(
            actual_ds,
            experiment_name=experiment_name,
            trial=args.trial,
            level=args.level,
            pid_sid_set=pid_sid_set,
            num_correct_trials=args.num_correct_trials,
            num_perf_trials=args.num_perf_trials,
            atol=args.atol,
            rtol=args.rtol,
            flush_l2=args.flush_l2,
            timeout=args.timeout,
            eval_mem_gb=args.eval_mem_gb,
            log_level=args.log_level,
        )
        logger.info("done with evaluate all")
    else:
        logger.warning("no successful compilations found for %s trial_%s", experiment_name, args.trial)

    _ensure_evaluations_csv(experiment_name, args.trial)

    if record_failures:
        logger.error("%s evaluation result row(s) could not be recorded", record_failures)
        return 1
    logger.info("exiting")
    return 0


def cli() -> None:
    """Console-script entry point (``robust-kernelbench-eval``)."""
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
