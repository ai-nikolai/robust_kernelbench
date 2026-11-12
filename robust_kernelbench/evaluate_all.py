# Previously Working Version 05.11.2025 (without subprocess). Current version from 05.11.2025
import os
import json
import io
import sys

import subprocess

from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.cpp_extension
import numpy as np

import gc

from datasets import load_dataset

from pydantic import BaseModel

from tqdm import tqdm

import pandas as pd

import shutil

from datetime import datetime

import multiprocessing as mp
from multiprocessing import Pool,Lock

import traceback

from typing import Any

from robust_kernelbench.logging_utils import get_logger, setup_logging
from robust_kernelbench.utils.utils import (
    construct_generation_dict,
    find_ending_digit_and_string,
    get_folder_path,
    get_file_name,
    get_problem_id_from_file_name,
    write_code,
)

from robust_kernelbench.evaluate_single import (
    memory_cleanup, 
    eval_kernel_against_ref,
    get_dataframe_results,
    read_file,
    evaluate_single,
    eval_wrapper
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


def _ensure_trailing_newline(path):
    """Make sure ``path`` ends with a newline before a row is appended.

    A child killed mid-append can leave a truncated last line; appending another
    row straight after it would merge the two and corrupt the CSV.  Best effort:
    a failure here is logged and does not stop the run.
    """
    try:
        with open(path, "rb+") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                return
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
    except OSError as e:
        logger.warning("could not normalise the trailing newline of %s: %s", path, e)


def write_timeout_result(experiment_name, trial, problem_id, sample_id, timeout, log_tail=""):
    """Append a real evaluation row for a kernel that had to be killed.

    Without this the problem simply vanished from evaluations.csv, so the
    evaluated count no longer matched the compiled count.

    The flags record only what was actually observed.  The compile stage did
    succeed (the kernel is listed in compilations.csv), but the process was
    killed before it could load the module, execute the kernel, or produce a
    timing -- so every runtime/loading/correctness/cuda flag is False and
    ``runtime`` is -1.0.  Claiming ``cuda_success=True`` here would have
    inflated the CUDA and compile aggregates in the analysis.
    """
    from robust_kernelbench.utils.utils import get_folder_path
    from robust_kernelbench.utils.utils_data import KernelExecResult

    # `get_dataframe_results` is imported at module level (from evaluate_single);
    # re-importing it here shadows that name (ruff F811).

    result = KernelExecResult(
        cuda_success=False,
        compiled=True,
        runtime_success=False,
        loading_success=False,
        correctness_success=False,
        timing_success=False,
        correctness=False,
        runtime=-1.0,
        runtime_stats={},
        runtime_original=-1.0,
        runtime_original_stats={},
        main_error=f"eval subprocess timed out after {timeout}s (process group killed)",
        main_traceback=(log_tail or "")[-4000:],
    )

    row = result.__dict__.copy()
    row["problem_id"] = problem_id
    row["sample_id"] = sample_id

    folder_path = get_folder_path(experiment_name, trial=trial)
    eval_filename = os.path.join(folder_path, "evaluations.csv")
    results_df = get_dataframe_results([row])
    if os.path.exists(eval_filename):
        _ensure_trailing_newline(eval_filename)
        results_df.to_csv(eval_filename, index=False, header=False, mode="a")
    else:
        results_df.to_csv(eval_filename, index=False)
    logger.error(
        "recorded timeout row for pid=%s sid=%s in %s", problem_id, sample_id, eval_filename
    )


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
    captured_text = "No Capture."
    device = torch.cuda.current_device() if torch.cuda.is_available() else None


    # folder_path = get_folder_path(experiment_name, trial)
    folder_path_code = get_folder_path(experiment_name, trial, postfix="code")
    folder_path_kernel = get_folder_path(experiment_name, trial, postfix="kernel")

    dict_of_files_to_evaluate = construct_generation_dict(folder_path_code)

    out_results = []

    # Path("kernels").mkdir(parents=True, exist_ok=True)
    # build_dir = os.path.join("kernels", f"ref_kernels_{CURRENT_DATETIME}")


    for sample_file_name, list_of_solutions in (pbar := tqdm(dict_of_files_to_evaluate.items())):
        problem_id = get_problem_id_from_file_name(sample_file_name)
        print("\n======Start\n[Main] Main Loop start...")
        if compiled_problem_ids:
            if not problem_id in compiled_problem_ids:
                print(f"Skipping problem_id {problem_id} as it's not in the compiled list.")
                continue

        print(f"[Main Evaluate] Currently evaluating problem_id: {problem_id} with {len(list_of_solutions)} solutions.")

        file_name_reference = get_file_name(problem_id)
        ref_kernel_folder_name = file_name_reference+"_ref_kernel"

        build_dir = os.path.join(folder_path_kernel, ref_kernel_folder_name)
        Path(build_dir).mkdir(parents=True, exist_ok=True)

        tmp = dataset.filter(lambda x: x["problem_id"] == problem_id)
        if tmp:
            sample = tmp[0]
        else:
            print(f"Problem with problem_id {problem_id} is not in the dataset. Skipping.")
            continue

        reference_code_path = os.path.join(build_dir,f"{sample['name']}.py")
        write_code(reference_code_path,sample["code"])


        for sample_id, solution in (pbar2 := tqdm(enumerate(list_of_solutions))):
            if (problem_id, sample_id) in pid_sid_set:
                print(f"[Main Evaluate] Skipping, pid_sid already evaluated: {(problem_id, sample_id)}")
                continue
            
            # pbar2.set_description(f"-------\nProblemID: {problem_id} -- evaluating specific solution id: {sample_id+1} of {len(list_of_solutions)}")
            print(f"-------\n[Main Evaluate] ProblemID: {problem_id} -- evaluating specific solution id: {sample_id+1} of {len(list_of_solutions)}")
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
                        proc.wait(timeout=timeout)
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
                    # Keep the per-problem logs for post-mortem when a kernel hung.
                    try:
                        write_timeout_result(
                            experiment_name, trial, problem_id, sample_id, timeout, log_tail=captured_text
                        )
                    except Exception:  # noqa: BLE001 - losing one row must not abort the batch
                        logger.exception(
                            "could not record the timeout row for pid=%s sid=%s; "
                            "evaluations.csv will be missing this problem",
                            problem_id,
                            sample_id,
                        )
                    continue

                # Keep the logs only for timeouts (post-mortem): otherwise two files
                # per problem accumulate inside every kernel build directory.
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

                ##################################
                # WITH Multi Processing
                # reference_src = read_file(reference_code_path)
                # custom_src = read_file(code_path)

                # print(f"\n---:\n[Main Evaluate] Running Actual Evaluation now for l:{level}, t: {trial}, p:{problem_id}, s:{sample_id}")
                # start = datetime.now()

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

                # with Pool(processes=1) as pool:
                #     print(f"Entering Pool {kwargs}")
                #     result = pool.apply_async(evaluate_single, kwds=kwargs)
                #     result.get(timeout=600)  # 10 minutes timeout
        
                # end = datetime.now()
                # print(f"[Main Evaluate] Got Results. Took time: {end-start}") 
                
                # NOTE: OLD CODE
                # custom_src = read_file(code_path)
                
                # print("\n---:\n[Main] Running Actual Evaluation now:")
                # results = eval_kernel_against_ref(
                #     original_model_src = sample["code"],
                #     custom_model_src = custom_src,
                #     seed_num = 42,
                #     num_correct_trials = 3,
                #     num_perf_trials = 3,
                #     verbose  = True,
                #     measure_performance = True,
                #     build_dir = build_dir,
                #     build_dir_new = build_dir_new,
                #     device = torch.cuda.current_device() if torch.cuda.is_available() else None, # have to run on GPU
                # )

                # memory_cleanup(custom_string=f"Final Clean-up: problem_id: {problem_id}, sample_id: {sample_id}", verbose=True)
                # Cleaning-up Evaluation
                # if clean_build_dir:
                #     pass
                #     Skipping this for now...
                #     shutil.rmtree(build_dir_new)
                
                # END OF CLEAN_UP
                # END OF OLD CODE 


            except Exception as e:
                print(f"[Main] THIS EVAL for problem_id:{problem_id} and sample_id:{sample_id} FULLY FAILED")
                print(captured_text)
                print("----")
                print(e)
                # memory_cleanup(custom_string=f"Final Clean-up:(fail) problem_id: {problem_id}, sample_id: {sample_id}", verbose=True)

    print("Done overall with eval loop.")
    return out_results



def get_successful_compilation(experiment_name, trial):
    # Load compilations.csv and filter for successful compilations
    compilations_csv_path = os.path.join(get_folder_path(experiment_name, trial=trial), 'compilations.csv')
    successful_problem_sample_pairs = None

    if os.path.exists(compilations_csv_path):
        compilations_df = pd.read_csv(compilations_csv_path)

        # Filter for rows where model_new_available is True
        successful_compilations = compilations_df[compilations_df['model_new_available'] == True]

        # Extract unique problem_ids and sample_ids
        successful_problem_sample_pairs = list(zip(successful_compilations['problem_id'], successful_compilations['sample_id']))

        print(f"[Main] Found {len(successful_problem_sample_pairs)} successful compilations with model_new_available=True")
    else:
        print(f"[Main] Warning: compilations.csv not found at {compilations_csv_path}. Proceeding with full dataset.")   

    return successful_problem_sample_pairs


def get_evaluation_pid_sid(experiment_name, trial):
    pid_sid_list = {}
    try:
        folder_path = get_folder_path(experiment_name, trial=trial)
        eval_filename = os.path.join(folder_path,'evaluations.csv')

        if os.path.exists(eval_filename):
            existing_compilations = pd.read_csv(eval_filename)
            pid_sid_list = set(
                [
                    x for x in \
                    existing_compilations[["problem_id", "sample_id"]].itertuples(index=False, name=None)
                ]
            )
    
    except Exception as e:
        print(f"[get_evaluations pid_sid] failed::{e}")
    
    return pid_sid_list

def get_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate every compiled kernel for one trial.")
    parser.add_argument("--trial", type=int, default=1, help="Which trial you are currently running...")
    parser.add_argument("--experiment_name", type=str, default="exp_test_run", help="experiment names")
    parser.add_argument("--experiment", type=str, default=None, help="Alias for --experiment_name.")
    parser.add_argument("--level", type=int, default=1, help="Which level to run...")

    # Correctness / timing policy (forwarded to every evaluate_single subprocess)
    parser.add_argument(
        "--num_correct_trials",
        type=int,
        default=5,
        help="Correctness trials per kernel; all must match the reference.",
    )
    parser.add_argument("--num_perf_trials", type=int, default=3, help="Performance trials per kernel.")
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
    parser.add_argument("--timeout", type=int, default=600, help="Per-kernel eval timeout (s).")
    parser.add_argument(
        "--eval_mem_gb",
        type=float,
        default=24.0,
        help="Address-space cap for one eval child in GB (0 disables the cap).",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG/INFO/WARNING/...).")

    args = parser.parse_args(argv)
    if not args.experiment_name or args.experiment_name == "exp_test_run":
        if args.experiment:
            args.experiment_name = args.experiment
    return args

def main(argv=None) -> int:
    args = get_args(argv)
    setup_logging(args.log_level, force=True)

    ds = load_dataset("ai-nikolai/KernelBench")
    level=args.level
    actual_ds = ds[f"level_{level}"]

    experiment_name = args.experiment_name

    successful_problem_sample_pairs = get_successful_compilation(experiment_name, args.trial)
    # Filter dataset to only include successful compilations
    if successful_problem_sample_pairs and len(successful_problem_sample_pairs) > 0:
        successful_problem_ids = set([pid for pid, _ in successful_problem_sample_pairs])
        actual_ds = actual_ds.filter(lambda x: x["problem_id"] in successful_problem_ids)
        logger.info("filtered dataset to %s problems with successful compilations", len(actual_ds))

        pid_sid_set = get_evaluation_pid_sid(experiment_name,args.trial)
        logger.info("found %s existing evaluations", len(pid_sid_set))
        eval_main_loop(
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


    try:
        # Ensure compilations.csv and evaluations.csv exist if not present
        output_dir = get_folder_path(experiment_name, args.trial)
        # Path(output_dir).mkdir(parents=True, exist_ok=True)

        compilations_csv = os.path.join(output_dir, 'compilations.csv')
        if os.path.exists(compilations_csv):
            evaluations_csv = os.path.join(output_dir, 'evaluations.csv')
            if not os.path.exists(evaluations_csv):
                print("[Main] WARNING: GENERATING EMPTY evaluations.csv, since compilations.csv exists, but no evaluations.csv.")
                empty_evaluations_df = pd.DataFrame(columns=[
                    'problem_id', 
                    'sample_id', 
                    'cuda_success', 
                    'compiled', 
                    'runtime_success', 
                    'loading_success', 
                    'correctness_success', 
                    'timing_success', 
                    'correctness', 
                    'metadata', 
                    'runtime', 
                    'runtime_stats', 
                    'runtime_original', 
                    'runtime_original_stats', 
                    'main_output', 
                    'main_error', 
                    'main_traceback'
                ])
                empty_evaluations_df.to_csv(evaluations_csv, index=False)
                logger.warning("created empty evaluations.csv at %s", evaluations_csv)
            else:
                logger.info("compilations.csv and evaluations.csv files exist. All seems ok.")
    except Exception as e:
        logger.error("filling in blanks for evaluations.csv did not work: %s", e)

    logger.info("exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
