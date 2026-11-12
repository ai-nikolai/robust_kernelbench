"""Compile a single kernel (one ``problem_id``/``sample_id`` pair).

Thin wrapper over :mod:`robust_kernelbench.compile_driver`.  Invoked directly, or by
``compile_all_sequential`` as an isolated subprocess.

Usage::

    python -m robust_kernelbench.compile_single \
        --code_path experiments/exp_v3/trial_1/code/88__sample_0.py \
        --build_dir_new experiments/exp_v3/trial_1/kernel/88__sample_0_solution_kernel \
        --experiment exp_v3 --trial 1 --problem_id 88 --sample_id 0
"""

from __future__ import annotations

import argparse
import sys

from robust_kernelbench.compile_driver import (
    configure_multiprocessing,
    resolve_experiment_name,
    single_compile_job,
)
from robust_kernelbench.logging_utils import setup_logging
from robust_kernelbench.utils.utils import read_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compile exactly one generated kernel.")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name.")
    parser.add_argument("--experiment_name", type=str, default=None, help="Alias for --experiment.")
    parser.add_argument("--trial", type=int, default=1, help="Which trial you are currently running.")
    parser.add_argument("--level", type=int, default=1, help="Which KernelBench level to run.")
    parser.add_argument("--code_path", type=str, required=True, help="Path to the generated kernel source.")
    parser.add_argument("--build_dir_new", type=str, required=True, help="Build directory for the kernel.")
    parser.add_argument("--problem_id", type=int, default=None, help="Problem ID.")
    parser.add_argument("--sample_id", type=int, default=0, help="Sample ID.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG/INFO/WARNING/...).")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level, force=True)

    configure_multiprocessing()

    experiment_name = resolve_experiment_name(args)
    custom_model_src = read_file(args.code_path)

    single_compile_job(
        device=None,
        custom_model_src=custom_model_src,
        build_dir_new=args.build_dir_new,
        trial=args.trial,
        problem_id=args.problem_id,
        sample_id=args.sample_id,
        experiment_name=experiment_name,
        len_list_of_solutions=1,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
