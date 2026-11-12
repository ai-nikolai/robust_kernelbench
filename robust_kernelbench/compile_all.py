"""Parallel compilation entry point.

Thin wrapper over :mod:`robust_kernelbench.compile_driver`; keep the logic there so the
parallel/sequential/single variants cannot drift apart again.

Usage::

    python -m robust_kernelbench.compile_all --experiment_name exp_v3 --trial 1 --num_processes 8
"""

from __future__ import annotations

import argparse
import sys

from robust_kernelbench.compile_driver import (
    add_common_compile_args,
    compile_all,
    configure_multiprocessing,
    get_compiled_pid_sid,
    resolve_experiment_name,
)
from robust_kernelbench.logging_utils import setup_logging

COMPILE_MODE = "parallel"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compile generated kernels in parallel (multiprocessing pool).")
    add_common_compile_args(parser)
    args = parser.parse_args(argv)
    if args.mode is None:
        args.mode = COMPILE_MODE
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level, force=True)

    configure_multiprocessing()

    experiment_name = resolve_experiment_name(args)
    pid_sid_set = get_compiled_pid_sid(experiment_name, args.trial)

    failures = compile_all(
        experiment_name=experiment_name,
        trial=args.trial,
        num_processes=args.num_processes,
        target_problem_id=args.problem_id,
        pid_sid_set=pid_sid_set,
        mode=args.mode,
        timeout=args.timeout,
        limit=args.limit,
    )
    return 0 if failures == 0 else 1


def cli() -> None:
    """Console-script entry point (``robust-kernelbench-compile``)."""
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
