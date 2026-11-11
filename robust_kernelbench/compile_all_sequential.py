"""Sequential (crash-isolated) compilation entry point.

Thin wrapper over :mod:`robust_kernelbench.compile_driver`.  Each kernel is compiled in
its own subprocess, so a kernel that aborts the CUDA context or hangs only loses itself.

Usage::

    python -m robust_kernelbench.compile_all_sequential --experiment_name exp_v3 --trial 1
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

COMPILE_MODE = "sequential"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compile generated kernels one subprocess at a time.")
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


if __name__ == "__main__":
    sys.exit(main())
