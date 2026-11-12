"""Kernel generation, stage 1 of the pipeline (SGLang entry point).

Same pipeline as ``run_inference_test_time_scaling.py``; only the engine construction
differs, and that lives behind a backend class in :mod:`robust_kernelbench.inference_common`.

Usage::

    python -m robust_kernelbench.run_inference_test_time_scaling_v2 \
        --experiment_name exp_local_1 --trial 1 --num_items 3
"""

from __future__ import annotations

import sys

from robust_kernelbench.agent.inference_common import (
    build_arg_parser,
    run_main as _run_main,
)

BACKEND = "sglang"


def parse_args(argv=None):
    return build_arg_parser("Kernel generation (SGLang backend)", backend_default=BACKEND).parse_args(argv)


def main(argv=None) -> int:
    return _run_main(argv, backend_kind=BACKEND, description="Kernel generation (SGLang backend)")


if __name__ == "__main__":
    sys.exit(main())
