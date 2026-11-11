"""Kernel generation, stage 1 of the pipeline (vLLM entry point).

Sets the vLLM environment variables that must be in place *before* ``vllm`` is imported,
then delegates everything else to :mod:`robust_kernelbench.inference_common`.

Usage::

    python -m robust_kernelbench.run_inference_test_time_scaling \
        --experiment_name exp_local_1 --trial 1 --num_items 3
"""

from __future__ import annotations

import os
import sys

# Must be set before `vllm` (or anything that imports it) is imported.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ALLREDUCE_USE_SYMM_MEM", "0")
os.environ.setdefault("VLLM_SHARED_MEM_BROADCAST_TIMEOUT", "300")  # 5 minutes
os.environ.setdefault("VLLM_FLASHINFER_ALLREDUCE", "0")

from robust_kernelbench.inference_common import (  # noqa: E402
    build_arg_parser,
    run_main as _run_main,
)

BACKEND = "vllm"


def parse_args(argv=None):
    return build_arg_parser("Kernel generation (vLLM backend)", backend_default=BACKEND).parse_args(argv)


def main(argv=None) -> int:
    return _run_main(argv, backend_kind=BACKEND, description="Kernel generation (vLLM backend)")


if __name__ == "__main__":
    sys.exit(main())
