"""Shared pytest configuration.

The project is a flat-layout package that is expected to be used via ``python -m
robust_kernelbench.<module>`` (or after ``pip install -e .``).  When the tests run straight
from a checkout, make sure the repository root is importable so ``import robust_kernelbench``
works without installing anything.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def cuda_available() -> bool:
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard dependency, but be safe
        return False
    return bool(torch.cuda.is_available())


def has_cuda_toolkit() -> bool:
    """Whether a CUDA compiler is available for ``torch.utils.cpp_extension.load_inline``."""
    import shutil

    if shutil.which("nvcc"):
        return True

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and os.path.exists(os.path.join(cuda_home, "bin", "nvcc")):
        return True

    # torch ships a CUDA runtime but not necessarily the compiler.
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.version.cuda and shutil.which("gcc"))
    except Exception:  # pragma: no cover
        return False


requires_gpu = pytest.mark.skipif(
    not cuda_available(), reason="requires a CUDA device"
)
requires_cuda_toolkit = pytest.mark.skipif(
    not (cuda_available() and has_cuda_toolkit()),
    reason="requires a CUDA device and a CUDA compiler (nvcc)",
)


@pytest.fixture
def experiment_dir(tmp_path, monkeypatch):
    """Point the pipeline at a throwaway ``experiments/`` root inside ``tmp_path``."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "experiments").mkdir()
    return tmp_path
