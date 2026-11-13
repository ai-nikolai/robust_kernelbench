"""The evaluation core must remain importable without the agentic layer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_core_modules_import_without_agent_package():
    code = r"""
import sys

class BlockAgent:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "robust_kernelbench.agent" or fullname.startswith("robust_kernelbench.agent."):
            raise ImportError("agent layer is intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockAgent())

import robust_kernelbench.artifacts
import robust_kernelbench.compile_driver
import robust_kernelbench.evaluate_all
import robust_kernelbench.evaluate_single
import robust_kernelbench.tasks
print("core imports ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "core imports ok" in completed.stdout
