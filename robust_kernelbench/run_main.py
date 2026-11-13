"""Compatibility shim for the moved agentic orchestrator.

New code should import :mod:`robust_kernelbench.agent.orchestrator`; the
``python -m robust_kernelbench.run_main`` invocation remains supported.
"""

from __future__ import annotations

import sys

from robust_kernelbench.agent.orchestrator import *  # noqa: F401,F403
from robust_kernelbench.agent.orchestrator import main

if __name__ == "__main__":
    sys.exit(main())
