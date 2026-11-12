"""Compatibility shim for the moved agentic generation module.

New code should import :mod:`robust_kernelbench.agent.inference_common`; this
module keeps the historical import path working for existing scripts.
"""

from robust_kernelbench.agent.inference_common import *  # noqa: F401,F403
