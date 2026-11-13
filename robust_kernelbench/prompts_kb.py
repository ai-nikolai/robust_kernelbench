"""Compatibility shim for the moved KernelBench prompt templates.

New code should import :mod:`robust_kernelbench.agent.prompts_kb`.
"""

from robust_kernelbench.agent.prompts_kb import *  # noqa: F401,F403
