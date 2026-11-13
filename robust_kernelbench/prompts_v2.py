"""Compatibility shim for the moved agentic prompt templates.

New code should import :mod:`robust_kernelbench.agent.prompts_v2`.
"""

from robust_kernelbench.agent.prompts_v2 import *  # noqa: F401,F403
