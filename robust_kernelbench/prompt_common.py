"""Compatibility shim for the moved agentic prompt-selection module.

New code should import :mod:`robust_kernelbench.agent.prompt_common`.
"""

from robust_kernelbench.agent.prompt_common import *  # noqa: F401,F403
