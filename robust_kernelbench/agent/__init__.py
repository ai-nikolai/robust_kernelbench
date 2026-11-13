"""Agentic generation/repair loop.

This subpackage is deliberately separate from the evaluation core in
``robust_kernelbench``.  Installing the PyPI package without the ``agent`` extra
gives you the compiler/evaluator/analysis harness; the LLM-serving, prompting and
multi-round orchestration code lives here so an evaluation checkout never needs
vLLM/SGLang/OpenAI installed.

The public entry point is :func:`robust_kernelbench.agent.orchestrator.cli`
(also exposed as the ``robust-kernelbench`` console script).
"""
