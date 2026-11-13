# Architecture

The repository is split into three layers with a one-way dependency rule:

```text
robust_kernelbench.agent  ->  robust_kernelbench core  ->  task sources (HF/local)
        |                              |
        +---- optional serving deps     +---- torch / pandas / datasets
```

## Core package

`robust_kernelbench` contains everything needed to evaluate kernels:

- `artifacts.py` — trial layout, CSV reads/appends, stage-completeness keys
- `compile_*`, `evaluate_*` — compile and evaluation drivers
- `tasks.py` — HF/directory/JSONL/Parquet task sources and validation
- `utils/` — filesystem, exec/compile and result-model helpers

Core modules **must not import** `robust_kernelbench.agent`. This is enforced
by `tests/test_agent_boundary.py`.

## Agent layer

`robust_kernelbench.agent` contains generation, prompting and multi-round
orchestration:

- LLM backends (OpenAI-compatible, vLLM, SGLang)
- prompt templates and failure classification
- `run_main` / `robust-kernelbench` orchestration

It is installed with the `agent` extra:

```bash
pip install "robust-kernelbench[agent]"
```

The evaluation core is installed by:

```bash
pip install robust-kernelbench
```

## Decision: keep the agent as an optional subpackage

We are **not** moving `agent/` into the core evaluation package, and we are
**not** moving it to `experiments/` or `baselines/`.

Reasons:

1. The core already has no runtime dependency on the agent; the only coupling is
   the `[agent]` extra and the compatibility CLI.
2. A single wheel with optional extras keeps installation and release simple.
3. `experiments/` is runtime output and is pruned; source must never live there.
4. `baselines/` describes comparison methods, not the experiment driver.
5. The current import boundary is explicit and tested, so it can be split later
   without changing the evaluation code.

Keep the agent in `robust_kernelbench/agent/` until there is a concrete need for
an independent release cadence, separate maintainers, or a strictly eval-only
wheel with no agent bytes.

## Migration path if the agent is split out

Move to a two-project repository:

```text
packages/
├── robust-kernelbench/          # core, no agent imports
│   └── pyproject.toml
└── robust-kernelbench-agent/    # depends on robust-kernelbench
    └── pyproject.toml
```

The agent project should:

- depend on `robust-kernelbench` at the compatible version;
- move `kb_prompts/` with it;
- continue to provide the `robust-kernelbench` console script;
- keep `robust-kernelbench-eval` / `-compile` in the core project;
- update old import paths through small re-export shims or a one-time deprecation.

## Dataset layer

Dataset **sources** live in `data/<name>/`, outside the wheel. They are exported
to Hugging Face with a per-dataset `to_hf.py` script. The package only contains
the loaders and validators in `tasks.py`; it never bundles a dataset.
