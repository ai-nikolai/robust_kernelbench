"""Report for the ScienceKernelBench five-step benchmark run.

Reads the experiments created by ``scripts/slurm_science_kb_5steps.sh``:

    experiments/exp_science_v0_level_<L>_<mode>/trial_<1..6>/evaluations.csv

and writes:

    analysis_output/science_v0_rounds.csv
    analysis_output/science_v0_table.md
    analysis_output/science_v0_correctness.png
    analysis_output/science_v0_final_comparison.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis.run_analysis_summary_clean import output_analysis

DEFAULT_MODES = ("kernelbench", "kb_multi_stage")
DEFAULT_LEVELS = (1, 2, 3)
DEFAULT_ROUNDS = 6


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="science_v0", help="Experiment version prefix.")
    parser.add_argument("--levels", type=int, nargs="+", default=list(DEFAULT_LEVELS))
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_output"))
    parser.add_argument("--name", default="science_v0")
    return parser.parse_args(argv)


def _metrics(frame: pd.DataFrame) -> dict:
    if frame is None or frame.empty:
        return {
            "count": 0,
            "compiled_global": 0.0,
            "correctness_global": 0.0,
            "check_ran_global": 0.0,
            "cuda_success_global": 0.0,
            "fast_1_global": 0.0,
            "fast_2_global": 0.0,
            "runtime_mean_compiled": 0.0,
        }
    try:
        return output_analysis(frame)
    except Exception as exc:  # noqa: BLE001 - a malformed/empty run should not abort the report
        print(f"[report] could not analyse {len(frame)} row(s): {exc}")
        return _metrics(pd.DataFrame())


def _read_metadata(experiment_dir: Path, trial: int) -> dict:
    path = experiment_dir / f"trial_{trial}" / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}


def collect(args) -> pd.DataFrame:
    rows = []
    for level in args.levels:
        for mode in args.modes:
            experiment = args.experiments_dir / f"exp_{args.version}_level_{level}_{mode}"
            for trial in range(1, args.rounds + 1):
                path = experiment / f"trial_{trial}" / "evaluations.csv"
                if not path.exists():
                    continue
                frame = pd.read_csv(path)
                metrics = _metrics(frame)
                metadata = _read_metadata(experiment, trial)
                rows.append(
                    {
                        "level": level,
                        "mode": mode,
                        "round": trial,
                        "trial": trial,
                        "rows": len(frame),
                        "dataset_version": metadata.get("dataset_version"),
                        "dataset_revision": metadata.get("dataset_revision"),
                        "compiled_global": metrics.get("compiled_global", 0.0),
                        "correctness_global": metrics.get("correctness_global", 0.0),
                        "check_ran_global": metrics.get("check_ran_global", 0.0),
                        "cuda_success_global": metrics.get("cuda_success_global", 0.0),
                        "fast_1_global": metrics.get("fast_1_global", 0.0),
                        "fast_2_global": metrics.get("fast_2_global", 0.0),
                        "runtime_mean_compiled": metrics.get("runtime_mean_compiled", 0.0),
                    }
                )
    return pd.DataFrame(rows)


def write_summary(rounds: pd.DataFrame, args) -> pd.DataFrame:
    if rounds.empty:
        return rounds
    first = rounds[rounds["round"] == 1][
        ["level", "mode", "correctness_global", "compiled_global"]
    ].rename(
        columns={
            "correctness_global": "baseline_correctness",
            "compiled_global": "baseline_compiled",
        }
    )
    last = rounds[rounds["round"] == args.rounds][
        ["level", "mode", "correctness_global", "compiled_global", "fast_1_global", "fast_2_global"]
    ].rename(
        columns={
            "correctness_global": "final_correctness",
            "compiled_global": "final_compiled",
            "fast_1_global": "final_fast_1",
            "fast_2_global": "final_fast_2",
        }
    )
    summary = first.merge(last, on=["level", "mode"], how="outer")
    summary["absolute_gain"] = summary["final_correctness"] - summary["baseline_correctness"]
    summary["relative_gain"] = summary["absolute_gain"] / summary["baseline_correctness"].replace(0, pd.NA)
    return summary.sort_values(["level", "mode"]).reset_index(drop=True)


def markdown_table(summary: pd.DataFrame) -> str:
    lines = [
        "| Level | Prompt | Baseline acc. | Final acc. | Gain | Final fast-1 | Final fast-2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.level} | `{row.mode}` | {row.baseline_correctness:.1%} | "
            f"{row.final_correctness:.1%} | {row.absolute_gain:+.1%} | "
            f"{row.final_fast_1:.1%} | {row.final_fast_2:.1%} |"
        )
    return "\n".join(lines) + "\n"


def plot_rounds(rounds: pd.DataFrame, path: Path) -> None:
    levels = sorted(rounds["level"].unique())
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4), sharey=True)
    if len(levels) == 1:
        axes = [axes]
    for axis, level in zip(axes, levels):
        subset = rounds[rounds["level"] == level]
        for mode in sorted(subset["mode"].unique()):
            mode_rows = subset[subset["mode"] == mode].sort_values("round")
            axis.plot(
                mode_rows["round"],
                mode_rows["correctness_global"],
                marker="o",
                label=mode,
            )
        axis.set_title(f"Level {level}")
        axis.set_xlabel("Round (1 = baseline, 6 = after 5 improvements)")
        axis.set_ylabel("Correctness")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("ScienceKernelBench: correctness across improvement rounds")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_final(summary: pd.DataFrame, path: Path) -> None:
    levels = sorted(summary["level"].unique())
    modes = sorted(summary["mode"].unique())
    width = 0.8 / max(len(modes), 1)
    fig, axis = plt.subplots(figsize=(7, 4))
    for index, mode in enumerate(modes):
        values = [
            summary[(summary["level"] == level) & (summary["mode"] == mode)]["final_correctness"].mean()
            for level in levels
        ]
        positions = [level + (index - (len(modes) - 1) / 2) * width for level in levels]
        axis.bar(positions, values, width=width, label=mode)
    axis.set_xticks(levels)
    axis.set_xticklabels([f"Level {level}" for level in levels])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Correctness after 5 improvements")
    axis.set_title("ScienceKernelBench: final correctness by prompt family")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rounds = collect(args)
    if rounds.empty:
        print(
            "No evaluations found. Expected experiments like "
            f"{args.experiments_dir}/exp_{args.version}_level_1_kernelbench/trial_1/evaluations.csv"
        )
        return 1

    rounds_path = args.output_dir / f"{args.name}_rounds.csv"
    rounds.to_csv(rounds_path, index=False)
    summary = write_summary(rounds, args)
    summary_path = args.output_dir / f"{args.name}_summary.csv"
    summary.to_csv(summary_path, index=False)
    table_path = args.output_dir / f"{args.name}_table.md"
    table_path.write_text(markdown_table(summary))

    plot_rounds(rounds, args.output_dir / f"{args.name}_correctness.png")
    plot_final(summary, args.output_dir / f"{args.name}_final_comparison.png")

    print(f"wrote {rounds_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {table_path}")
    print(markdown_table(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
