#!/usr/bin/env python3
"""Plot the CS839 paired-seed PPO/SAC learning curves from W&B panel CSV exports."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEEDS = (1, 2, 3, 4, 5)
EPISODE_LENGTH = 1000
BIN_STEPS = 10_000
T_95_DF4 = 2.7764451051977987
METRICS = {
    "charts/episodic_return": "Episodic return",
    "charts/true_performance": "Completed inversion-to-upright cycles per episode",
}
RUN_PATTERN = re.compile(r"__(ppo|sac)_continuous_action__(\d+)__(\d+)")
COLORS = {"ppo": "#2878B5", "sac": "#E1812C"}
LABELS = {"ppo": "PPO", "sac": "SAC"}


@dataclass
class RunMetric:
    run_id: str
    panel_steps: np.ndarray
    values: np.ndarray
    source: Path


def parse_exports(data_dir: Path) -> dict[str, dict[int, dict[str, RunMetric]]]:
    grouped: dict[str, dict[int, dict[str, RunMetric]]] = {"ppo": {}, "sac": {}}
    seen: set[tuple[str, int, str, str]] = set()
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV exports found in {data_dir}")

    for path in csv_files:
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames or []
            if not fieldnames:
                raise ValueError(f"{path.name} has no CSV header")
            step_column = next((name for name in fieldnames if name.lower() == "step"), None)
            if step_column is None:
                continue

            metric_columns = [
                name
                for name in fieldnames
                if name.rsplit(" - ", 1)[-1] in METRICS
            ]
            if not metric_columns:
                continue

            rows = list(reader)
            for column in metric_columns:
                match = RUN_PATTERN.search(column)
                if match is None:
                    raise ValueError(f"Could not identify algorithm/seed/run in header: {column}")
                algorithm, seed_text, run_id = match.groups()
                seed = int(seed_text)
                metric = column.rsplit(" - ", 1)[-1]
                key = (algorithm, seed, run_id, metric)
                if key in seen:
                    raise ValueError(f"Duplicate export for {key}: {path.name}")
                seen.add(key)

                panel_steps: list[float] = []
                values: list[float] = []
                for row_number, row in enumerate(rows, start=2):
                    step_text = row.get(step_column, "").strip()
                    value_text = row.get(column, "").strip()
                    if not step_text or not value_text:
                        continue
                    try:
                        panel_steps.append(float(step_text))
                        values.append(float(value_text))
                    except ValueError as exc:
                        raise ValueError(
                            f"Non-numeric Step/metric at {path.name}:{row_number}"
                        ) from exc

                if not values:
                    raise ValueError(f"No metric values found in {path.name}: {metric}")

                run_metrics = grouped[algorithm].setdefault(seed, {})
                if metric in run_metrics:
                    raise ValueError(
                        f"Multiple run IDs for {algorithm.upper()} seed {seed}; "
                        "remove pilot/old-run exports so each seed uses one final run."
                    )
                run_metrics[metric] = RunMetric(
                    run_id=run_id,
                    panel_steps=np.asarray(panel_steps, dtype=float),
                    values=np.asarray(values, dtype=float),
                    source=path,
                )

    expected_metrics = set(METRICS)
    for algorithm, runs in grouped.items():
        actual_seeds = set(runs)
        if actual_seeds != set(SEEDS):
            raise ValueError(
                f"{algorithm.upper()} exports have seeds {sorted(actual_seeds)}; "
                f"expected exactly {list(SEEDS)}."
            )
        for seed, metrics in runs.items():
            if set(metrics) != expected_metrics:
                raise ValueError(
                    f"{algorithm.upper()} seed {seed} has metrics {sorted(metrics)}; "
                    f"expected {sorted(expected_metrics)}."
                )
            returns = metrics["charts/episodic_return"]
            performance = metrics["charts/true_performance"]
            if returns.run_id != performance.run_id:
                raise ValueError(
                    f"{algorithm.upper()} seed {seed} return and performance exports "
                    "come from different W&B runs."
                )
            if len(returns.values) != len(performance.values):
                raise ValueError(
                    f"{algorithm.upper()} seed {seed} has different episode counts "
                    "between return and true-performance exports."
                )
            if not np.array_equal(returns.panel_steps, performance.panel_steps):
                raise ValueError(
                    f"{algorithm.upper()} seed {seed} panel Step values do not align "
                    "between its two metric exports."
                )

    if set(grouped["ppo"]) != set(grouped["sac"]):
        raise ValueError("PPO and SAC seed sets are not paired.")
    return grouped


def bin_seed_curve(values: np.ndarray, episode_length: int, edges: np.ndarray) -> np.ndarray:
    episode_steps = np.arange(1, len(values) + 1, dtype=float) * episode_length
    binned = np.full(len(edges) - 1, np.nan, dtype=float)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (episode_steps > left) & (episode_steps <= right)
        if np.any(mask):
            binned[index] = float(np.mean(values[mask]))
    return binned


def aggregate_algorithm(
    runs: dict[int, dict[str, RunMetric]],
    metric: str,
    episode_length: int,
    bin_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    common_episodes = min(len(runs[seed][metric].values) for seed in SEEDS)
    max_steps = common_episodes * episode_length
    edges = list(range(0, max_steps, bin_steps))
    if not edges or edges[0] != 0:
        edges.insert(0, 0)
    if edges[-1] != max_steps:
        edges.append(max_steps)
    edge_array = np.asarray(edges, dtype=float)
    x_steps = (edge_array[:-1] + edge_array[1:]) / 2.0

    seed_curves: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        values = runs[seed][metric].values[:common_episodes]
        seed_curves[seed] = bin_seed_curve(values, episode_length, edge_array)

    matrix = np.vstack([seed_curves[seed] for seed in SEEDS])
    if np.isnan(matrix).any():
        raise ValueError(f"Missing binned values for metric {metric}")
    mean = np.mean(matrix, axis=0)
    sample_sd = np.std(matrix, axis=0, ddof=1)
    ci_half_width = T_95_DF4 * sample_sd / np.sqrt(len(SEEDS))
    return x_steps, mean, sample_sd, ci_half_width, seed_curves


def plot_metric(
    metric: str,
    title: str,
    ylabel: str,
    grouped: dict[str, dict[int, dict[str, RunMetric]]],
    output_dir: Path,
    episode_length: int,
    bin_steps: int,
) -> list[dict[str, float | int | str]]:
    fig, ax = plt.subplots(figsize=(10, 6))
    summary_rows: list[dict[str, float | int | str]] = []
    max_x = 0.0

    for algorithm in ("ppo", "sac"):
        x_steps, mean, sample_sd, ci_half_width, seed_curves = aggregate_algorithm(
            grouped[algorithm], metric, episode_length, bin_steps
        )
        x_millions = x_steps / 1_000_000.0
        color = COLORS[algorithm]
        max_x = max(max_x, float(x_millions[-1]))

        for seed in SEEDS:
            ax.plot(
                x_millions,
                seed_curves[seed],
                color=color,
                alpha=0.28,
                linewidth=0.9,
                label=f"{LABELS[algorithm]} seeds" if seed == SEEDS[0] else "_nolegend_",
            )
        ax.fill_between(
            x_millions,
            mean - ci_half_width,
            mean + ci_half_width,
            color=color,
            alpha=0.18,
            linewidth=0,
            label=f"{LABELS[algorithm]} 95% Student-t CI",
        )
        ax.plot(
            x_millions,
            mean,
            color=color,
            linewidth=2.4,
            label=f"{LABELS[algorithm]} mean",
        )

        for index, x_step in enumerate(x_steps):
            row: dict[str, float | int | str] = {
                "metric": metric,
                "algorithm": algorithm.upper(),
                "env_steps": int(round(x_step)),
                "mean": float(mean[index]),
                "sample_sd": float(sample_sd[index]),
                "ci_low": float(mean[index] - ci_half_width[index]),
                "ci_high": float(mean[index] + ci_half_width[index]),
            }
            for seed in SEEDS:
                row[f"seed_{seed}"] = float(seed_curves[seed][index])
            summary_rows.append(row)

    ax.set_title(title)
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0, right=max_x)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncols=2, fontsize=9)
    fig.text(
        0.5,
        0.015,
        f"Thin lines: individual seeds; shaded bands: mean ± t(0.975, 4)·s/√5. "
        f"Each seed was binned in {bin_steps:,}-environment-step windows before aggregation.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    stem = "episodic_return" if metric == "charts/episodic_return" else "true_performance"
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "Downloads" / "CS839_Data",
        help="Folder containing per-run W&B panel CSV exports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <data-dir>/plots).",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=EPISODE_LENGTH,
        help="Environment steps per completed episode (default: 1000).",
    )
    parser.add_argument(
        "--bin-steps",
        type=int,
        default=BIN_STEPS,
        help="Environment-step bin width used per seed before aggregation (default: 10000).",
    )
    args = parser.parse_args()

    if args.episode_length <= 0 or args.bin_steps <= 0:
        parser.error("--episode-length and --bin-steps must be positive.")
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_dir / "plots"
    )
    grouped = parse_exports(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    summary_rows.extend(
        plot_metric(
            "charts/episodic_return",
            "Episodic Return During Training",
            "Mean episodic return",
            grouped,
            output_dir,
            args.episode_length,
            args.bin_steps,
        )
    )
    summary_rows.extend(
        plot_metric(
            "charts/true_performance",
            "True Performance During Training",
            "Completed inversion-to-upright cycles per episode",
            grouped,
            output_dir,
            args.episode_length,
            args.bin_steps,
        )
    )

    summary_path = output_dir / "aggregated_curves.csv"
    fieldnames = [
        "metric",
        "algorithm",
        "env_steps",
        "mean",
        "sample_sd",
        "ci_low",
        "ci_high",
        *(f"seed_{seed}" for seed in SEEDS),
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Loaded five paired seeds for PPO and SAC from {data_dir}")
    print(f"Plots and aggregated data saved in {output_dir}")
    print(
        "Environment steps are reconstructed as completed episode index × "
        f"{args.episode_length}; the W&B panel 'Step' column is not used as environment steps."
    )


if __name__ == "__main__":
    main()
