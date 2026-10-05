#!/usr/bin/env python3
"""Evaluate MobileNet noES seed robustness from completed result folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DEFAULT_ROOT = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_noes50_clean_3seed/20260707_123320"
)


def parse_pipe_array(value: str, dtype=float) -> list[float]:
    out: list[float] = []
    for item in str(value).split("|"):
        if item == "nan":
            out.append(np.nan)
        else:
            out.append(dtype(item))
    return out


def weighted_from_auc_summary(path: Path) -> float:
    df = pd.read_csv(path)
    return float(np.average(df["study_auroc_binary"], weights=df["study_valid_count"]))


def discover_complete_seeds(root: Path, loops: list[int]) -> list[int]:
    seeds: list[int] = []
    for seed_dir in sorted(root.glob("seed_*")):
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        baseline = seed_dir / "baseline_no_clean" / "test_study_auroc_summary.csv"
        loop_csv = seed_dir / "sample20_remove_loop" / "remove_only" / "loop_metrics.csv"
        if not baseline.exists() or not loop_csv.exists():
            continue
        loop_df = pd.read_csv(loop_csv)
        completed = set(loop_df["loop_id"].astype(int).tolist())
        if all(loop in completed for loop in loops):
            seeds.append(seed)
    return sorted(seeds)


def stage_prediction_path(root: Path, seed: int, stage: str, loop: int | None = None) -> Path:
    if stage == "baseline":
        return root / f"seed_{seed}" / "baseline_no_clean" / "test_study_predictions.csv"
    if loop is None:
        raise ValueError("loop is required for remove stage")
    return (
        root
        / f"seed_{seed}"
        / "sample20_remove_loop"
        / "remove_only"
        / f"loop_{loop:02d}"
        / "train_eval"
        / "test_study_predictions.csv"
    )


def load_prediction_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    labels = np.array([parse_pipe_array(v, float) for v in df["binary_labels_for_metric"]], dtype=float)
    valid = np.array([parse_pipe_array(v, int) for v in df["valid_label_mask"]], dtype=bool)
    probs = np.array([parse_pipe_array(v, float) for v in df["pred_probs"]], dtype=float)
    return labels, valid, probs


def weighted_auc(labels: np.ndarray, valid: np.ndarray, probs: np.ndarray, idx: np.ndarray) -> float:
    label_scores: list[float] = []
    label_weights: list[int] = []
    for label_idx in range(labels.shape[1]):
        mask = valid[idx, label_idx]
        if not np.any(mask):
            continue
        y = labels[idx, label_idx][mask]
        p = probs[idx, label_idx][mask]
        if np.unique(y).size < 2:
            continue
        label_scores.append(float(roc_auc_score(y, p)))
        label_weights.append(int(mask.sum()))
    if not label_scores:
        return float("nan")
    return float(np.average(label_scores, weights=label_weights))


def collect_seed_metrics(root: Path, seeds: list[int], loops: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        base_summary = seed_root / "baseline_no_clean" / "test_study_auroc_summary.csv"
        baseline_weighted = weighted_from_auc_summary(base_summary)
        rows.append(
            {
                "seed": seed,
                "stage": "baseline",
                "loop": 0,
                "weighted_auroc": baseline_weighted,
                "delta_vs_baseline": 0.0,
            }
        )
        loop_df = pd.read_csv(seed_root / "sample20_remove_loop" / "remove_only" / "loop_metrics.csv")
        for loop in loops:
            match = loop_df.loc[loop_df["loop_id"].astype(int) == loop].iloc[0]
            weighted = float(match["test_study_weighted_auroc"])
            rows.append(
                {
                    "seed": seed,
                    "stage": "remove",
                    "loop": loop,
                    "weighted_auroc": weighted,
                    "delta_vs_baseline": weighted - baseline_weighted,
                    "oof_sample_issue_rate": float(match["oof_sample_issue_rate"]),
                    "oof_entry_issue_rate": float(match["oof_entry_issue_rate"]),
                    "cumulative_removed_samples": int(match["cumulative_removed_samples"]),
                }
            )
    per_seed = pd.DataFrame(rows).sort_values(["seed", "loop", "stage"])
    group_rows: list[dict[str, float | int | str]] = []
    for loop in [0] + loops:
        stage = "baseline" if loop == 0 else "remove"
        vals = per_seed.loc[(per_seed["stage"] == stage) & (per_seed["loop"] == loop), "weighted_auroc"]
        deltas = per_seed.loc[(per_seed["stage"] == stage) & (per_seed["loop"] == loop), "delta_vs_baseline"]
        group_rows.append(
            {
                "stage": stage,
                "loop": loop,
                "n": int(vals.count()),
                "mean_weighted_auroc": float(vals.mean()),
                "std_weighted_auroc": float(vals.std(ddof=1)) if vals.count() > 1 else np.nan,
                "mean_delta_vs_baseline": float(deltas.mean()),
                "std_delta_vs_baseline": float(deltas.std(ddof=1)) if deltas.count() > 1 else np.nan,
                "positive_delta_seeds": int((deltas > 0).sum()),
            }
        )
    return per_seed, pd.DataFrame(group_rows)


def bootstrap_loop_deltas(
    root: Path,
    seeds: list[int],
    loops: list[int],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for loop in loops:
        per_seed_boot: list[np.ndarray] = []
        for seed in seeds:
            base_labels, base_valid, base_probs = load_prediction_arrays(stage_prediction_path(root, seed, "baseline"))
            loop_labels, loop_valid, loop_probs = load_prediction_arrays(stage_prediction_path(root, seed, "remove", loop))
            if base_labels.shape != loop_labels.shape:
                raise ValueError(f"Shape mismatch for seed {seed} loop {loop}")
            n = base_labels.shape[0]
            diffs = np.empty(n_bootstrap, dtype=float)
            for boot_idx in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                base_auc = weighted_auc(base_labels, base_valid, base_probs, idx)
                loop_auc = weighted_auc(loop_labels, loop_valid, loop_probs, idx)
                diffs[boot_idx] = loop_auc - base_auc
            per_seed_boot.append(diffs)
        boot = np.vstack(per_seed_boot)
        mean_delta = np.nanmean(boot, axis=0)
        rows.append(
            {
                "loop": loop,
                "n_seeds": len(seeds),
                "n_bootstrap": n_bootstrap,
                "valid_bootstrap": int(np.isfinite(mean_delta).sum()),
                "mean_delta": float(np.nanmean(mean_delta)),
                "ci_low_2p5": float(np.nanpercentile(mean_delta, 2.5)),
                "ci_high_97p5": float(np.nanpercentile(mean_delta, 97.5)),
                "bootstrap_p_delta_le_0": float(np.nanmean(mean_delta <= 0)),
            }
        )
    return pd.DataFrame(rows)


def write_plots(per_seed: pd.DataFrame, group: pd.DataFrame, bootstrap: pd.DataFrame, out_dir: Path) -> None:
    labels = ["baseline"] + [f"loop{loop}" for loop in sorted(per_seed.loc[per_seed["stage"] == "remove", "loop"].unique())]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=180)
    for seed, sdf in per_seed.groupby("seed"):
        y = []
        for pos, _label in enumerate(labels):
            loop = 0 if pos == 0 else pos
            stage = "baseline" if loop == 0 else "remove"
            y.append(float(sdf.loc[(sdf["stage"] == stage) & (sdf["loop"] == loop), "weighted_auroc"].iloc[0]))
        ax.plot(x, y, marker="o", linewidth=1.3, alpha=0.75, label=f"seed {seed}")
    means = []
    stds = []
    for pos, _label in enumerate(labels):
        loop = 0 if pos == 0 else pos
        row = group.loc[group["loop"] == loop].iloc[0]
        means.append(float(row["mean_weighted_auroc"]))
        stds.append(float(row["std_weighted_auroc"]) if np.isfinite(row["std_weighted_auroc"]) else 0.0)
    ax.errorbar(x, means, yerr=stds, color="black", marker="s", linewidth=2.2, capsize=4, label="mean +/- sd")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Study weighted AUROC")
    ax.set_title("MobileNetV3 noES: seed robustness")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "weighted_auroc_seed_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=180)
    bx = bootstrap["loop"].to_numpy()
    y = bootstrap["mean_delta"].to_numpy()
    lower = y - bootstrap["ci_low_2p5"].to_numpy()
    upper = bootstrap["ci_high_97p5"].to_numpy() - y
    ax.axhline(0, color="0.35", linewidth=1)
    ax.errorbar(bx, y, yerr=[lower, upper], marker="o", linewidth=2, capsize=4)
    ax.set_xticks(bx, [f"loop{int(v)}" for v in bx])
    ax.set_ylabel("Delta weighted AUROC vs baseline")
    ax.set_title("Paired study-bootstrap CI for mean seed delta")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "paired_delta_bootstrap_ci.png")
    plt.close(fig)


def write_report(
    out_dir: Path,
    root: Path,
    seeds: list[int],
    group: pd.DataFrame,
    bootstrap: pd.DataFrame,
    n_bootstrap: int,
) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        headers = list(df.columns)
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df.iterrows():
            cells: list[str] = []
            for col in headers:
                value = row[col]
                if isinstance(value, (float, np.floating)):
                    cells.append("nan" if np.isnan(value) else f"{value:.4f}")
                else:
                    cells.append(str(value))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    lines = [
        "# MobileNet noES Seed Robustness Evaluation",
        "",
        f"Result root: `{root}`",
        f"Completed seeds included: `{', '.join(map(str, seeds))}`",
        f"Bootstrap iterations: `{n_bootstrap}`",
        "",
        "## Mean / SD",
        "",
        markdown_table(group),
        "",
        "## Paired Study-Bootstrap Delta CI",
        "",
        markdown_table(bootstrap),
        "",
        "## Interpretation Notes",
        "",
        "- Treat this as an evaluation dry run if only the original three seeds are complete.",
        "- The bootstrap resamples test studies and estimates paired deltas against each seed's own baseline.",
        "- Very rare labels can be unstable under bootstrap because some resamples may lack both classes.",
        "- Do not report best-loop-only results as the primary claim; use fixed loops and paired deltas.",
    ]
    (out_dir / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--loops", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    root = args.root
    loops = sorted(args.loops)
    seeds = sorted(args.seeds or discover_complete_seeds(root, loops))
    if not seeds:
        raise SystemExit(f"No complete seeds found under {root}")
    out_dir = args.out_dir or (root / f"evaluation_{len(seeds)}seed")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed, group = collect_seed_metrics(root, seeds, loops)
    rng = np.random.default_rng(args.seed)
    bootstrap = bootstrap_loop_deltas(root, seeds, loops, args.bootstrap_iters, rng)

    per_seed.to_csv(out_dir / "per_seed_weighted_auroc.csv", index=False)
    group.to_csv(out_dir / "group_weighted_auroc_summary.csv", index=False)
    bootstrap.to_csv(out_dir / "bootstrap_loop_delta_ci.csv", index=False)
    write_plots(per_seed, group, bootstrap, out_dir)
    write_report(out_dir, root, seeds, group, bootstrap, args.bootstrap_iters)

    print(f"Wrote evaluation outputs to {out_dir}")
    print(f"Included seeds: {seeds}")
    print(group.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
