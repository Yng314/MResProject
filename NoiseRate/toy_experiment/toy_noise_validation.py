"""
Toy experiment for validating whether cleanlab-estimated noise rate tracks
injected label noise in controlled K-class classification settings.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class TrialResult:
    n_classes: int
    seed: int
    injected_noise_rate: float
    true_noise_rate: float
    estimated_noise_rate: float
    issue_precision: float
    issue_recall: float
    oof_auroc_vs_clean: float


def rank_array(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt((x_centered**2).sum() * (y_centered**2).sum())
    if denom == 0:
        return np.nan
    return float((x_centered * y_centered).sum() / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rank_array(x), rank_array(y))


def inject_label_noise(
    y_clean: np.ndarray,
    noise_rate: float,
    n_classes: int,
    rng: np.random.RandomState,
):
    """Inject symmetric label noise by changing labels to random other classes."""
    y_noisy = y_clean.copy()
    flip_mask = rng.rand(len(y_clean)) < noise_rate
    flip_indices = np.where(flip_mask)[0]

    for idx in flip_indices:
        current = int(y_noisy[idx])
        choices = [c for c in range(n_classes) if c != current]
        y_noisy[idx] = rng.choice(choices)

    return y_noisy, flip_mask


def get_oof_probabilities(
    x: np.ndarray,
    y_train_labels: np.ndarray,
    n_splits: int,
    seed: int,
    n_classes: int,
) -> np.ndarray:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probs = np.zeros((len(y_train_labels), n_classes), dtype=float)

    for train_idx, val_idx in skf.split(x, y_train_labels):
        model = LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="auto")
        model.fit(x[train_idx], y_train_labels[train_idx])
        fold_probs = model.predict_proba(x[val_idx])

        # Ensure probability columns align with global class indices.
        probs[np.ix_(val_idx, model.classes_)] = fold_probs

    return probs


def calc_auroc_vs_clean(y_clean: np.ndarray, oof_probs: np.ndarray, n_classes: int) -> float:
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_clean, oof_probs[:, 1]))
        return float(roc_auc_score(y_clean, oof_probs, multi_class="ovr", average="macro"))
    except ValueError:
        return np.nan


def run_single_trial(
    n_classes: int,
    seed: int,
    noise_rate: float,
    n_samples: int,
    n_features: int,
    class_sep: float,
    n_splits: int,
) -> TrialResult:
    x, y_clean = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=max(1, n_features // 5),
        n_classes=n_classes,
        n_clusters_per_class=1,
        weights=[1.0 / n_classes] * n_classes,
        class_sep=class_sep,
        flip_y=0.0,
        random_state=seed,
    )

    rng = np.random.RandomState(seed + 10_000)
    y_noisy, flip_mask = inject_label_noise(y_clean, noise_rate, n_classes, rng)

    oof_probs = get_oof_probabilities(
        x=x,
        y_train_labels=y_noisy,
        n_splits=n_splits,
        seed=seed,
        n_classes=n_classes,
    )
    auroc_vs_clean = calc_auroc_vs_clean(y_clean, oof_probs, n_classes)

    issue_indices = find_label_issues(
        labels=y_noisy,
        pred_probs=oof_probs,
        return_indices_ranked_by="self_confidence",
    )

    issue_mask = np.zeros(len(y_noisy), dtype=bool)
    issue_mask[issue_indices] = True

    tp = int((issue_mask & flip_mask).sum())
    fp = int((issue_mask & ~flip_mask).sum())
    fn = int((~issue_mask & flip_mask).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    return TrialResult(
        n_classes=n_classes,
        seed=seed,
        injected_noise_rate=float(noise_rate),
        true_noise_rate=float(flip_mask.mean()),
        estimated_noise_rate=float(issue_mask.mean()),
        issue_precision=float(precision),
        issue_recall=float(recall),
        oof_auroc_vs_clean=float(auroc_vs_clean),
    )


def plot_kclass_comparison(summary_df: pd.DataFrame, output_path: Path):
    """Single chart for K=2/3/4: estimated noise (left y) and AUROC (right y)."""
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax2 = ax1.twinx()

    color_map = {2: "tab:blue", 3: "tab:green", 4: "tab:orange"}
    classes_sorted = sorted(summary_df["n_classes"].unique())

    for n_classes in classes_sorted:
        data = summary_df[summary_df["n_classes"] == n_classes].sort_values("injected_noise_rate")
        color = color_map.get(int(n_classes), None)

        ax1.errorbar(
            data["injected_noise_rate"],
            data["estimated_noise_rate_mean"],
            yerr=data["estimated_noise_rate_std"],
            fmt="o-",
            capsize=3,
            color=color,
            alpha=0.9,
            label=f"K={int(n_classes)} estimated",
        )
        ax2.plot(
            data["injected_noise_rate"],
            data["oof_auroc_vs_clean_mean"],
            "s--",
            color=color,
            alpha=0.9,
            label=f"K={int(n_classes)} AUROC",
        )

    ax1.plot([0, 0.5], [0, 0.5], linestyle=":", color="gray", alpha=0.6, label="Ideal y=x")

    ax1.set_xlabel("Injected Noise Rate")
    ax1.set_ylabel("Estimated Noise Rate")
    ax2.set_ylabel("OOF AUROC")
    ax1.set_xlim(0.0, 0.5)
    ax1.set_ylim(0.0, 0.55)
    ax2.set_ylim(0.0, 1.02)
    ax1.grid(alpha=0.3)
    ax1.set_title("Toy Comparison (K=2,3,4): Injected vs Estimated + AUROC")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(0.01, 0.98), fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_noise_levels(noise_levels_text: str) -> List[float]:
    values = [float(x.strip()) for x in noise_levels_text.split(",") if x.strip()]
    for v in values:
        if v < 0 or v > 1:
            raise ValueError(f"Noise rate must be in [0, 1], got {v}")
    return values


def build_noise_levels_from_range(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError(f"noise step must be > 0, got {step}")
    if start < 0 or end > 1 or start > end:
        raise ValueError(
            f"noise range must satisfy 0 <= start <= end <= 1, got start={start}, end={end}"
        )
    values = np.arange(start, end + step * 0.1, step)
    values = np.clip(values, 0.0, 1.0)
    values = np.round(values, 6)
    return [float(v) for v in values]


def parse_n_classes_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    for v in values:
        if v < 2:
            raise ValueError(f"n_classes must be >= 2, got {v}")
    return values


def main():
    parser = argparse.ArgumentParser(description="Run toy noise-rate validation experiment.")
    parser.add_argument("--output-dir", type=str, default="NoiseRate/toy_results")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--class-sep", type=float, default=2.0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-classes-list", type=str, default="2,3,4")
    parser.add_argument(
        "--noise-levels",
        type=str,
        default=None,
        help="Comma-separated levels in [0,1], e.g. '0,0.1,0.2'. Overrides range args.",
    )
    parser.add_argument("--noise-start", type=float, default=0.0)
    parser.add_argument("--noise-end", type=float, default=0.5)
    parser.add_argument("--noise-step", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_classes_list = parse_n_classes_list(args.n_classes_list)
    if args.noise_levels is not None:
        noise_levels = parse_noise_levels(args.noise_levels)
    else:
        noise_levels = build_noise_levels_from_range(
            start=args.noise_start, end=args.noise_end, step=args.noise_step
        )

    seeds = list(range(args.n_seeds))
    all_results: List[TrialResult] = []

    for n_classes in n_classes_list:
        for seed in seeds:
            for noise_rate in noise_levels:
                result = run_single_trial(
                    n_classes=n_classes,
                    seed=seed,
                    noise_rate=noise_rate,
                    n_samples=args.n_samples,
                    n_features=args.n_features,
                    class_sep=args.class_sep,
                    n_splits=args.n_splits,
                )
                all_results.append(result)
                print(
                    f"K={n_classes} seed={seed} noise={noise_rate:.2f} "
                    f"true={result.true_noise_rate:.3f} est={result.estimated_noise_rate:.3f} "
                    f"auroc={result.oof_auroc_vs_clean:.3f}"
                )

    trial_df = pd.DataFrame([r.__dict__ for r in all_results])
    trial_csv = output_dir / "toy_trials.csv"
    trial_df.to_csv(trial_csv, index=False)

    summary_df = (
        trial_df.groupby(["n_classes", "injected_noise_rate"], as_index=False)
        .agg(
            true_noise_rate_mean=("true_noise_rate", "mean"),
            true_noise_rate_std=("true_noise_rate", "std"),
            estimated_noise_rate_mean=("estimated_noise_rate", "mean"),
            estimated_noise_rate_std=("estimated_noise_rate", "std"),
            issue_precision_mean=("issue_precision", "mean"),
            issue_recall_mean=("issue_recall", "mean"),
            oof_auroc_vs_clean_mean=("oof_auroc_vs_clean", "mean"),
        )
        .sort_values(["n_classes", "injected_noise_rate"])
    )
    summary_csv = output_dir / "toy_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    report_txt = output_dir / "toy_report.txt"
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("Toy Noise Validation Report (K-class)\n")
        f.write("=====================================\n\n")
        f.write(f"n_samples={args.n_samples}\n")
        f.write(f"n_features={args.n_features}\n")
        f.write(f"class_sep={args.class_sep}\n")
        f.write(f"n_seeds={args.n_seeds}\n")
        f.write(f"n_splits={args.n_splits}\n")
        f.write(f"n_classes_list={n_classes_list}\n")
        f.write(f"noise_levels={noise_levels}\n\n")
        f.write("Per-class-count correlations (injected vs estimated_mean):\n")

        for n_classes in n_classes_list:
            group = summary_df[summary_df["n_classes"] == n_classes].sort_values("injected_noise_rate")
            x = group["injected_noise_rate"].to_numpy()
            y = group["estimated_noise_rate_mean"].to_numpy()
            pearson = pearson_corr(x, y)
            spearman = spearman_corr(x, y)
            f.write(f"- K={n_classes}: Pearson={pearson:.4f}, Spearman={spearman:.4f}\n")

        f.write("\nSee toy_summary.csv for full aggregated results.\n")

    plot_path = output_dir / "toy_injected_vs_estimated.png"
    plot_kclass_comparison(summary_df, plot_path)

    print("\nSaved:")
    print(f"- Trials:  {trial_csv}")
    print(f"- Summary: {summary_csv}")
    print(f"- Plot:    {plot_path}")
    print(f"- Report:  {report_txt}")


if __name__ == "__main__":
    main()
