#!/usr/bin/env python3
"""
Generate a reusable "meeting follow-up" analysis package from existing runs.

This script focuses on the non-poster todo items from the supervisor meeting:
1. weighted metrics
2. more rigorous summaries with uncertainty estimates
3. digging into concrete noisy samples
4. explaining sample-level vs entry-level cleaning trends
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score


ROOT = Path("/vol/gpudata/yz3522-llmtest")
EXPERIMENT_ROOT = ROOT / "MRes" / "NoiseRate" / "cxr_real_experiment"
FIGURE_ROOT = ROOT / "MRes" / "NoiseRate" / "figures"
IMAGE_ROOT = ROOT / "MedSoul" / "datasets" / "mimic-cxr-jpg-224"

COMMON12 = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
]
SHORT = {
    "Atelectasis": "Ate",
    "Cardiomegaly": "Card",
    "Consolidation": "Cons",
    "Edema": "Edem",
    "Enlarged Cardiomediastinum": "ECM",
    "Fracture": "Frac",
    "Lung Lesion": "Les",
    "Lung Opacity": "Opac",
    "Pleural Effusion": "Eff",
    "Pleural Other": "PO",
    "Pneumonia": "PNA",
    "Pneumothorax": "PTX",
}


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    family: str
    predictions_csv: Path
    baseline_sample_details: Path | None = None
    baseline_entry_details: Path | None = None


RUNS: dict[str, RunInfo] = {
    "appa_densenet_baseline": RunInfo(
        "appa_densenet_baseline",
        "DenseNet",
        EXPERIMENT_ROOT
        / "results_appa_pipeline/slurm_236304/01_baseline/test_study_predictions.csv",
        baseline_sample_details=EXPERIMENT_ROOT
        / "results_appa_pipeline/slurm_236304/01_baseline/train_cleanlab_sample_details.csv",
        baseline_entry_details=EXPERIMENT_ROOT
        / "results_appa_pipeline/slurm_236304/01_baseline/train_cleanlab_entry_details.csv",
    ),
    "appa_densenet_top10_sample": RunInfo(
        "appa_densenet_top10_sample",
        "DenseNet",
        EXPERIMENT_ROOT
        / "results_appa_densenet_topk/slurm_236927/top10_issue_fraction/test_study_predictions.csv",
    ),
    "appa_densenet_top05_entry": RunInfo(
        "appa_densenet_top05_entry",
        "DenseNet",
        EXPERIMENT_ROOT
        / "results_appa_densenet_entry_topk/slurm_237240/top05_entry_fraction/test_study_predictions.csv",
    ),
    "appa_xrv_linear_baseline": RunInfo(
        "appa_xrv_linear_baseline",
        "XRV linear-head",
        EXPERIMENT_ROOT
        / "results_appa_xrv12_linearhead_pipeline/slurm_236337/01_baseline/test_study_predictions.csv",
        baseline_sample_details=EXPERIMENT_ROOT
        / "results_appa_xrv12_linearhead_pipeline/slurm_236337/01_baseline/train_cleanlab_sample_details.csv",
        baseline_entry_details=EXPERIMENT_ROOT
        / "results_appa_xrv12_linearhead_pipeline/slurm_236337/01_baseline/train_cleanlab_entry_details.csv",
    ),
    "appa_xrv_linear_top20_sample": RunInfo(
        "appa_xrv_linear_top20_sample",
        "XRV linear-head",
        EXPERIMENT_ROOT
        / "results_appa_xrv12_linearhead_topk/slurm_236928/top20_issue_fraction/test_study_predictions.csv",
    ),
    "appa_xrv_linear_top05_entry": RunInfo(
        "appa_xrv_linear_top05_entry",
        "XRV linear-head",
        EXPERIMENT_ROOT
        / "results_appa_xrv12_linearhead_entry_topk/slurm_237241/top05_entry_fraction/test_study_predictions.csv",
    ),
}


def parse_pipe_array(series: pd.Series) -> np.ndarray:
    rows = []
    for value in series:
        arr = []
        for token in str(value).split("|"):
            token = token.strip()
            if token == "nan":
                arr.append(np.nan)
            else:
                arr.append(float(token))
        rows.append(arr)
    return np.asarray(rows, dtype=float)


def compute_study_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    per_label = {}
    micro_true = []
    micro_prob = []
    weights = []
    values = []
    valid_counts = []
    for idx, label in enumerate(COMMON12):
        mask = ~np.isnan(y_true[:, idx])
        yt = y_true[mask, idx]
        yp = y_prob[mask, idx]
        valid_counts.append(int(mask.sum()))
        if len(yt) == 0 or len(np.unique(yt)) < 2:
            per_label[label] = np.nan
            continue
        auc = roc_auc_score(yt, yp)
        per_label[label] = float(auc)
        values.append(float(auc))
        weights.append(int(mask.sum()))
        micro_true.append(yt)
        micro_prob.append(yp)
    macro = float(np.nanmean(values))
    weighted = float(np.average(values, weights=weights))
    micro = float(roc_auc_score(np.concatenate(micro_true), np.concatenate(micro_prob)))
    out = {
        "study_micro": micro,
        "study_macro": macro,
        "study_weighted": weighted,
    }
    for label, auc in per_label.items():
        out[f"study_{label}"] = auc
    return out


def bootstrap_metrics(predictions_csv: Path, n_boot: int = 300, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv)
    y_true = parse_pipe_array(df["binary_labels_for_metric"])
    y_prob = parse_pipe_array(df["pred_probs"])
    rng = np.random.default_rng(seed)
    rows = []
    n = len(df)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rows.append(compute_study_metrics(y_true[idx], y_prob[idx]))
    return pd.DataFrame(rows)


def ci_row(run_id: str, family: str, boot_df: pd.DataFrame) -> dict[str, float | str]:
    row: dict[str, float | str] = {"run_id": run_id, "family": family}
    for metric in ["study_micro", "study_macro", "study_weighted"]:
        vals = boot_df[metric].dropna().to_numpy()
        row[f"{metric}_mean"] = float(np.mean(vals))
        row[f"{metric}_ci_low"] = float(np.quantile(vals, 0.025))
        row[f"{metric}_ci_high"] = float(np.quantile(vals, 0.975))
    return row


def load_supports(predictions_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv)
    y_true = parse_pipe_array(df["binary_labels_for_metric"])
    rows = []
    for idx, label in enumerate(COMMON12):
        mask = ~np.isnan(y_true[:, idx])
        yt = y_true[mask, idx]
        rows.append(
            {
                "label": label,
                "valid_count": int(mask.sum()),
                "positive_count": int((yt == 1).sum()),
                "negative_count": int((yt == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def load_common12_table() -> pd.DataFrame:
    path = EXPERIMENT_ROOT / "common12_tables_20260511" / "common12_study_metrics_all_runs.csv"
    return pd.read_csv(path)


def label_delta_table(common12_df: pd.DataFrame, run_ids: list[str]) -> pd.DataFrame:
    sub = common12_df.set_index("run_id").loc[run_ids].copy()
    baseline = sub.loc[run_ids[0]]
    rows = []
    for label in COMMON12:
        base_value = float(baseline[f"study_{label}"])
        row = {"label": label, "baseline": base_value}
        for rid in run_ids[1:]:
            value = float(sub.loc[rid, f"study_{label}"])
            row[rid] = value
            row[f"{rid}_delta_vs_baseline"] = value - base_value
        rows.append(row)
    return pd.DataFrame(rows)


def counts_from_sample_details(sample_details_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(sample_details_csv)
    binary = parse_pipe_array(df["binary_labels_for_detection"])
    valid = np.array([[tok == "1" for tok in str(v).split("|")] for v in df["valid_label_mask"]], dtype=bool)
    rows = []
    for idx, label in enumerate(COMMON12):
        rows.append(
            {
                "label": label,
                "positive_count": int(np.sum((binary[:, idx] == 1) & valid[:, idx])),
                "negative_count": int(np.sum((binary[:, idx] == 0) & valid[:, idx])),
            }
        )
    return pd.DataFrame(rows)


def counts_from_sample_subset(sample_subset_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(sample_subset_csv)
    binary = parse_pipe_array(df["binary_labels_for_detection"])
    valid = np.array([[tok == "1" for tok in str(v).split("|")] for v in df["valid_label_mask"]], dtype=bool)
    rows = []
    for idx, label in enumerate(COMMON12):
        rows.append(
            {
                "label": label,
                "positive_removed": int(np.sum((binary[:, idx] == 1) & valid[:, idx])),
                "negative_removed": int(np.sum((binary[:, idx] == 0) & valid[:, idx])),
            }
        )
    return pd.DataFrame(rows)


def counts_from_entry_details(entry_details_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(entry_details_csv)
    sub = df[df["valid_label"] == 1].copy()
    rows = []
    for label in COMMON12:
        label_df = sub[sub["label_name"] == label]
        rows.append(
            {
                "label": label,
                "positive_count": int(np.sum(label_df["binary_label"] == 1)),
                "negative_count": int(np.sum(label_df["binary_label"] == 0)),
            }
        )
    return pd.DataFrame(rows)


def counts_from_entry_subset(entry_subset_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(entry_subset_csv)
    rows = []
    for label in COMMON12:
        label_df = df[df["label_name"] == label]
        rows.append(
            {
                "label": label,
                "positive_removed": int(np.sum(label_df["binary_label"] == 1)),
                "negative_removed": int(np.sum(label_df["binary_label"] == 0)),
            }
        )
    return pd.DataFrame(rows)


def removal_percentage_table(
    family: str,
    baseline_sample_details_csv: Path,
    baseline_entry_details_csv: Path,
    sample_subset_map: dict[str, Path],
    entry_subset_map: dict[str, Path],
    focus_labels: Iterable[str],
) -> pd.DataFrame:
    sample_base = counts_from_sample_details(baseline_sample_details_csv).set_index("label")
    entry_base = counts_from_entry_details(baseline_entry_details_csv).set_index("label")
    rows = []
    for label in focus_labels:
        row = {"family": family, "label": label}
        for name, path in sample_subset_map.items():
            removed = counts_from_sample_subset(path).set_index("label").loc[label]
            row[f"{name}_sample_pos_removed_pct"] = 100.0 * removed["positive_removed"] / max(
                1, sample_base.loc[label, "positive_count"]
            )
            row[f"{name}_sample_neg_removed_pct"] = 100.0 * removed["negative_removed"] / max(
                1, sample_base.loc[label, "negative_count"]
            )
        for name, path in entry_subset_map.items():
            removed = counts_from_entry_subset(path).set_index("label").loc[label]
            row[f"{name}_entry_pos_removed_pct"] = 100.0 * removed["positive_removed"] / max(
                1, entry_base.loc[label, "positive_count"]
            )
            row[f"{name}_entry_neg_removed_pct"] = 100.0 * removed["negative_removed"] / max(
                1, entry_base.loc[label, "negative_count"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def prettify_raw_label(raw: float | int | None) -> str:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return "NA"
    if raw == 1.0:
        return "+"
    if raw == 0.0:
        return "-"
    if raw == -1.0:
        return "U"
    return str(raw)


def render_issue_grid(
    sample_details_csv: Path,
    out_png: Path,
    out_csv: Path,
    top_n: int = 12,
) -> None:
    sample_df = pd.read_csv(sample_details_csv)
    sample_df = sample_df[sample_df["est_issue_sample"] == 1].copy()
    sample_df = sample_df.sort_values("issue_rank_self_confidence", na_position="last").head(top_n).copy()
    entry_masks = []
    raw_labels = []
    pred_probs = []
    for _, row in sample_df.iterrows():
        entry_masks.append([int(x) for x in str(row["issue_entry_mask"]).split("|")])
        raw_labels.append(
            [np.nan if x == "nan" else float(x) for x in str(row["raw_labels_4class"]).split("|")]
        )
        pred_probs.append([float(x) for x in str(row["pred_probs"]).split("|")])
    sample_df["entry_masks_parsed"] = entry_masks
    sample_df["raw_labels_parsed"] = raw_labels
    sample_df["pred_probs_parsed"] = pred_probs

    sample_df[
        [
            "subject_id",
            "study_id",
            "dicom_id",
            "image_path",
            "est_issue_entry_count",
            "issue_rank_self_confidence",
            "sample_quality_self_confidence",
            "raw_labels_4class",
            "pred_probs",
            "issue_entry_mask",
        ]
    ].to_csv(out_csv, index=False)

    ncols = 3
    nrows = math.ceil(len(sample_df) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(sample_df) :]:
        ax.axis("off")

    for ax, (_, row) in zip(axes, sample_df.iterrows()):
        img = Image.open(IMAGE_ROOT / row["image_path"]).convert("L")
        ax.imshow(np.asarray(img), cmap="gray")
        ax.axis("off")
        ax.set_title(
            f"study {int(row['study_id'])} | issue rank {int(row['issue_rank_self_confidence'])}\n"
            f"entry_cnt={int(row['est_issue_entry_count'])} | q={row['sample_quality_self_confidence']:.4f}",
            fontsize=10,
        )
        y = 1.02
        for label, raw, prob, flag in zip(
            COMMON12,
            row["raw_labels_parsed"],
            row["pred_probs_parsed"],
            row["entry_masks_parsed"],
        ):
            ax.text(
                0.01,
                y,
                f"{SHORT[label]}:{prettify_raw_label(raw)}->{prob:.2f}",
                transform=ax.transAxes,
                fontsize=7,
                color="red" if flag == 1 else "white",
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 1},
            )
            y += 0.06
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = EXPERIMENT_ROOT / "meeting_followup_20260520"
    out_dir.mkdir(parents=True, exist_ok=True)

    common12_df = load_common12_table()

    # 1. Weighted + CI summary for the most representative runs.
    ci_rows = []
    for run_id in [
        "appa_densenet_baseline",
        "appa_densenet_top10_sample",
        "appa_densenet_top05_entry",
        "appa_xrv_linear_baseline",
        "appa_xrv_linear_top20_sample",
        "appa_xrv_linear_top05_entry",
    ]:
        info = RUNS[run_id]
        boot = bootstrap_metrics(info.predictions_csv, n_boot=300, seed=42)
        ci_rows.append(ci_row(run_id, info.family, boot))
    pd.DataFrame(ci_rows).to_csv(out_dir / "study_metric_bootstrap_ci_summary.csv", index=False)

    # 2. Test support counts for common12.
    support_df = load_supports(RUNS["appa_xrv_linear_baseline"].predictions_csv)
    support_df.to_csv(out_dir / "study_label_support_common12.csv", index=False)

    # 3. Label trend tables for the two strongest families.
    densenet_trend = label_delta_table(
        common12_df,
        [
            "appa_densenet_baseline",
            "appa_densenet_top10_sample",
            "appa_densenet_top20_sample",
            "appa_densenet_top05_entry",
            "appa_densenet_top10_entry",
            "appa_densenet_top20_entry",
        ],
    )
    densenet_trend = densenet_trend.merge(support_df, on="label", how="left")
    densenet_trend.to_csv(out_dir / "densenet_label_trend_vs_baseline.csv", index=False)

    xrv_trend = label_delta_table(
        common12_df,
        [
            "appa_xrv_linear_baseline",
            "appa_xrv_linear_top10_sample",
            "appa_xrv_linear_top20_sample",
            "appa_xrv_linear_top05_entry",
            "appa_xrv_linear_top10_entry",
            "appa_xrv_linear_top20_entry",
        ],
    )
    xrv_trend = xrv_trend.merge(support_df, on="label", how="left")
    xrv_trend.to_csv(out_dir / "xrv_linear_label_trend_vs_baseline.csv", index=False)

    # 4. Removal-percentage analysis for focus labels.
    focus_labels = ["Lung Lesion", "Pleural Other", "Atelectasis", "Fracture", "Lung Opacity"]
    xrv_removal = removal_percentage_table(
        family="XRV linear-head",
        baseline_sample_details_csv=RUNS["appa_xrv_linear_baseline"].baseline_sample_details,  # type: ignore[arg-type]
        baseline_entry_details_csv=RUNS["appa_xrv_linear_baseline"].baseline_entry_details,  # type: ignore[arg-type]
        sample_subset_map={
            "top05": EXPERIMENT_ROOT
            / "results_appa_xrv12_linearhead_topk/slurm_236928/top05_issue_fraction_issue_subset.csv",
            "top10": EXPERIMENT_ROOT
            / "results_appa_xrv12_linearhead_topk/slurm_236928/top10_issue_fraction_issue_subset.csv",
            "top20": EXPERIMENT_ROOT
            / "results_appa_xrv12_linearhead_topk/slurm_236928/top20_issue_fraction_issue_subset.csv",
        },
        entry_subset_map={
            "top05": EXPERIMENT_ROOT
            / "results_appa_xrv12_linearhead_entry_topk/slurm_237241/top05_entry_fraction_issue_subset.csv",
            "top10": EXPERIMENT_ROOT
            / "results_appa_xrv12_linearhead_entry_topk/slurm_237241/top10_entry_fraction_issue_subset.csv",
            "top20": EXPERIMENT_ROOT
            / "results_appa_xrv12_linearhead_entry_topk/slurm_237241/top20_entry_fraction_issue_subset.csv",
        },
        focus_labels=focus_labels,
    )
    dense_removal = removal_percentage_table(
        family="DenseNet",
        baseline_sample_details_csv=RUNS["appa_densenet_baseline"].baseline_sample_details,  # type: ignore[arg-type]
        baseline_entry_details_csv=RUNS["appa_densenet_baseline"].baseline_entry_details,  # type: ignore[arg-type]
        sample_subset_map={
            "top05": EXPERIMENT_ROOT
            / "results_appa_densenet_topk/slurm_236927/top05_issue_fraction_issue_subset.csv",
            "top10": EXPERIMENT_ROOT
            / "results_appa_densenet_topk/slurm_236927/top10_issue_fraction_issue_subset.csv",
            "top20": EXPERIMENT_ROOT
            / "results_appa_densenet_topk/slurm_236927/top20_issue_fraction_issue_subset.csv",
        },
        entry_subset_map={
            "top05": EXPERIMENT_ROOT
            / "results_appa_densenet_entry_topk/slurm_237240/top05_entry_fraction_issue_subset.csv",
            "top10": EXPERIMENT_ROOT
            / "results_appa_densenet_entry_topk/slurm_237240/top10_entry_fraction_issue_subset.csv",
            "top20": EXPERIMENT_ROOT
            / "results_appa_densenet_entry_topk/slurm_237240/top20_entry_fraction_issue_subset.csv",
        },
        focus_labels=focus_labels,
    )
    removal_df = pd.concat([dense_removal, xrv_removal], ignore_index=True)
    removal_df.to_csv(out_dir / "focus_label_removal_percentages.csv", index=False)

    # 5. Real noisy-sample inspection artifact.
    render_issue_grid(
        sample_details_csv=RUNS["appa_xrv_linear_baseline"].baseline_sample_details,  # type: ignore[arg-type]
        out_png=out_dir / "xrv_linear_baseline_top_noisy_samples.png",
        out_csv=out_dir / "xrv_linear_baseline_top_noisy_samples.csv",
    )

    # 6. Lightweight markdown summary to reuse in meetings.
    ci_df = pd.read_csv(out_dir / "study_metric_bootstrap_ci_summary.csv")
    best_row = common12_df.loc[common12_df["study_macro"].idxmax()]
    report = f"""# Meeting Follow-up Analysis

Generated on 2026-05-20.

## What this package covers

- weighted study-level AUROC summaries
- bootstrap uncertainty estimates for representative runs
- per-label trend tables for DenseNet and XRV linear-head
- focus-label removal percentages for sample vs entry top-k
- real noisy-sample inspection figure from the XRV linear baseline

## Best common12 study-level macro AUROC

- `{best_row['run_id']}` = `{best_row['study_macro']:.4f}`

## Representative bootstrap-CI runs

| run_id | family | study_macro_mean | study_macro_95CI | study_weighted_mean | study_weighted_95CI |
|---|---|---:|---|---:|---|
"""
    for _, row in ci_df.iterrows():
        report += (
            f"| {row['run_id']} | {row['family']} | {row['study_macro_mean']:.4f} | "
            f"[{row['study_macro_ci_low']:.4f}, {row['study_macro_ci_high']:.4f}] | "
            f"{row['study_weighted_mean']:.4f} | "
            f"[{row['study_weighted_ci_low']:.4f}, {row['study_weighted_ci_high']:.4f}] |\n"
        )

    report += """

## Suggested interpretation anchors

- weighted AUROC is less sensitive than macro AUROC to small-class fluctuations
- entry-level top-k can hurt when it removes too much negative supervision for rare or hard labels
- sample-level top-k more often removes globally suspicious samples, so it can improve several labels at once
"""
    (out_dir / "README.md").write_text(report)


if __name__ == "__main__":
    main()
