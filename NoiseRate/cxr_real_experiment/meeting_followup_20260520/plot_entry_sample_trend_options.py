#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


ROOT = Path("/vol/gpudata/yz3522-llmtest/MRes/NoiseRate/cxr_real_experiment/meeting_followup_20260520")
XRV = pd.read_csv(ROOT / "xrv_linear_label_metrics_with_removal_percentages.csv")
DENSE = pd.read_csv(ROOT / "densenet_label_metrics_with_removal_percentages.csv")
SUPPORT = pd.read_csv(ROOT / "study_label_support_common12.csv").set_index("label")

TOPKS = ["05", "10", "20"]
LABELS = XRV["label"].tolist()
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


def add_support(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["test_neg"] = out["label"].map(SUPPORT["negative_count"])
    out["test_pos"] = out["label"].map(SUPPORT["positive_count"])
    return out


XRV = add_support(XRV)
DENSE = add_support(DENSE)


def scatter_figure():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    cmap = plt.cm.viridis_r
    norm = mcolors.LogNorm(vmin=1, vmax=max(SUPPORT["negative_count"]))

    for row_idx, (family_name, df) in enumerate([("XRV linear-head", XRV), ("DenseNet", DENSE)]):
        for col_idx, topk in enumerate(TOPKS):
            ax = axes[row_idx, col_idx]
            x = df[f"top{topk}_entry_neg_pct"] - df[f"top{topk}_sample_neg_pct"]
            y = df[f"top{topk}_entry_auroc"] - df[f"top{topk}_sample_auroc"]
            neg = df["test_neg"]
            ax.axhline(0, color="0.75", lw=1)
            ax.axvline(0, color="0.75", lw=1)
            sc = ax.scatter(
                x,
                y,
                c=neg,
                cmap=cmap,
                norm=norm,
                s=70,
                edgecolor="black",
                linewidth=0.4,
            )
            for _, row in df.iterrows():
                ax.annotate(
                    SHORT[row["label"]],
                    (
                        row[f"top{topk}_entry_neg_pct"] - row[f"top{topk}_sample_neg_pct"],
                        row[f"top{topk}_entry_auroc"] - row[f"top{topk}_sample_auroc"],
                    ),
                    fontsize=8,
                    xytext=(4, 4),
                    textcoords="offset points",
                )
            ax.set_title(f"{family_name} | top{topk}")
            ax.set_xlabel("Entry neg removal % - Sample neg removal %")
            ax.set_ylabel("Entry AUROC - Sample AUROC")
    cbar = fig.colorbar(sc, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("Test negative count")
    fig.suptitle("Option 1: Summary scatter", fontsize=16)
    out = ROOT / "option1_summary_scatter.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def small_multiples(df: pd.DataFrame, family_name: str, filename: str):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    axes = axes.ravel()
    xs = np.array([5, 10, 20])
    for ax, (_, row) in zip(axes, df.iterrows()):
        sample_neg = np.array([row[f"top05_sample_neg_pct"], row[f"top10_sample_neg_pct"], row[f"top20_sample_neg_pct"]])
        entry_neg = np.array([row[f"top05_entry_neg_pct"], row[f"top10_entry_neg_pct"], row[f"top20_entry_neg_pct"]])
        sample_auc = np.array([row[f"top05_sample_auroc"], row[f"top10_sample_auroc"], row[f"top20_sample_auroc"]])
        entry_auc = np.array([row[f"top05_entry_auroc"], row[f"top10_entry_auroc"], row[f"top20_entry_auroc"]])

        ax2 = ax.twinx()
        ax.plot(xs, sample_neg, marker="o", color="#1f77b4", label="sample neg %")
        ax.plot(xs, entry_neg, marker="o", color="#d62728", label="entry neg %")
        ax2.plot(xs, sample_auc, marker="s", linestyle="--", color="#1f77b4", alpha=0.75, label="sample AUROC")
        ax2.plot(xs, entry_auc, marker="s", linestyle="--", color="#d62728", alpha=0.75, label="entry AUROC")

        ax.set_title(f"{row['label']} | test {int(row['test_pos'])}/{int(row['test_neg'])}", fontsize=10)
        ax.set_xticks(xs)
        ax.set_xlabel("top-k (%)")
        ax.set_ylabel("negative removal %")
        ax2.set_ylabel("AUROC")
        ax.grid(alpha=0.2)

    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[0].twinx().get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(f"Option 2: Per-label dual-axis small multiples | {family_name}", fontsize=16)
    out = ROOT / filename
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def heatmap_figure():
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
    families = [("XRV linear-head", XRV), ("DenseNet", DENSE)]
    for row_idx, (family_name, df) in enumerate(families):
        # neg removal gap heatmap
        neg_gap = np.array(
            [[row[f"top{topk}_entry_neg_pct"] - row[f"top{topk}_sample_neg_pct"] for topk in TOPKS] for _, row in df.iterrows()]
        )
        auc_gap = np.array(
            [[row[f"top{topk}_entry_auroc"] - row[f"top{topk}_sample_auroc"] for topk in TOPKS] for _, row in df.iterrows()]
        )
        neg_ax = axes[row_idx, 0]
        auc_ax = axes[row_idx, 1]
        ratio_ax = axes[row_idx, 2]
        im1 = neg_ax.imshow(neg_gap, aspect="auto", cmap="OrRd")
        im2 = auc_ax.imshow(auc_gap, aspect="auto", cmap="PuOr", vmin=-0.3, vmax=0.3)
        ratio = np.array([[row["test_neg"] for _ in TOPKS] for _, row in df.iterrows()])
        im3 = ratio_ax.imshow(ratio, aspect="auto", cmap="Blues")
        for ax, title in [
            (neg_ax, f"{family_name} | Entry-Sample negative removal gap"),
            (auc_ax, f"{family_name} | Entry-Sample AUROC gap"),
            (ratio_ax, f"{family_name} | Test negative count"),
        ]:
            ax.set_title(title, fontsize=11)
            ax.set_xticks(range(len(TOPKS)), [f"top{t}" for t in TOPKS])
            ax.set_yticks(range(len(LABELS)), [SHORT[l] for l in LABELS])
        fig.colorbar(im1, ax=neg_ax, shrink=0.8)
        fig.colorbar(im2, ax=auc_ax, shrink=0.8)
        fig.colorbar(im3, ax=ratio_ax, shrink=0.8)
    fig.suptitle("Option 3: Compact heatmaps", fontsize=16)
    out = ROOT / "option3_compact_heatmaps.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    scatter_figure()
    small_multiples(XRV, "XRV linear-head", "option2_small_multiples_xrv.png")
    small_multiples(DENSE, "DenseNet", "option2_small_multiples_densenet.png")
    heatmap_figure()


if __name__ == "__main__":
    main()
