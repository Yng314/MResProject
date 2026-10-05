#!/usr/bin/env python3
"""Evaluate pre-locked own-top20 refinement checkpoints across five seeds."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


DEFAULT_BASE_ROOT = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_noes50_clean_3seed/20260707_123320"
)
DEFAULT_REFINE_ROOT = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_own_top20_llm_refine_5seed/20260713_184706"
)
SEEDS = [7, 13, 42, 97, 123]
KEY_COLUMNS = [
    "study_id",
    "subject_id",
    "n_images_in_study",
    "binary_labels_for_metric",
    "valid_label_mask",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--refine-root", type=Path, default=DEFAULT_REFINE_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--remove-loops", nargs="+", type=int, default=list(range(1, 6)))
    parser.add_argument("--refine-loops", nargs="+", type=int, default=list(range(1, 9)))
    parser.add_argument("--checkpoint-loop", type=int, default=5)
    parser.add_argument("--extension-loop", type=int, default=8)
    parser.add_argument("--remove-comparator-loop", type=int, default=5)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--refine-layout",
        choices=["own", "fixed"],
        default="own",
        help="own: REFINE_ROOT/seed_N/llm_refine; fixed: REFINE_ROOT/seed_N/sample20_llm_refine_fixed_xrv_s13",
    )
    return parser.parse_args()


def parse_pipe_matrix(values: pd.Series, dtype: type) -> np.ndarray:
    return np.vstack([np.fromstring(str(value), sep="|", dtype=dtype) for value in values])


def prediction_path(
    base_root: Path,
    refine_root: Path,
    refine_layout: str,
    seed: int,
    method: str,
    loop: int,
) -> Path:
    if method == "baseline":
        return base_root / f"seed_{seed}" / "baseline_no_clean" / "test_study_predictions.csv"
    if method == "remove":
        return (
            base_root
            / f"seed_{seed}"
            / "sample20_remove_loop"
            / "remove_only"
            / f"loop_{loop:02d}"
            / "train_eval"
            / "test_study_predictions.csv"
        )
    refine_branch = (
        "llm_refine" if refine_layout == "own" else "sample20_llm_refine_fixed_xrv_s13"
    )
    return (
        refine_root
        / f"seed_{seed}"
        / refine_branch
        / f"loop_{loop:02d}"
        / "train_eval"
        / "test_study_predictions.csv"
    )


def stage_specs(remove_loops: list[int], refine_loops: list[int]) -> list[tuple[str, int]]:
    return (
        [("baseline", 0)]
        + [("remove", loop) for loop in remove_loops]
        + [("refine", loop) for loop in refine_loops]
    )


def stage_name(method: str, loop: int) -> str:
    return "baseline" if method == "baseline" else f"{method}_L{loop}"


def load_prediction_arrays(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    labels = parse_pipe_matrix(frame["binary_labels_for_metric"], float)
    valid = parse_pipe_matrix(frame["valid_label_mask"], int).astype(bool)
    probs = parse_pipe_matrix(frame["pred_probs"], float)
    return frame, labels, valid, probs


def metric_row(labels: np.ndarray, valid: np.ndarray, probs: np.ndarray) -> dict[str, float | int]:
    aucs: list[float] = []
    aps: list[float] = []
    briers: list[float] = []
    nlls: list[float] = []
    weights: list[int] = []
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    for label_index in range(labels.shape[1]):
        mask = valid[:, label_index]
        y_true = labels[mask, label_index].astype(int)
        y_prob = probs[mask, label_index]
        if y_true.size == 0 or np.unique(y_true).size < 2:
            continue
        clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
        aucs.append(float(roc_auc_score(y_true, y_prob)))
        aps.append(float(average_precision_score(y_true, y_prob)))
        briers.append(float(np.mean((y_prob - y_true) ** 2)))
        nlls.append(float(log_loss(y_true, clipped, labels=[0, 1])))
        weights.append(int(mask.sum()))
        all_labels.append(y_true)
        all_probs.append(y_prob)
    y_flat = np.concatenate(all_labels)
    p_flat = np.concatenate(all_probs)
    return {
        "study_weighted_auroc": float(np.average(aucs, weights=weights)),
        "study_macro_auroc": float(np.mean(aucs)),
        "macro_average_precision": float(np.mean(aps)),
        "micro_brier": float(np.mean((p_flat - y_flat) ** 2)),
        "micro_nll": float(
            log_loss(y_flat, np.clip(p_flat, 1e-7, 1 - 1e-7), labels=[0, 1])
        ),
        "valid_test_entries": int(len(y_flat)),
    }


def validate_and_collect(
    *,
    base_root: Path,
    refine_root: Path,
    refine_layout: str,
    seeds: list[int],
    remove_loops: list[int],
    refine_loops: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, int | list]]:
    specs = stage_specs(remove_loops, refine_loops)
    metric_rows: list[dict] = []
    label_rows: list[dict] = []
    label_names: list[str] | None = None
    global_reference: pd.DataFrame | None = None
    files_checked = 0

    for seed in seeds:
        seed_reference: pd.DataFrame | None = None
        seed_labels: np.ndarray | None = None
        seed_valid: np.ndarray | None = None
        for method, loop in specs:
            path = prediction_path(base_root, refine_root, refine_layout, seed, method, loop)
            if not path.is_file():
                raise FileNotFoundError(path)
            frame, labels, valid, probs = load_prediction_arrays(path)
            files_checked += 1
            keys = frame[KEY_COLUMNS]
            if seed_reference is None:
                seed_reference = keys
                seed_labels = labels
                seed_valid = valid
            elif not keys.equals(seed_reference):
                raise ValueError(f"Prediction alignment mismatch for seed={seed}, {method} L{loop}")
            elif not np.array_equal(labels, seed_labels, equal_nan=True) or not np.array_equal(
                valid, seed_valid
            ):
                raise ValueError(f"Test-label mismatch for seed={seed}, {method} L{loop}")
            if global_reference is None:
                global_reference = keys
            elif not keys.equals(global_reference):
                raise ValueError(f"Shared test-set mismatch for seed={seed}, {method} L{loop}")

            metrics = metric_row(labels, valid, probs)
            metric_rows.append(
                {"seed": seed, "method": method, "loop": loop, "stage": stage_name(method, loop), **metrics}
            )

            summary_path = path.with_name("test_study_auroc_summary.csv")
            summary = pd.read_csv(summary_path).sort_values("label_index")
            current_names = summary["label_name"].astype(str).tolist()
            if label_names is None:
                label_names = current_names
            elif current_names != label_names:
                raise ValueError(f"Label-order mismatch in {summary_path}")
            if summary["label_index"].astype(int).tolist() != list(range(labels.shape[1])):
                raise ValueError(f"Unexpected label indices in {summary_path}")
            for label_index, row in enumerate(summary.itertuples(index=False)):
                mask = valid[:, label_index]
                y_true = labels[mask, label_index].astype(int)
                y_prob = probs[mask, label_index]
                valid_count = int(mask.sum())
                positive_count = int(y_true.sum())
                negative_count = int(valid_count - positive_count)
                if (
                    valid_count != int(row.study_valid_count)
                    or positive_count != int(row.study_positive_count)
                    or negative_count != int(row.study_negative_count)
                ):
                    raise ValueError(f"Saved support mismatch in {summary_path}, label {label_index}")
                study_auroc = (
                    float(roc_auc_score(y_true, y_prob))
                    if y_true.size and np.unique(y_true).size == 2
                    else np.nan
                )
                label_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "loop": loop,
                        "label_index": label_index,
                        "label_name": str(row.label_name),
                        "study_auroc": study_auroc,
                        "valid_count": valid_count,
                        "positive_count": positive_count,
                        "negative_count": negative_count,
                    }
                )

    assert global_reference is not None and label_names is not None
    qa = {
        "prediction_files_checked": files_checked,
        "studies_per_file": int(len(global_reference)),
        "duplicate_study_ids": int(global_reference["study_id"].duplicated().sum()),
        "seeds": seeds,
        "remove_loops": remove_loops,
        "refine_loops": refine_loops,
    }
    return pd.DataFrame(metric_rows), pd.DataFrame(label_rows), label_names, qa


def summarize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "study_weighted_auroc",
        "study_macro_auroc",
        "macro_average_precision",
        "micro_brier",
        "micro_nll",
        "valid_test_entries",
    ]
    rows: list[dict] = []
    for (method, loop), group in frame.groupby(["method", "loop"], sort=True):
        row: dict[str, float | int | str] = {
            "method": method,
            "loop": int(loop),
            "n_seeds": int(len(group)),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "loop"])


def exact_signflip_p(delta: np.ndarray) -> float:
    observed = abs(float(np.mean(delta)))
    permuted = [
        abs(float(np.mean(np.asarray(signs) * delta)))
        for signs in product([-1, 1], repeat=len(delta))
    ]
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


def holm_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_t_p(delta: np.ndarray) -> float:
    if np.allclose(delta, delta[0], rtol=0.0, atol=1e-15):
        return 1.0 if float(np.mean(delta)) == 0.0 else 0.0
    return float(stats.ttest_1samp(delta, 0).pvalue)


def contrast_definitions(
    checkpoint_loop: int, extension_loop: int, remove_loop: int
) -> list[tuple[str, tuple[str, int], tuple[str, int]]]:
    return [
        (
            f"refine_L{checkpoint_loop}_vs_baseline",
            ("refine", checkpoint_loop),
            ("baseline", 0),
        ),
        (
            f"refine_L{extension_loop}_vs_baseline",
            ("refine", extension_loop),
            ("baseline", 0),
        ),
        (
            f"refine_L{extension_loop}_vs_refine_L{checkpoint_loop}",
            ("refine", extension_loop),
            ("refine", checkpoint_loop),
        ),
        (
            f"refine_L{checkpoint_loop}_vs_remove_L{remove_loop}",
            ("refine", checkpoint_loop),
            ("remove", remove_loop),
        ),
    ]


def paired_seed_statistics(
    metrics: pd.DataFrame,
    comparisons: list[tuple[str, tuple[str, int], tuple[str, int]]],
) -> pd.DataFrame:
    pivot = metrics.pivot(index="seed", columns=["method", "loop"], values="study_weighted_auroc")
    rows: list[dict] = []
    for name, left, right in comparisons:
        delta = (pivot[left] - pivot[right]).to_numpy(dtype=float)
        mean = float(delta.mean())
        sd = float(delta.std(ddof=1))
        half_width = float(stats.t.ppf(0.975, len(delta) - 1) * sd / np.sqrt(len(delta)))
        rows.append(
            {
                "comparison": name,
                "n_seeds": len(delta),
                "mean_delta": mean,
                "sd_delta": sd,
                "t_ci_low_2p5": mean - half_width,
                "t_ci_high_97p5": mean + half_width,
                "paired_effect_dz": mean / sd if sd else np.nan,
                "positive_delta_seeds": int((delta > 0).sum()),
                "exact_signflip_p_two_sided": exact_signflip_p(delta),
                "paired_t_p_two_sided": paired_t_p(delta),
            }
        )
    frame = pd.DataFrame(rows)
    frame["holm4_exact_p"] = holm_adjust(frame["exact_signflip_p_two_sided"].to_numpy())
    frame["holm4_paired_t_p"] = holm_adjust(frame["paired_t_p_two_sided"].to_numpy())
    return frame


def bootstrap_one_seed(
    base_root_text: str,
    refine_root_text: str,
    refine_layout: str,
    seed: int,
    stages: list[tuple[str, int]],
    scope_indices: dict[str, list[int]],
    n_bootstrap: int,
    bootstrap_seed: int,
) -> tuple[int, np.ndarray]:
    base_root = Path(base_root_text)
    refine_root = Path(refine_root_text)
    reference, labels, valid, baseline_probs = load_prediction_arrays(
        prediction_path(base_root, refine_root, refine_layout, seed, "baseline", 0)
    )
    probabilities = [baseline_probs]
    for method, loop in stages[1:]:
        frame, other_labels, other_valid, probs = load_prediction_arrays(
            prediction_path(base_root, refine_root, refine_layout, seed, method, loop)
        )
        if not frame[KEY_COLUMNS].equals(reference[KEY_COLUMNS]):
            raise ValueError(f"Bootstrap alignment mismatch for seed={seed}, {method} L{loop}")
        if not np.array_equal(labels, other_labels, equal_nan=True) or not np.array_equal(
            valid, other_valid
        ):
            raise ValueError(f"Bootstrap test-label mismatch for seed={seed}, {method} L{loop}")
        probabilities.append(probs)
    probs_array = np.stack(probabilities)
    output = np.full((len(scope_indices), len(stages), n_bootstrap), np.nan)
    rng = np.random.default_rng(bootstrap_seed)
    for boot_index in range(n_bootstrap):
        sample = rng.integers(0, len(reference), size=len(reference))
        label_scores = np.full((len(stages), labels.shape[1]), np.nan)
        label_weights = np.zeros(labels.shape[1], dtype=int)
        for label_index in range(labels.shape[1]):
            mask = valid[sample, label_index]
            y_true = labels[sample, label_index][mask]
            if y_true.size == 0 or np.unique(y_true).size < 2:
                continue
            label_weights[label_index] = int(mask.sum())
            for stage_index in range(len(stages)):
                y_score = probs_array[stage_index, sample, label_index][mask]
                label_scores[stage_index, label_index] = roc_auc_score(y_true, y_score)
        for scope_pos, selected_list in enumerate(scope_indices.values()):
            selected = np.asarray(selected_list, dtype=int)
            finite = np.isfinite(label_scores[0, selected]) & (label_weights[selected] > 0)
            selected = selected[finite]
            output[scope_pos, :, boot_index] = np.average(
                label_scores[:, selected], axis=1, weights=label_weights[selected]
            )
    return seed, output


def bootstrap_one_seed_from_args(args: tuple) -> tuple[int, np.ndarray]:
    return bootstrap_one_seed(*args)


def observed_scope_contrasts(
    label_rows: pd.DataFrame,
    scope_indices: dict[str, list[int]],
    comparisons: list[tuple[str, tuple[str, int], tuple[str, int]]],
) -> dict[tuple[str, str], float]:
    observed: dict[tuple[str, str], float] = {}
    for scope, selected_indices in scope_indices.items():
        scoped = label_rows.loc[label_rows["label_index"].isin(selected_indices)].copy()
        score_rows: list[dict[str, float | int | str]] = []
        for (seed, method, loop), group in scoped.groupby(
            ["seed", "method", "loop"], sort=False
        ):
            usable = group.loc[
                np.isfinite(group["study_auroc"]) & group["valid_count"].gt(0)
            ]
            if usable.empty:
                raise ValueError(
                    f"No evaluable labels for scope={scope}, seed={seed}, {method} L{loop}"
                )
            score_rows.append(
                {
                    "seed": int(seed),
                    "method": str(method),
                    "loop": int(loop),
                    "weighted_auroc": float(
                        np.average(usable["study_auroc"], weights=usable["valid_count"])
                    ),
                }
            )
        scores = pd.DataFrame(score_rows)
        pivot = scores.pivot(
            index="seed", columns=["method", "loop"], values="weighted_auroc"
        )
        for name, left, right in comparisons:
            delta = pivot[left] - pivot[right]
            if delta.isna().any():
                raise ValueError(f"Missing observed score for scope={scope}, comparison={name}")
            observed[(scope, name)] = float(delta.mean())
    return observed


def hierarchical_bootstrap(
    *,
    base_root: Path,
    refine_root: Path,
    refine_layout: str,
    seeds: list[int],
    comparisons: list[tuple[str, tuple[str, int], tuple[str, int]]],
    label_names: list[str],
    label_rows: pd.DataFrame,
    n_bootstrap: int,
    bootstrap_seed: int,
    workers: int,
) -> pd.DataFrame:
    stages: list[tuple[str, int]] = [("baseline", 0)]
    for _, left, right in comparisons:
        for stage in [left, right]:
            if stage not in stages:
                stages.append(stage)
    support = label_rows.loc[
        (label_rows["seed"] == seeds[0]) & (label_rows["method"] == "baseline")
    ].sort_values("label_index")
    minority = support[["positive_count", "negative_count"]].min(axis=1).to_numpy()
    scope_indices = {
        "all_labels": list(range(len(label_names))),
        "support_ge_5": np.flatnonzero(minority >= 5).astype(int).tolist(),
    }
    observed = observed_scope_contrasts(label_rows, scope_indices, comparisons)
    worker_args = [
        (
            str(base_root),
            str(refine_root),
            refine_layout,
            seed,
            stages,
            scope_indices,
            n_bootstrap,
            bootstrap_seed,
        )
        for seed in seeds
    ]
    if workers == 1:
        results = [bootstrap_one_seed(*item) for item in worker_args]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(seeds))) as executor:
            results = list(executor.map(bootstrap_one_seed_from_args, worker_args))
    results.sort(key=lambda item: seeds.index(item[0]))
    boot = np.stack([item[1] for item in results])
    stage_pos = {stage: index for index, stage in enumerate(stages)}
    seed_rng = np.random.default_rng(bootstrap_seed + 1)
    seed_draws = seed_rng.integers(0, len(seeds), size=(n_bootstrap, len(seeds)))
    rows: list[dict] = []
    for scope_pos, scope in enumerate(scope_indices):
        family: list[dict] = []
        for name, left, right in comparisons:
            delta_by_seed = (
                boot[:, scope_pos, stage_pos[left], :] - boot[:, scope_pos, stage_pos[right], :]
            )
            for uncertainty, values in [
                ("conditional_shared_study", np.nanmean(delta_by_seed, axis=0)),
                (
                    "hierarchical_seed_and_study",
                    np.asarray(
                        [
                            np.nanmean(delta_by_seed[seed_draws[index], index])
                            for index in range(n_bootstrap)
                        ]
                    ),
                ),
            ]:
                values = values[np.isfinite(values)]
                p_le = float((np.sum(values <= 0) + 1) / (len(values) + 1))
                p_ge = float((np.sum(values >= 0) + 1) / (len(values) + 1))
                family.append(
                    {
                        "scope": scope,
                        "comparison": name,
                        "uncertainty": uncertainty,
                        "n_seeds": len(seeds),
                        "n_bootstrap": n_bootstrap,
                        "valid_bootstrap": len(values),
                        "observed_mean_delta": observed[(scope, name)],
                        "bootstrap_mean_delta": float(np.mean(values)),
                        "ci_low_2p5": float(np.percentile(values, 2.5)),
                        "ci_high_97p5": float(np.percentile(values, 97.5)),
                        "bootstrap_probability_delta_le_zero": p_le,
                        "bootstrap_probability_delta_ge_zero": p_ge,
                    }
                )
        rows.extend(family)
    return pd.DataFrame(rows).sort_values(["scope", "uncertainty", "comparison"])


def summarize_per_label(
    label_rows: pd.DataFrame,
    comparisons: list[tuple[str, tuple[str, int], tuple[str, int]]],
) -> pd.DataFrame:
    pivot = label_rows.pivot(
        index=["seed", "label_index", "label_name", "valid_count", "positive_count", "negative_count"],
        columns=["method", "loop"],
        values="study_auroc",
    )
    rows: list[dict] = []
    for name, left, right in comparisons:
        delta = (pivot[left] - pivot[right]).rename("delta").reset_index()
        for keys, group in delta.groupby(
            ["label_index", "label_name", "valid_count", "positive_count", "negative_count"]
        ):
            rows.append(
                {
                    "comparison": name,
                    "label_index": int(keys[0]),
                    "label_name": str(keys[1]),
                    "valid_count": int(keys[2]),
                    "positive_count": int(keys[3]),
                    "negative_count": int(keys[4]),
                    "minority_support": int(min(keys[3], keys[4])),
                    "mean_delta": float(group["delta"].mean()),
                    "sd_delta": float(group["delta"].std(ddof=1)),
                    "positive_delta_seeds": int((group["delta"] > 0).sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["comparison", "mean_delta"], ascending=[True, False])


def write_plots(
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    checkpoint_loop: int,
    extension_loop: int,
    out_dir: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 4.8), dpi=200)
    baseline = summary.loc[summary["method"].eq("baseline")].iloc[0]
    axis.axhline(
        baseline["study_weighted_auroc_mean"],
        color="#4F5663",
        linestyle="--",
        linewidth=1.6,
        label=f"Baseline ({baseline['study_weighted_auroc_mean']:.3f})",
    )
    for method, color, label in [
        ("remove", "#2B73B6", "Simple removal"),
        ("refine", "#C6372F", "Own-top20 LLM refinement"),
    ]:
        frame = summary.loc[summary["method"].eq(method)].sort_values("loop")
        axis.errorbar(
            frame["loop"],
            frame["study_weighted_auroc_mean"],
            yerr=frame["study_weighted_auroc_sd"],
            marker="o",
            linewidth=2.4,
            elinewidth=1.8,
            capsize=5,
            color=color,
            label=label,
        )
    axis.axvline(checkpoint_loop, color="#A9AFB8", linestyle=":", linewidth=1)
    axis.axvline(extension_loop, color="#A9AFB8", linestyle=":", linewidth=1)
    axis.set_xlabel("Cleaning loop")
    axis.set_ylabel("Study-weighted AUROC")
    axis.set_title("Pre-locked five-seed own-top20 refinement trajectory", fontweight="bold")
    axis.grid(axis="y", color="#DFE4EA")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "own_top20_five_seed_trajectory.png", bbox_inches="tight")
    plt.close(fig)

    frame = bootstrap.loc[
        bootstrap["scope"].eq("all_labels")
        & bootstrap["uncertainty"].eq("hierarchical_seed_and_study")
    ].copy()
    frame = frame.iloc[::-1]
    y = np.arange(len(frame))
    values = frame["observed_mean_delta"].to_numpy()
    lower = values - frame["ci_low_2p5"].to_numpy()
    upper = frame["ci_high_97p5"].to_numpy() - values
    fig, axis = plt.subplots(figsize=(9.2, 4.8), dpi=200)
    axis.axvline(0, color="#4F5663", linewidth=1)
    axis.errorbar(
        values,
        y,
        xerr=[lower, upper],
        fmt="o",
        color="#2E5C92",
        capsize=5,
        linewidth=2,
    )
    axis.set_yticks(y, frame["comparison"])
    axis.set_xlabel("Mean study-weighted AUROC difference")
    axis.set_title("Hierarchical seed + study uncertainty", fontweight="bold")
    axis.grid(axis="x", color="#DFE4EA")
    fig.tight_layout()
    fig.savefig(out_dir / "own_top20_prelocked_endpoint_forest.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    seeds = sorted(args.seeds)
    remove_loops = sorted(set(args.remove_loops))
    refine_loops = sorted(set(args.refine_loops))
    for required in [args.checkpoint_loop, args.extension_loop]:
        if required not in refine_loops:
            raise ValueError(f"Required refinement endpoint Loop {required} is not in refine-loops")
    if args.remove_comparator_loop not in remove_loops:
        raise ValueError("remove-comparator-loop is not in remove-loops")
    if len(seeds) != 5:
        raise ValueError("The pre-locked analysis requires exactly five seeds")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics, label_rows, label_names, qa = validate_and_collect(
        base_root=args.base_root,
        refine_root=args.refine_root,
        refine_layout=args.refine_layout,
        seeds=seeds,
        remove_loops=remove_loops,
        refine_loops=refine_loops,
    )
    comparisons = contrast_definitions(
        args.checkpoint_loop, args.extension_loop, args.remove_comparator_loop
    )
    summary = summarize_metrics(metrics)
    seed_stats = paired_seed_statistics(metrics, comparisons)
    bootstrap = hierarchical_bootstrap(
        base_root=args.base_root,
        refine_root=args.refine_root,
        refine_layout=args.refine_layout,
        seeds=seeds,
        comparisons=comparisons,
        label_names=label_names,
        label_rows=label_rows,
        n_bootstrap=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
        workers=args.workers,
    )
    all_label_points = bootstrap.loc[
        bootstrap["scope"].eq("all_labels")
        & bootstrap["uncertainty"].eq("hierarchical_seed_and_study"),
        ["comparison", "observed_mean_delta"],
    ]
    point_check = seed_stats[["comparison", "mean_delta"]].merge(
        all_label_points, on="comparison", validate="one_to_one"
    )
    if not np.allclose(
        point_check["mean_delta"], point_check["observed_mean_delta"], atol=1e-12, rtol=0
    ):
        raise AssertionError("All-label scope point estimates do not match seed statistics")
    per_label = summarize_per_label(label_rows, comparisons)

    metrics.to_csv(args.out_dir / "per_seed_stage_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "five_seed_stage_summary.csv", index=False)
    seed_stats.to_csv(args.out_dir / "prelocked_seed_paired_statistics.csv", index=False)
    bootstrap.to_csv(args.out_dir / "prelocked_hierarchical_bootstrap.csv", index=False)
    per_label.to_csv(args.out_dir / "prelocked_per_label_deltas.csv", index=False)
    metadata = {
        **qa,
        "status": "passed",
        "base_root": str(args.base_root),
        "refine_root": str(args.refine_root),
        "refine_layout": args.refine_layout,
        "checkpoint_loop": args.checkpoint_loop,
        "extension_loop": args.extension_loop,
        "remove_comparator_loop": args.remove_comparator_loop,
        "prelocked_comparisons": [item[0] for item in comparisons],
        "bootstrap_iterations": args.bootstrap_iters,
        "bootstrap_seed": args.bootstrap_seed,
        "shared_study_resamples_across_seeds": True,
        "multiplicity": "Holm adjustment over the four pre-locked headline contrasts",
        "bootstrap_tail_mass_status": "descriptive empirical tail mass, not a hypothesis-test p-value",
        "formal_p_values": "exact paired seed sign-flip and paired t tests, Holm-adjusted across four contrasts",
        "interpretation": "Five seeds give limited exact-test resolution; report effects and hierarchical intervals with p-values",
    }
    (args.out_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_plots(summary, bootstrap, args.checkpoint_loop, args.extension_loop, args.out_dir)
    print(json.dumps(metadata, indent=2))
    print("\nPre-locked seed statistics:")
    print(seed_stats.to_string(index=False))
    print("\nHierarchical all-label statistics:")
    print(
        bootstrap.loc[
            bootstrap["scope"].eq("all_labels")
            & bootstrap["uncertainty"].eq("hierarchical_seed_and_study")
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
