#!/usr/bin/env python3
"""Run the outcome-informed four-seed sensitivity analysis excluding seed 7."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluate_unified_5seed_8loop import (
    CONTRASTS,
    DEFAULT_BASE_ROOT,
    DEFAULT_LOOPS,
    DEFAULT_REFINE_ROOT,
    PERFORMANCE_DIRECTIONS,
    QUALITY_DIRECTIONS,
    build_checks,
    build_quality_outputs,
    contrast_tuples,
    hierarchical_bootstrap,
    paired_metric_statistics,
    plot_bootstrap_forest,
    plot_coverage,
    plot_performance,
    plot_quality,
    summarize_metrics,
    summarize_per_label,
    summarize_quality,
    validate_and_collect,
)


INCLUDED_SEEDS = [13, 42, 97, 123]
EXCLUDED_SEED = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--refine-root", type=Path, default=DEFAULT_REFINE_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_root = args.base_root.resolve()
    refine_root = args.refine_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    performance, label_rows, label_names, qa = validate_and_collect(
        base_root=base_root,
        refine_root=refine_root,
        refine_layout="own",
        seeds=INCLUDED_SEEDS,
        remove_loops=DEFAULT_LOOPS,
        refine_loops=DEFAULT_LOOPS,
    )
    performance_summary = summarize_metrics(performance)
    performance_stats = paired_metric_statistics(
        performance,
        PERFORMANCE_DIRECTIONS,
    )
    comparisons = contrast_tuples()
    bootstrap = hierarchical_bootstrap(
        base_root=base_root,
        refine_root=refine_root,
        refine_layout="own",
        seeds=INCLUDED_SEEDS,
        comparisons=comparisons,
        label_names=label_names,
        label_rows=label_rows,
        n_bootstrap=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
        workers=args.workers,
    )
    per_label = summarize_per_label(label_rows, comparisons)

    quality, quality_labels, actions = build_quality_outputs(
        base_root,
        refine_root,
        INCLUDED_SEEDS,
        DEFAULT_LOOPS,
        args.workers,
    )
    quality_summary = summarize_quality(quality)
    quality_stats = paired_metric_statistics(quality, QUALITY_DIRECTIONS)

    checks = build_checks(
        performance=performance,
        labels=label_rows,
        quality=quality,
        quality_labels=quality_labels,
        qa=qa,
        bootstrap=bootstrap,
        base_root=base_root,
        refine_root=refine_root,
        seeds=INCLUDED_SEEDS,
        loops=DEFAULT_LOOPS,
    )
    if not checks["status"].eq("passed").all():
        raise RuntimeError(
            f"Post-hoc four-seed checks failed:\n{checks.to_string(index=False)}"
        )

    performance.to_csv(out_dir / "performance_per_seed_stage.csv", index=False)
    performance_summary.to_csv(
        out_dir / "performance_four_seed_summary.csv",
        index=False,
    )
    performance_stats.to_csv(
        out_dir / "performance_paired_statistics.csv",
        index=False,
    )
    bootstrap.to_csv(
        out_dir / "performance_hierarchical_bootstrap.csv",
        index=False,
    )
    per_label.to_csv(out_dir / "performance_per_label_deltas.csv", index=False)
    quality.to_csv(out_dir / "quality_per_seed_stage.csv", index=False)
    quality_summary.to_csv(out_dir / "quality_four_seed_summary.csv", index=False)
    quality_stats.to_csv(out_dir / "quality_paired_statistics.csv", index=False)
    quality_labels.to_csv(out_dir / "quality_per_label.csv", index=False)
    actions.to_csv(out_dir / "refinement_action_oof_support.csv", index=False)
    checks.to_csv(out_dir / "validation_checks.csv", index=False)

    plot_performance(performance_summary, out_dir)
    plot_quality(quality_summary, out_dir)
    plot_coverage(quality_summary, out_dir)
    plot_bootstrap_forest(bootstrap, out_dir)

    metadata = {
        "status": "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(
            Path(__file__).with_name(
                "posthoc_4seed_8loop_excluding_seed7_protocol_20260723.md"
            ).resolve()
        ),
        "analysis_label": "post-hoc four-seed sensitivity",
        "base_root": str(base_root),
        "refine_root": str(refine_root),
        "included_seeds": INCLUDED_SEEDS,
        "excluded_seed": EXCLUDED_SEED,
        "exclusion_timing": "after inspection of complete five-seed outcomes",
        "exclusion_validity_reason": None,
        "exclusion_reason": "user-requested outcome-informed sensitivity analysis",
        "loops": DEFAULT_LOOPS,
        "primary_endpoint": "study_weighted_auroc",
        "primary_comparison": CONTRASTS[0].name,
        "contrasts": [
            {
                "name": item.name,
                "left": list(item.left),
                "right": list(item.right),
                "tier": item.tier,
            }
            for item in CONTRASTS
        ],
        "bootstrap_iterations": args.bootstrap_iters,
        "bootstrap_seed": args.bootstrap_seed,
        "shared_study_resamples_across_seeds": True,
        "quality_evidence": "unchanged seed-specific initial OOF probabilities",
        "multiplicity": "Holm across eight frozen contrasts within each metric",
        "inference_status": (
            "post-hoc outcome-informed sensitivity; not a replacement for "
            "the formal five-seed analysis"
        ),
        "checks": checks.to_dict("records"),
    }
    (out_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / ".evaluation_complete").touch()

    print(json.dumps(metadata, indent=2))
    print("\nPrimary performance statistics:")
    print(
        performance_stats.loc[
            performance_stats["comparison"].eq(CONTRASTS[0].name)
        ].to_string(index=False)
    )
    print("\nPrimary hierarchical AUROC statistics:")
    print(
        bootstrap.loc[
            bootstrap["comparison"].astype(str).eq(CONTRASTS[0].name)
            & bootstrap["uncertainty"].eq("hierarchical_seed_and_study")
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
