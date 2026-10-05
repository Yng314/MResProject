#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from export_sample_topk_entries_for_llm import expand_sample_rows, rank_samples
from run_entry_level_llm_review import validate_review_payload


DEFAULT_SOURCE_ROOT = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_noes50_clean_3seed/20260707_123320"
)
SEEDS = (7, 13, 42, 97, 123)
RESPONSE_COLUMNS = [
    "mismatch_type",
    "analysis_note",
    "recommended_action",
    "attempt_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a fresh own-top20 v5 root using binary-label entry expansion and "
            "strictly validated reuse of method-equivalent raw 0/1 Loop1 LLM responses."
        )
    )
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_loop(source_loop: Path, seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    summary_path = source_loop / "oof" / "oof_cleanlab_smoke_summary.csv"
    issues_path = source_loop / "oof" / "train_cleanlab_sample_issues_only.csv"
    selected_path = source_loop / "sample_top_fraction_issue_subset.csv"
    for path in [summary_path, issues_path, selected_path]:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing reusable Loop1 source artifact: {path}")

    summary = pd.read_csv(summary_path).iloc[0]
    expected_config = {
        "seed": seed,
        "n_splits": 4,
        "epochs": 100,
        "batch_size": 32,
        "model_backbone": "mobilenet_v3_small_scratch",
    }
    for field, expected in expected_config.items():
        observed = summary[field]
        if str(observed) != str(expected):
            raise ValueError(
                f"Seed {seed} reusable OOF mismatch for {field}: {observed!r} != {expected!r}"
            )

    all_issues = rank_samples(pd.read_csv(issues_path))
    selected = rank_samples(pd.read_csv(selected_path))
    expected_selected_count = int(round(len(all_issues) * 0.20))
    if len(selected) != expected_selected_count:
        raise ValueError(
            f"Seed {seed} selection count mismatch: {len(selected)} != {expected_selected_count}"
        )
    expected_ids = all_issues.head(expected_selected_count)["pool_row_id"].astype(int).tolist()
    selected_ids = selected["pool_row_id"].astype(int).tolist()
    if selected_ids != expected_ids:
        raise ValueError(f"Seed {seed} selection is not the exact ranked top 20% issue set")

    source_metadata = {
        "source_loop": str(source_loop),
        "source_oof_summary_sha256": sha256(summary_path),
        "source_issue_table_sha256": sha256(issues_path),
        "source_selection_sha256": sha256(selected_path),
        "issue_samples": int(len(all_issues)),
        "selected_samples": int(len(selected)),
    }
    return selected, source_metadata


def validate_and_import_responses(
    pilot_results_path: Path,
    expanded: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not pilot_results_path.is_file() or pilot_results_path.stat().st_size == 0:
        return pd.DataFrame(), {
            "pilot_results_path": str(pilot_results_path),
            "pilot_results_sha256": None,
            "reused_response_rows": 0,
        }

    pilot = pd.read_csv(pilot_results_path)
    required = {
        "entry_key",
        "pool_row_id",
        "label_index",
        "label_name",
        "raw_label",
        "binary_label",
        "valid_label",
        "pred_prob",
        "study_id",
        "sample_selected_rank",
        "original_report_full",
        *RESPONSE_COLUMNS,
    }
    missing = sorted(required - set(pilot.columns))
    if missing:
        raise ValueError(f"Pilot response file is missing columns {missing}: {pilot_results_path}")
    if pilot["entry_key"].astype(str).duplicated().any():
        raise ValueError(f"Pilot response file contains duplicate entry keys: {pilot_results_path}")

    for column in ["raw_label", "binary_label", "valid_label"]:
        pilot[column] = pd.to_numeric(pilot[column], errors="coerce")
    if not pilot["raw_label"].isin([0.0, 1.0]).all():
        raise ValueError("Pilot reuse is restricted to explicit raw 0/1 response rows")
    if not pilot["binary_label"].eq(pilot["raw_label"]).all():
        raise ValueError("Pilot raw 0/1 responses are not target-equivalent to their binary labels")
    if not pilot["valid_label"].eq(1).all():
        raise ValueError("Pilot response rows contain invalid labels")
    for row in pilot.itertuples(index=False):
        validate_review_payload(
            {
                "mismatch_type": row.mismatch_type,
                "analysis_note": row.analysis_note,
                "recommended_action": row.recommended_action,
            }
        )

    expanded_by_key = expanded.set_index("entry_key", drop=False)
    pilot_keys = pilot["entry_key"].astype(str)
    missing_keys = sorted(set(pilot_keys) - set(expanded_by_key.index.astype(str)))
    if missing_keys:
        raise ValueError(
            f"Pilot response keys are absent from corrected expansion: {missing_keys[:10]}"
        )

    pilot_by_key = pilot.assign(entry_key=pilot_keys).set_index("entry_key", drop=False)
    new_rows = expanded_by_key.loc[pilot_keys].reset_index(drop=True)
    old_rows = pilot_by_key.loc[pilot_keys].reset_index(drop=True)
    exact_columns = [
        "pool_row_id",
        "label_index",
        "label_name",
        "raw_label",
        "binary_label",
        "valid_label",
        "study_id",
        "subject_id",
        "dicom_id",
        "image_path",
        "sample_selected_rank",
    ]
    for column in exact_columns:
        if not new_rows[column].astype(str).eq(old_rows[column].astype(str)).all():
            raise ValueError(f"Pilot/new source mismatch in {column}: {pilot_results_path}")
    if not pd.to_numeric(new_rows["pred_prob"]).sub(
        pd.to_numeric(old_rows["pred_prob"])
    ).abs().le(1e-12).all():
        raise ValueError(f"Pilot/new OOF probability mismatch: {pilot_results_path}")

    imported = new_rows.copy()
    imported["original_report_full"] = old_rows["original_report_full"].tolist()
    for column in RESPONSE_COLUMNS:
        imported[column] = old_rows[column].tolist()

    expected_columns = list(expanded.columns) + ["original_report_full"] + RESPONSE_COLUMNS
    imported = imported[expected_columns]
    if imported["entry_key"].duplicated().any():
        raise ValueError("Imported response rows contain duplicate entry keys")

    metadata = {
        "pilot_results_path": str(pilot_results_path),
        "pilot_results_sha256": sha256(pilot_results_path),
        "reused_response_rows": int(len(imported)),
        "reuse_scope": "Loop1 raw 0/1 rows only",
        "reuse_equivalence": (
            "raw label equals post-binarization target; report, OOF probability, label, and "
            "entry key match the corrected expansion"
        ),
    }
    return imported, metadata


def main() -> None:
    args = parse_args()
    pilot_root = args.pilot_root.resolve()
    source_root = args.source_root.resolve()
    new_root = args.new_root
    if not pilot_root.is_dir():
        raise ValueError(f"Pilot root does not exist: {pilot_root}")
    if new_root.exists():
        raise ValueError(f"Refusing to overwrite existing v5 root: {new_root}")

    new_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = new_root.with_name(f".{new_root.name}.preparing")
    if staging_root.exists():
        raise ValueError(f"Preparation staging path already exists: {staging_root}")
    staging_root.mkdir(parents=True)

    seed_records: list[dict[str, object]] = []
    for seed in SEEDS:
        source_loop = source_root / f"seed_{seed}" / "sample20_remove_loop" / "remove_only" / "loop_01"
        selected, source_metadata = validate_source_loop(source_loop, seed)
        expanded = expand_sample_rows(selected)

        loop_dir = staging_root / f"seed_{seed}" / "llm_refine" / "loop_01"
        loop_dir.mkdir(parents=True)
        (loop_dir / "oof").symlink_to((source_loop / "oof").resolve(), target_is_directory=True)
        (loop_dir / "sample_top_fraction_issue_subset.csv").symlink_to(
            (source_loop / "sample_top_fraction_issue_subset.csv").resolve()
        )
        (loop_dir / ".oof_complete").touch()
        (loop_dir / ".selection_complete").touch()

        expanded_path = loop_dir / "sample_top_fraction_expanded_entries.csv"
        expanded.to_csv(expanded_path, index=False)
        (loop_dir / ".entry_expansion_complete").touch()

        pilot_results = (
            pilot_root
            / f"seed_{seed}"
            / "llm_refine"
            / "loop_01"
            / "llm_review"
            / "results.csv"
        )
        imported, reuse_metadata = validate_and_import_responses(pilot_results, expanded)
        if not imported.empty:
            review_dir = loop_dir / "llm_review"
            review_dir.mkdir()
            imported.to_csv(review_dir / "results.csv", index=False)

        selected_count = int(len(selected))
        represented_count = int(expanded["pool_row_id"].nunique())
        if represented_count != selected_count:
            raise AssertionError(
                f"Seed {seed} corrected expansion coverage is {represented_count}/{selected_count}"
            )
        reused_count = int(reuse_metadata["reused_response_rows"])
        record = {
            "seed": seed,
            **source_metadata,
            "expanded_binary_entries": int(len(expanded)),
            "raw_uncertain_entries": int(expanded["raw_label"].eq(-1.0).sum()),
            "raw_explicit_entries": int(expanded["raw_label"].isin([0.0, 1.0]).sum()),
            "represented_selected_samples": represented_count,
            "selected_sample_review_coverage": represented_count / selected_count,
            **reuse_metadata,
            "new_llm_rows_pending": int(len(expanded) - reused_count),
            "new_expansion_sha256": sha256(expanded_path),
        }
        seed_records.append(record)
        print(
            f"seed {seed}: selected={selected_count}, expanded={len(expanded)}, "
            f"raw_-1={record['raw_uncertain_entries']}, reused={reused_count}, "
            f"pending={record['new_llm_rows_pending']}"
        )

    records_df = pd.DataFrame(seed_records)
    records_df.to_csv(staging_root / "binary_v5_reuse_manifest.csv", index=False)
    manifest = {
        "protocol_version": "own_top20_binary_v5",
        "prepared_at": datetime.now().astimezone().isoformat(),
        "pilot_root": str(pilot_root),
        "source_root": str(source_root),
        "new_root": str(new_root),
        "label_basis": "post-binarization binary label",
        "uncertain_projection": "raw -1 -> binary 1",
        "pilot_status": "invalidated; no action, training, metric, or outcome reused",
        "response_reuse_policy": (
            "Only Loop1 raw 0/1 responses with identical binary target, entry key, label, "
            "report, OOF probability, and source identity are reused."
        ),
        "seeds": seed_records,
    }
    (staging_root / "binary_v5_reuse_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    os.rename(staging_root, new_root)
    print(f"Prepared corrected v5 root: {new_root}")


if __name__ == "__main__":
    main()
