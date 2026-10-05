#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from export_sample_topk_entries_for_llm import (
    LABEL_NAMES,
    expand_sample_rows,
    rank_samples,
)
from select_topk_unreviewed_issue_samples import (
    load_review_history,
    select_unreviewed_samples,
)


def pipe(values: list[object]) -> str:
    return "|".join("nan" if value is None else str(value) for value in values)


def sample_row(pool_row_id: int, rank: int, issue_indices: list[int]) -> dict[str, object]:
    raw = [None] * len(LABEL_NAMES)
    binary = [None] * len(LABEL_NAMES)
    valid = [0] * len(LABEL_NAMES)
    probs = [None] * len(LABEL_NAMES)
    issue = [0] * len(LABEL_NAMES)
    for label_index in issue_indices:
        raw[label_index] = 0.0
        binary[label_index] = 0.0
        valid[label_index] = 1
        probs[label_index] = 0.99 - rank / 1000
        issue[label_index] = 1
    return {
        "sample_local_index": pool_row_id,
        "pool_row_id": pool_row_id,
        "subject_id": 1000 + pool_row_id,
        "study_id": 2000 + pool_row_id,
        "dicom_id": f"dicom-{pool_row_id}",
        "image_path": f"image-{pool_row_id}.jpg",
        "est_issue_sample": 1,
        "est_issue_entry_count": len(issue_indices),
        "issue_rank_self_confidence": float(rank),
        "issue_rank_normalized_margin": float(rank),
        "issue_rank_confidence_weighted_entropy": float(rank),
        "sample_quality_self_confidence": rank / 100,
        "sample_quality_normalized_margin": rank / 100,
        "sample_quality_confidence_weighted_entropy": rank / 100,
        "raw_labels_4class": pipe(raw),
        "binary_labels_for_detection": pipe(binary),
        "valid_label_mask": pipe(valid),
        "pred_probs": pipe(probs),
        "issue_entry_mask": pipe(issue),
    }


def main() -> None:
    issues = pd.DataFrame(
        [
            sample_row(1, 1, [0]),
            sample_row(2, 2, [0, 1]),
            sample_row(3, 3, [0]),
            sample_row(4, 4, [0]),
            sample_row(5, 5, [0]),
            sample_row(6, 6, [0]),
        ]
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        run_root = Path(temp_dir)
        history_dir = run_root / "loop_01"
        history_dir.mkdir()
        pd.DataFrame(
            {"entry_key": ["1::0", "2::0", "4::0"]}
        ).to_csv(
            history_dir / "sample_top_fraction_expanded_entries.csv",
            index=False,
        )
        history, per_loop = load_review_history(run_root, current_loop=2)

    assert history == {"1::0", "2::0", "4::0"}
    assert per_loop[0]["history_rows"] == 3

    with tempfile.TemporaryDirectory() as temp_dir:
        run_root = Path(temp_dir)
        results_dir = run_root / "loop_01" / "llm_review"
        results_dir.mkdir(parents=True)
        pd.DataFrame({"entry_key": ["2::1", "3::0"]}).to_csv(
            results_dir / "results.csv",
            index=False,
        )
        successful_history, successful_per_loop = load_review_history(
            run_root,
            current_loop=2,
            history_source="successful-results",
        )
    assert successful_history == {"2::1", "3::0"}
    assert successful_per_loop[0]["history_rows"] == 2

    empty_history, empty_per_loop = load_review_history(
        Path("/not/read/for/loop1"),
        current_loop=1,
        history_source="successful-results",
    )
    assert empty_history == set()
    assert empty_per_loop == []

    selected, audit = select_unreviewed_samples(
        issues,
        history_keys=history,
        top_fraction=0.5,
    )
    assert selected["pool_row_id"].astype(int).tolist() == [2, 3, 5]
    assert selected["original_sample_rank"].astype(int).tolist() == [2, 3, 5]
    assert selected["est_issue_entry_count"].astype(int).tolist() == [1, 1, 1]
    assert selected["issue_entry_mask"].tolist()[0].startswith("0|1|")
    assert audit["target_selected_samples"] == 3
    assert audit["selected_samples"] == 3
    assert audit["samples_scanned_from_original_ranking"] == 5
    assert audit["backfill_samples_scanned_beyond_original_cutoff"] == 2
    assert audit["history_entries_filtered_inside_selected_samples"] == 1

    expanded = expand_sample_rows(rank_samples(selected))
    keys = set(expanded["entry_key"].astype(str))
    assert keys == {"2::1", "3::0", "5::0"}
    assert not keys.intersection(history)
    print("unreviewed top-fraction selection tests passed")


if __name__ == "__main__":
    main()
