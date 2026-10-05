#!/usr/bin/env python3
from __future__ import annotations

import math
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from build_llm_refinement_tables import build_refinement_tables
from export_sample_topk_entries_for_llm import LABEL_NAMES, expand_sample_rows
from run_entry_level_llm_review import (
    build_user_prompt,
    filter_and_rank_entries,
    prepare_tasks,
)


def pipe(values: list[object]) -> str:
    return "|".join("nan" if value is None else str(value) for value in values)


def make_sample_row(
    *,
    pool_row_id: int,
    study_id: int,
    selected_rank: int,
    raw_prefix: list[float],
) -> dict[str, object]:
    raw = raw_prefix + [None] * (len(LABEL_NAMES) - len(raw_prefix))
    binary = [0.0 if value == 0.0 else 1.0 for value in raw_prefix]
    binary += [None] * (len(LABEL_NAMES) - len(binary))
    valid = [1] * len(raw_prefix) + [0] * (len(LABEL_NAMES) - len(raw_prefix))
    probabilities = [0.05, 0.95, 0.05, 0.50, 0.95, 0.95][: len(raw_prefix)]
    probabilities += [None] * (len(LABEL_NAMES) - len(probabilities))
    issue = [1] * len(raw_prefix) + [0] * (len(LABEL_NAMES) - len(raw_prefix))
    return {
        "sample_selected_rank": selected_rank,
        "est_issue_entry_count": len(raw_prefix),
        "issue_rank_self_confidence": float(selected_rank),
        "issue_rank_normalized_margin": float(selected_rank),
        "issue_rank_confidence_weighted_entropy": float(selected_rank),
        "sample_quality_self_confidence": 0.1,
        "sample_quality_normalized_margin": 0.1,
        "sample_quality_confidence_weighted_entropy": 0.1,
        "sample_local_index": pool_row_id,
        "pool_row_id": pool_row_id,
        "subject_id": 1000 + pool_row_id,
        "study_id": study_id,
        "dicom_id": f"dicom-{pool_row_id}",
        "image_path": f"image-{pool_row_id}.jpg",
        "raw_labels_4class": pipe(raw),
        "binary_labels_for_detection": pipe(binary),
        "valid_label_mask": pipe(valid),
        "pred_probs": pipe(probabilities),
        "issue_entry_mask": pipe(issue),
    }


def expect_value_error(callable_obj: object, expected_text: str) -> None:
    try:
        callable_obj()  # type: ignore[operator]
    except ValueError as exc:
        assert expected_text in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected ValueError containing {expected_text!r}")


def main() -> None:
    samples = pd.DataFrame(
        [
            make_sample_row(
                pool_row_id=11,
                study_id=101,
                selected_rank=1,
                raw_prefix=[-1.0],
            ),
            make_sample_row(
                pool_row_id=12,
                study_id=102,
                selected_rank=2,
                raw_prefix=[0.0, 1.0, 0.0, 1.0, 0.0],
            ),
        ]
    )
    expanded = expand_sample_rows(samples)
    assert len(expanded) == 6
    assert expanded["pool_row_id"].nunique() == len(samples)
    assert set(expanded["raw_label"]) == {-1.0, 0.0, 1.0}
    expected = expanded["raw_label"].map({-1.0: 1.0, 0.0: 0.0, 1.0: 1.0})
    assert expanded["binary_label"].eq(expected).all()

    ranked = filter_and_rank_entries(expanded, top_k=100)
    assert len(ranked) == len(expanded)
    assert ranked["raw_label"].eq(-1.0).sum() == 1

    with tempfile.TemporaryDirectory() as temp_dir:
        archive = Path(temp_dir) / "reports.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("reports/s101.txt", "FINDINGS: Test.\nIMPRESSION: Test.")
            zf.writestr("reports/s102.txt", "FINDINGS: Test.\nIMPRESSION: Test.")
        tasks = prepare_tasks(ranked, archive)
    uncertain_task = next(task for task in tasks if float(task.source_row["raw_label"]) == -1.0)
    prompt = build_user_prompt(uncertain_task.llm_input)
    assert "Current binary label (decision target): 1.0" in prompt
    assert "Raw CheXpert label (provenance only): -1.0" in prompt

    decisions = {
        (11, 0): ("chexpert_wrong_cl_right", "review_or_relabel"),
        (12, 0): ("chexpert_wrong_cl_right", "review_or_relabel"),
        (12, 1): ("chexpert_wrong_cl_right", "review_or_relabel"),
        (12, 2): ("report_ambiguous", "mark_ambiguous"),
        (12, 3): ("chexpert_right_cl_wrong", "keep_chexpert"),
        (12, 4): ("chexpert_wrong_cl_right", "keep_chexpert"),
    }
    reviewed = expanded.copy()
    reviewed["mismatch_type"] = [
        decisions[(int(row.pool_row_id), int(row.label_index))][0]
        for row in reviewed.itertuples()
    ]
    reviewed["recommended_action"] = [
        decisions[(int(row.pool_row_id), int(row.label_index))][1]
        for row in reviewed.itertuples()
    ]
    relabel, mask, summary = build_refinement_tables(reviewed)

    relabel_map = {
        (int(row.pool_row_id), int(row.label_index)): (
            float(row.old_raw_label),
            float(row.old_binary_label),
            float(row.new_raw_label),
        )
        for row in relabel.itertuples()
    }
    assert relabel_map[(11, 0)] == (-1.0, 1.0, 0.0)
    assert relabel_map[(12, 0)] == (0.0, 0.0, 1.0)
    assert relabel_map[(12, 1)] == (1.0, 1.0, 0.0)
    assert len(mask) == 2
    assert set(mask["action_resolution"]) == {
        "mask_report_ambiguous",
        "mask_response_field_conflict",
    }
    assert int(summary.iloc[0]["keep_rows"]) == 1
    assert int(summary.iloc[0]["raw_uncertain_review_rows"]) == 1
    assert summary.iloc[0]["action_label_basis"] == "post_binarization_binary_label"

    bad_projection = expanded.copy()
    bad_projection.loc[bad_projection["raw_label"].eq(-1.0), "binary_label"] = 0.0
    expect_value_error(
        lambda: filter_and_rank_entries(bad_projection, top_k=100),
        "fixed U-Ones binary projection",
    )

    uncovered = samples.copy()
    uncovered.loc[1, "issue_entry_mask"] = pipe([0] * len(LABEL_NAMES))
    expect_value_error(
        lambda: expand_sample_rows(uncovered),
        "did not cover every selected sample",
    )

    assert not math.isnan(float(expanded.iloc[0]["entry_quality_self_confidence"]))
    print("binary-label refinement pipeline tests passed")


if __name__ == "__main__":
    main()
