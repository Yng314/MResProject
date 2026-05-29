#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import textwrap
import zipfile
from pathlib import Path

import pandas as pd


LABEL_PATTERNS = {
    "Pleural Effusion": {
        "positive": [
            r"\bpleural effusions?\b",
            r"\beffusions?\b",
        ],
        "negative": [
            r"\bno (large )?effusions?\b",
            r"\bno pleural effusions?\b",
            r"\bwithout pleural effusions?\b",
        ],
    },
    "Cardiomegaly": {
        "positive": [
            r"\bcardiomegaly\b",
            r"\b(enlarged|moderately enlarged|markedly enlarged) heart\b",
            r"\bheart is (mildly |moderately |markedly )?enlarged\b",
            r"\bcardiac enlargement\b",
        ],
        "negative": [
            r"\bno cardiomegaly\b",
            r"\bheart size (is )?normal\b",
            r"\bcardiomediastinal silhouette (is )?normal\b",
            r"\bcardiac silhouette (is )?normal\b",
            r"\bheart is not enlarged\b",
        ],
    },
    "Atelectasis": {
        "positive": [r"\batelectasis\b", r"\batelectatic\b"],
        "negative": [r"\bno atelectasis\b", r"\bwithout atelectasis\b"],
    },
    "Lung Opacity": {
        "positive": [
            r"\bopacit(y|ies)\b",
            r"\bopacification(s)?\b",
            r"\bpulmonary opacification(s)?\b",
            r"\binfiltrate(s)?\b",
        ],
        "negative": [
            r"\bno focal consolidation\b",
            r"\bno focal opacit(y|ies)\b",
            r"\blungs are clear\b",
            r"\bno acute pulmonary abnormality\b",
        ],
    },
    "Lung Lesion": {
        "positive": [
            r"\blung lesion\b",
            r"\bpulmonary lesion\b",
            r"\bnodule(s)?\b",
            r"\bmass(es)?\b",
            r"\bmass-like\b",
        ],
        "negative": [
            r"\bno lung lesion\b",
            r"\bno pulmonary lesion\b",
            r"\bno nodule(s)?\b",
            r"\bno mass(es)?\b",
        ],
    },
    "Pneumonia": {
        "positive": [r"\bpneumonia\b"],
        "negative": [r"\bno pneumonia\b", r"\bwithout pneumonia\b"],
    },
}


def normalize_text(text: str) -> str:
    text = text.lower().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_report_map(report_archive: Path, study_ids: list[int]) -> dict[int, str]:
    reports: dict[int, str] = {}
    with zipfile.ZipFile(report_archive, "r") as zf:
        names = zf.namelist()
        index = {name.rsplit("/", 1)[-1]: name for name in names if name.endswith(".txt")}
        for study_id in study_ids:
            key = f"s{study_id}.txt"
            if key in index:
                reports[study_id] = zf.read(index[key]).decode("utf-8", errors="replace")
            else:
                reports[study_id] = ""
    return reports


def report_relation(label_name: str, report_text: str) -> tuple[str, str]:
    patterns = LABEL_PATTERNS.get(label_name)
    if not patterns:
        return "no_rule", ""

    text = normalize_text(report_text)
    for pat in patterns["negative"]:
        m = re.search(pat, text)
        if m:
            return "negated", m.group(0)
    for pat in patterns["positive"]:
        m = re.search(pat, text)
        if m:
            return "supported", m.group(0)
    return "unclear", ""


def shorten(text: str, width: int = 240) -> str:
    text = normalize_text(text)
    return textwrap.shorten(text, width=width, placeholder=" ...")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-details-csv", type=Path, required=True)
    parser.add_argument("--report-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k-per-label", type=int, default=20)
    parser.add_argument("--min-pred-prob", type=float, default=0.95)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.entry_details_csv)
    df = df[(df["raw_label"] == 0.0) & (df["pred_prob"] >= args.min_pred_prob)].copy()
    df = df[df["entry_issue_rank_self_confidence"].notna()].copy()

    # Keep a manageable high-suspicion slice for each label.
    df = (
        df.sort_values(["label_name", "entry_issue_rank_self_confidence"], ascending=[True, True])
        .groupby("label_name", group_keys=False)
        .head(args.top_k_per_label)
        .reset_index(drop=True)
    )

    reports = extract_report_map(args.report_archive, sorted(df["study_id"].astype(int).unique().tolist()))
    df["report_text"] = df["study_id"].astype(int).map(reports)

    relations = df.apply(lambda r: report_relation(r["label_name"], r["report_text"]), axis=1)
    df["report_relation"] = [x[0] for x in relations]
    df["report_match_phrase"] = [x[1] for x in relations]
    df["report_snippet"] = df["report_text"].map(shorten)

    all_path = args.output_dir / "high_suspicion_entries_with_report_support.csv"
    df.to_csv(all_path, index=False)

    # "Likely false alarm" here means model very confident, but report does not support the label.
    candidates = df[df["report_relation"].isin(["negated", "unclear"])].copy()
    candidates = candidates.sort_values(
        ["label_name", "report_relation", "pred_prob", "entry_issue_rank_self_confidence"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    cand_path = args.output_dir / "likely_false_alarm_candidates_from_reports.csv"
    candidates.to_csv(cand_path, index=False)

    summary = (
        df.groupby(["label_name", "report_relation"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("label_name")
    )
    summary_path = args.output_dir / "report_support_summary_by_label.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {all_path}")
    print(f"Saved: {cand_path}")
    print(f"Saved: {summary_path}")
    print("\nTop likely false-alarm candidates:")
    cols = [
        "study_id",
        "label_name",
        "pred_prob",
        "entry_quality_self_confidence",
        "entry_issue_rank_self_confidence",
        "report_relation",
        "report_match_phrase",
    ]
    print(candidates[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
