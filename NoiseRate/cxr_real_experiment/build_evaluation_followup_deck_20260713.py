#!/usr/bin/env python3
"""Build the post-meeting evaluation follow-up deck and speaker notes."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "cxr_real_experiment"
FOLLOWUP_DIR = SCRIPT_DIR / "evaluation_followup_20260713"
OUTPUT_DIR = SCRIPT_DIR / "evaluation_followup_deck_20260713"
OUTPUT_PPTX = OUTPUT_DIR / "evaluation_followup_20260713.pptx"

SYNTHETIC_DIR = FOLLOWUP_DIR / "synthetic_dqs_validation"
QUALITY_DIR = FOLLOWUP_DIR / "real_data_quality_figures"
SYNTHETIC_TRAJECTORY = SYNTHETIC_DIR / "synthetic_dqs_known_truth_trajectories.png"
SYNTHETIC_SENSITIVITY = SYNTHETIC_DIR / "synthetic_dqs_oof_quality_sensitivity.png"
ENTRY_DQS_FIGURE = QUALITY_DIR / "entry_dqs_common_scale_five_seed.png"
SAMPLE_HEALTH_FIGURE = QUALITY_DIR / "sample_issue_free_rate_five_seed.png"
QUALITY_METADATA = QUALITY_DIR / "quality_figure_metadata.json"
OLD_HIERARCHICAL_FIGURE = (
    SCRIPT_DIR
    / "evaluation_update_deck_20260712"
    / "hierarchical_uncertainty_4contrast.png"
)

DEFAULT_OWN_ROOT = (
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_own_top20_binary_v5_llm_refine_5seed/20260714_065539"
)
OWN_ROOT = Path(os.environ.get("OWN_TOP20_ROOT", DEFAULT_OWN_ROOT))
PERFORMANCE_DIR = OWN_ROOT / "evaluation_prelocked_loop5_loop8" / "performance"
SEEDS = [7, 13, 42, 97, 123]


def load_seed_jobs() -> dict[int, str]:
    path = OWN_ROOT / "slurm_submission_manifest.json"
    if not path.is_file():
        return {seed: "not-submitted" for seed in SEEDS}
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("seed_jobs", {})
    return {seed: str(jobs.get(str(seed), jobs.get(seed, "unknown"))) for seed in SEEDS}


SEED_JOBS = load_seed_jobs()

SLIDE_W = 13.333333
SLIDE_H = 7.5

INK = "202735"
MUTED = "687386"
BLUE = "2B73B6"
RED = "C6372F"
GREEN = "287A57"
AMBER = "A56A16"
LIGHT = "F5F7FA"
LINE = "D9E0E8"
WHITE = "FFFFFF"
DARK_BLUE = "203D67"

TITLES = [
    "We now test the metric itself, not only the cleaning outcome",
    "Synthetic labels make the denominator question directly testable",
    "Adjustment fixes the denominator interpretation, but DQS remains model-dependent",
    "Simple removal looks cleaner only on what remains",
    "Five independent own-top20 pipelines remove the fixed-table limitation",
    "Effect sizes and hierarchical uncertainty lead; p-values support them",
    "The story is stronger, but clinical correctness is still a separate question",
    "Backup: hierarchical uncertainty was not presented last time",
    "Backup: proper scoring rules were not presented last time",
    "Backup: DQS inherits the quality of its OOF probabilities",
    "Backup: exact sources and frozen protocol",
]

SOURCES = [
    "Source: meeting transcript, 13 July 2026; follow-up analyses listed on this slide.",
    "Source: validate_synthetic_dqs_adjustment.py; N=5,000, 10 seeds, Loops 0-5.",
    "Source: synthetic_dqs_summary.csv and synthetic_dqs_error_summary.csv.",
    "Source: five-seed frozen-OOF quality summaries; error bars are seed SD.",
    "Source: locked binary-label v5 protocol snapshot and reuse manifest under the corrected result root.",
    "Source: pre-locked four-contrast analysis; the table is populated only from completed evaluator CSVs.",
    "Source: combined synthetic, five-seed quality, and held-out evaluation evidence.",
    "Source: fixed-table five-seed analysis shown in the previous deck; this page was not presented.",
    "Source: held-out supplementary metrics from the previous five-seed analysis; not presented last time.",
    "Source: synthetic_dqs_oof_quality_sensitivity.png.",
    "Source: protocol snapshot checksums and scripts listed on this slide.",
]

NOTES = [
    (
        "Last time, the central question was whether the current evaluation is reliable, not whether "
        "we can add another cleaning method. I therefore separated five issues: whether adjusted DQS "
        "has a meaningful target, whether the plots make the denominator visible, whether the same "
        "story appears at sample level, whether refinement continues after Loop 5, and how uncertainty "
        "should be reported. The hierarchical uncertainty and proper-scoring pages in my previous deck "
        "were not presented, so I treat them here as supporting analysis rather than prior feedback."
    ),
    (
        "The synthetic experiment is deliberately small and interpretable. I generate 5,000 binary "
        "entries with exactly 12 percent symmetric label noise. Because the true clean label is known, "
        "raw DQS can be compared with current-subset accuracy, while adjusted DQS can be compared with "
        "the amount of correct and retained label mass relative to the original denominator. These are "
        "different questions, so they need different truth targets."
    ),
    (
        "The key comparison is removal. After all known errors are removed, the surviving subset is "
        "fully correct, so a raw score near one is reasonable. However, only 88 percent of the original "
        "entries remain, so adjusted DQS stays near 0.88. Random removal makes the same distinction clear: "
        "raw cleanliness changes little, while adjusted healthy mass falls with coverage. The weak-OOF "
        "condition also shows the limit: denominator adjustment does not turn weak model evidence into truth."
    ),
    (
        "The entry and sample views now use comparable scales and show individual seed traces plus mean "
        "and standard deviation. The corrected sample denominator excludes rows with no valid target labels: "
        "those rows are unevaluable, not automatically issue-free. At Loop 5, removal rises from 0.736 to 0.865 "
        "among evaluable survivors, but falls to 0.698 on the baseline-evaluable denominator. Fixed refinement "
        "reaches 0.756 raw and 0.754 adjusted at 99.69 percent evaluable coverage, above the 0.736 baseline in "
        "all five seeds. This is a strict custom diagnostic, not an official Cleanlab DQS, and it is supporting "
        "fixed-table evidence rather than the pending own-top20 outcome."
    ),
    (
        "The previous refinement comparison varied the final training seed but reused one fixed set of "
        "refinement tables. The corrected experiment removes that limitation. Each seed starts from its own "
        "four-fold MobileNet OOF probabilities, selects its own top 20 percent of suspicious samples, sends "
        "every flagged binary entry for GPT review, and repeats the complete pipeline through Loop 8. An audit "
        "found that the invalid pilot excluded raw uncertain labels even though detection and training mapped "
        "them to binary positive. I stopped that run, discarded its actions and outcomes, and corrected the "
        "review target. Loop 5 and Loop 8 are carried over unchanged, so I will not select a best-looking loop. "
        "A pre-action QA check also found occasional conflicts between the two redundant LLM decision fields. "
        "Those raw responses are retained, but the affected entries are conservatively masked and counted."
    ),
    (
        "This is the uncertainty page that matters for the new result. Method comparisons are paired within "
        "five seeds, and the hierarchical bootstrap varies both training seed and held-out study. I report "
        "the observed effect, the interval, seed directions, exact sign-flip p-value, and Holm-adjusted p-value. "
        "With five seeds, the smallest possible two-sided exact p-value is 0.0625, so p greater than 0.05 cannot "
        "be interpreted as evidence of no effect. The current slide contains no estimated result while jobs run."
    ),
    (
        "If the own-top20 result agrees, the defensible advance is from training-seed robustness to repeatability "
        "of the complete model-based cleaning pipeline. It still does not establish that every LLM relabel is "
        "clinically correct, that the result generalises to another dataset or backbone, or that DQS is independent "
        "of the OOF model. The final claim should remain a repeatable quality-coverage improvement conditional on "
        "model-based evidence."
    ),
    (
        "Use this page only if uncertainty from the previous result is discussed. It was not presented in the "
        "last meeting. A study-only bootstrap treats the five trained seeds as fixed. The hierarchical analysis "
        "also varies the training seed, which is the broader and more appropriate uncertainty target here. The "
        "previous fixed-table Loop 5 effect was positive in all five seeds, but its hierarchical interval still "
        "included zero and the exact p-value was limited to 0.0625."
    ),
    (
        "Use this page only if someone asks whether AUROC tells the whole story. It was not presented last time. "
        "AUROC and average precision assess ranking, while Brier score and negative log-likelihood assess the "
        "quality of the predicted probabilities. Late removal retained a modest AUROC gain but had substantially "
        "worse NLL, which is consistent with more high-confidence errors. These are held-out utility measures and "
        "still do not directly prove that the training labels are clinically correct."
    ),
    (
        "This sensitivity analysis is the caveat to the synthetic validation. Informative OOF probabilities rank "
        "the known label errors well and give a mean matching-target error around 0.019. Weak OOF probabilities "
        "reduce truth AUROC and increase the matching-target error to around 0.055. Therefore adjusted DQS has a "
        "defensible denominator interpretation, but its numerical accuracy remains conditional on model quality."
    ),
    (
        "This is a provenance slide for questions about exact implementation. The protocol, endpoints, action rules, "
        "LLM model, and analysis scripts were frozen under the corrected root before any v5 action or outcome existed. "
        "The document also states that the semantic correction followed an invalid pilot outcome, so this is not "
        "presented as an outcome-blind preregistration. "
        "The same builder regenerates this deck after the locked-v5 CSVs exist, so the final statistics can be inserted "
        "without changing the comparisons or wording boundaries."
    ),
]


def rgb(hex_value: str) -> RGBColor:
    return RGBColor.from_string(hex_value.lstrip("#"))


def add_text(
    slide: Any,
    text: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    size: float = 16,
    color: str = INK,
    bold: bool = False,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    valign: MSO_ANCHOR = MSO_ANCHOR.TOP,
    name: str | None = None,
) -> Any:
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    if name:
        shape.name = name
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.03)
    frame.margin_right = Inches(0.03)
    frame.margin_top = Inches(0.02)
    frame.margin_bottom = Inches(0.02)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    return shape


def add_bullets(
    slide: Any,
    items: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    size: float = 15,
    color: str = INK,
    spacing: float = 8,
) -> Any:
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.08)
    frame.margin_right = Inches(0.03)
    frame.margin_top = Inches(0.02)
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = item
        paragraph.level = 0
        bullet = OxmlElement("a:buChar")
        bullet.set("char", "\u2022")
        paragraph._p.get_or_add_pPr().append(bullet)
        paragraph.font.name = "Arial"
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = rgb(color)
        paragraph.space_after = Pt(spacing)
    return shape


def add_panel(
    slide: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str = LIGHT,
    line: str = LINE,
    radius: bool = False,
) -> Any:
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    panel = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(width), Inches(height))
    panel.fill.solid()
    panel.fill.fore_color.rgb = rgb(fill)
    panel.line.color.rgb = rgb(line)
    panel.line.width = Pt(0.8)
    return panel


def add_title(slide: Any, title: str, number: int, *, backup: bool = False) -> None:
    add_text(slide, title, 0.55, 0.30, 11.95, 0.72, size=21, bold=True, name="slide-title")
    add_text(
        slide,
        "BACKUP" if backup else "EVALUATION FOLLOW-UP",
        10.85,
        0.08,
        1.55,
        0.20,
        size=7.5,
        color=DARK_BLUE,
        bold=True,
        align=PP_ALIGN.RIGHT,
    )
    add_text(slide, f"{number:02d}", 12.62, 0.08, 0.28, 0.20, size=7.5, color=MUTED, bold=True)
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.03), Inches(12.20), Inches(0.025))
    rule.fill.solid()
    rule.fill.fore_color.rgb = rgb(LINE)
    rule.line.fill.background()


def add_footer(slide: Any, text: str) -> None:
    add_text(slide, text, 0.58, 7.10, 12.10, 0.24, size=6.7, color=MUTED, name="source-note")


def add_note(slide: Any, text: str) -> bool:
    try:
        slide.notes_slide.notes_text_frame.text = text
        return True
    except (AttributeError, NotImplementedError):
        return False


def add_picture_fit(slide: Any, path: Path, x: float, y: float, width: float, height: float) -> Any:
    with Image.open(path) as image:
        source_ratio = image.width / image.height
    box_ratio = width / height
    if source_ratio >= box_ratio:
        pic_width = width
        pic_height = width / source_ratio
    else:
        pic_height = height
        pic_width = height * source_ratio
    return slide.shapes.add_picture(
        str(path),
        Inches(x + (width - pic_width) / 2),
        Inches(y + (height - pic_height) / 2),
        Inches(pic_width),
        Inches(pic_height),
    )


def style_cell(
    cell: Any,
    text: str,
    *,
    fill: str,
    color: str,
    size: float,
    bold: bool = False,
    align: PP_ALIGN = PP_ALIGN.LEFT,
) -> None:
    cell.text = text
    cell.fill.solid()
    cell.fill.fore_color.rgb = rgb(fill)
    cell.margin_left = Inches(0.06)
    cell.margin_right = Inches(0.04)
    cell.margin_top = Inches(0.025)
    cell.margin_bottom = Inches(0.025)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for paragraph in cell.text_frame.paragraphs:
        paragraph.alignment = align
        for run in paragraph.runs:
            run.font.name = "Arial"
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = rgb(color)


def add_table(
    slide: Any,
    headers: list[str],
    rows: list[list[str]],
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    col_widths: list[float] | None = None,
    font_size: float = 10.5,
    first_col_bold: bool = True,
) -> Any:
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(width), Inches(height))
    table = shape.table
    if col_widths:
        total = sum(col_widths)
        for index, value in enumerate(col_widths):
            table.columns[index].width = Inches(width * value / total)
    for index, header in enumerate(headers):
        style_cell(
            table.cell(0, index),
            header,
            fill=DARK_BLUE,
            color=WHITE,
            size=font_size,
            bold=True,
            align=PP_ALIGN.CENTER,
        )
    for row_index, row in enumerate(rows, start=1):
        fill = WHITE if row_index % 2 else LIGHT
        for col_index, value in enumerate(row):
            style_cell(
                table.cell(row_index, col_index),
                value,
                fill=fill,
                color=INK,
                size=font_size,
                bold=first_col_bold and col_index == 0,
                align=PP_ALIGN.LEFT if col_index == 0 else PP_ALIGN.CENTER,
            )
    return shape


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def matching_row(rows: list[dict[str, str]], **criteria: Any) -> dict[str, str]:
    matches = [
        row for row in rows if all(str(row.get(key)) == str(value) for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row for {criteria}; found {len(matches)}")
    return matches[0]


def current_job_snapshot() -> dict[int, dict[str, Any]]:
    statuses = {seed: "NOT IN QUEUE" for seed in SEEDS}
    submitted_ids = [job_id for job_id in SEED_JOBS.values() if job_id.isdigit()]
    try:
        if submitted_ids:
            result = subprocess.run(
                ["squeue", "-h", "-j", ",".join(submitted_ids), "-o", "%i|%T"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.splitlines():
                job_id, state = line.strip().split("|", 1)
                seed = next(seed for seed, known_job in SEED_JOBS.items() if known_job == job_id)
                statuses[seed] = state
    except (OSError, subprocess.SubprocessError, ValueError, StopIteration):
        pass

    snapshot: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        branch = OWN_ROOT / f"seed_{seed}" / "llm_refine"
        completed = sum(
            (branch / f"loop_{loop:02d}" / ".train_eval_complete").exists()
            for loop in range(1, 9)
        )
        review_dir = branch / f"loop_{min(completed + 1, 8):02d}" / "llm_review"
        results = review_dir / "results.csv"
        errors = review_dir / "errors.csv"
        success_count = len(read_csv(results)) if results.exists() and results.stat().st_size else 0
        error_count = len(read_csv(errors)) if errors.exists() and errors.stat().st_size else 0
        snapshot[seed] = {
            "job_id": SEED_JOBS[seed],
            "state": statuses[seed],
            "completed_loops": completed,
            "active_loop": min(completed + 1, 8),
            "review_successes": success_count,
            "review_errors": error_count,
        }
    return snapshot


def endpoint_rows() -> tuple[list[list[str]], bool]:
    seed_path = PERFORMANCE_DIR / "prelocked_seed_paired_statistics.csv"
    bootstrap_path = PERFORMANCE_DIR / "prelocked_hierarchical_bootstrap.csv"
    labels = [
        ("refine_L5_vs_baseline", "Own refine L5 vs baseline"),
        ("refine_L8_vs_baseline", "Own refine L8 vs baseline"),
        ("refine_L8_vs_refine_L5", "Own refine L8 vs refine L5"),
        ("refine_L5_vs_remove_L5", "Own refine L5 vs remove L5"),
    ]
    if not seed_path.exists() or not bootstrap_path.exists():
        return [[label, "pending", "pending", "pending", "pending", "pending"] for _, label in labels], False

    seed_rows = read_csv(seed_path)
    boot_rows = read_csv(bootstrap_path)
    output: list[list[str]] = []
    for key, label in labels:
        seed_row = matching_row(seed_rows, comparison=key)
        boot_row = matching_row(
            boot_rows,
            comparison=key,
            scope="all_labels",
            uncertainty="hierarchical_seed_and_study",
        )
        output.append(
            [
                label,
                f"{float(seed_row['mean_delta']):+.4f}",
                f"[{float(boot_row['ci_low_2p5']):+.4f}, {float(boot_row['ci_high_97p5']):+.4f}]",
                f"{int(seed_row['positive_delta_seeds'])}/5",
                f"{float(seed_row['exact_signflip_p_two_sided']):.4f}",
                f"{float(seed_row['holm4_exact_p']):.4f}",
            ]
        )
    return output, True


def quality_anchor_rows() -> list[list[str]]:
    metadata = json.loads(QUALITY_METADATA.read_text(encoding="utf-8"))
    definition = metadata.get("definitions", {}).get("sample_coverage", "")
    if "zero-valid samples are not counted as issue-free" not in definition:
        raise ValueError("Quality metadata does not use the corrected evaluable-sample denominator")
    anchors = metadata["anchors"]

    def value(key: str) -> str:
        return f"{float(anchors[key]):.4f}"

    return [
        [
            "Entry DQS",
            value("baseline_l0_raw_dqs"),
            f"{value('remove_l5_raw_dqs')} / {value('remove_l5_adjusted_dqs')}",
            f"{value('refine_l5_raw_dqs')} / {value('refine_l5_adjusted_dqs')}",
        ],
        [
            "Strict issue-free samples",
            value("baseline_l0_sample_issue_free_rate"),
            f"{value('remove_l5_sample_issue_free_rate')} / {value('remove_l5_adjusted_sample_health')}",
            f"{value('refine_l5_sample_issue_free_rate')} / {value('refine_l5_adjusted_sample_health')}",
        ],
    ]


def new_slide(presentation: Presentation, index: int, *, backup: bool = False) -> Any:
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    background = slide.background.fill
    background.solid()
    background.fore_color.rgb = rgb(WHITE)
    add_title(slide, TITLES[index], index + 1, backup=backup)
    add_footer(slide, SOURCES[index])
    return slide


def build_slide_1(presentation: Presentation) -> None:
    slide = new_slide(presentation, 0)
    add_text(
        slide,
        "The follow-up separates metric validity, pipeline repeatability, and held-out utility.",
        0.65,
        1.18,
        11.9,
        0.38,
        size=14,
        color=MUTED,
    )
    rows = [
        ["Is adjusted DQS meaningful when truth is known?", "10-seed synthetic known-truth validation"],
        ["Are DQS plots visually comparable?", "Shared scale, visible seed traces, mean +/- SD"],
        ["Does entry aggregation hide sample behavior?", "Strict sample-level issue-free rate + coverage"],
        ["Does refinement continue after Loop 5?", "Five own-top20 pipelines locked through Loop 8"],
        ["Where are the p-values?", "Four pre-locked contrasts + hierarchical uncertainty"],
    ]
    add_table(
        slide,
        ["Meeting concern", "Evidence in this update"],
        rows,
        0.65,
        1.72,
        12.0,
        4.65,
        col_widths=[1.02, 0.98],
        font_size=13,
    )
    add_text(
        slide,
        "No new cleaning method was added.",
        0.72,
        6.50,
        4.0,
        0.28,
        size=11,
        color=RED,
        bold=True,
    )


def build_slide_2(presentation: Presentation) -> None:
    slide = new_slide(presentation, 1)
    add_text(slide, "Known setup", 0.65, 1.28, 2.1, 0.30, size=11, color=DARK_BLUE, bold=True)
    add_bullets(
        slide,
        [
            "5,000 binary entries with exactly 12% symmetric label noise",
            "10 synthetic seeds; Loops 0-5; four-fold logistic-regression OOF",
            "Frozen OOF evidence across actions, matching the real-data audit",
        ],
        0.65,
        1.68,
        5.45,
        2.15,
        size=14,
    )
    add_panel(slide, 6.40, 1.25, 6.20, 2.62, fill="F1F5FA", line="C9D7E7")
    add_text(slide, "Two truth targets", 6.70, 1.52, 2.6, 0.28, size=12, color=DARK_BLUE, bold=True)
    add_text(
        slide,
        "Raw target\ncorrect labels / currently covered entries",
        6.72,
        2.00,
        2.55,
        1.10,
        size=14,
        bold=True,
        valign=MSO_ANCHOR.MIDDLE,
    )
    add_text(
        slide,
        "Adjusted target\ncorrect and retained labels / original N",
        9.60,
        2.00,
        2.55,
        1.10,
        size=14,
        bold=True,
        valign=MSO_ANCHOR.MIDDLE,
    )
    action_titles = ["Correct known errors", "Remove known errors", "Remove random entries", "Corrupt clean entries"]
    action_colors = [GREEN, BLUE, AMBER, RED]
    for index, (title, color) in enumerate(zip(action_titles, action_colors)):
        x = 0.65 + index * 3.04
        add_panel(slide, x, 4.25, 2.78, 1.62, fill=WHITE, line=color)
        circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.18), Inches(4.52), Inches(0.40), Inches(0.40))
        circle.fill.solid()
        circle.fill.fore_color.rgb = rgb(color)
        circle.line.fill.background()
        add_text(slide, str(index + 1), x + 0.18, 4.56, 0.40, 0.20, size=9, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, title, x + 0.70, 4.42, 1.85, 0.72, size=12, bold=True, valign=MSO_ANCHOR.MIDDLE)
    add_text(
        slide,
        "The aim is measurement validation, not a miniature chest X-ray benchmark.",
        0.68,
        6.20,
        10.5,
        0.35,
        size=12,
        color=MUTED,
    )


def build_slide_3(presentation: Presentation) -> None:
    slide = new_slide(presentation, 2)
    add_picture_fit(slide, SYNTHETIC_TRAJECTORY, 0.45, 1.18, 8.85, 5.62)
    add_panel(slide, 9.45, 1.35, 3.25, 4.95, fill="F6F8FB", line=LINE)
    add_text(slide, "Loop 5 anchors", 9.72, 1.64, 2.55, 0.30, size=12, color=DARK_BLUE, bold=True)
    anchors = [
        ("Known-error removal", "coverage 0.88", "raw 0.994", "adjusted 0.875"),
        ("Random removal", "coverage 0.88", "raw 0.903", "adjusted 0.795"),
        ("Correct known errors", "coverage 1.00", "raw 0.994", "adjusted 0.994"),
    ]
    for index, values in enumerate(anchors):
        y = 2.16 + index * 1.18
        add_text(slide, values[0], 9.72, y, 2.62, 0.25, size=10.5, bold=True)
        add_text(slide, " | ".join(values[1:]), 9.72, y + 0.34, 2.65, 0.42, size=9.5, color=MUTED)
    add_text(slide, "Matching-target MAE", 9.72, 5.70, 1.85, 0.22, size=9, color=MUTED, bold=True)
    add_text(slide, "0.019 informative OOF\n0.055 weak OOF", 11.45, 5.54, 1.02, 0.62, size=10.5, color=RED, bold=True, align=PP_ALIGN.RIGHT)
    add_text(
        slide,
        "Adjustment repairs the interpretation of the denominator; it does not create ground truth.",
        9.55,
        6.48,
        3.05,
        0.38,
        size=10.2,
        color=DARK_BLUE,
        bold=True,
    )


def build_slide_4(presentation: Presentation) -> None:
    slide = new_slide(presentation, 3)
    add_text(slide, "Entry-level DQS", 0.62, 1.15, 2.4, 0.24, size=10, color=DARK_BLUE, bold=True)
    add_text(slide, "Strict sample-level diagnostic", 6.76, 1.15, 3.3, 0.24, size=10, color=DARK_BLUE, bold=True)
    add_picture_fit(slide, ENTRY_DQS_FIGURE, 0.45, 1.38, 6.18, 2.70)
    add_picture_fit(slide, SAMPLE_HEALTH_FIGURE, 6.63, 1.38, 6.18, 2.70)
    rows = quality_anchor_rows()
    add_table(
        slide,
        ["Loop 5 view", "Baseline", "Remove raw / adjusted", "Refine raw / adjusted"],
        rows,
        0.72,
        4.40,
        11.90,
        1.42,
        col_widths=[1.40, 0.85, 1.35, 1.35],
        font_size=11.3,
    )
    add_text(
        slide,
        "Sample metric: among samples with >=1 valid target, a sample is unhealthy if any valid entry is flagged; zero-valid samples are unevaluable.",
        0.74,
        6.02,
        11.8,
        0.45,
        size=10,
        color=MUTED,
    )
    add_text(slide, "Removal: cleaner survivors, less healthy mass", 0.78, 6.54, 5.2, 0.28, size=11, color=BLUE, bold=True)
    add_text(slide, "Refinement: modest gain without sample deletion", 6.82, 6.54, 5.1, 0.28, size=11, color=RED, bold=True)


def build_slide_5(presentation: Presentation, jobs: dict[int, dict[str, Any]]) -> None:
    slide = new_slide(presentation, 4)
    steps = ["Own 4-fold\nMobileNet OOF", "Own top 20%\nsamples", "Expand suspicious\nentries", "GPT-5.4\nreview", "Relabel / mask\nand train", "Repeat to\nLoop 8"]
    for index, step in enumerate(steps):
        x = 0.52 + index * 2.08
        add_panel(slide, x, 1.38, 1.72, 1.05, fill="F5F8FC", line="BFCFE0", radius=True)
        add_text(slide, step, x + 0.10, 1.56, 1.52, 0.62, size=10.5, bold=True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
        if index < len(steps) - 1:
            add_text(slide, ">", x + 1.77, 1.72, 0.25, 0.24, size=15, color=MUTED, bold=True, align=PP_ALIGN.CENTER)

    protocol_rows = [
        ["Seeds", "7, 13, 42, 97, 123"],
        ["Locked endpoints", "Loop 5 and Loop 8; no best-loop substitution"],
        ["Binary review target", "U-Ones: raw -1/1 -> 1; raw 0 -> 0"],
        ["Coverage gate", "100% of selected samples represented or fail"],
        ["Fixed training", "MobileNetV3-small, 50 epochs, no early stopping"],
    ]
    add_table(slide, ["Protocol", "Frozen value"], protocol_rows, 0.62, 2.88, 7.00, 2.75, col_widths=[0.85, 2.15], font_size=10.8)
    add_panel(slide, 7.95, 2.88, 4.72, 3.48, fill="FAFBFC", line=LINE)
    add_text(slide, "Live run snapshot", 8.20, 3.15, 2.3, 0.26, size=11, color=DARK_BLUE, bold=True)
    y = 3.58
    for seed in SEEDS:
        item = jobs[seed]
        state_color = GREEN if item["state"] == "RUNNING" else AMBER if item["state"] == "PENDING" else MUTED
        detail = (
            f"seed {seed:<3}  job {item['job_id']}  {item['state']:<11}  "
            f"completed L{item['completed_loops']}"
        )
        add_text(slide, detail, 8.20, y, 4.12, 0.26, size=9.5, color=state_color, bold=True)
        y += 0.44
    add_text(
        slide,
        "Each seed now varies both detection/selection and final training.",
        8.20,
        5.92,
        4.05,
        0.32,
        size=10.2,
        color=RED,
        bold=True,
    )


def build_slide_6(presentation: Presentation, rows: list[list[str]], complete: bool) -> None:
    slide = new_slide(presentation, 5)
    status = "LOCKED V5 RESULTS AVAILABLE" if complete else "LOCKED V5 RESULTS PENDING"
    status_color = GREEN if complete else AMBER
    add_text(slide, status, 0.67, 1.18, 4.0, 0.26, size=10, color=status_color, bold=True)
    add_table(
        slide,
        ["Contrast", "Mean delta", "Hierarchical 95% CI", "+ seeds", "Exact p", "Holm p"],
        rows,
        0.58,
        1.60,
        12.20,
        3.12,
        col_widths=[2.25, 0.82, 1.55, 0.60, 0.66, 0.66],
        font_size=9.2,
    )
    add_panel(slide, 0.68, 5.10, 3.75, 1.40, fill="F2F6FB", line="C6D6E6")
    add_text(slide, "Paired seed effect", 0.94, 5.35, 2.2, 0.22, size=10, color=DARK_BLUE, bold=True)
    add_text(slide, "method differences are paired within each of five seeds", 0.94, 5.72, 3.12, 0.48, size=11.2, bold=True)
    add_panel(slide, 4.78, 5.10, 3.75, 1.40, fill="F2F6FB", line="C6D6E6")
    add_text(slide, "Hierarchical interval", 5.04, 5.35, 2.4, 0.22, size=10, color=DARK_BLUE, bold=True)
    add_text(slide, "resamples both training seed and held-out study", 5.04, 5.72, 3.12, 0.48, size=11.2, bold=True)
    add_panel(slide, 8.88, 5.10, 3.75, 1.40, fill="FFF8EE", line="E5C792")
    add_text(slide, "Exact-test resolution", 9.14, 5.35, 2.4, 0.22, size=10, color=AMBER, bold=True)
    add_text(slide, "minimum two-sided p with five seeds = 0.0625", 9.14, 5.72, 3.12, 0.48, size=11.2, bold=True)
    if not complete:
        add_text(slide, "No effect estimate is shown before the five Loop 8 pipelines finish.", 0.72, 6.68, 9.0, 0.28, size=10.5, color=RED, bold=True)


def build_slide_7(presentation: Presentation) -> None:
    slide = new_slide(presentation, 6)
    rows = [
        ["Adjusted DQS has a defensible denominator interpretation", "Every LLM relabel is clinically correct"],
        ["Trends repeat across own OOF selections and training seeds", "Expert adjudication or external-data generalisation"],
        ["Refinement preserves coverage better than repeated removal", "Causal superiority for every label or model family"],
        ["Held-out ranking and proper scores improve at locked loops", "DQS is independent of the OOF model"],
    ]
    add_table(slide, ["Supported if final pipeline agrees", "Still not established"], rows, 0.68, 1.45, 11.95, 4.38, col_widths=[1, 1], font_size=12.2)
    add_panel(slide, 0.70, 6.12, 11.90, 0.66, fill="F1F5FA", line="C7D4E2")
    add_text(
        slide,
        "Defensible claim: repeatable quality-coverage improvement, conditional on model-based evidence.",
        0.98,
        6.30,
        11.25,
        0.26,
        size=13,
        color=DARK_BLUE,
        bold=True,
        align=PP_ALIGN.CENTER,
    )


def build_slide_8(presentation: Presentation) -> None:
    slide = new_slide(presentation, 7, backup=True)
    add_picture_fit(slide, OLD_HIERARCHICAL_FIGURE, 0.52, 1.26, 8.05, 5.60)
    add_panel(slide, 8.78, 1.38, 3.85, 4.95, fill="F6F8FB", line=LINE)
    add_text(slide, "Why this matters", 9.06, 1.70, 2.7, 0.28, size=12, color=DARK_BLUE, bold=True)
    add_bullets(
        slide,
        [
            "Study-only bootstrap keeps the five trained seeds fixed.",
            "Hierarchical bootstrap also varies training seed.",
            "Fixed-table refine L5 vs baseline: +0.0341.",
            "Hierarchical 95% CI: [-0.0005, 0.0783].",
            "5/5 positive; exact p = 0.0625; Holm p = 0.2797.",
        ],
        9.00,
        2.25,
        3.22,
        3.55,
        size=11.3,
        spacing=7,
    )
    add_text(slide, "Promising direction; unresolved seed uncertainty.", 9.05, 6.50, 3.25, 0.32, size=10.5, color=RED, bold=True)


def build_slide_9(presentation: Presentation) -> None:
    slide = new_slide(presentation, 8, backup=True)
    add_text(
        slide,
        "AUROC/AP test ranking; Brier/NLL test the numerical quality of probabilities.",
        0.72,
        1.22,
        11.7,
        0.36,
        size=13,
        color=MUTED,
    )
    rows = [
        ["Weighted AUROC", "0.7231", "0.7312", "0.7572", "higher"],
        ["Macro AP", "0.8560", "0.8581", "0.8629", "higher"],
        ["Micro Brier", "0.1610", "0.1641", "0.1568", "lower"],
        ["Micro NLL", "0.6954", "0.9477", "0.7005", "lower"],
    ]
    add_table(
        slide,
        ["Metric", "Baseline", "Remove L5", "Refine L5", "Better"],
        rows,
        0.72,
        1.82,
        11.85,
        3.35,
        col_widths=[1.35, 0.90, 0.95, 0.95, 0.70],
        font_size=12.2,
    )
    add_panel(slide, 0.72, 5.52, 5.65, 1.00, fill="FFF5F3", line="E8C0BA")
    add_text(slide, "Late removal", 1.00, 5.74, 1.65, 0.25, size=11, color=RED, bold=True)
    add_text(slide, "modest ranking gain, substantially worse NLL", 1.00, 6.08, 4.80, 0.24, size=11.2, bold=True)
    add_panel(slide, 6.68, 5.52, 5.88, 1.00, fill="F2F8F5", line="BFD8CA")
    add_text(slide, "Refinement", 6.96, 5.74, 1.65, 0.25, size=11, color=GREEN, bold=True)
    add_text(slide, "better ranking and Brier; NLL near baseline", 6.96, 6.08, 5.05, 0.24, size=11.2, bold=True)
    add_text(slide, "Held-out utility is not direct label-correctness evidence.", 0.76, 6.72, 7.4, 0.27, size=10.5, color=MUTED, bold=True)


def build_slide_10(presentation: Presentation) -> None:
    slide = new_slide(presentation, 9, backup=True)
    add_picture_fit(slide, SYNTHETIC_SENSITIVITY, 0.55, 1.25, 9.00, 5.45)
    add_panel(slide, 9.72, 1.44, 2.95, 4.60, fill="F6F8FB", line=LINE)
    add_text(slide, "Informative OOF", 9.98, 1.80, 2.30, 0.25, size=11, color=GREEN, bold=True)
    add_text(slide, "truth AUROC 0.979\nmatching-target MAE 0.019", 9.98, 2.22, 2.30, 0.72, size=13, bold=True)
    add_text(slide, "Weak OOF", 9.98, 3.44, 2.30, 0.25, size=11, color=RED, bold=True)
    add_text(slide, "truth AUROC 0.668\nmatching-target MAE 0.055", 9.98, 3.86, 2.30, 0.72, size=13, bold=True)
    add_text(slide, "Metric validity and model validity are separate questions.", 9.98, 5.30, 2.30, 0.52, size=10.5, color=DARK_BLUE, bold=True)


def build_slide_11(presentation: Presentation, jobs: dict[int, dict[str, Any]]) -> None:
    slide = new_slide(presentation, 10, backup=True)
    rows = [
        ["Synthetic validation", "validate_synthetic_dqs_adjustment.py; Slurm 260284"],
        ["Real quality figures", "build_evaluation_followup_quality_figures.py"],
        ["Own-top20 protocol", "evaluation_followup_20260713/own_top20_refinement_protocol.md"],
        ["Performance evaluator", "evaluate_own_top20_refinement_endpoints.py"],
        ["Quality evaluator", "evaluate_own_top20_refinement_quality.py"],
        ["One-command final evaluation", "run_own_top20_followup_evaluation.sh"],
        ["Frozen code", "protocol_snapshot_v5_binary_label_locked/checksums.sha256"],
    ]
    add_table(slide, ["Evidence", "Exact source"], rows, 0.62, 1.34, 12.05, 4.78, col_widths=[1.10, 2.90], font_size=10.3)
    status_text = " | ".join(f"s{seed}:{jobs[seed]['job_id']} {jobs[seed]['state']}" for seed in SEEDS)
    add_panel(slide, 0.66, 6.32, 11.96, 0.58, fill="F2F6FB", line="C7D5E4")
    add_text(slide, status_text, 0.88, 6.49, 11.52, 0.22, size=8.8, color=DARK_BLUE, bold=True, align=PP_ALIGN.CENTER)


def build_deck() -> dict[str, Any]:
    required = [SYNTHETIC_TRAJECTORY, SYNTHETIC_SENSITIVITY, ENTRY_DQS_FIGURE, SAMPLE_HEALTH_FIGURE, OLD_HIERARCHICAL_FIGURE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing slide evidence: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = current_job_snapshot()
    statistics, complete = endpoint_rows()

    presentation = Presentation()
    presentation.slide_width = Inches(SLIDE_W)
    presentation.slide_height = Inches(SLIDE_H)

    build_slide_1(presentation)
    build_slide_2(presentation)
    build_slide_3(presentation)
    build_slide_4(presentation)
    build_slide_5(presentation, jobs)
    build_slide_6(presentation, statistics, complete)
    build_slide_7(presentation)
    build_slide_8(presentation)
    build_slide_9(presentation)
    build_slide_10(presentation)
    build_slide_11(presentation, jobs)

    notes_added = sum(add_note(slide, NOTES[index]) for index, slide in enumerate(presentation.slides))
    presentation.save(OUTPUT_PPTX)

    plan = [
        {
            "slide_number": index + 1,
            "title": title,
            "section": "main" if index < 7 else "backup",
            "speaker_note": NOTES[index],
            "source": SOURCES[index],
        }
        for index, title in enumerate(TITLES)
    ]
    (OUTPUT_DIR / "deck_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    build_notes = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "endpoint_results_available": complete,
        "own_top20_root": str(OWN_ROOT),
        "jobs": jobs,
        "interpretation": (
            "The previous hierarchical-uncertainty and proper-score pages were not presented. "
            "Hierarchical uncertainty is now part of the main follow-up; proper scores remain backup evidence."
        ),
    }
    (OUTPUT_DIR / "build_notes.json").write_text(json.dumps(build_notes, indent=2) + "\n", encoding="utf-8")
    return {"notes_added": notes_added, "endpoint_complete": complete, "jobs": jobs}


def postflight(build: dict[str, Any]) -> dict[str, Any]:
    presentation = Presentation(OUTPUT_PPTX)
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "status": "passed" if passed else "failed", "detail": detail})

    add("slide_count", len(presentation.slides) == len(TITLES), len(presentation.slides))
    add("aspect_ratio", abs(presentation.slide_width / presentation.slide_height - 16 / 9) < 0.01, presentation.slide_width / presentation.slide_height)
    titles = []
    all_in_bounds = True
    shape_counts = []
    picture_count = 0
    table_count = 0
    source_count = 0
    for slide in presentation.slides:
        title_shape = next((shape for shape in slide.shapes if shape.name == "slide-title"), None)
        titles.append(title_shape.text.strip() if title_shape is not None else "")
        shape_counts.append(len(slide.shapes))
        source_count += int(any(shape.name == "source-note" for shape in slide.shapes))
        for shape in slide.shapes:
            all_in_bounds = all_in_bounds and shape.left >= 0 and shape.top >= 0
            all_in_bounds = all_in_bounds and shape.left + shape.width <= presentation.slide_width + 10
            all_in_bounds = all_in_bounds and shape.top + shape.height <= presentation.slide_height + 10
            picture_count += int(shape.shape_type == 13)
            table_count += int(getattr(shape, "has_table", False))
    add("titles", titles == TITLES, titles)
    add("nonblank_slides", min(shape_counts) >= 8, shape_counts)
    add("shape_bounds", all_in_bounds, "all shapes inside 16:9 canvas")
    add("source_notes", source_count == len(TITLES), source_count)
    add("speaker_notes", build["notes_added"] == len(TITLES), build["notes_added"])
    add("evidence_images", picture_count == 5, picture_count)
    add("native_tables", table_count == 7, table_count)
    try:
        with zipfile.ZipFile(OUTPUT_PPTX) as archive:
            bad_member = archive.testzip()
        add("pptx_zip_integrity", bad_member is None, bad_member or "all entries readable")
    except zipfile.BadZipFile as exc:
        add("pptx_zip_integrity", False, str(exc))

    failed = [check for check in checks if check["status"] != "passed"]
    result = {
        "status": "passed" if not failed else "failed",
        "summary": {"checks": len(checks), "passed": len(checks) - len(failed), "failed": len(failed)},
        "checks": checks,
    }
    (OUTPUT_DIR / "postflight_checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if failed:
        raise RuntimeError(f"Deck postflight failed: {[item['name'] for item in failed]}")
    return result


def main() -> int:
    build = build_deck()
    checks = postflight(build)
    print(
        json.dumps(
            {
                "output": str(OUTPUT_PPTX),
                "endpoint_results_available": build["endpoint_complete"],
                "postflight": checks["summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
