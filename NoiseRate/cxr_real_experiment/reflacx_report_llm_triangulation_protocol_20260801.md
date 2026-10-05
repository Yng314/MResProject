# REFLACX Report-Only LLM Triangulation Protocol

## Material Passport

- Origin skill: Academic Research Suite, experiment-agent
- Mode: run/validate
- Date locked: 2026-08-01
- Version: `reflacx_report_llm_triangulation_v1`
- Status before outcome: protocol defined before any API review

## Objective

For the same 3,172 valid six-label entries used by the REFLACX Phase-3
confident-learning pilot, independently extract the target finding from the
original MIMIC-CXR report with the project LLM, then compare three sources:

1. the current report-derived CheXpert/U-Ones binary label;
2. the independent report-only LLM label;
3. the REFLACX image-reader label.

This is a source-triangulation audit. It is not an expert adjudication of which
source is definitively correct.

## Blinding

The API receives only:

- entry key and identifiers needed for bookkeeping;
- target finding;
- original report text.

The API does not receive:

- current CheXpert raw or binary label;
- confident-learning score, issue flag, or selection status;
- REFLACX certainty or binary reference;
- image pixels.

The model returns `report_positive`, `report_negative`, or
`report_ambiguous`. After review, positive/negative are mapped to binary 1/0;
ambiguous maps to the existing refinement action `mask`. Relative to the
hidden current label, a matching definite answer becomes `keep` and the
opposite answer becomes `relabel`.

## Cohort

- REFLACX Phase 3 strict ontology mapping.
- Six findings: Atelectasis, Cardiomegaly, Consolidation, Edema, Lung Lesion,
  and Pneumothorax.
- Include exactly the entries with a valid current CheXpert raw label in
  `{-1, 0, 1}` under U-Ones (`-1/1 -> 1`, `0 -> 0`).
- Expected scope: 1,646 studies, 1,386 subjects, and 3,172 entries.

## Reference Scopes

- Primary: REFLACX certainty `>=3` is positive and `<3` is negative.
- Sensitivity: exclude certainty `3`; certainty `>=4` is positive and `<3` is
  negative.

REFLACX is treated as an image-reader reference, not incontrovertible ground
truth. Phase-3 reader variation and report-versus-image task mismatch remain
interpretive limitations.

## Primary Three-Way Outcomes

For definite LLM outputs, report:

- all three sources agree;
- LLM and REFLACX agree while CheXpert differs;
- LLM and CheXpert agree while REFLACX differs;
- CheXpert and REFLACX agree while LLM differs;
- LLM report is ambiguous.

Within entries where CheXpert and REFLACX disagree, the second pattern is
evidence consistent with a CheXpert report-extraction mismatch; the third is
evidence consistent with a report/image-reader mismatch. Neither pattern alone
proves causation because the LLM and REFLACX reader can also be wrong.

## Quantitative Outputs

- LLM definite coverage.
- LLM-CheXpert retained and coverage-adjusted agreement.
- CheXpert-REFLACX baseline agreement.
- LLM-REFLACX retained accuracy and coverage-adjusted correct mass.
- Difference between LLM coverage-adjusted correct mass and the original
  CheXpert-REFLACX agreement.
- Three-way pattern counts and proportions overall and per finding.
- Subject-cluster bootstrap 95% intervals for primary aggregate metrics.
- Secondary stratification by the frozen CL issue flag and selected top-20
  candidate status from the completed pilot.

## Execution Lock

- Model: `gpt-5.4`.
- Temperature: `0`.
- One entry-specific API call per report-finding pair.
- Concurrency: `10`.
- Append-only successful responses and resumable retries.
- API review must finish with 3,172 unique entries and zero unresolved errors
  before private comparison is allowed.
