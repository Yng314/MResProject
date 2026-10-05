# Dual-Reference Confident-Learning Detection Benchmark

## Status and scope

This protocol is locked before running the formal dual-reference evaluation,
but after the frozen CL scores, the Med-PaLM analysis, and rough feasibility
counts from the MIMIC-CXR-JPG 2.1.0 file were already available. It is therefore
a transparent retrospective validation plan, not a prospective preregistration.

The experiment asks whether the current confident-learning (CL) evidence finds
expert-confirmed label issues under the actual study-level hard-flag/top-20%
selection policy. It runs no new model training, no LLM review, no iterative
cleaning, and no held-out performance evaluation.

## Frozen evidence

- Input: the outcome-blind `full_test_cl_entry_scores.csv` produced by the
  completed Med-PaLM detection benchmark.
- Candidate population: 3,050 official-test AP/PA studies and 8,288 valid
  study-finding entries under the current 12-label U-Ones task.
- Estimators: frozen baseline seeds `13`, `42`, `97`, and `123`, plus their
  probability ensemble.
- All baseline checkpoints were trained on the MIMIC-CXR training split. The
  score file was generated before either expert reference is joined and its
  manifest states `expert_reference_used=false`.

## Independent references

References are evaluated separately because their sampling and annotation
protocols differ.

### Primary: MIMIC-CXR-JPG 2.1.0 radiologist panel

- Source: `mimic-cxr-2.1.0-test-set-labeled.csv`.
- Annotation: one radiologist; entries follow the four-state CheXpert schema.
- AP/PA/current-schema overlap expected after joining the frozen score file:
  1,796 valid entries, 563 studies, 251 subjects.
- Fixed U-Ones mapping: expert/current `-1 -> 1`, `0 -> 0`, `1 -> 1`.
- Expected current-label issues: 109/1,796.

### Secondary: Med-PaLM hard-case panel

- Source: 498 protocol-compatible study-finding entries selected from
  Med-PaLM-vs-CheXpert disagreements and adjudicated by three radiologists.
- Expected overlap: 498 entries, 457 studies, 179 subjects.
- Expected current-label issues: 127/498 under the same U-Ones mapping.
- This panel is explicitly conditional on external hard-case selection.

## Outcome isolation

The program has separate `select` and `evaluate` stages.

1. `select` may read only the frozen score file and its outcome-blind manifest.
2. It writes the complete suspicious-study ranking, exact top-20% selection,
   expanded hard-flagged entries, input/output hashes, and a completion marker.
3. `evaluate` verifies those hashes before reading either expert reference.
4. No selection parameter, score, threshold, checkpoint, or cohort definition
   may be changed after the references are joined.

Blind-stage artifacts must not contain expert labels, reference labels,
true-issue indicators, expected actions, or reader agreement.

## Exact study-level selection policy

For each estimator independently:

1. A study is suspicious when at least one valid entry has `cl_hard_issue=1`.
2. Study quality is the mean entry-level self-confidence over all currently
   valid entries in that study.
3. Suspicious studies are sorted by ascending study quality, then descending
   number of hard-flagged entries, then ascending `study_id` for deterministic
   tie breaking.
4. Select `round(0.20 * number_of_suspicious_studies)` studies, with a minimum
   of one when the pool is non-empty.
5. Expand only the CL-hard-flagged entries from those selected studies.

This reproduces the current top-fraction policy at study level. It does not rank
only the expert-labelled entries and does not force expert-reference coverage.

## Endpoints

The ensemble is primary; four individual seeds are robustness analyses.

### Entry-level detection

- expert issue prevalence;
- AUPRC and AUPRC minus prevalence for continuous CL suspiciousness;
- AUROC;
- CL hard-flag precision, recall, and enrichment over prevalence.

### Exact operational policy

- number of suspicious studies, selected studies, and expanded entries in the
  complete test candidate pool;
- expert-reference entries covered by the expanded selection;
- true issues captured, precision, recall, and enrichment within each reference;
- 10,000 matched random-policy replicates that select the same number of studies
  uniformly from that estimator's CL-suspicious study pool and expand their
  hard-flagged entries;
- one-sided empirical random-policy tail probability for capturing at least the
  observed number of known issues.

The random-policy comparator tests prioritization within the CL-suspicious pool.
Hard-flag enrichment separately tests whether CL flagging itself enriches issues.

### Heterogeneity

- per-finding entry count, issue count, prevalence, AUPRC, hard-flag metrics,
  and exact-policy capture;
- findings with inadequate outcome support remain descriptive.

## Uncertainty

- 2,000 percentile bootstrap replicates resample subjects and retain all their
  annotated entries.
- Report 95% intervals for prevalence, AUPRC, AUPRC-minus-prevalence, AUROC,
  hard-flag precision/recall/enrichment, and exact-policy precision/recall/
  enrichment.
- Report exact empirical random-policy tail probabilities but do not use a
  binary `p<0.05` gate as the sole conclusion.
- Effect size, interval width, estimator consistency, and reference-cohort
  differences must be reported together.

## Interpretation rules

Strong conditional evidence requires AUPRC above prevalence, hard-flag
enrichment above one, and exact-policy capture above matched random expectation,
with directionally consistent individual seeds. Mixed endpoints must be reported
as mixed evidence rather than collapsed into a pass/fail label.

The defensible population is the corresponding expert-reference panel. This
experiment cannot estimate precision or recall for all MIMIC-CXR training
entries because neither reference is a dense, independently sampled annotation
of that population.

## Reproducibility

- Primary output directory must be immutable after `.evaluation_complete`.
- The frozen score input, protocol, program, blind selections, and reference
  inputs are recorded by SHA-256.
- All estimators must have exactly 8,288 score rows and identical entry keys.
- Reference and selection joins are one-to-one by `entry_key`.
- Formal outputs include machine-readable metrics, bootstrap replicates,
  random-policy replicates, per-finding results, a figure, and a summary JSON.
