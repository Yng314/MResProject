# VinDr Noise-Direction and DQS-Calibration Sensitivity Protocol

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-04
- Verification Status: UNVERIFIED
- Version Label: vindr_direction_sensitivity_protocol_v1
- Study Status: PLANNED, PRE-OUTCOME FOR ALL NEW SCENARIOS
- Parent Study: `vindr_known_gt_cl_protocol_20260804.md`
- Parent Result: `vindr_known_gt_cl_validation_20260804_zh.md`

## Experiment Overview

- **Title**: VinDr noise-direction robustness and DQS calibration
- **Type**: controlled label-corruption benchmark
- **Objective**: determine whether OOF confident-learning evidence detects
  false-positive and false-negative label errors equally well, and whether raw
  entry DQS tracks known dataset quality across multiple corruption levels.
- **Primary motivation**: the parent 20%-balanced experiment found mean hard-flag
  recall `0.9107` for `0 -> 1` errors but only `0.1772` for `1 -> 0` errors.
- **Prospective boundary**: this protocol is a post-result sensitivity study.
  The parent four-seed result is known and is not relabelled as unseen
  confirmatory evidence.

## Research Questions

1. Does the `0 -> 1` versus `1 -> 0` recall gap persist when the two regimes
   contain exactly the same number of injected errors?
2. Does CL-first remain better than random review across noise rates and
   corruption regimes?
3. Is any incremental CL-hard-filter benefit over global self-confidence stable
   across the new scenario matrix?
4. Does raw entry DQS vary monotonically with known true quality, with low
   absolute calibration error?

## Inputs

| Input | Locked source | Role |
|---|---|---|
| VinDr consensus labels | verified `image_labels_test.csv` | private clean reference |
| Image index | parent benchmark `image_index.csv` | immutable image order and IDs |
| Frozen XRV features | parent `xrv_features.npz` | shared outcome-blind image representation |
| OOF engine | parent `vindr_known_gt_cl_benchmark.py` | unchanged nested four-fold training and CL evidence |

The source DICOMs and PNGs are not reprocessed. No LLM API is used.

## Labels and Unit of Analysis

- Unit: one `(image_id, label_name)` entry.
- Samples: 3,000 images.
- Entries: 18,000 per scenario and seed.
- Labels: Atelectasis, Cardiomegaly, Consolidation, Lung Opacity, Pleural
  effusion, and Pneumonia.
- The consensus reference remains a benchmark reference rather than infallible
  clinical truth.

## Scenario Matrix

### Seeds

Six seeds are frozen before outcome generation:

`13, 42, 97, 123, 211, 307`

Seeds `13/42/97/123` overlap the parent study; `211/307` are new prospective
replication seeds. Results must identify these two blocks rather than treating
all six as an untouched replication of the parent finding.

### Error-Rate Anchors

- clean anchor: `0%`
- noisy anchors: `10%`, `20%`, and `30%` of positive support per direction in
  the balanced regime

For label `l` with positive support `P_l`, define `k_l(r) = round(r * P_l)`.
Every noisy regime uses exactly `2 * k_l(r)` errors for that label:

| Regime | `0 -> 1` errors | `1 -> 0` errors | Total |
|---|---:|---:|---:|
| balanced | `k_l(r)` | `k_l(r)` | `2k_l(r)` |
| false-positive-only | `2k_l(r)` | `0` | `2k_l(r)` |
| false-negative-only | `0` | `2k_l(r)` | `2k_l(r)` |

Thus all three regimes at a given rate have the same total error count and true
quality. The `30%` level is feasible because `2k_l(0.30) < P_l` for every
included label.

There are ten unique scenarios per seed: one clean anchor plus three rates by
three noisy regimes, for 60 blind OOF runs in total.

## Corruption Assignment

- Corruptions are sampled before OOF evidence and without using image features,
  OOF probabilities, CL scores, or prior detection outcomes.
- Deterministic scenario-specific random streams are derived from
  `(study_version, seed, label, regime, rate, direction)`.
- Selection is without replacement.
- Fold assignment is fixed within seed and shared across all scenarios, so
  scenario contrasts are paired by seed and image partition.
- Exact selected-entry keys, counts, private references, blind cohorts, and
  hashes are saved before any OOF task starts.

## Outcome Isolation

1. `prepare` creates blind noisy cohorts and physically separate private
   references for all scenarios.
2. `prepare` also writes a dedicated `blind_run_manifest.csv` containing only
   array index, scenario condition, seed, and blind input/output paths. It
   excludes injected-error counts, true quality, private hashes, and reference
   columns.
3. Each OOF array task reads only this blind manifest and receives one blind
   cohort plus the frozen features.
4. The private evaluator starts only after all 60 blind-run completion markers
   exist.
5. Evaluation joins private outcomes by `(image_id, label_name)` and then writes
   metrics.
6. `verify` checks exact scenario/seed coverage, row counts, hashes, noise counts,
   folds, outputs, and completion markers.

Clean/reference/injected/error columns are forbidden from blind tables and
blind evidence artifacts.

## OOF and CL Configuration

- Frozen XRV DenseNet-121 features from the parent benchmark.
- Four image-level outer folds, fixed within seed across scenarios.
- Independent six-label linear head per outer fold.
- Inner split chosen by the first deterministic candidate with at least three
  positive and three negative examples for every label in inner train and inner
  validation.
- Adam, learning rate `1e-3`, batch size 128, maximum 50 epochs, patience 8.
- CL estimated independently per binary label from OOF probabilities.

## Comparators and Budgets

- CL-first: hard-flagged entries first, ordered by observed-label
  self-confidence within strata.
- Global self-confidence.
- Predictive entropy.
- Exact hypergeometric matched-random reference.

Primary prioritization uses a review budget equal to the known number of
injected errors. Secondary budgets are 1%, 2%, 5%, and 10% of entries.

## Primary Endpoints

1. Hard-flag recall for false-positive-only and false-negative-only scenarios.
2. Seed-level mean direction gap across rates:
   `recall(false-positive-only) - recall(false-negative-only)`.
3. CL-first injected-error AUPRC minus injected-error prevalence.
4. CL-hard enrichment over injected-error prevalence.
5. CL-first minus self-confidence recall at the true-error-count budget.
6. DQS calibration MAE and Spearman monotonicity across clean/10%/20%/30%
   quality anchors within each seed and regime.

## Secondary Endpoints

- Per-label recall, precision, AUPRC, AUROC, and enrichment.
- Noisy-label and clean-reference OOF AUROC.
- Hard issue count and DQS signed error.
- CL-first minus predictive-entropy equal-budget recall.
- 1%, 2%, 5%, and 10% budget curves.
- Image-cluster bootstrap intervals for hard recall, precision, direction recall,
  true quality, and raw DQS.

## Statistical Analysis

- Seed is the primary independent repetition unit.
- Scenario/rate contrasts are paired within seed.
- Report all six seed values and distinguish the parent-overlap block from the
  two new replication seeds.
- Use a two-sided exact sign-flip test on six seed-level mean contrasts.
- With six nonzero, fully concordant seed differences, the minimum attainable
  two-sided exact p-value is `0.03125`.
- Apply Holm correction to the two method-contrast family members: CL-first
  versus self-confidence and CL-first versus predictive entropy.
- Use 1,000 image-cluster bootstrap replicates per scenario/seed for descriptive
  within-cohort uncertainty. These intervals do not replace seed-level exact
  inference.
- Use exact hypergeometric random-selection expectations and tail probabilities;
  no Monte Carlo random-review approximation is needed.

## Predeclared Decision Rules

### Detection Robustness

Supported only if, for every noisy scenario and all six seeds:

1. CL-first AUPRC exceeds injected-error prevalence;
2. CL-hard enrichment exceeds 1; and
3. CL-first true-error-budget recall exceeds exact random expected recall.

### Directional Asymmetry

- **Confirmed** if every seed's mean direction gap across `10/20/30%` exceeds
  `0.20` and the six-seed exact sign-flip p-value is below `0.05`.
- **Direction-invariant performance supported** only if the absolute mean gap is
  at most `0.10` at every rate and neither direction has mean recall below `0.50`.
- Otherwise the result is classified as mixed.

### Incremental CL-Hard-Filter Value

Supported only if the seed-mean CL-first minus self-confidence recall difference
across the nine noisy scenarios:

1. exceeds `0.02` in every seed;
2. has exact sign-flip `p < 0.05`; and
3. survives Holm correction within the two-comparator family.

### DQS Calibration

Supported only if, for every seed and each corruption regime:

1. Spearman correlation between raw DQS and true quality is at least `0.95`;
2. MAE is at most `0.01`; and
3. raw DQS decreases from clean to 10% to 20% to 30% without reversal.

Slope and intercept are reported as descriptive calibration diagnostics, not
used as hard gates because the true-quality range is narrow.

## Interpretation Boundaries

- The experiment tests controlled corruption mechanisms, not naturally produced
  report-label errors.
- Direction-only regimes intentionally alter class prevalence; matched total
  error count controls workload and true quality, not prevalence shift.
- The parent result motivated this study. New outcomes can prospectively test
  the follow-up hypotheses, but pooled six-seed results are not an untouched
  replication of the parent experiment.
- The experiment evaluates detection and DQS. It does not test LLM correction,
  iterative convergence, downstream retraining benefit, or clinical utility.

## Expected Outputs and Success Checks

| Output | Success criterion |
|---|---|
| scenario manifest | 10 scenarios, six seeds, exact matched counts |
| blind-run manifest | 60 rows and no outcome/reference columns |
| blind runs | 60 markers, 3,000 OOF rows and 18,000 evidence rows each |
| overall and budget tables | complete scenario-seed-method grid |
| per-label/direction tables | complete six-label grid |
| bootstrap table | 60,000 rows |
| calibration table | 18 seed-regime rows |
| contrast table | direction and two comparator contrasts with exact/Holm p-values |
| figures | direction-recall, DQS-calibration, budget and AUPRC plots |
| final summary | all decision gates and interpretation labels |
| verification marker | written only after every check passes |

## Monitoring

- Prepare smoke timeout: 30 minutes.
- OOF array timeout: two hours per task, maximum three concurrent tasks.
- Evaluation timeout: three hours.
- Slurm `END,FAIL` email is required.
- No long-running process may execute on the login server.
