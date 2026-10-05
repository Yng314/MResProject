# VinDr Iterative CL Oracle-Cleaning Protocol

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-05
- Verification Status: PLANNED
- Version Label: vindr_iterative_oracle_cleaning_v1

## Objective

Test whether repeated dynamic OOF confident-learning selection discovers
remaining known label errors efficiently, whether selected-only perfect
correction progressively improves fixed-denominator true dataset quality, and
whether raw entry DQS follows that known quality trajectory.

This experiment isolates the detection and selection stage. It assumes a
perfect downstream correction action and does not evaluate an LLM.

## Frozen Starting State

- Source: completed VinDr exact-global symmetric-noise benchmark.
- Cohort: 3,000 images and six binary findings, giving 18,000 entries.
- Initial corruption: exact 20% symmetric entry noise, or 3,600 known errors.
- Initial known quality: 0.80.
- Seeds: `13, 42, 97, 123, 211, 307`.
- Loop-0 MobileNet OOF probabilities, folds and CL evidence are reused exactly.

## Dynamic Eight-Loop Procedure

For each seed and loop:

1. Read the current OOF/CL evidence and cumulative blind review history.
2. Exclude every entry reviewed in any earlier loop.
3. Restrict candidates to the current CL hard-issue pool.
4. Select the top 20% of that unreviewed pool by `cl_first_score`, rounding the
   count upward and using frozen deterministic tie-breaking.
5. Write and hash the outcome-blind selection before accessing private truth.
6. Privately replace selected erroneous labels with their consensus GT value;
   selected correct labels remain unchanged. Mark every selected entry reviewed.
7. Rerun the same four-fold nested MobileNet OOF and CL evidence on the updated
   blind cohort before the next loop.

The fixed 18,000-entry denominator never changes. No entry is removed or
masked, and no reviewed entry can re-enter a later selection.

## Model and Training

- Same verified one-channel, six-output MobileNetV3-small OOF engine as the
  global-noise benchmark.
- Four outer folds, maximum 100 epochs, inner patience 10 and best-state recovery.
- Unweighted masked BCE-with-logits, Adam `1e-3`, batch size 32.
- Shared fold assignments remain fixed within seed across all loops.

## Comparators

- `dynamic_cl`: retrain OOF and rerank after every selected-only correction.
- `frozen_loop0_cl`: consume the original Loop-0 ranking at the same cumulative
  review budgets without reranking.
- `random_review`: 10,000 full-entry random permutations at the same budgets.

## Primary Outcomes

1. Cumulative recall of the fixed 3,600 injected errors versus review budget.
2. Per-loop selection precision and newly corrected true errors.
3. Fixed-denominator true quality after every action.
4. Dynamic-minus-frozen and dynamic-minus-random discovery-curve area by seed.
5. Raw DQS versus true quality, including Spearman correlation and the sign
   agreement of every adjacent-loop change.
6. Remaining `0 -> 1` and `1 -> 0` errors after each loop.

## Statistical Analysis

- Training/corruption seed is the repetition unit.
- Report all six seed trajectories, means, standard deviations and effect sizes.
- Use two-sided six-seed exact sign-flip tests for dynamic-minus-frozen and
  dynamic-minus-random discovery-curve area.
- Treat those two tests as one family and apply Holm correction in the final
  interpretation.

## Outcome Isolation

- Selection commands receive no clean labels, injected flags or true quality.
- Private GT is read only after the selected-entry CSV and its hash are frozen.
- Post-action OOF training receives only the updated blind cohort and never the
  private reference or private metrics.
- Per-seed private evaluation runs only after all eight loop markers pass.

## Interpretation Boundary

- Positive results establish an oracle upper bound for iterative CL-guided
  cleaning under controlled VinDr corruption.
- They do not prove that an LLM always applies the correct action; that component
  is supported separately by the Med-PaLM correction benchmark.
- Symmetric VinDr corruption remains a stress test and does not reproduce the
  prevalence or direction mixture of natural MIMIC-CXR report-label errors.
