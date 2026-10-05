# VinDr Iterative Oracle-Cleaning Two-Seed Replication Protocol

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-06
- Verification Status: PLANNED
- Version Label: vindr_iterative_oracle_replication_2seed_v1

## Status and Motivation

This is a post-result, fixed-size precision extension of the completed six-seed
VinDr iterative oracle-cleaning experiment. It was initiated after observing
that both predeclared six-seed contrasts had raw exact `p = 0.03125` and
Holm-adjusted `p = 0.0625`.

The extension must not be described as part of the original confirmatory design
or as an outcome-independent sample-size decision. The combined eight-seed
exact/Holm p-values are descriptive post-result evidence.

## Frozen Replication Block

- New seeds: `509` and `701`.
- Both IDs were verified absent from all existing experiment result directories
  before any new corruption, OOF prediction, selection, or outcome was created.
- Both seeds must be retained in every analysis regardless of effect direction.
- No interim analysis is permitted between the two seeds.
- No additional seed may be added to this replication block based on its p-value.

## Inputs and Corruption

- Same verified VinDr cohort: 3,000 images and six consensus binary findings.
- Same exact-global symmetric-noise generator and protocol as the completed
  benchmark.
- Same four anchors are prepared (`clean`, 10%, 20%, 30%) so fold selection and
  preparation logic exactly match the prior six seeds.
- The iterative experiment uses only the exact 20% anchor: 18,000 entries,
  3,600 injected errors, and known initial quality 0.80.
- Each seed receives a newly generated support-gated four-fold assignment and a
  newly trained Loop-0 MobileNet OOF model. No prior seed output is reused.

## Iterative Procedure

For each new seed, run the unchanged eight-loop procedure from
`vindr_iterative_oracle_cleaning_protocol_20260805.md`:

1. Exclude every entry reviewed in any earlier loop.
2. Restrict to the current CL hard-issue pool.
3. Select the top 20% by `cl_first_score`, rounding upward with deterministic
   tie-breaking.
4. Freeze and hash the outcome-blind selection.
5. Restore selected erroneous labels to consensus GT; leave selected correct
   labels unchanged and mark every selected entry reviewed.
6. Rerun nested four-fold MobileNet OOF on the updated blind cohort.

No entry is removed or masked. The denominator remains 18,000 throughout.

## Model and Training

- Same one-channel, six-output scratch MobileNetV3-small OOF engine.
- Four outer folds with support-gated inner early stopping.
- Maximum 100 epochs, patience 10, best-state recovery.
- Unweighted masked BCE-with-logits, Adam `1e-3`, batch size 32.

## Outcomes and Comparators

- Same cumulative review budgets within each seed.
- `dynamic_cl`: rerun OOF and ranking after every correction loop.
- `frozen_loop0_cl`: consume the original Loop-0 ranking.
- `random_review`: 10,000 matched full-entry random permutations.
- Report discovery AUDC, final recall, known quality, raw DQS, per-loop precision,
  and remaining `0 -> 1` and `1 -> 0` errors.

## Analysis

1. Preserve the original six-seed analysis unchanged.
2. Report seed509 and seed701 effects individually as the replication block.
3. Report replication-block direction agreement and effect sizes without claiming
   a standalone significance test; two seeds cannot support a useful two-sided
   exact sign-flip threshold.
4. Produce a combined eight-seed descriptive analysis using the same two-sided
   exact sign-flip tests for dynamic-minus-frozen and dynamic-minus-random AUDC.
5. Treat the two contrasts as one family and apply Holm correction.
6. Report raw and adjusted p-values regardless of whether they cross 0.05.

## Interpretation Boundary

- Positive results strengthen robustness and precision for the controlled
  perfect-correction upper bound.
- They do not retroactively make the original sample-size choice confirmatory.
- They do not validate an LLM correction stage or natural MIMIC-CXR noise.
- Direction-specific failure, especially unchanged `1 -> 0` errors, must remain
  visible in the final interpretation.
