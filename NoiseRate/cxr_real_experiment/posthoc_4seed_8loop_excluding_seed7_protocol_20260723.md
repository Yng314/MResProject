# Post-hoc Four-Seed, Eight-Loop Sensitivity Protocol

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + validate
- Origin Date: 2026-07-23
- Verification Status: PLANNED
- Version Label: posthoc_4seed_excluding_seed7_v1

## Status

This analysis was requested after the complete five-seed results had been
inspected and after seed 7 was identified as the only seed where Loop 8 simple
removal exceeded Loop 8 LLM refinement. Seed 7 has no known data, execution, or
protocol invalidity.

The exclusion is therefore outcome-informed and post hoc. This analysis is a
sensitivity description of seeds `13, 42, 97, 123`; it does not replace the
formal five-seed analysis and must not be described as preregistered or
confirmatory.

## Fixed Inputs

- Included seeds: `13, 42, 97, 123`
- Excluded seed: `7`
- Exclusion reason: user-requested outcome-informed sensitivity analysis
- Loops: `1-8`
- Shared held-out test set: `605` studies and `12` labels
- Methods: baseline, simple removal, seed-specific own-top20 LLM refinement
- Training, OOF, action, and prediction artifacts: unchanged from the formal
  five-seed experiment

## Endpoints and Contrasts

The endpoint definitions, support sensitivity, proper scores, frozen-evidence
DQS metrics, and eight contrasts are identical to
`unified_5seed_8loop_evaluation_protocol_20260723.md`.

The primary contrast remains `refine_L8_vs_remove_L8` for study-weighted AUROC.
Full Loop1-8 trajectories must be reported; no best-loop-only selection is
allowed.

## Statistical Analysis

- Four-seed paired mean, SD, paired standardized effect, and
  assumption-sensitive paired t interval
- Two-sided exact sign-flip p-value
- Paired t-test as supporting evidence
- Holm adjustment across the same eight contrasts within each metric
- `2,000` conditional shared-study bootstrap draws
- `2,000` hierarchical seed-and-study bootstrap draws
- Bootstrap seed `20260723`
- All-label and minority-support-at-least-five scopes

With four seeds, the minimum attainable non-zero two-sided exact sign-flip
p-value is `0.125`. A narrow interval after excluding the only discordant seed
must be interpreted as conditional on that exclusion, not as evidence that the
excluded seed was invalid.

## Required Reporting

1. Label every output as `post-hoc four-seed sensitivity`.
2. State that seed 7 was excluded after viewing the five-seed outcomes.
3. Report the four-seed result without presenting it as the primary experiment.
4. Preserve the five-seed report and output directory unchanged.
5. Run the same dynamic QA checks for 68 stage rows, 816 per-label rows, eight
   Loop 8 transaction markers, and bootstrap point alignment.
