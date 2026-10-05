# Five-Seed, Eight-Loop Unified Evaluation Protocol

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + validate
- Origin Date: 2026-07-23
- Verification Status: PLANNED
- Version Label: unified_5seed_8loop_protocol_v1

## Status and Scope

This protocol is frozen after all model outcomes were generated but before the
new unified analysis is executed. It is not a prospective preregistration and
does not make the resulting inference confirmatory. Its purpose is to prevent
further endpoint and contrast selection while producing one auditable analysis
of the completed five-seed, eight-loop experiment.

No model training, OOF detection, label action, or LLM call is performed by this
evaluation.

## Research Question

Under the completed MobileNetV3 pipeline, does seed-specific own-top20 LLM
refinement provide a better held-out performance and data-quality/coverage
tradeoff than simple removal after the same number of iterative cleaning loops?

## Inputs

- Seeds: `7, 13, 42, 97, 123`
- Loops: `1-8`
- Shared held-out test set: 605 studies and 12 labels
- Baseline/removal root:
  `/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_noes50_clean_3seed/20260707_123320`
- Corrected-binary own-top20 refinement root:
  `/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_own_top20_binary_v5_llm_refine_5seed/20260714_065539`

## Endpoints

### Primary performance endpoint

- Study-weighted AUROC across all 12 labels.

### Required performance sensitivity

- Study-weighted AUROC restricted to labels with minority test support at least
  five.

### Supporting performance metrics

- Study macro AUROC
- Macro average precision
- Micro Brier score
- Micro negative log-likelihood

### Data-quality and coverage metrics

- Raw entry DQS
- Coverage-adjusted entry DQS
- Valid-entry coverage
- Raw sample DQS, defined as the fraction of retained evaluable samples with no
  Confident Learning issue
- Coverage-adjusted sample DQS
- Sample coverage

Entry and sample quality comparisons use each seed's unchanged initial OOF
probabilities for all post-action stages. This frozen evidence isolates changes
in labels and coverage but remains a model-consistency diagnostic rather than
expert-adjudicated label accuracy.

## Frozen Contrasts

The left stage minus right stage is reported. For lower-is-better metrics, the
sign is reversed in the `mean_improvement` field so that positive always means
the left stage is better.

1. Primary: `refine_L8_vs_remove_L8`
2. Secondary: `refine_L8_vs_baseline`
3. Secondary: `remove_L8_vs_baseline`
4. Secondary: `refine_L5_vs_remove_L5`
5. Secondary: `refine_L5_vs_baseline`
6. Secondary: `remove_L5_vs_baseline`
7. Secondary: `refine_L8_vs_refine_L5`
8. Secondary: `remove_L8_vs_remove_L5`

The full Loop1-8 trajectories are reported for both methods. No best-loop-only
claim is permitted.

## Statistical Analysis

For every seed-paired metric contrast:

- Mean paired difference and SD
- Paired standardized effect `dz`
- 95% paired t interval, labelled assumption-sensitive because `n=5`
- Two-sided exact sign-flip p-value
- Paired t-test p-value as supporting evidence only
- Holm adjustment across the eight frozen contrasts within each metric

For study-weighted AUROC:

- Conditional shared-study bootstrap, preserving the same resampled study
  indices across all five seeds
- Hierarchical seed-and-study bootstrap
- 2,000 bootstrap iterations with seed `20260723`
- Both all-label and minority-support-at-least-five scopes

Bootstrap tail masses are descriptive empirical probabilities and are not
labelled as formal p-values.

## Interpretation Rules

- Effect estimates and intervals take precedence over thresholded p-values.
- With five seeds, the minimum attainable non-zero two-sided exact sign-flip
  p-value is `0.0625`.
- A positive conditional study interval with a hierarchical interval crossing
  zero supports the observed trained models, not generalization to new pipeline
  seeds.
- Full-label and support-filtered results must be presented together.
- DQS improvement cannot be described as clinical label correctness.
- Same-loop comparison does not imply equal intervention amount or cost.

## Multiplicity

Holm adjustment is applied separately within each metric across the eight
frozen contrasts. Per-label results are exploratory and receive no confirmatory
interpretation.

## Required Quality Checks

1. Exactly 85 aligned prediction files: 5 baseline, 40 removal, 40 refinement.
2. Identical test studies, labels, valid masks, and label ordering.
3. Exactly 85 frozen-evidence quality rows and 1,020 per-label quality rows.
4. Complete Loop8 transaction markers for all ten seed-method runs.
5. No missing or infinite headline metrics.
6. All-label hierarchical point estimates match seed-paired point estimates.
7. All generated figures are non-empty.
8. ARS validation report checks all 11 statistical fallacy types.

## Planned Outputs

- Machine-readable performance, quality, bootstrap, per-label, and QA tables
- Performance, proper-score, DQS, coverage, and interval figures
- ARS-compatible Chinese validation report with Material Passport
