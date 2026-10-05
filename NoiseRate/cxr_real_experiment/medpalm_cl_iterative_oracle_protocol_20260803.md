# Med-PaLM Dynamic-OOF Iterative CL Oracle Benchmark

## Material Passport

- Origin skill: Academic Research Suite, experiment-agent
- Lock date: 2026-08-03
- Version: `medpalm_dynamic_cl_oracle_v1`
- Status before dynamic outcomes: UNVERIFIED

This plan is locked after the Med-PaLM reference and the one-shot four-model
detection result were already known. It is not a prospective preregistration.
No dynamic-loop OOF result is available at lock time.

## Research Question

At the same cumulative review budget, does recomputing subject-grouped OOF
confident-learning evidence after oracle corrections discover expert-confirmed
issues faster than keeping the initial CL ranking frozen?

The endpoint is detection efficiency, not the tautological observation that
reviewing all 498 entries eventually reveals all 127 known issues.

## Cohort and State

- Complete official MIMIC-CXR test AP/PA pool: expected 3,414 images and 3,050
  studies.
- Med-PaLM benchmark: 498 entries, 457 studies, 179 subjects, 127 expert issues,
  and 371 expert-correct entries.
- Initial labels use the same U-Ones binary projection as the main pipeline.
- A reviewed expert-correct entry remains unchanged.
- A reviewed expert issue is set to its expert binary reference before the next
  OOF round.
- Unreviewed expert outcomes are unavailable to blind scoring and selection.

This converts the official test pool into a development benchmark for this
experiment. It must not subsequently be treated as an untouched held-out set
for model-performance claims. No held-out AUROC is calculated here.

## Dynamic OOF Design

- One cost-controlled pilot trajectory: seed 13, chosen as the smallest ID in
  the current four-seed analysis set rather than from this outcome.
- Five selection rounds.
- Four OOF folds, grouped by `subject_id`; every image and study from one
  patient remains in one fold.
- MobileNetV3-small scratch, up to 100 epochs, batch size 32, learning rate
  0.001, early-stopping patience 10, and best-weight recovery.
- Fold assignments and fold-specific random initialization are fixed across
  loops, so label-state changes rather than changing folds drive trajectory
  differences.
- Study probability is the maximum across held-out images, matching the main
  pipeline.
- CL is fit independently per finding on all estimable valid study-label
  entries. Cross-finding comparison uses the within-finding percentile of
  self-confidence suspiciousness.

## Review Rule

- Each round selects the 100 highest-ranked benchmark entries not reviewed in
  any prior round; the final round selects the remaining 98.
- Previously reviewed entries are never selected again.
- The continuous ranking fills the fixed budget even when a candidate is not a
  Cleanlab hard flag. Hard-flag membership is reported separately.
- Selection receives entry identifiers, current labels, dynamic CL evidence,
  and review history keys only. It does not receive unreviewed expert outcomes.
- Oracle correction is applied only after the blind selection file and its
  SHA-256 are frozen.

## Comparators

- `dynamic_cl`: rerun subject-grouped OOF and CL after every cumulative oracle
  correction, then rerank the unreviewed entries.
- `frozen_cl`: use the complete Loop-1 ranking once and consume it in the same
  100/100/100/100/98 review chunks.
- `random_review`: 10,000 random entry permutations, summarized by mean and 95%
  empirical interval at the same cumulative budgets.

Dynamic and frozen share the exact Loop-1 ranking and must match at the first
100-entry checkpoint.

## Outcomes

At cumulative review budgets 100, 200, 300, 400, and 498, report:

- newly discovered true issues;
- cumulative discovered true issues;
- cumulative recall out of the fixed 127 issues;
- cumulative precision;
- remaining true issues;
- hard-flag support among the reviewed entries.

The trajectory summary is area under the discovery curve (AUDC), calculated by
trapezoidal integration of cumulative issue recall against reviewed fraction,
including the origin `(0, 0)` and endpoint `(1, 1)`. Report dynamic minus frozen
AUDC and checkpoint differences at 40%, 60%, and 80% review.

Because this is one deterministic training trajectory, differences are
descriptive and do not receive seed-level significance claims. A positive
dynamic-minus-frozen AUDC is evidence that retraining reprioritizes remaining
issues usefully in this pilot; a non-positive value means the dynamic loop has
not improved on one-shot ranking.

## Interpretation Boundary

Oracle correction isolates upstream CL behavior under perfect actions. It does
not measure LLM correction error, clinical utility, population-wide MIMIC-CXR
precision/recall, or repeatability across training seeds. Existing LLM actions
may be evaluated later as a separate end-to-end sensitivity analysis, but they
do not alter this benchmark state or primary trajectory.
