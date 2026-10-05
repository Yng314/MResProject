# Locked Follow-up: CL versus Simple Uncertainty Baselines

## Status

This protocol is locked before computing outcomes for the non-CL baselines, but
after the dual-reference CL benchmark and candidate-prioritization follow-up
were observed. It is therefore a post-result, outcome-blind follow-up rather
than a prospective preregistration.

## Research question

At the same entry-review budget, does the CL hard-issue filter add incremental
ability to identify expert-confirmed current-label issues beyond simple rankings
derived from the same frozen out-of-sample predictions?

The experiment performs no model training, no new CL fitting, no LLM review, no
label correction, and no iterative loops.

## Frozen evidence and references

- Frozen outcome-blind entry scores for baseline seeds `13`, `42`, `97`, and
  `123`, plus their probability ensemble.
- Primary reference: MIMIC-CXR-JPG 2.1 radiologist panel, expected to contain
  1,796 compatible entries and 109 current-label issues.
- Secondary reference: Med-PaLM hard-case panel, expected to contain 498 entries
  and 127 issues.
- Ensemble is the primary estimator. Individual seeds are robustness analyses.

## Locked entry budgets

Each estimator uses the number of entries selected by the reproduced current
whole-study policy and by the direct-entry follow-up:

- seed 13: 326 entries;
- seed 42: 290 entries;
- seed 97: 289 entries;
- seed 123: 332 entries;
- ensemble: 186 entries.

Every policy for an estimator selects exactly this many entries.

## Locked policies

Lower label quality or higher uncertainty receives higher priority.

1. `cl_hard_direct` (method under test): restrict to `cl_hard_issue = 1`, then
   rank by ascending given-label self-confidence.
2. `self_confidence_global`: rank all valid entries by ascending given-label
   self-confidence without the CL hard filter.
3. `self_confidence_per_label_percentile` (primary non-CL comparator): rank all
   valid entries by descending within-finding suspiciousness percentile. This
   controls for finding-specific score scales without using CL hard flags.
4. `predictive_entropy`: rank all valid entries by descending binary predictive
   entropy without using the current label for ranking.
5. `seed_disagreement`: ensemble-only sensitivity that ranks all entries by
   descending standard deviation of predicted probability across the four
   individual seeds.

Deterministic tie breakers use self-confidence, finding name, and `entry_key`.

In this binary task, normalized-margin quality is numerically identical to
self-confidence for every frozen row, and fixed 0.5 prediction-label mismatch
produces the same top-budget prefix as global self-confidence because every
budget is smaller than the number of entries with self-confidence below 0.5.
These duplicate rankings are excluded rather than reported as independent
baselines.

## Outcome isolation

The program separates `select` and `evaluate` stages.

1. `select` reads only frozen scores and their outcome-blind manifest.
2. It computes seed disagreement, complete policy rankings, fixed-budget entry
   selections, duplicate-baseline audits, hashes, and a completion marker.
3. `evaluate` verifies all blind artifacts and hashes before reading either
   expert reference.
4. No policy, tie breaker, budget, endpoint, or reference scope may change after
   reference access.

Blind artifacts must contain no reader label, reference label, true-issue flag,
expected action, or reader agreement.

## Endpoints

For every available reference-estimator-policy combination:

- selected entries and selected studies in the full candidate pool;
- expert-reference entries covered;
- expert-confirmed issues captured;
- precision among covered entries, recall, reference coverage, and enrichment;
- per-finding descriptive results;
- 5,000 subject-cluster bootstrap replicates;
- paired bootstrap differences `CL minus baseline`;
- one-sided bootstrap tail probabilities and Holm-adjusted values across all
  available CL-versus-baseline comparisons for each estimator/reference;
- 10,000 full-pool random entry selections at the same estimator-specific entry
  budget.

The full-pool random comparator tests screening utility. Paired CL-versus-score
comparisons test whether CL contributes beyond a simple ranking.

## Primary decision rule

The primary comparator is `self_confidence_per_label_percentile` on the
MIMIC-CXR 2.1 reference with the ensemble estimator. Incremental CL value is
called `supported` only if all conditions hold:

1. CL captures more known issues at the same entry budget;
2. CL precision is no lower;
3. the paired subject-bootstrap 95% CI for issue-capture difference has lower
   bound greater than zero;
4. the paired precision-difference CI has lower bound at least zero;
5. the one-sided paired bootstrap tail probability for capture is below 0.05;
6. CL captures more issues than the primary comparator in at least three of the
   four individual seeds; and
7. CL capture exceeds full-pool random expectation with empirical `p < 0.05`.

Failure of this rule means the current evidence does not isolate incremental CL
value from simpler uncertainty ranking. It does not negate the previously
established enrichment of the complete CL-hard pool.

## Interpretation boundaries

- The MIMIC reference is sparse and single-reader; Med-PaLM is externally
  selected for difficult disagreements.
- The same MIMIC reference motivated this follow-up, so even a positive result
  requires confirmation on a new reference or known-ground-truth dataset.
- Metrics apply to the labelled panels, not the full training population.
- This ablation tests frozen screening and ranking only, not correction quality,
  iterative convergence, DQS validity, or downstream model performance.

## Reproducibility

- Freeze program, protocol, imported helper, and frozen score hashes before
  formal execution.
- Formal outputs become immutable after `.evaluation_complete`.
- Verify policy availability, exact budgets, duplicate-baseline audits, blind
  columns, unique keys, replicate counts, and all output hashes.
