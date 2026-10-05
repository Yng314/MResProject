# VinDr matched-budget refinement protocol (2026-08-24)

## Purpose

This experiment separates two questions that were previously mixed by different VinDr configurations:

1. Under the same total review quantity, does retraining and reranking between review rounds identify more known label errors than retaining the initial ranking?
2. After the complete initial issue pool has been reviewed once, do further full-pool refinement loops identify additional errors and improve the evidence-model outcomes?

The first question is answered by the new matched-budget comparison. The second is answered by the existing full-issue-pool experiment. The two analyses share exactly the same prepared cohorts, corruption seeds, MobileNetV3-small configuration, and Loop-0 OOF evidence.

## Locked inputs

- Scenario: `hard_r30`.
- Seeds: `11003, 13007, 17011, 19001, 23003, 27011, 31013, 37003`.
- Action cohort: 2,400 VinDr-CXR images and 14,400 binary finding entries per seed.
- Training-excluded sentinel: 600 images and 3,600 entries per seed.
- Injected action-set errors: 30% hard-instance corruption (4,320 entries).
- Evidence model: scratch MobileNetV3-small, four subject-independent OOF folds, maximum 50 epochs, patience 8, Adam learning rate 0.001, batch size 32.
- The initial OOF evidence is reused byte-for-byte from the corresponding full-issue-pool run.

## Per-seed review quantity

Let `N0` be the number of entries marked as label issues by Confident Learning in Loop 0 for a seed. `N0` is the total review quantity for every matched policy.

Five cumulative review endpoints are fixed as `floor(k * N0 / 5)` for `k = 1,...,5`. The per-round quantities are the successive differences between these endpoints. This deterministic allocation ensures that the five rounds contain positive integer quantities and sum exactly to `N0`, including when `N0` is not divisible by five.

## Policies

### Dynamic reranking

At each endpoint, the current cohort is used to generate new four-fold OOF evidence. Previously reviewed entries are excluded, all remaining entries are ordered by the CL-first score, and the next fixed quantity is reviewed. Current CL issues are always ordered before non-issues by this score. If fewer current issues remain than the fixed quantity, the highest-ranked remaining non-issues fill the endpoint so that review quantity stays matched.

### Frozen Loop-0 ranking

The Loop-0 ordering is retained for all five endpoints. Successive non-overlapping portions are consumed without retraining or reranking. Its fifth endpoint is exactly the complete Loop-0 issue pool.

### One-shot full initial pool

The existing full-issue-pool experiment reviews the complete Loop-0 issue pool in its first round. Its first-round selected-entry set must be identical to the frozen fifth endpoint. Its post-correction MobileNet evidence is reused as the one-shot endpoint; it is not recomputed.

## Outcomes

The primary matched-budget outcome is known initial-error recall across the five cumulative endpoints, summarized by the area under the discovery curve for dynamic versus frozen ranking. Secondary outcomes are endpoint errors corrected and true label quality, plus action-set and sentinel macro AUROC for the dynamic endpoint versus the reused one-shot endpoint.

The existing full-issue-pool Loops 2 onward are reported separately as additional review effort. Their natural stopping rule is the absence of any new, previously unreviewed current CL issue. A natural stop is a completed endpoint, not a failed fixed-loop run.

All comparisons are paired by seed. Exact sign-flip tests summarize the eight paired differences. The matched-budget design was finalized after partial full-pool results were available, so it is treated as a protocol-locked follow-up comparison rather than a prospectively preregistered experiment.
