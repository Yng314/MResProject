# VinDr detector alternatives benchmark protocol

## Question

Under the same review budget, does a detector based on evidence that differs from the current OOF label-incompatibility score rank known corrupted chest X-ray label entries more effectively?

## Frozen inputs

- Dataset: the verified 3,000-image VinDr consensus cohort with six binary findings.
- Corruption: exact global symmetric entry noise at 10%, 20%, and 30%, plus a clean control.
- Replication: six fixed seeds (`13, 42, 97, 123, 211, 307`).
- OOF evidence: the completed MobileNetV3-small four-fold runs from `20260805_v1`.
- Feature evidence: the previously frozen 1,024-dimensional TorchXRayVision DenseNet features for the same images.

## Locked methods

1. **OOF label incompatibility**: the existing probability assigned to the alternative label. This is the current CL-compatible ranking reference.
2. **Active Label Cleaning (ALC)**: cross-entropy of the observed label minus predictive entropy, computed from the same OOF probability.
3. **SimiFeat-style feature neighbourhood**: cosine-neighbour label support in the frozen XRV feature space. The primary setting is the paper's default `k=10`; `k=20` and `k=50` are sensitivity analyses. The query image is excluded from its own neighbourhood. Because the project is multi-label, the binary score is computed separately for each finding and converted to a percentile within each finding and observed-label stratum.
4. **OOF + feature rank fusion**: the unweighted mean of the OOF rank and primary SimiFeat-style rank. The weight is fixed at 0.5 before private evaluation.

The SimiFeat-style method is an adaptation of its feature-neighbour cross-entropy idea, not a claim to reproduce the complete transition-matrix estimation and iterative algorithm in the original paper.

## Outcome separation

The scoring stage may read image identifiers, noisy labels, OOF probabilities, and frozen image features only. It must write and hash all detector scores before the evaluator reads `private_reference.csv`. Columns containing injected-error or clean-label outcomes are forbidden in blind score files.

## Metrics

- Primary: recall of injected errors when the review count equals the number of injected errors in that scenario.
- Secondary: AUPRC, AUROC, area under the review-fraction/error-recall curve, and precision/recall at fixed 5%, 10%, 20%, and 30% review budgets.
- Stratified: per-finding AUPRC and matched-budget recall.
- Statistical summary: paired seed differences against OOF label incompatibility and an exact two-sided sign-flip test, reported separately by corruption level. All six seeds are retained.

## Interpretation

A method supports the proposed selective-review pipeline if it retrieves more true errors than random review at the same budget. It provides incremental evidence beyond the current approach only if it also improves over the OOF reference across seeds. Results from the synthetic corruption benchmark establish controlled detection efficiency; they do not by themselves prove stable downstream improvement on MIMIC-CXR.
