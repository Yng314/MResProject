# VinDr XRV OOF comparison protocol

## Question

On the same six fresh VinDr corruption/training seeds and at the same entry-review budget, does replacing the scratch MobileNet OOF model with a frozen-XRV-feature OOF head improve true label-error detection, and does a fixed combination with direct XRV evidence improve it further?

## Locked data and methods

- Reuse the verified detector-v2 grid: seeds `887, 1597, 3253, 5003, 7867, 10007`; clean plus balanced, false-positive-only, and false-negative-only corruption at 10%, 20%, and 30%.
- **MobileNet OOF/CL**: existing verified detector-v2 `score_oof_cl`.
- **XRV OOF/CL**: four-fold nested OOF linear heads trained on the official TorchXRayVision `densenet121-res224-all` `features2()` representation and the current noisy labels; CL scoring is unchanged.
- **Direct XRV**: existing verified label-incompatibility score from the same frozen XRV model without fitting to the current noisy VinDr labels.
- **XRV OOF + Direct XRV**: equal-weight average of their global percentile ranks within each run. The 50/50 rule is fixed before private evaluation and will not be tuned.

The direct and OOF XRV branches share a pretrained encoder, so their fusion is a two-view scoring comparison, not evidence from two independent pretrained models.

## Outcome isolation

Every OOF model and all four detector scores are produced using only blind noisy cohorts, folds, images/features, and existing blind scores. Each 18,000-row comparison score file is written and hashed before any `private_reference.csv` is read.

## Endpoints

- Primary: balanced 20% corruption, overall true-error recall when the review budget equals the number of injected errors. Three paired six-seed contrasts against MobileNet OOF/CL use exact two-sided sign-flip tests and Holm correction across these three primary comparisons.
- Secondary: all regimes/rates, observed-positive `0 -> 1` and observed-negative `1 -> 0` queues, AUPRC, AUROC, precision, enrichment, and review-recall AUC.

## Interpretation boundary

This is a controlled VinDr detector comparison using known injected errors. The `all` XRV weights include MIMIC-CXR during pretraining; success here supports the model/evidence choice on VinDr but does not establish independence when later applied to MIMIC-CXR. A MIMIC-facing deployment claim requires a model not pretrained on MIMIC-CXR.
