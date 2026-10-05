# VinDr detector benchmark v2 protocol

## Research question

At an identical entry-review budget, do independent feature or image-model signals improve known label-error detection over the current OOF/CL ranking, particularly for observed-negative entries that contain missing-positive (`1 -> 0`) errors?

## Development and confirmation separation

- The old seeds `13, 42, 97, 123, 211, 307` are development data only. One old run may be used as an execution smoke test.
- Formal evaluation uses six previously unused corruption/training seeds: `887, 1597, 3253, 5003, 7867, 10007`.
- All six formal seeds are retained regardless of result. No method, score normalization, endpoint, or seed count is changed after private evaluation begins.

## Data and corruption grid

- Cohort: the verified 3,000-image VinDr-CXR consensus test cohort.
- Entries: six binary findings per image, 18,000 entries per run.
- Conditions: clean plus balanced, false-positive-only, and false-negative-only corruption at 10%, 20%, and 30% of the original positive count per finding, matching the previous direction-sensitivity construction.
- OOF model: the unchanged scratch one-channel, six-output MobileNetV3-small with four outer folds, nested early stopping, unweighted BCE, Adam `1e-3`, batch 32, maximum 100 epochs, and patience 10.

## Locked detectors

1. **OOF/CL**: the current CL-compatible hard-issue-first label-incompatibility score.
2. **Active Label Cleaning**: observed-label cross-entropy minus predictive entropy from the same OOF probability. This is a same-evidence comparator.
3. **FINE-GMM**: for each finding and observed-label stratum, obtain the leading singular vector of frozen XRV latent features, compute absolute alignment, fit a two-component Gaussian mixture, and use one minus the posterior probability of the higher-alignment component as suspicion. This is the core FINE detector adapted to binary multi-label entries; it is not presented as the paper's full training framework.
4. **External XRV**: label incompatibility from the frozen `densenet121-res224-all` finding probability after TorchXRayVision operating-point normalization. The external model never reads the injected truth or the current noisy labels during inference.

PU learning is not included in this symmetric/directional confirmation run. Standard PU learning assumes that labelled positives are reliable, but the balanced and false-positive corruption conditions deliberately violate that assumption. It should be evaluated separately in a missing-positive setting with a defensible outcome-blind class-prior estimator.

## Outcome isolation

External features/probabilities are extracted from images only. The blind scoring stage may read image ids, noisy labels, OOF probabilities, and frozen external evidence. It must write and hash every score file before the evaluator reads any `private_reference.csv`. Blind outputs reject columns containing clean labels, injected-error flags, true outcomes, or references.

## Endpoints

- Primary: total injected-error recall when the number of reviewed entries equals the number of injected errors in that run.
- Direction guardrail: `1 -> 0` recall within the observed-negative queue when its review budget equals the number of `1 -> 0` errors.
- Secondary: `0 -> 1` queue recall, AUPRC, AUROC, precision, enrichment over random, review-recall AUC, and selected-set overlap with OOF/CL.
- Statistics: paired seed differences against OOF/CL, exact two-sided sign-flip tests, and Holm adjustment within each scope/metric family. Effect sizes and seed direction are reported even when the discrete six-seed test cannot resolve small effects.

## Claim boundary

This benchmark tests controlled error detection and review efficiency. A positive result supports adding the corresponding evidence source to the selective-review pipeline. It does not by itself establish MIMIC-CXR error prevalence, LLM correction accuracy, or stable downstream AUROC improvement.
