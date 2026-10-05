# VinDr Exact Global Symmetric-Noise Calibration Protocol

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-05
- Verification Status: PLANNED
- Version Label: vindr_global_symmetric_entry_noise_v1

## Objective

Test whether confident learning identifies known corrupted entries and whether
raw entry DQS tracks known dataset quality when the full 18,000-entry VinDr
matrix has exact global noise rates of 10%, 20%, and 30%.

The target is known true quality, not a preselected DQS value. DQS remains an
evaluated model-consistency proxy.

## Data and Unit

- 3,000 verified VinDr-CXR test images with radiologist consensus labels.
- Six binary findings, giving 18,000 `(image_id, label_name)` entries.
- Clean plus three corrupted conditions per seed.
- Seeds: `13, 42, 97, 123, 211, 307`.
- Known quality anchors: `1.00, 0.90, 0.80, 0.70`.

## Corruption Mechanism

For each label independently:

1. Set the exact number of flipped entries to `rate * 3,000`, giving 300, 600,
   or 900 flips.
2. Allocate `1 -> 0` flips as `round(rate * positive support)`.
3. Allocate all remaining flips to `0 -> 1` entries.
4. Sample without replacement with a deterministic scenario-specific seed.

Thus each label and the full matrix have the exact requested entry-noise rate,
while positive and negative entries have approximately the same conditional
flip probability. The resulting positive-prevalence inflation is measured and
reported rather than hidden.

This is a controlled stress test. It is not presented as a realistic model of
natural clinical-report errors.

## Outcome Isolation

1. Blind OOF jobs receive image paths, fold IDs and noisy labels only.
2. Clean labels, flip locations, true quality and direction are kept in private
   reference files outside the model inputs.
3. Private evaluation starts only after all blind-run completion markers pass.
4. Each seed uses shared outer folds across all four quality anchors.

## OOF and CL Configuration

- One-channel, six-output `torchvision` MobileNetV3-small trained from scratch.
- Unweighted masked BCE-with-logits, Adam `1e-3`, batch size 32.
- Four-fold nested OOF prediction.
- Maximum 100 epochs, inner early-stopping patience 10, best-state recovery.
- Same preprocessing and CL evidence implementation as the completed MobileNet
  direction-sensitivity benchmark.

## Primary Outcomes

1. Injected-error ranking AUPRC versus injected-error prevalence.
2. Recall at a review budget equal to the known number of errors versus random
   expected recall.
3. Hard CL issue enrichment and recall.
4. Raw entry DQS versus known true quality across the four anchors.
5. DQS Spearman correlation, MAE, slope and monotonicity within each seed.

## Statistical Unit and Uncertainty

- Training seed is the independent repetition unit.
- Report every seed, mean and standard deviation.
- Use a two-sided six-seed exact sign-flip test for positive AUPRC lift over
  prevalence and recall lift over random expectation.
- Image-cluster bootstrap intervals are descriptive and do not replace
  seed-level inference.

## Interpretation Boundaries

- Symmetric entry flips substantially increase observed positive prevalence in
  this sparse multi-label matrix; the prevalence trajectory must accompany the
  quality results.
- Success supports CL/DQS under controlled high noise, not under realistic
  MIMIC-CXR report-label noise.
- Failure can reflect weak OOF discrimination, class imbalance, CL calibration,
  or their interaction and cannot by itself identify one cause.
