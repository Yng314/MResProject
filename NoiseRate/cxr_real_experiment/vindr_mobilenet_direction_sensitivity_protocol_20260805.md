# VinDr MobileNet Direction-Sensitivity Replay Protocol

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-05
- Verification Status: PLANNED
- Version Label: vindr_mobilenet_direction_sensitivity_v1

## Objective

Repeat the locked VinDr known-error direction-sensitivity experiment with the
project's original MobileNetV3-small-scratch image-training environment. The
experiment tests whether the previously observed weak `1 -> 0` detection is a
consequence of the frozen-XRV linear OOF classifier rather than a robust
property of the noisy-label/CL setting.

The frozen-XRV outcomes are already known. This is a prospective paired model
ablation, not a new preregistration of the earlier result.

## Frozen Inputs

- Source experiment:
  `/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_noise_direction_sensitivity/20260804_v2`
- Exactly the same 60 blind cohorts, private references, image order, injected
  error locations, outer-fold assignments, scenario conditions and seeds are
  replayed without regeneration.
- Seeds: `13, 42, 97, 123, 211, 307`.
- Scenarios per seed: clean plus `10/20/30%` balanced, `0 -> 1` only and
  `1 -> 0` only corruption.
- Images: the already verified 3,000 lossless PNG-224 conversions.
- Unit: one `(image_id, label_name)` entry across six VinDr labels.

## MobileNet OOF Configuration

The model and image environment match the original project MobileNet OOF path:

- `torchvision.models.mobilenet_v3_small(weights=None)`;
- first convolution changed from three channels to one channel;
- final classifier changed to six outputs;
- grayscale load, XRayCenterCrop, XRayResizer(224), and
  `torchxrayvision.datasets.normalize(maxval=255)`;
- no augmentation;
- unweighted masked BCE-with-logits over all six valid labels;
- Adam, learning rate `1e-3`;
- batch size 32;
- maximum 100 epochs;
- early-stopping patience 10 and recovery of the best inner-validation state;
- CUDA execution with four data-loader workers.

The original architecture, loss, preprocessing and optimizer are retained.
The existing nested OOF safeguard is also retained: epoch selection uses only a
support-gated inner split of the outer-training fold. Outer-fold labels are not
used for early stopping.

## Outcome Isolation

1. Blind array tasks read only the locked outcome-free manifest and one blind
   noisy cohort.
2. The MobileNet program receives no clean labels, injected-error flags, true
   quality or private reference paths.
3. Each task writes only OOF probabilities, CL evidence, training history,
   support/provenance and a blind completion marker.
4. Private evaluation starts only after every formal MobileNet blind marker is
   present.
5. Model outputs use the same schema as the frozen-XRV experiment so all
   detection, budget, DQS and bootstrap metrics are computed identically.

## Primary Comparisons

1. MobileNet `1 -> 0` hard recall averaged over the three noise rates versus
   the paired frozen-XRV result within each seed.
2. MobileNet direction gap (`0 -> 1` minus `1 -> 0`) and its six-seed exact
   sign-flip test.
3. CL-first recall at the true-error-count budget versus random,
   self-confidence and predictive entropy.
4. Clean-reference and noisy-label OOF macro AUROC.
5. Raw DQS calibration across clean/10/20/30% anchors within each corruption
   regime.

## Decision Rules

### Model-Limitation Explanation

Supported only if:

1. the MobileNet minus frozen-XRV seed-mean `1 -> 0` hard-recall difference is
   positive in all six seeds;
2. the mean improvement is at least `0.10`;
3. the two-sided six-seed exact sign-flip `p < 0.05`; and
4. MobileNet clean-reference macro AUROC is no more than `0.02` below the
   frozen-XRV clean anchor.

### Directional Asymmetry Robustness

Confirmed for MobileNet if every seed's mean direction gap exceeds `0.20` and
the six-seed exact sign-flip `p < 0.05`.

### Detection Utility

Supported only if CL-first AUPRC exceeds injected-error prevalence and
true-error-budget recall exceeds random expectation in every noisy run.

### DQS Calibration

Uses the existing locked gate: every seed-regime requires Spearman `rho >=
0.95`, MAE `<= 0.01`, and strict decrease from clean through 30% noise.

## Statistical Analysis

- Seed is the independent repetition unit.
- Scenario and model comparisons are paired by seed, corruption positions and
  outer folds.
- Report all seed values and effect sizes before p-values.
- Use a two-sided exact sign-flip test for the single primary cross-model
  `1 -> 0` contrast.
- The two within-MobileNet method contrasts use Holm correction, matching the
  parent protocol.
- Image-cluster bootstrap intervals remain descriptive and do not replace
  seed-level inference.

## Interpretation Boundaries

- This experiment changes model architecture/training while holding the
  controlled corruption benchmark fixed.
- It does not test class weighting, balanced sampling, probability calibration,
  LLM correction, iterative retraining or natural MIMIC label errors.
- A stronger MobileNet result supports model limitation as one contributor; it
  does not prove model capacity is the only cause.
- A persistent direction gap argues against the frozen-linear classifier being
  the sole explanation.

## Execution Gates

1. Unit tests and Python compilation pass.
2. One full four-fold, three-epoch `fn_only_r30`, seed211 smoke passes on GPU.
3. Formal jobs use immutable output paths, code/protocol/input hashes and
   `END,FAIL` email.
4. No formal private evaluator starts before all blind runs succeed.
5. No long-running process executes on a login server.
