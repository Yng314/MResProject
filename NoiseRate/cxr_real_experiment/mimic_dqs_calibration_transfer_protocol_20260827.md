# MIMIC-CXR DQS calibration-transfer experiment

## Status and question

This protocol is locked before any new fold-model prediction is joined to an
expert reference. The experiment asks whether an expert-labelled MIMIC-CXR
subset can provide a numerical anchor for the frozen-evidence DQS trajectory
used in the end-to-end analysis.

## Target trajectory

- Target data: the AP/PA official MIMIC-CXR training split used by the existing
  five-seed full-issue-pool experiment.
- Findings: the same twelve findings and U-Ones mapping as the end-to-end
  analysis.
- Seeds: 7, 13, 42, 97, and 123.
- Evidence model: scratch MobileNetV3-small, four image-level folds, batch size
  32, Adam learning rate 0.001, at most 100 epochs, early-stopping patience 10,
  and recovery of the lowest-validation-loss weights.
- Dataset states: the uncleaned state and completed full-pool refinement Rounds
  1--5. As in the existing analysis, each seed's Round-0 OOF probability matrix
  is held fixed while labels and masks are updated.

Only Round-0 OOF models are replayed. The existing LLM reviews, label actions,
refinement models, and downstream models are not rerun.

## Outcome-blind scoring stage

Each fold model produces predictions for its held-out training fold and for all
AP/PA images in the official MIMIC-CXR test split. Expert reference files are
not read during this stage. Action-set row identity, the fold assignment,
training OOF probabilities, test-image probabilities, source hashes, and model
checkpoints are saved before calibration begins.

The replay is compared with the archived Round-0 OOF evidence. Exact row
identity is required. Numerical agreement is reported using probability error,
per-finding correlation, and the resulting uncalibrated DQS; exact floating
point equality is not assumed across GPU models.

## Calibration anchors

Two anchors are evaluated separately and are never pooled.

1. **Radiologist-panel anchor.** The MIMIC-CXR-JPG 2.1.0 expert-labelled test
   subset is the main calibration candidate. Four-state expert labels are
   mapped with the same U-Ones rule. Its broader sampling makes it more relevant
   to the full archive than a disagreement-only cohort, although it is not
   treated as exhaustive training-set ground truth.
2. **Med-PaLM hard-case anchor.** The 498 retained expert-adjudicated
   report--finding disagreements form a predeclared sensitivity analysis. This
   anchor is explicitly conditional on hard-case selection and cannot by itself
   identify population-wide MIMIC-CXR label quality.

Study-level expert labels are joined to every corresponding AP/PA image so that
the calibration unit matches the image--finding entries used by the target DQS.
For each seed and fold model, one global self-confidence threshold is selected
so that the estimated correct fraction in the calibration anchor most closely
matches its expert-observed correct fraction. No finding-specific threshold is
fitted because several findings have limited expert support.

Thresholds and their source hashes are frozen before they are applied to the
training trajectory. Results from the two anchors are reported side by side;
the anchor producing the more favourable trajectory is not selected post hoc.

## Interpretation boundary

The experiment can establish computational feasibility, threshold stability,
and the sensitivity of the MIMIC-CXR DQS trajectory to two in-domain expert
anchors. Because complete training-label ground truth is unavailable, it cannot
directly measure the absolute error of either calibrated trajectory over the
full training set. A substantial difference between the two anchors is evidence
of calibration-cohort dependence, not a reason to choose one retrospectively.

## Resource scope

The computational stage contains 20 fold fits: five seeds by four folds. Test
inference is negligible relative to training. No LLM API call is made.
