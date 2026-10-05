# VinDr symmetric-r20 top-20% calibration-transfer refinement

## Material passport

- Experiment ID: `vindr_symmetric_r20_top20_calibrated_v1`
- Origin: ARS experiment-agent follow-up validation
- Protocol lock date: 2026-08-27
- Status: locked before the new Loop-0 predictions and calibration outcomes

## Question

When the original symmetric-noise refinement protocol is repeated with an
independent calibration cohort, does a threshold fixed on that cohort track
known label quality more accurately than the uncalibrated dataset-quality
proxy across five top-20% refinement rounds?

## Locked cohorts and corruption

The 3,000 VinDr-CXR images with consensus labels are divided once into a fixed
600-image calibration cohort and a disjoint 2,400-image action cohort. The same
image split is used for all eight old protocol seeds:
`13, 42, 97, 123, 211, 307, 509, 701`.

For each seed, the old symmetric-entry corruption rule is applied separately to
the two cohorts. Exactly 20% of the binary finding labels are flipped in each:

- action cohort: 14,400 entries, 2,880 errors, known initial quality 0.80;
- calibration cohort: 3,600 entries, 720 errors, known initial quality 0.80.

The calibration images never enter model fitting, inner validation, candidate
selection, or correction. The action cohort retains the seed-specific four-fold
assignment from the old experiment.

## Evidence model and refinement

The model and training settings match the old experiment: a scratch
MobileNetV3-small with a one-channel input and six-output head, four-fold OOF
prediction, at most 100 epochs, early-stopping patience 10, Adam learning rate
0.001, and batch size 32.

At each of five rounds, Confident Learning identifies the current issue pool.
Previously reviewed entries are excluded, and the highest-ranked
`ceil(0.20 x current unreviewed issue-pool size)` entries are reviewed. Selected
errors are restored to their consensus labels; selected correct labels are
retained. No entry is deleted, masked, or reviewed twice. The corrected action
cohort is then used to retrain the four OOF models and recompute the next
ranking.

## Calibration lock

At Loop 0, each of the four action-trained fold models predicts every image in
the calibration cohort. For each seed and fold model, one global
self-confidence threshold is chosen so that the estimated correct-label mass
is closest to the known 0.80 calibration quality. A higher threshold resolves
ties.

All 32 thresholds are written and hash-locked before the action reference
labels are used for refinement or evaluation. The thresholds remain fixed for
Loops 0--5; they are not recalibrated after seeing the action trajectory.

## Outcomes

Every action-set state reports:

1. known label quality;
2. the uncalibrated DQS, defined as Cleanlab
   `overall_label_health_score` over all flattened binary entries; and
3. calibrated DQS, defined as the fraction of entries whose label
   self-confidence meets the frozen threshold associated with their OOF fold.

The primary comparison is the paired, seed-level trajectory mean absolute
error of calibrated versus uncalibrated DQS over Loops 0--5. Supporting
outcomes are seed-level and pooled rank association, adjacent-loop direction
agreement, endpoint absolute error, the number of entries reviewed and errors
corrected per round, and the known-quality trajectory.

The old 3,000-image curve is contextual only. It is not a comparator in the
primary analysis because the new experiment reserves 600 images for
calibration. The new result is retained and reported irrespective of whether
calibration improves the trajectory.
