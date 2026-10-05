# VinDr DQS calibration-transfer simulation

## Material Passport

- Experiment ID: `vindr_dqs_calibration_transfer_v1`
- Origin: ARS experiment-agent
- Mode: run / validation
- Protocol lock date: 2026-08-27
- Status: locked before fold-specific sentinel outcomes

## Question

Can a self-confidence threshold calibrated on a small expert-labelled cohort be
fixed and transferred to a larger noisy training cohort, where it is used to
estimate label quality across refinement rounds without access to that cohort's
reference labels?

## Cohorts

The experiment reuses the eight locked `hard_r30` VinDr-CXR preparations from
the completed full-issue-pool MobileNet experiment. Within each seed, the 3,000
consensus-labelled images are already partitioned into:

- a 600-image sentinel cohort (3,600 label entries), excluded from all model
  fitting, inner validation, review, and correction;
- a 2,400-image action cohort (14,400 label entries), used for four-fold OOF
  fitting and iterative oracle correction, with known initial quality 0.70.

The archived sentinel labels contain 20% hard errors. Before calibration, the
same locked hardness ordering is used to reconstruct both 20% and 30% hard-error
states from the sentinel consensus labels. The reconstructed 20% state must
match every archived sentinel label, error indicator, and error direction. The
matched 30% state, with known quality 0.70, is then used for calibration so that
its corruption mechanism and overall error rate match the initial action state.

The sentinel cohort represents the small audited calibration set. The action
cohort represents the larger dataset to which the calibrated estimate is
transferred. Action-set reference labels are not read until the calibration
thresholds have been written and hashed.

## Fold-matched prediction

For each of the eight locked seeds and each of the four outer folds, the same
MobileNetV3-small configuration used in the completed full-pool experiment is
retrained from scratch on the locked action data. Each fold model produces:

1. predictions for its held-out action fold; and
2. predictions for all 600 sentinel images.

The rerun action OOF probabilities must reproduce the archived Round-0 action
OOF file exactly. The mean of the four rerun sentinel predictions, together
with the training history and fold-support table, must also reproduce the
corresponding archived files exactly. Failure of any replay check stops the
analysis before calibration. For calibration, sentinel predictions are
retained separately for each fold model rather than averaged across models.

## Threshold calibration

For a binary label $y$ and predicted positive probability $p$, self-confidence
is the probability assigned to the existing label:

\[
s = yp + (1-y)(1-p).
\]

For each seed and fold model, a single global threshold is selected using all
3,600 matched-quality sentinel label entries. Candidate thresholds are
evaluated by the absolute difference between the fraction of sentinel entries
with $s\geq\tau$ and the known fraction of correct sentinel labels. The
threshold with minimum absolute error is selected; ties are resolved by
choosing the higher threshold. This calibrates the total estimated correct
mass; it does not train an entry-level correctness classifier. No
finding-specific threshold is fitted.

The resulting 32 seed-by-fold thresholds are written and hashed before any
action-set reference outcome is read.

## Transfer evaluation

The fixed threshold from fold $k$ is applied to the action entries whose OOF
prediction was produced by fold model $k$. For every available Loop 0--5 state,
the transferred threshold score is:

\[
\mathrm{DQS}_{\tau,t}=\frac{1}{N_0}\sum_{e=1}^{N_0}
\mathbf{1}(s_{e,t}\geq\tau_{\mathrm{seed},\mathrm{fold}(e)}),
\]

where all 14,400 action entries remain available, so coverage is one. The
primary outcome is the mean of the eight seed-specific mean absolute errors.
Loop 0--3, which are available for every seed before submission, form the
complete-case trajectory. Later loops are reported as available-case results
with their seed counts. Supporting outcomes are Loop-0 transfer error,
seed-specific endpoint error, seed-specific Spearman association,
adjacent-loop direction agreement, and pooled associations.

## Interpretation boundary

This is a controlled internal simulation of calibration transfer. A successful
result supports the feasibility of calibrating a DQS threshold on a small
expert-labelled cohort before applying it to a larger dataset. The numerical
threshold is dataset- and prediction-pipeline-specific; applying this design to
another archive requires a representative local calibration cohort.
