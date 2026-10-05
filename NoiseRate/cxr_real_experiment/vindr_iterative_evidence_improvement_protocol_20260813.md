# VinDr iterative evidence-improvement protocol

## Research question

Does oracle correction of labels selected by cross-fitted confident-learning evidence improve the evidence produced in the next cleaning loop, and does this let later loops recover errors that the first loop missed?

The experiment is mechanistic. It evaluates OOF evidence and error discovery under known synthetic ground truth. It does not estimate downstream generalization or the accuracy of LLM correction.

## Paired experiment branches

Both branches use the same TorchXRayVision single-channel DenseNet121 with a six-output classifier. All layers are trainable.

1. `scratch`: random initialization, Adam learning rate `1e-3`.
2. `xrv_pretrained`: `densenet121-res224-all` initialization with a new six-output head, Adam learning rate `1e-4`.

XRV pretraining includes several chest-radiograph sources but not VinDr-CXR. Architecture, data, corruption, folds, review budgets, loop count, training seeds, and evaluation are otherwise paired. The branch comparison therefore isolates initialization, subject to the predeclared learning rates required by the two initialization regimes.

## Locked data and corruption

- VinDr-CXR test set: 3,000 images and six binary findings.
- Six existing corruption seeds: `13, 42, 97, 123, 211, 307`.
- Global symmetric entry corruption: exactly 3,600 of 18,000 entries are flipped (20%).
- Four outer folds are inherited from each locked corruption seed.
- Every image receives one OOF prediction from a model that did not train on that image.

The same corruption seed is used in both branches. Synthetic clean labels are private to split scoring, oracle correction, and final evaluation; they are not available to model training, CL ranking, or review selection.

## Fixed probe and action pool

For each corruption seed, 600 image IDs are selected once by a deterministic seeded random draw. This selection does not inspect clean labels or injected-error indicators. The split is shared by both initialization branches.

- Fixed probe: 600 images, 3,600 entries.
- Action pool: 2,400 images, 14,400 entries.

Probe images remain in model training so that all 3,000 images participate in four-fold OOF estimation, but their entries are never reviewed or corrected. Consequently, changes in probe error ranking can only arise from changes in the training labels outside the probe. This is a fixed diagnostic probe, not an independent downstream test set.

## Training lock

For each seed, fold, and branch:

- Outer folds remain fixed across loops.
- Inner early-stopping splits are built from the locked Loop-0 cohort and remain fixed across loops.
- The model is reset to the same branch-specific initial state at every loop; it is not warm-started.
- Maximum epochs: 50; early-stopping patience: 8; batch size: 32.
- Training loss: unweighted binary cross-entropy across all six entries.

Resetting the model and freezing all splits prevents extra training time or changing validation membership from being mistaken for an effect of cleaner labels.

## Cleaning action

At each of eight loops, all currently unreviewed action-pool entries are ranked by a CL-first score:

1. entries in the confident-learning issue set;
2. within that group, lower OOF self-confidence first;
3. if fewer than 360 hard issues remain, the same self-confidence ranking continues into the remaining entries.

Exactly 360 new entries are reviewed per loop. Probe entries and all previously reviewed entries are excluded. The oracle restores each selected true error to its clean label and leaves a selected correct entry unchanged. Eight loops therefore review 2,880 unique action entries, or 20% of the action pool.

## Comparators at the same review budget

- `dynamic_cl`: retrain OOF models and recompute CL ranking after every correction loop.
- `frozen_loop0_cl`: retain the initial Loop-0 ordering throughout.
- `random_review_expected`: expected discoveries under uniform random review.

All three are evaluated at cumulative budgets from 0 to 2,880 entries. The dynamic-versus-frozen contrast is the direct test of whether iteration adds value beyond one initial ranking.

## Outcomes

Primary mechanistic outcomes:

1. Change from Loop 0 to Loop 8 in fixed-probe error AUPRC.
2. Dynamic-minus-frozen area under the error-discovery curve (AUDC).

Supporting outcomes:

- clean-reference OOF macro and micro AUROC;
- fixed-probe error AUROC and recall/precision at 72 reviewed entries (2% of probe entries);
- recovery and mean rank of Loop-0-missed probe errors;
- cumulative true errors found at each matched review budget;
- true entry quality and raw entry DQS;
- direction-specific probe results for `0_to_1` and `1_to_0` corruption.

OOF AUROC is an evidence-quality diagnostic. It will not be described as downstream held-out performance.

## Statistical analysis

All effects are paired by corruption seed. Report per-seed trajectories, mean differences, direction counts, two-sided exact sign-flip p-values, and Holm-adjusted p-values within each declared family. With six paired seeds, the smallest attainable non-zero two-sided exact p-value is 0.03125. Effect magnitude and consistency will be reported even when multiplicity-adjusted significance is not reached.

## Execution gate

A one-seed, one-loop, two-epoch smoke run is executed for each branch. It is used only to verify data alignment, GPU execution, markers, output schemas, probe exclusion, and no-repeat review. Formal parameters and outcomes will not be changed in response to smoke performance. The 12 formal jobs start only if both smoke branches pass.

## Interpretation boundary

Evidence supporting the proposed mechanism requires both improved fixed-probe ranking and a dynamic discovery advantage over the frozen ordering. A model-quality increase alone is insufficient. Oracle correction is an upper-bound isolation experiment: it tests the CL/iteration mechanism under correct corrections, not whether an LLM would always make those corrections in practice.
