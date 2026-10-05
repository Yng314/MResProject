# VinDr self-optimization stress experiment

## Material Passport

- Experiment ID: `vindr_self_optimization_stress_v1`
- Status: protocol locked before formal outcomes
- Purpose: test whether iterative confident-learning reranking adds value when label noise is larger and concentrated in hard instances
- Correction model: selected-only oracle correction, used to isolate detection and iteration from LLM correction error

## Research question

Does retraining after partial correction improve the evidence used to find the next errors, relative to spending the same review budget once with the frozen Loop-0 ranking?

The primary hypothesis is that dynamic reranking has higher error-discovery area under the review-budget curve than frozen Loop-0 ranking under `30% hard-instance noise` with Scratch DenseNet121 evidence.

## Controlled factors

- Noise rate: `20%`, `30%`, `40%` of action-set entries.
- Noise structure: `uniform` versus `hard`.
- Uniform corruption: entries are sampled randomly within each label and flip direction.
- Hard corruption: the same number of entries per label and flip direction are selected from the lowest clean-label self-confidence values produced by an independent MobileNetV3 OOF model.
- The corruption order is nested across rates. Every 20% error remains an error at 30%, and every 30% error remains an error at 40%.
- Eight previously unused seeds: `11003, 13007, 17011, 19001, 23003, 27011, 31013, 37003`.

## Data split and leakage control

- Each seed partitions the 3,000 VinDr images into 2,400 action images and 600 fixed sentinel images without looking at outcomes.
- The action set is the only set used for model fitting, OOF evidence, review selection, and correction.
- Sentinel images never enter outer training, inner validation, review selection, or correction.
- Sentinel corruption is fixed at 20% hard-instance noise within a seed and is identical across action-noise conditions and model initializations.
- The sentinel predictions are averaged over the four independently trained outer-fold models.

## Cleaning procedure

- Five loops.
- Each loop reviews exactly 288 previously unreviewed action entries, equal to 2% of the original 14,400-entry action pool.
- Dynamic CL reranks after every correction and model retraining.
- Frozen Loop-0 CL and random review are evaluated at the identical cumulative review budgets.
- All model branches reset from their locked initialization at each loop; only the corrected training labels change.

## Model branches

- Scratch DenseNet121: all six noise conditions across eight seeds (`48` runs).
- XRV-pretrained DenseNet121: only predeclared corner controls, `uniform 20%` and `hard 40%`, across eight seeds (`16` runs).
- Scratch learning rate: `1e-3`; XRV-pretrained learning rate: `1e-4`.
- Four-fold OOF, maximum 50 epochs, early-stopping patience 8, unweighted BCE-with-logits.

## Outcomes

Primary:

- Dynamic minus frozen discovery AUDC in Scratch `hard_r30`, paired across eight seeds.

Mechanism outcomes:

- Marginal errors corrected per loop and marginal action yield.
- Held-out sentinel error AUPRC, AUROC, and recall at fixed budget.
- Held-out sentinel clean-reference macro AUROC.
- Action-set clean-reference OOF macro AUROC.

Dataset outcomes:

- Known action-set quality and remaining errors.
- Raw entry DQS and its within-run rank correlation with known quality.
- Final dynamic, frozen, and random error recall at matched cumulative review budget.

## Statistical plan

- The primary paired contrast uses an exact two-sided sign-flip test over the eight locked seeds and reports the mean paired effect and direction count.
- Secondary condition and XRV-corner contrasts are labeled secondary and Holm-adjusted as one secondary family.
- Effect sizes and trajectories are reported regardless of p-value.
- A positive result requires more than final quality improvement: dynamic reranking must outperform the frozen ranking at the same review budgets and the held-out sentinel evidence should improve consistently with cleaning.

## Interpretation boundary

This experiment isolates the CL detection and iterative-retraining mechanism. Oracle correction does not establish real LLM end-to-end performance; targeted LLM correction is supported by the separate Med-PaLM experiment. VinDr is a controlled external validation dataset and does not replace the real-world MIMIC-CXR evaluation.
