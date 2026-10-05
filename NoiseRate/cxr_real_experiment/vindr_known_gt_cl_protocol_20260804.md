# VinDr-CXR Known-GT Confident-Learning Benchmark Protocol

## Material Passport

- Artifact type: pre-outcome code experiment protocol
- Workflow: ARS experiment-agent (`plan -> run -> validate`)
- Status: `PLANNED`
- Data access: local credentialed VinDr-CXR v1.0.0 test cohort only
- Protocol date: 2026-08-04
- Primary purpose: validate whether OOF confident-learning evidence prioritizes known label errors

## Research Question

When controlled label errors are injected into expert-consensus VinDr-CXR
image-level labels, does label-wise confident learning using out-of-fold image
predictions identify and prioritize the injected error entries better than random
review at the same entry budget?

This experiment evaluates the **detection stage** only. It does not evaluate LLM
correction, iterative cleaning convergence, clinical utility, or population-wide
MIMIC-CXR error prevalence.

## Data and Unit of Analysis

- Source: VinDr-CXR v1.0.0 consensus-labelled test cohort.
- Images: 3,000 PA chest radiographs in DICOM format.
- Reference: the released test image labels, formed by consensus of five
  radiologists.
- Unit of detection and review: one `(image_id, label_name)` entry.
- The released consensus labels are treated as the benchmark reference, not as
  infallible clinical truth.
- `No finding` is excluded because it is logically coupled to all abnormality
  labels. Labels with zero or very low positive support are also excluded.

Primary labels are the six sufficiently supported labels that map directly to
the current MIMIC-CXR task vocabulary:

| Label | Consensus positives (n=3,000) |
|---|---:|
| Atelectasis | 86 |
| Cardiomegaly | 309 |
| Consolidation | 96 |
| Lung Opacity | 84 |
| Pleural effusion | 111 |
| Pneumonia | 246 |

These labels were selected from support counts before any OOF or CL outcome was
generated.

## Image Preprocessing

The source DICOMs are converted once into deterministic 224 x 224, 8-bit,
lossless grayscale PNGs. The converter must:

1. decode the DICOM pixel array;
2. apply modality rescale before any VOI LUT/window operation;
3. apply the first available VOI LUT/window when present;
4. invert `MONOCHROME1` images after grayscale processing;
5. center-crop to a square and resize to 224 x 224;
6. scale the displayed grayscale range to `[0, 255]` and save without JPEG loss;
7. write source/output hashes and DICOM processing metadata.

A 20-image smoke and contact sheet are mandatory before the 3,000-image batch.
The full conversion is admitted only if every consensus `image_id` has exactly
one readable PNG, all outputs are finite and non-constant, all hashes are
recorded, and no source DICOM is modified.

## Controlled Noise Injection

The consensus reference is copied to a private table. The blind training table
contains only the injected/noisy labels.

For each label and experiment seed:

- sample `round(0.20 * positive_support)` true-positive entries and flip `1 -> 0`;
- sample the same number of true-negative entries and flip `0 -> 1`;
- keep all other entries unchanged.

This is a prevalence-preserving, bidirectional corruption. It prevents a method
from succeeding only because the injected process changes class prevalence and
ensures both false-negative and false-positive errors are evaluated. The noise
realization, fold assignment, and model initialization are controlled by the
same predeclared seed set: `13, 42, 97, 123`.

## Outcome Isolation

The benchmark is split into physically separate transactions:

1. `prepare`: reads consensus labels, freezes the corruption, writes a blind
   noisy-label cohort and a private reference.
2. `run`: receives only PNGs and the blind cohort; it cannot receive the
   consensus-label CSV or private-reference path.
3. `evaluate`: runs only after OOF/CL evidence is complete and then joins the
   private reference by `(image_id, label_name)`.
4. `verify`: checks hashes, row counts, seed coverage, fold coverage, budgets,
   and required outputs before exposing a completion marker.

Reference columns, injected-error flags, and clean labels are forbidden in blind
training and evidence files.

## OOF Evidence

- Four folds per experiment seed.
- Patient-grouped folds are used if stable de-identified patient identifiers are
  available; otherwise the released one-image-per-study unit is used and the
  limitation is recorded.
- Primary estimator: frozen radiography-pretrained TorchXRayVision DenseNet-121
  encoder with a newly initialized six-label linear head in every outer fold.
  Image features are extracted without access to any labels.
- Maximum 50 epochs, Adam, learning rate `1e-3`, linear-head batch size 128,
  early-stopping patience 8.
- Early stopping uses a fixed inner split drawn only from the outer-fold training
  images; the outer OOF fold is never used for fitting or epoch selection.
- The inner split is the first split in a deterministic seed sequence for which
  every target has at least three positive and three negative examples in both
  inner-train and inner-validation sets. The accepted split seed is recorded;
  labels are used only for support feasibility, never for outcome selection.
- Every entry is predicted exactly once by a model that did not train on that
  image/group.
- CL is run independently for each binary label using the OOF probabilities.

## Predeclared Comparators

All prioritization comparisons use exactly the same entry budget.

1. `CL-first`: CL hard-flagged entries first, ranked within strata by label
   self-confidence.
2. `Self-confidence`: all entries ranked by OOF probability assigned to their
   observed label, without the CL hard filter.
3. `Predictive entropy`: all entries ranked by binary predictive entropy.
4. `Random`: matched entry-budget random selection.

The comparison with self-confidence is necessary because prior project evidence
did not isolate an incremental benefit from the CL hard filter.

## Endpoints

### Primary

- Injected-error AUPRC of the continuous label-quality score.
- CL hard-flag precision, recall, F1, and enrichment over injected-error
  prevalence.
- Precision and recall at review budgets of 1%, 2%, 5%, and 10% of all entries.
- Precision and recall at a budget equal to the number of injected errors.
- Directional consistency across the four experiment seeds.

### Secondary

- Equal-budget differences between `CL-first` and each comparator.
- Per-label and flip-direction (`0 -> 1`, `1 -> 0`) detection metrics.
- True dataset quality `1 - injected_error_rate` versus raw entry DQS
  `1 - estimated_CL_issue_rate`, including signed and absolute calibration error.
- OOF AUROC against the blind noisy labels and, only during private evaluation,
  against the consensus labels.

## Statistical Analysis

- Report effect sizes and seed-wise values before p-values.
- Use image-cluster bootstrap within seed for detection metrics and a
  hierarchical seed/image bootstrap for aggregate contrasts.
- Use matched random selections for budget-based enrichment.
- Apply Holm correction within each declared comparator family.
- Four seeds limit the minimum two-sided exact sign-flip p-value to 0.125; this
  limitation is stated rather than treating non-significance as no effect.

## Decision Rules

The primary claim "OOF label-incompatibility evidence prioritizes known errors"
is supported only if all of the following hold:

1. AUPRC exceeds injected-error prevalence in all four seeds.
2. CL hard-flag enrichment exceeds 1 in all four seeds.
3. At the injected-error-count review budget, `CL-first` captures more errors
   than the matched-random mean in all four seeds.
4. No label with at least 80 positive consensus examples shows enrichment below
   1 in three or more seeds.

Incremental benefit from the **CL hard filter** is a separate claim and is
supported only if equal-budget `CL-first` consistently exceeds global
self-confidence. The first claim must not be rewritten as the second.

## Interpretation Boundaries

- Synthetic corruption establishes known error locations but may not reproduce
  the mechanism of report-derived MIMIC-CXR errors.
- The dataset is from Vietnamese hospitals and has different prevalence and
  acquisition characteristics from MIMIC-CXR.
- A positive result validates detection under this controlled benchmark; it
  does not estimate real-world precision on the full MIMIC-CXR training set.
- A negative result can reflect weak OOF predictions on the small 3,000-image
  cohort as well as a limitation of CL; OOF model quality must be reported.

## Sources

- VinDr-CXR v1.0.0 data description and five-radiologist test consensus:
  https://physionet.org/content/vindr-cxr/1.0.0/
- Northcutt, Jiang, and Chuang, *Confident Learning: Estimating Uncertainty in
  Dataset Labels*, JAIR 70 (2021), 1373-1411:
  https://arxiv.org/abs/1911.00068
- pydicom VOI processing order and API:
  https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_voi_lut.html
