# Med-PaLM Confident-Learning Detection Benchmark

## Material Passport

- Origin skill: Academic Research Suite, experiment-agent
- Origin mode: plan -> run
- Lock date: 2026-08-03
- Version: `medpalm_cl_detection_v1`
- Verification status before CL result generation: UNVERIFIED

This is an analysis-plan lock before CL inference/scoring results, not a
prospective preregistration before the Med-PaLM reference existed. Benchmark
class counts and the prior LLM correction result were already known; neither
is used to fit the frozen image models or calculate the blinded CL scores.

## Research Question

Within the externally selected Med-PaLM hard-case cohort, can confident-learning
evidence rank expert-confirmed incorrect CheXpert/U-Ones entries above
expert-confirmed correct entries?

The expert outcome is:

`true_issue = current_binary_label != expert_binary_label`.

The compatible benchmark is fixed at 498 entries from 457 studies and 179
subjects: 127 expert-confirmed issues and 371 expert-confirmed correct entries.

## Separation of Pipeline Questions

- This benchmark tests upstream detection only.
- It does not retrain a model on Med-PaLM.
- It does not run iterative cleaning loops.
- Existing LLM reviews are used only for a separately labelled secondary
  end-to-end analysis after the CL detection result is locked.

## Frozen Models and Inference Pool

- Frozen no-clean MobileNetV3-small checkpoints: seeds 13, 42, 97, and 123.
- The checkpoints predate this benchmark and are not selected using Med-PaLM
  detection outcomes.
- Inference is run on the complete official MIMIC-CXR test AP/PA pool rather
  than only the 498 preselected entries.
- Study probabilities use the existing maximum-over-images aggregation.
- Because this pool is an official held-out split, these probabilities are
  out of sample and do not require K-fold OOF construction within Med-PaLM.

## Outcome Blinding

Inference and CL scoring receive official split membership, AP/PA images,
current CheXpert/U-Ones labels, and frozen checkpoints only. Med-PaLM expert
labels, expected keep/relabel actions, and prior LLM outcomes are unavailable.

All full-test probabilities, per-label CL scores, hard issue flags, ranks,
manifests, and SHA-256 hashes must be complete before evaluation can join the
private expert reference.

## CL Evidence

For each seed and the arithmetic-mean four-seed ensemble:

- primary label-quality method: self-confidence;
- sensitivity method: normalized margin;
- hard issue flag: `find_label_issues`, fit per finding on the complete official
  test pool;
- pooled benchmark rank: within-finding percentile of suspiciousness, so a
  finding-specific score scale cannot by itself create aggregate separation.

A finding requires at least ten valid official-test entries and both binary
classes to estimate CL evidence. A non-estimable finding is skipped during
blind scoring, and evaluation still requires all 498 benchmark entries to join;
therefore a benchmark finding cannot be silently omitted. `Pleural Other` is
non-estimable in this test pool but is absent from the compatible benchmark.

## Confirmatory Analysis

Primary cohort: all 498 compatible entries.

Primary endpoint: AUPRC from the ensemble self-confidence suspiciousness
percentile. The no-skill reference is issue prevalence, 127/498 = 0.25502.

Secondary endpoints:

- AUROC;
- precision, recall, and enrichment among the top 20% benchmark entries;
- precision, recall, and enrichment for the hard CL issue flag;
- raw self-confidence and normalized-margin sensitivity analyses;
- each frozen seed shown separately.

Subject-cluster bootstrap with 2,000 replicates provides 95% intervals. The
primary gate is supported only if:

1. the 95% interval for `AUPRC - prevalence` is above zero;
2. the 95% interval for top-20 enrichment is above one; and
3. at least three of four individual seeds have AUPRC above their no-skill
   prevalence.

If point estimates exceed the references but either interval crosses the
reference, label the result suggestive rather than supported.

## Label Support

Findings with at least 20 entries, 10 true issues, and 10 correct entries form
the pre-specified supported-label sensitivity cohort. Current counts imply:

- Pneumonia;
- Pleural Effusion;
- Pneumothorax.

Other per-finding results are exploratory because one outcome side is too
small or absent. They cannot independently support a general detection claim.

## Positive Control

On the 371 expert-confirmed correct entries, inject known 5%, 10%, and 20%
binary flips under fixed frozen probabilities. Repeat each rate 100 times and
report AUPRC and top-20 enrichment. This checks whether the implementation can
recover controlled corruption; it cannot replace the real expert benchmark.

## Secondary End-to-End Analysis

After the CL result is fixed, apply the already completed Med-PaLM LLM actions
only to ensemble top-20 CL entries; leave all unselected entries unchanged.
Report fixed-denominator correct mass, retained accuracy, coverage, and change
from the original 371/498 baseline. This analysis does not alter the detection
gate.

## Interpretation Boundary

Med-PaLM is an externally selected disagreement/hard-case cohort. Passing this
benchmark supports conditional discrimination among these adjudicated hard
cases. It does not estimate population precision or recall over all MIMIC-CXR
entries and does not prove performance on an unselected random clinical sample.
