# Pipeline Update 2026-06-18

## 1. What changed

The earlier pipeline was a one-shot correction flow:

1. Train on the noisy dataset.
2. Use confident learning to identify suspicious labels.
3. Remove or mask those suspicious labels/samples once.
4. Retrain one more time.

This is useful, but it is not a true iterative self-optimization loop.

The updated pipeline is:

1. Start from the current training dataset.
2. Run `K-fold OOF` predictions instead of in-sample full-train predictions.
3. Use confident learning on the OOF outputs to rank suspicious samples or entries.
4. Select a suspicious pool such as top `5%`, `10%`, or `20%`.
5. Use the original report text plus LLM review to refine those suspicious labels.
6. Build an updated training dataset.
7. Retrain on the refined dataset.
8. Optionally repeat the same loop again on the refined dataset.

So the main conceptual update is:

- old pipeline: `train -> CL -> one correction step -> retrain`
- new pipeline: `train -> OOF -> CL -> LLM refinement -> dataset update -> retrain -> repeatable loop`

## 2. Latest K4 OOF results

Model: `XRV linear-head`, `AP/PA`, `12 labels`

### Main comparison

| Setting | Image macro AUROC | Study macro AUROC | Study weighted AUROC |
|---|---:|---:|---:|
| Baseline | 0.8095 | 0.8158 | 0.8330 |
| Sample 5% | 0.8514 | 0.8557 | 0.8555 |
| Sample 10% | 0.8013 | 0.8114 | 0.8412 |
| Sample 20% | 0.8301 | 0.8385 | 0.8410 |
| Entry 5% | 0.8224 | 0.8272 | 0.8417 |
| Entry 10% | 0.8137 | 0.8182 | 0.8384 |
| Entry 20% | 0.8027 | 0.8082 | 0.8312 |

### Current takeaways from K4

- Under the `K4 OOF` setup, the strongest direct removal result is `sample 5%`.
- Sample-level removal is clearly stronger than entry-level removal in this run.
- Entry-level removal can still beat baseline in some settings, but the gain is smaller.

## 3. Top-5% LLM refinement results

This refinement experiment used the strongest suspicious pool from the K4 results:

- suspicious pool source: `sample top 5%`
- suspicious sample count: `2110`
- expanded entry-level review rows: `2063`

### LLM review distribution

LLM review result counts:

- `chexpert_right_cl_wrong = 1394`
- `chexpert_wrong_cl_right = 553`
- `report_ambiguous = 109`

Training-time refinement mapping:

- `chexpert_right_cl_wrong`: keep original label
- `chexpert_wrong_cl_right`: flip the original label
- `report_ambiguous`: mask that entry

This means the first refinement recipe applied:

- relabeled entries: `553`
- masked ambiguous entries: `109`

### Top-5% LLM-refined retraining with multiple seeds

| Seed | Image macro AUROC | Study macro AUROC | Study weighted AUROC |
|---|---:|---:|---:|
| 42 | 0.8062 | 0.8107 | 0.8335 |
| 13 | 0.8215 | 0.8235 | 0.8464 |
| 97 | 0.8033 | 0.8078 | 0.8301 |
| 123 | 0.8284 | 0.8326 | 0.8389 |

### Top-5% refinement takeaways

- The LLM-refined recipe is somewhat seed-sensitive.
- Some seeds beat baseline.
- But none of the tested seeds reaches the direct `sample 5% remove` upper bound.

In other words:

- direct `sample 5% remove` remains the strongest result in this branch
- `LLM relabel + ambiguous mask` is more interpretable and more editable
- but in the current form it does not preserve the full gain from aggressive sample removal

## 4. Why the new pipeline is different from the old one

The difference is not only that we added an LLM.

### Old pipeline

- one-shot
- correction is mainly remove-or-mask
- once the second training is done, the pipeline effectively ends

### New pipeline

- starts from `OOF` predictions rather than in-sample full-train predictions
- introduces report-grounded `LLM refinement`
- allows `label correction`, not only removal
- can produce a refined dataset `D_(t+1)`
- that refined dataset can be sent back into the same loop again

So the new version is not just:

`train twice with CL in the middle`

It is closer to:

`iterative dataset refinement with model feedback and report-grounded correction`

## 5. Current interpretation

At the moment, the strongest evidence is:

- `K4 OOF + sample 5% removal` gives the best performance in the current branch
- `LLM refinement` gives a more principled and editable dataset update mechanism
- but the first `top 5%` refinement recipe does not yet outperform the simple removal upper bound

This is why the next reasonable test is to start refinement from a less aggressive suspicious pool such as `sample 20%`, rather than starting from the already very strong `sample 5%` removal point.
