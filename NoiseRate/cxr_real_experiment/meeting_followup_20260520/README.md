# Meeting Follow-up Analysis

Generated on 2026-05-20.

## What this package covers

- weighted study-level AUROC summaries
- bootstrap uncertainty estimates for representative runs
- per-label trend tables for DenseNet and XRV linear-head
- focus-label removal percentages for sample vs entry top-k
- real noisy-sample inspection figure from the XRV linear baseline

## Best common12 study-level macro AUROC

- `appa_xrv_linear_top20_sample` = `0.8386`

## Representative bootstrap-CI runs

| run_id | family | study_macro_mean | study_macro_95CI | study_weighted_mean | study_weighted_95CI |
|---|---|---:|---|---:|---|
| appa_densenet_baseline | DenseNet | 0.7772 | [0.7339, 0.8269] | 0.8359 | [0.8055, 0.8611] |
| appa_densenet_top10_sample | DenseNet | 0.8225 | [0.7870, 0.8579] | 0.8448 | [0.8156, 0.8691] |
| appa_densenet_top05_entry | DenseNet | 0.8245 | [0.7875, 0.8597] | 0.8433 | [0.8174, 0.8685] |
| appa_xrv_linear_baseline | XRV linear-head | 0.8091 | [0.7707, 0.8488] | 0.8313 | [0.8052, 0.8568] |
| appa_xrv_linear_top20_sample | XRV linear-head | 0.8313 | [0.7953, 0.8657] | 0.8474 | [0.8230, 0.8682] |
| appa_xrv_linear_top05_entry | XRV linear-head | 0.8207 | [0.7777, 0.8563] | 0.8371 | [0.8024, 0.8669] |


## Suggested interpretation anchors

- weighted AUROC is less sensitive than macro AUROC to small-class fluctuations
- entry-level top-k can hurt when it removes too much negative supervision for rare or hard labels
- sample-level top-k more often removes globally suspicious samples, so it can improve several labels at once
