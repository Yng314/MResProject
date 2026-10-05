# Med-PaLM Confident Learning 检测验证

## 一句话结论

在 Med-PaLM 外部筛出的 498 个 hard-case entries 中，四 seed 冻结模型的
confident-learning evidence 能把专家确认的错误标签排在正确标签之前。预先锁定的
三项 gate 全部通过，因此当前证据支持“CL 能在这类已筛选困难样本中识别真正可疑
标签”，但不能把这里的 precision/recall 外推到全部 MIMIC-CXR entries。

## 实验回答什么

本实验只回答 upstream detection：给定当前 CheXpert/U-Ones label 和不接触专家答案的
held-out model probability，CL 是否会给真实错误 entry 更高的 suspiciousness。

- 冻结模型：未清理 MobileNetV3 baseline，seeds `13/42/97/123`。
- inference pool：完整 official test AP/PA pool，3,414 images、3,050 studies。
- benchmark：498 entries、457 studies、179 subjects；127 个专家确认错误，371 个正确。
- primary score：四 seed probability ensemble 后的 self-confidence suspiciousness；跨类别
  汇总前转换为 finding 内 percentile。
- 隔离：先完成 probabilities、CL scores、manifests 和 hashes，再由 evaluation stage
  连接 expert reference。
- 这是一份在 CL 结果生成前锁定的 analysis plan，不是 expert reference 出现前的
  prospective preregistration；benchmark counts 和先前 LLM correction 结果已经知道。

## Primary result

| Metric | Result | Reference / interpretation |
|---|---:|---|
| AUPRC | 0.638 (95% CI 0.561-0.719) | no-skill prevalence = 0.255 |
| AUPRC - prevalence | +0.383 (95% CI +0.313 to +0.459) | interval entirely above 0 |
| AUROC | 0.799 (95% CI 0.752-0.845) | clear ranking separation |
| Top-20% precision | 0.650 (95% CI 0.546-0.740) | 65 true issues among 100 reviews |
| Top-20% recall | 0.512 (95% CI 0.440-0.573) | captures 65/127 known issues |
| Top-20% enrichment | 2.549 (95% CI 2.196-2.857) | 2.55x random selection |
| CL hard-flag precision | 0.596 | 59 true issues among 99 flags |
| CL hard-flag recall | 0.465 | captures 59/127 known issues |
| CL hard-flag enrichment | 2.337 (95% CI 2.005-2.713) | interval entirely above 1 |

Intervals use 2,000 subject-cluster bootstrap replicates. The primary gate passed because:

1. the interval for `AUPRC - prevalence` is above zero;
2. the interval for top-20 enrichment is above one; and
3. all four individual seeds, not merely the required three, have AUPRC above prevalence.

Individual-seed AUPRC values were `0.548`, `0.608`, `0.655`, and `0.587` for seeds
`13`, `42`, `97`, and `123`, respectively. This means the conclusion is not produced by
one favorable seed.

## Sensitivity and heterogeneity

The supported-label cohort was fixed as Pleural Effusion, Pneumonia, and Pneumothorax
(at least 20 entries, 10 issues, and 10 correct entries). Its ensemble AUPRC was `0.534`
against prevalence `0.247`; top-20 enrichment was `2.287`. This keeps the aggregate direction
after removing labels with extremely small outcome support.

The result is not uniform by finding. Pleural Effusion and Pneumonia show useful enrichment,
whereas Pneumothorax has AUPRC `0.689` against high prevalence `0.600`, AUROC `0.628`, and
top-20 enrichment `0.833`. All other per-finding results are exploratory because one outcome
side is too small or absent. Therefore the defensible claim is aggregate conditional ranking,
not reliable detection for every individual pathology.

Self-confidence and normalized-margin sensitivity analyses gave the same percentile ranking in
this binary setting. Using raw rather than within-finding scores slightly improved AUPRC to
`0.646` and top-20 enrichment to `2.627`, so the primary result is not dependent on percentile
normalization.

## Positive control and end-to-end secondary result

The synthetic positive control recovered injected flips at 5%, 10%, and 20% rates. Mean AUPRC
was `0.402`, `0.539`, and `0.698`, versus corresponding prevalences `0.051`, `0.100`, and
`0.199`; mean top-20 enrichment was `4.48`, `4.33`, and `3.63`. This confirms that the
implementation responds correctly to controlled corruption, but it is supporting evidence rather
than a substitute for the real expert benchmark.

As a secondary analysis, reviewing only the ensemble CL top 20% with the already completed LLM
actions raised fixed-denominator correct mass from `0.7450` to `0.8313` (`+0.0863`) while
retaining `0.9699` coverage. It used 100 reviews instead of 498. For context, reviewing all 498
entries previously produced a `+0.1125` gain, so CL prioritization recovered about 77% of that
gain with about 20% of the review workload. This efficiency comparison was not part of the
primary detection gate and has no dedicated uncertainty interval.

## Interpretation boundary

The result validates both parts of the proposed module-level story on the same external benchmark:

- upstream: CL meaningfully prioritizes expert-confirmed issues;
- downstream: the previously tested LLM can usually correct an externally supplied issue.

It does not yet validate iterative-loop convergence, population-wide MIMIC-CXR precision/recall,
or every pathology separately. Med-PaLM is an externally selected disagreement/hard-case cohort,
so the headline should remain: **CL detects real errors better than chance within these adjudicated
hard cases**.

## Reproducibility

- Protocol: `cxr_real_experiment/medpalm_cl_detection_protocol_20260803.md`
- Runner: `cxr_real_experiment/run_medpalm_cl_detection_benchmark.sh`
- Job: `269695`, completed in approximately five minutes with empty stderr.
- Canonical outputs: `/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/medpalm_cl_detection_benchmark/20260803_frozen4seed`
- Completion markers: inference, blind scoring, and private evaluation all present.
- Main machine-readable result: `medpalm_cl_detection_summary.json`.
