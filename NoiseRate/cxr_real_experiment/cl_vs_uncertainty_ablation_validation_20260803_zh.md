# CL 与简单不确定性排序的同预算消融验证

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-03
- Verification Status: ANALYZED
- Version Label: cl_vs_uncertainty_validation_v1

## Validation Report

- **Source**: Slurm job `269809`
- **Type**: outcome-blind equal-entry-budget ablation
- **Status**: completed
- **Overall Confidence**: CAUTION
- **Formal output**: `/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/cl_vs_uncertainty_ablation/20260803_frozen4seed_equal_entry_budget`

## 要回答的问题

在完全相同的 entry 审核预算下，CL 的 hard-issue 筛选是否比直接使用同一组 OOF 概率产生的简单不确定性排序，更能找到专家确认的当前标签错误？

这个问题包含两个不同层次，必须分开回答：

1. 当前 CL-derived 队列是否比随机审核更集中地包含真实错误？
2. 这种集中能力是否能独立归因于 CL hard filter，而不是更简单的 label-aware self-confidence？

## 实验设计

- 冻结四个训练 seed `13/42/97/123` 及 probability ensemble 的 OOF entry scores。
- 每个 estimator 使用完全相同的 entry 预算：`326/290/289/332/186`。
- `cl_hard_direct`：先限制在 CL hard issues，再按 observed-label self-confidence 从低到高排序。
- `self_confidence_global`：不使用 hard filter，直接在全部 entries 中按 self-confidence 排序。
- `self_confidence_per_label_percentile`：在每种 finding 内对 self-confidence 转成 percentile，再跨 finding 排序；这是锁定的主要非 CL comparator。
- `predictive_entropy`：按预测熵排序，不使用当前标签。
- `seed_disagreement`：ensemble sensitivity，仅按四个 seed 的预测标准差排序。
- 所有选择在读取专家结果前完成并锁定；盲选文件不包含 expert label、issue flag 或 ground truth。
- Primary reference 是 MIMIC-CXR-JPG 2.1 radiologist panel：`1,796` 个 compatible entries，其中 `109` 个 current-label issues。
- Secondary reference 是 Med-PaLM hard-case panel：`498` 个 entries，其中 `127` 个 issues。
- 置信区间使用 `5,000` 次 subject-cluster bootstrap；随机审核基线使用 `10,000` 次全池同预算抽样。

## 核心结论

### 1. CL-derived 队列确实能够优先找到真实错误

在 primary MIMIC reference 的 ensemble 结果中：

- 总审核预算是 `186` entries，其中 `53` 个落在有专家参考的范围内。
- CL 队列找到 `16/109` 个已知错误。
- 在这 `53` 个可验证 entries 中，错误 precision 为 `0.3019`，而 reference prevalence 为 `0.0607`。
- Enrichment 为 `4.97`，subject-bootstrap 95% CI `[3.07, 7.36]`。
- 全池随机同预算平均只找到 `2.44` 个错误；经验 `p=0.00010`。

因此，可以说这套 frozen OOF label-incompatibility evidence 具有真实的 prioritization value。它不是在随机地挑 entries。

但这不是 exhaustive detection：在当前预算下，对 MIMIC reference 已知错误的 recall 是 `0.1468`；Med-PaLM 上是 `0.2047`。

### 2. 不能把这个优势独立归因于 CL hard filter

同一 MIMIC/ensemble/186-entry 预算下：

| Policy | Reference-covered entries | True issues | Precision | Recall | Enrichment |
|---|---:|---:|---:|---:|---:|
| CL hard + direct rank | 53 | 16 | 0.3019 | 0.1468 | 4.97 |
| Global self-confidence | 52 | 16 | 0.3077 | 0.1468 | 5.07 |
| Per-label percentile | 53 | 15 | 0.2830 | 0.1376 | 4.66 |
| Predictive entropy | 49 | 3 | 0.0612 | 0.0275 | 1.01 |
| Seed disagreement | 48 | 2 | 0.0417 | 0.0183 | 0.69 |

关键审计结果：

- 四个 individual seeds 中，`cl_hard_direct` 与 `self_confidence_global` 的 selected-entry sets 完全相同。
- Ensemble 中二者有 `185/186` 个 entries 相同，而且都找到 `16` 个错误。
- 对锁定的 per-label comparator，CL 只多找到 `1` 个错误；paired subject-bootstrap issue difference 的 95% CI 为 `[-3, 5]`，one-sided `p=0.4111`，Holm-adjusted `p=0.8222`。
- 四个 seeds 中 CL 对 per-label comparator 是 `0 wins / 3 ties / 1 loss`，没有达到锁定的 `>=3/4` wins gate。
- Med-PaLM 上，CL 找到 `26` 个错误，global self-confidence 也是 `26` 个，per-label percentile 则找到 `28` 个。

所以，锁定的 primary decision rule 给出：`incremental_cl_value = not_supported`。

最稳妥的科学表述是：

> Frozen OOF label-aware incompatibility scores can prioritize expert-confirmed label issues substantially better than random selection. Under the current review budgets, however, CL hard filtering does not add detectable value beyond directly ranking entries by observed-label self-confidence.

### 3. 为什么 entropy 和 seed disagreement 很差并不矛盾

Predictive entropy 只问模型是否接近 `p=0.5`，它寻找的是模型不确定的样本。标签错误则可能表现为模型非常确定、但与 observed label 强烈相反。

Self-confidence 是 label-aware 的：它直接衡量模型给当前 observed label 的支持度。因此，本实验支持的是 label-aware contradiction signal，而不是一般意义上的 uncertainty。

## Statistical Findings

| Metric | Test | Value | Effect | Confidence |
|---|---|---|---|---|
| CL enrichment vs MIMIC prevalence | Subject-cluster bootstrap | `4.97`, 95% CI `[3.07, 7.36]` | 大幅富集 | SOLID for screening signal |
| CL capture vs full-pool random | 10,000 equal-budget random draws | `16` vs mean `2.44`, empirical `p=0.00010` | 大幅优于随机 | SOLID for screening signal |
| CL vs global self-confidence capture | Paired descriptive comparison | `16` vs `16` | 无差异 | SOLID evidence of no observed gain at this budget |
| CL vs per-label capture | Subject-cluster paired bootstrap | mean diff `+1.01`, CI `[-3,5]`, Holm `p=0.8222` | 不确定且较小 | CAUTION |
| Seed direction vs per-label | Four frozen seeds | `0 wins / 3 ties / 1 loss` | 不支持一致优势 | CAUTION |
| Med-PaLM secondary comparison | Equal-budget descriptive comparison | CL `26`, global `26`, per-label `28` | 不支持 CL 独立优势 | CAUTION |

## 这项实验能证明什么

可以支持：

- OOF 模型对 observed label 的低支持度能够把真实标签问题富集到审核队列前部。
- 这种能力在 MIMIC single-reader panel 和 Med-PaLM hard-case panel 上方向一致。
- Label-aware self-confidence 明显优于 label-agnostic predictive entropy 和 seed disagreement。

不能支持：

- CL hard threshold 比直接使用 self-confidence 更好。
- 当前方法能找全所有错误。
- MIMIC reference 之外的全部训练 entries 具有相同 precision/recall。
- CL 检测会因果性地改善 LLM correction、DQS、iterative convergence 或 downstream AUROC；这些不是本实验的 endpoints。

## 下一步真正需要的验证

### Priority 1: known-ground-truth synthetic corruption benchmark

在完整可信标签集上注入已知错误，并从头运行 OOF detection。至少覆盖：

- symmetric flips；
- class-dependent/asymmetric flips；
- report-extraction-like structured errors，例如 positive-to-uncertain、finding-specific noise；
- 多个 noise rates 和多个 training seeds。

在每个相同审核预算下比较 CL hard、global self-confidence、per-label percentile、entropy 和 random，报告 AUPRC、precision@budget、recall@budget 与完整 recall-budget curve。这个实验最直接回答“CL 是否真的找到了已知被破坏的 entries”。

### Priority 2: budget sweep rather than a single budget

当前预算小于 CL hard pool，导致最低 self-confidence entries 几乎全部已经在 hard pool 内，hard filter 自然很难产生额外选择差异。应从小预算一直扫描到超过 hard-pool 边界，检查 hard filter 在什么位置开始改变 precision-recall trade-off。

### Priority 3: independent complete expert panel

使用一个未参与任何分析选择、覆盖完整 study-entry cohort 的多阅片者 adjudicated panel 做外部确认。Med-PaLM 是挑选过的 disagreement cases，不能估计真实总体筛查性能；MIMIC 2.1 panel 主要是 single-reader 且覆盖范围有限。

## Warnings

| Type | Detail | Affected |
|---|---|---|
| Post-result follow-up | 协议在先前 CL 结果之后锁定，但 non-CL outcomes 在锁定前未读取；它不是 prospective preregistration | 所有 confirmatory language |
| Sparse reference coverage | 186 个 selected entries 中只有 53 个在 MIMIC reference 内；precision 分母是这 53 个，而不是 186 | Precision, capture |
| Reference selection | Med-PaLM 是 hard-case panel，存在明显 spectrum/selection bias | Med-PaLM metrics |
| Reader uncertainty | MIMIC 2.1 compatible panel 主要是 single-reader reference，不等同于无误差真值 | Primary endpoint |
| Comparator redundancy | 当前预算下 CL 与 global self-confidence 几乎相同，无法隔离 hard filter | Incremental CL claim |
| Multiple comparisons | 所有 CL-vs-baseline paired comparisons 使用 Holm correction；未校正结果不作为主结论 | Paired p-values |

## Fallacy Scan

- **Coverage**: 11/11 statistical fallacy types checked

| Fallacy | Severity | Assessment |
|---|---|---|
| Simpson's paradox | NOTE | Per-label effects不一致；没有把 aggregate enrichment 表述为每个 finding 都有效 |
| Ecological fallacy | NOTE | 推断限定在 entry-level screening，不从 study-level结果推断每个 entry |
| Berkson's paradox | CAUTION | Med-PaLM 是经过 disagreement 选择的 hard-case sample |
| Collider bias | NOTE | 未通过结果变量筛选 policy，也未拟合控制变量模型；未见明确 collider adjustment |
| Base-rate neglect | PASS | 同时报告 prevalence、precision、recall 和 enrichment |
| Regression to the mean | PASS | 不是基于结果极值分组后的 pre-post 改善设计 |
| Survivorship bias | CAUTION | 只有 reference-covered entries 可被判定；未覆盖 entries 的真实状态未知 |
| Look-elsewhere effect | PASS | Primary comparator 和 gate 预先锁定；多比较使用 Holm adjustment |
| Garden of forking paths | CAUTION | 属于看过先前结果后的 follow-up；协议与结果隔离降低但不能消除该风险 |
| Correlation != causation | PASS | 结论使用 prioritization/enrichment language，不声称 CL 导致标签变正确 |
| Reverse causality | PASS | 本实验没有时间方向或因果方向推断 |

## Reproducibility

- **Method**: hash-locked formal run plus independent low-replicate smoke and prior-policy key reproduction
- **Verdict**: PARTIALLY_REPRODUCIBLE
- Formal job `269809` completed with empty stderr.
- Blind selection reproduced every prior `entry_direct` key for all four seeds and ensemble.
- Formal checks validated exact budgets, forbidden-column absence, unique keys, `5,000` bootstrap replicates per combination, `10,000` random draws per reference-estimator, completion markers, and output hashes.
- The full `5,000/10,000` stochastic analysis was not independently rerun into a second formal output directory, so the report does not use `Verification Status: VERIFIED`.

## 主要产物

- Protocol: `cxr_real_experiment/cl_vs_uncertainty_ablation_protocol_20260803.md`
- Metrics: `cl_uncertainty_metrics.csv`
- Paired differences: `cl_uncertainty_paired_differences.csv`
- Bootstrap intervals: `cl_uncertainty_subject_bootstrap_ci.csv`
- Random baseline: `cl_uncertainty_full_pool_random_summary.csv`
- Figure: `cl_vs_uncertainty_equal_budget.png`
- Machine-readable decision: `cl_vs_uncertainty_summary.json`
