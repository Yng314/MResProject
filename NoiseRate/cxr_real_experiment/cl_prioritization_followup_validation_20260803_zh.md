# CL 候选优先级策略 Follow-up Validation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-03
- Verification Status: VERIFIED
- Version Label: cl_prioritization_followup_validation_v1

## 一句话结论

**Direct-entry ranking 是本次最强的替代策略：在与旧 policy 相同的 186-entry review budget 下，它在主要 MIMIC-CXR 2.1 reference 中把真实错误捕获数从 6 提升到 16，precision 从 0.130 提升到 0.302，且四个 seed 全部改善。**但是，它相对 CL-hard entry pool 内 matched random 的 empirical `p=0.0595`，没有通过预锁定的 `<0.05` gate。因此当前最准确的结论是：direct-entry 明确修复了旧 whole-study aggregation 的损失，但现有稀疏 reference 还不足以确认它在 hard pool 内具有稳定的额外排序能力。

## 1. 实验隔离了什么

这次只改变 CL-hard candidates 的优先级，不重新训练模型、不重新计算 CL、不调用 LLM，也不运行 cleaning loops。

```text
同一 frozen CL-hard pool
    -> 四种 outcome-blind ranking policies
    -> selection 文件与 hash 固定
    -> 最后连接 expert references
```

主要 reference 是 MIMIC-CXR-JPG 2.1 radiologist panel；Med-PaLM hard cases 只作次要敏感性结果。Ensemble 是主要 estimator，seeds 13、42、97、123 检查方向一致性。

## 2. 四种策略和预算

| Policy | 排序单位 | 排序分数 | Ensemble budget |
|---|---|---|---:|
| Whole-study mean（旧方法） | study | study 内所有 valid entries 的平均 self-confidence | 135 studies / 186 entries |
| Worst entry | study | study 内最低 self-confidence | 135 studies / 192 entries |
| Hard-entry mean | study | 只对 CL-hard entries 求平均 self-confidence | 135 studies / 157 entries |
| Direct entry | entry | 直接按 CL-hard entry self-confidence 排序 | 168 studies / 186 entries |

三个 study policies 固定相同的 135-study budget；direct-entry 固定与旧方法相同的 186-entry budget。不同策略的 study overhead 和 entry workload 都单独报告。

## 3. 主要 MIMIC-CXR 2.1 结果

Reference 中共有 1,796 entries 和 109 个 current-label issues。

| Policy | Selected entries | Reference-covered | Issues captured | Precision | Recall | Matched-random issues | Empirical p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Whole-study mean | 186 | 46 | 6 | 0.130 | 0.055 | 9.57 | 0.9364 |
| Worst entry | 192 | 57 | 14 | 0.246 | 0.128 | 9.60 | 0.0769 |
| Hard-entry mean | 157 | 48 | 13 | 0.271 | 0.119 | 9.58 | 0.1432 |
| Direct entry | 186 | 53 | 16 | 0.302 | 0.147 | 10.95 | 0.0595 |

三种替代策略都优于旧方法的 point estimate，但证据强度不同。

### Direct entry

- entry workload 完全相同：186 vs 186；但覆盖 168 个 studies，而旧方法覆盖 135 个。
- 捕获 issues：16 vs 6，observed difference 为 `+10`。
- Precision：0.302 vs 0.130，observed difference 为 `+0.171`。
- Paired subject-bootstrap capture difference：mean `+9.88`，95% CI `[2, 18]`。
- Paired precision difference：mean `+0.171`，95% CI `[0.033, 0.307]`。
- Paired recall difference：mean `+0.091`，95% CI `[0.019, 0.167]`。

这些 paired intervals 支持 direct-entry 在当前 reference panel 内稳定优于旧 policy，而不只是因为覆盖了不同数量的 reference entries。

### Hard-entry mean

- 只 review 157 entries，比旧方法少 29 个，约减少 15.6% entry workload。
- 仍捕获 13 个 issues，是旧方法的两倍以上。
- Precision difference 的 paired 95% CI 为 `[0.006, 0.273]`，方向支持更高 issue yield。
- Capture difference 的 CI 下界为 0，因此捕获数改善的不确定性比 direct-entry 更大。

它不是捕获数最高的策略，但在 review efficiency 上最有吸引力。

### Worst entry

- 捕获 14 vs 6，但使用 192 entries，比旧方法多 6 个。
- Paired capture difference CI 为 `[0, 16]`；precision difference CI 为 `[-0.014, 0.255]`。
- 它明显改善 point estimate，但统计稳定性和 workload control 都弱于 direct-entry。

## 4. 四个 seed 的一致性

| Seed | Old policy | Worst entry | Hard-entry mean | Direct entry |
|---|---:|---:|---:|---:|
| 13 | 11 | 16 | 11 | 20 |
| 42 | 11 | 18 | 14 | 20 |
| 97 | 10 | 22 | 18 | 23 |
| 123 | 7 | 11 | 10 | 12 |

- Direct entry：4/4 seeds 捕获数高于旧方法。
- Worst entry：4/4 seeds 高于旧方法。
- Hard-entry mean：3/4 高于，1/4 持平。
- Direct entry 在 seeds 13、42、97 上也超过各自 matched-random expectation；seed 123 没有。

因此 direct-entry 的相对旧方法改善不是 ensemble 或单一 seed 造成的，但其相对随机 hard-pool prioritization 的强度仍随 seed 改变。

## 5. 为什么预锁定 gate 仍未通过

预锁定 decision rule 要求同时满足：

1. 比旧方法捕获更多 issues；
2. precision 不低于旧方法；
3. matched-random `p_capture < 0.05`；
4. 至少 3/4 seeds 的 capture 高于旧方法。

三种替代策略都只卡在第三项：

- Worst entry：`p=0.0769`
- Hard-entry mean：`p=0.1432`
- Direct entry：`p=0.0595`

所以按锁定规则，三者的 formal status 都是 `not_promising`。这个标签只表示“没有通过全部四个 gate”，不等于“没有改善”或“应该继续使用旧方法”。Direct-entry 相对旧方法的 paired evidence 是明确的，只是相对 hard-pool random 的 evidence 仍略弱于预设门槛。

## 6. 次要 Med-PaLM 结果

| Policy | Selected entries | Issues captured | Precision | Matched-random issues | Empirical p |
|---|---:|---:|---:|---:|---:|
| Whole-study mean | 186 | 19 | 0.613 | 11.82 | 0.0174 |
| Worst entry | 192 | 25 | 0.758 | 11.76 | 0.0002 |
| Hard-entry mean | 157 | 19 | 0.792 | 11.80 | 0.0157 |
| Direct entry | 186 | 26 | 0.813 | 13.50 | 0.0003 |

Med-PaLM 也把 direct-entry 排在最前，而且 paired precision difference 为 `+0.200`，95% CI `[0.041, 0.383]`。但 Med-PaLM 是 externally selected hard-case cohort，不能用来覆盖主要 reference 的 `p=0.0595`，也不能外推为完整 MIMIC population precision。

## 7. 类别异质性

Aggregate improvement 不是每个 finding 都一致：

- Direct-entry 在 Atelectasis 捕获 5 个、Enlarged Cardiomediastinum 捕获 4 个，明显高于旧方法各 1 个。
- Direct-entry 在 Edema 和 Pneumothorax 也增加捕获。
- 旧方法在 Pneumonia 捕获 3 个，而 direct-entry 在其覆盖的 5 个 entries 中捕获 0 个。
- Fracture 和 Lung Lesion 没有被任何 ensemble policy 覆盖到 reference entries。

因此不能把 aggregate direct-entry 结果表述为“对所有疾病标签都更好”。它目前是总体 workload-level prioritization 改善。

## 8. 最合理的技术解释

结果支持旧 policy 存在 aggregation mismatch：CL evidence 本来是 entry-level，但旧方法先对 study 中所有 valid entries 求平均。高置信、无问题 entries 会稀释少数真正可疑 entry 的分数。

Direct-entry 完全移除这次不必要的 aggregation，所以在相同 entry budget 下表现最好。Hard-entry mean 只聚合可疑 entries，也避免大部分稀释，因此用更少的 entries 仍捕获更多问题。这个机制与结果一致，但仍是 observational mechanism evidence，不是因果分解实验。

## 9. 统计与偏倚审计

Overall Confidence：`CAUTION`。策略相对旧方法的改善可信，但 reference sparsity、post-result design 和 matched-random gate 未通过限制了确认强度。

### Fallacy Scan

- Coverage：11/11 checked

| Fallacy | Severity | 本实验检查结果 |
|---|---|---|
| Simpson's paradox | CAUTION | Aggregate 改善并非所有 finding 同方向；已完整报告 per-label heterogeneity。 |
| Ecological fallacy | NOTE | Study budget、entry budget 和 entry outcome 分开报告，不把 study aggregate 直接解释为每个 entry 的效果。 |
| Berkson's paradox | CAUTION | 分析条件化在 CL-hard pool；Med-PaLM 还条件化在 external hard-case selection。 |
| Collider bias | CAUTION | 进入 CL-hard pool 同时受模型预测和 current label 影响，结论限定为该 pool 内 prioritization。 |
| Base-rate neglect | NOTE | 同时报告 prevalence、precision、recall、enrichment 和 matched random。 |
| Regression to the mean | NOTE | 使用同一 CL-hard pool、相同 selection unit/budget 的随机对照。 |
| Survivorship/verification bias | CAUTION | 只有 reference-labelled entries 可验证，未标注 entries 不能被当作 correct。 |
| Look-elsewhere effect | CAUTION | 三个替代策略和多个 endpoints 全部报告；没有只选择最好结果，但未做 multiplicity-adjusted confirmatory test。 |
| Garden of forking paths | CAUTION | 策略由旧 policy 结果触发；protocol 在 alternative outcomes 前 hash-lock，因此是 locked follow-up，不是初始 preregistration。 |
| Correlation != causation | NOTE | 结果验证 frozen ranking association，不声称策略已经造成下游 AUROC 改善。 |
| Reverse causality | NOTE | 不涉及时间方向因果推断；reference 只在 selection 固定后连接。 |

## 10. 能回答什么

现在可以更具体地回答：

1. CL hard flags 能富集真实错误，这由上一轮双 reference benchmark 支持。
2. 旧 whole-study mean 会损失 entry-level signal，这次三种替代策略都比它捕获更多错误。
3. Direct-entry 是当前最强候选：同样 186-entry budget，捕获数 6 -> 16，且 4/4 seeds 改善。
4. 现有 reference 仍不足以确认 direct-entry 明显优于 hard-pool 内随机排序，因为预锁定 empirical `p=0.0595`。
5. 所以工程上可以把 direct-entry 作为下一轮 pipeline 的候选 policy，但学术上应称为 `best supported follow-up candidate`，而不是已经确认的最优策略。

## 11. Reproducibility

- Formal Slurm job：`269800`
- Runtime：2026-08-03 15:34:57-15:36:26 BST，约 89 秒
- Protocol：`cxr_real_experiment/cl_prioritization_followup_protocol_20260803.md`
- Program：`cxr_real_experiment/cl_prioritization_followup_benchmark.py`
- Formal output：`/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/cl_prioritization_followup/20260803_frozen4seed_locked4policy`
- Uncertainty：2,000 subject-cluster bootstrap replicates
- Matched random：10,000 replicates × 2 references × 5 estimators × 4 policies = 400,000 rows
- Current-policy reproduction：5/5 estimators 的 selected entry keys 与上一轮 formal benchmark 完全一致
- Verification：blind-column audit、source/output hashes、policy budgets、unique keys、hard-entry membership、expected row counts 和 final verify 全部通过
- Formal stderr：0 bytes

## Reproducibility Verdict

`REPRODUCIBLE`：正式输出与冻结代码、协议、helper、scores、references 和 blind selections 的 SHA-256 绑定；独立执行 `verify` 后全部检查通过。
