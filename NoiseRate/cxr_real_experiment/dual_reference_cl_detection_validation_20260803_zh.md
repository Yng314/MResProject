# Confident Learning 真实错误检测：双参考集正式验证

## 一句话结论

当前结果支持：**confident learning（CL）的连续可疑度分数和 hard flags 确实能在两个与 CL selection 隔离的专家参考集中富集真实标签错误**，而且四个训练 seed 的方向一致；但是，当前在 CL-hard study pool 内按 study 平均 self-confidence 再取 top 20% 的优先级策略，在主要 MIMIC-CXR 2.1 参考集上没有优于随机选择。因此应把“CL 能找到问题”和“当前 top-20 study 排序能优先找到更多问题”作为两个不同结论。

## 1. 这次实验回答什么

研究问题是：在不使用专家答案参与模型训练、CL 打分或候选选择的前提下，CL evidence 能否识别专家确认的 current-label issues？

这不是 iterative cleaning、LLM correction 或下游 AUROC 实验。它只验证整个 pipeline 的第一个模块：

```text
冻结模型预测 -> CL 可疑度 / hard flag -> 专家 reference 事后验证
```

## 2. 实验设计

- 冻结输入：四个 baseline seeds（13、42、97、123）及其 probability ensemble。
- 完整候选池：official-test 中 3,050 个 AP/PA studies、8,288 个 valid study-finding entries。
- 主要参考集：MIMIC-CXR-JPG 2.1 radiologist labels，匹配 1,796 entries，其中 109 个 current-label issues（6.1%）。
- 次要参考集：Med-PaLM hard-case panel，498 entries，其中 127 个 issues（25.5%）。
- `select` 阶段完全不读取专家 reference；候选 selection 的文件与 hash 固定后，`evaluate` 阶段才连接 reference。
- 不重新训练模型，不调用 LLM，也不根据结果调 threshold。

两个参考集必须分别解释。MIMIC-CXR 2.1 更接近常规 test subset，但只由一位 radiologist 标注；Med-PaLM 由三位 radiologists 复核，但样本原本就是外部方法筛出的 disagreement/hard cases，不能代表完整 MIMIC population。

## 3. 先区分两个判断层级

### A. CL detection

判断 CL 分数或 hard flag 是否比数据集基础错误率更集中地覆盖真实错误。主要看：

- AUPRC 是否高于 issue prevalence；
- CL-hard precision、recall 和 enrichment；
- 四个独立 seed 是否同方向。

### B. 当前 top-20 prioritization

现有 study-level policy 是：

1. study 内至少一个 entry 被 CL hard-flag，才进入 suspicious-study pool；
2. 用该 study 所有 valid entries 的平均 self-confidence 排序；
3. 选择质量最低的 20% suspicious studies；
4. 只展开这些 studies 中的 CL-hard entries。

它的合理对照不是全数据集随机抽样，而是：**从同一个 CL-suspicious study pool 中随机抽取相同数量的 studies**。这样才能单独检验第二次排序是否有额外价值。

## 4. Ensemble 主结果

| Reference | Issue prevalence | CL AUPRC | AUROC | CL-hard precision | CL-hard recall | CL-hard enrichment |
|---|---:|---:|---:|---:|---:|---:|
| MIMIC-CXR 2.1 | 0.061 | 0.186 | 0.726 | 0.230 | 0.440 | 3.78x |
| Med-PaLM | 0.255 | 0.638 | 0.799 | 0.596 | 0.465 | 2.34x |

MIMIC-CXR 2.1 中，AUPRC 比 prevalence 高 0.125，subject-cluster bootstrap 95% CI 为 `[0.080, 0.199]`；hard enrichment 的 95% CI 为 `[3.03, 4.61]`。

Med-PaLM 中，AUPRC 比 prevalence 高 0.383，95% CI 为 `[0.308, 0.455]`；hard enrichment 的 95% CI 为 `[2.00, 2.72]`。

这两组结果都说明：真实错误并不是均匀分布在 entries 中，而是更集中出现在 CL 给出较高可疑度或 hard flag 的位置。

## 5. 四个 seed 的一致性

- MIMIC-CXR 2.1：4/4 seeds 的 AUPRC 都高于 prevalence，4/4 的 hard enrichment 都大于 1。
- Med-PaLM：4/4 seeds 的 AUPRC 都高于 prevalence，4/4 的 hard enrichment 都大于 1。
- MIMIC individual-seed AUPRC 为 0.135-0.208；hard enrichment 为 2.46x-2.89x。

因此，CL detection 的正向结果不是由一个特别好的 seed 或 ensemble 单独造成。不过效应大小会随 seed 改变，不能把单一数值当成固定属性。

## 6. 当前 top-20 policy 的结果

| Reference | Top-20 覆盖的 reference entries | 捕获 issues | Policy precision | 随机策略平均捕获 issues | Empirical p |
|---|---:|---:|---:|---:|---:|
| MIMIC-CXR 2.1 | 46 | 6 | 0.130 | 9.57 | 0.936 |
| Med-PaLM | 31 | 19 | 0.613 | 11.82 | 0.017 |

### MIMIC-CXR 2.1

当前 top-20 policy 捕获 6 个已知 issues，而同一 CL-hard pool 内的 matched random policy 平均捕获 9.57 个。四个 individual seeds 中也只有 1/4 高于各自随机均值。因此主要参考集不支持当前第二阶段 study ranking。

这不否定 CL hard flags：在不做第二次 study ranking 时，hard flags 的 precision 为 0.230、enrichment 为 3.78x。问题出现在把 hard-flagged entries 按 study 平均质量重新排序以后。

### Med-PaLM

top-20 捕获 19 个 issues，高于随机均值 11.82，empirical tail probability 为 0.017。但是 policy precision 为 0.613，几乎等于随机 hard-pool precision 0.595；top-20 选择覆盖了 31 个 Med-PaLM entries，而随机策略平均只覆盖约 19.8 个。捕获数量更高主要伴随更高的 reference coverage，并不能单独证明 within-covered-entry discrimination 更强。

Med-PaLM 本身又是预先选择的 hard-case cohort，因此该结果只能作为次要支持，不能覆盖主要参考集的负面 prioritization 结果。

## 7. 为什么会出现“CL 有效，但 top-20 排序无效”

CL hard flag 是 entry-level evidence。当前第二阶段却用一个 study 中**所有 valid entries 的平均 self-confidence**给 study 排序。一个非常可疑的 entry 可能被同一 study 中多个高置信、无问题 entries 稀释。

因此，这次结果更像是在定位一个 policy mismatch：

```text
entry-level signal 有效
    ↓
用 whole-study mean 重新聚合后，信号被稀释
    ↓
top-20 study prioritization 未必更有效
```

这是根据结果提出的机制解释，不是本次实验已经证明的因果机制。后续可预先锁定并比较 entry-level ranking、minimum flagged-entry self-confidence 或其他只聚合 hard entries 的策略。

## 8. 偏倚与稳健性审计

- **Base-rate fallacy**：已用 AUPRC 对比 prevalence，并直接报告 enrichment，避免只报告 accuracy 或 AUROC。
- **Verification/selection bias**：Med-PaLM 是 hard-case panel，因此不外推 population precision；MIMIC 2.1 结果作为主要参考。
- **Regression to the mean / 极端选择**：top-20 与同一 CL-hard pool 内 matched random selection 比较，而不是与不匹配的全数据随机选择比较。
- **Subject dependence**：区间通过 subject-cluster bootstrap 计算，同一 subject 的 entries 一起重采样。
- **Seed cherry-picking**：ensemble 是预定主 estimator，同时完整报告四个 seed 的方向，不选择 best seed。
- **Per-label heterogeneity**：类别结果差异明显，部分 finding 的 issue support 很少；aggregate 结果不能解释为每个类别均有效。
- **Forking paths**：protocol 在正式 dual-reference run 前锁定，但研究者已看过早期 Med-PaLM 结果和 MIMIC feasibility counts，因此它是透明的 retrospective validation，不是 prospective preregistration。

## 9. 能说和不能说的结论

### 可以说

1. 在两个 outcome-isolated 专家参考 panel 内，CL continuous scores 和 hard flags 都能富集真实 current-label issues。
2. 该 detection 方向在四个训练 seed 中一致。
3. 当前 whole-study-mean top-20 prioritization 没有在主要 MIMIC-CXR 2.1 参考集上带来额外收益。
4. pipeline 后续应优先修正候选聚合/排序策略，而不是否定 CL detection 本身。

### 不能说

1. 不能声称已经获得完整 MIMIC-CXR training population 的 precision 或 recall。
2. 不能声称 CL 对每个 finding 都有效。
3. 不能由本实验声称 iterative cleaning 一定收敛、DQS 一定对应 ground-truth quality，或下游 AUROC 一定提高。
4. 不能把 Med-PaLM 的显著 matched-random 结果当成无偏 population validation。

## 10. 最直接的下一步

在保持 frozen CL evidence 和相同 expert references 不变的前提下，预先锁定 2-3 个候选策略，专门比较第二阶段 prioritization：

- 直接按 hard-flagged entry 的 CL quality 排序；
- study score 取最可疑 hard entry，而不是所有 entries 的均值；
- study score 只在 hard-flagged entries 内聚合。

选择标准应同时考虑 true issues captured、precision、reference coverage 和 matched-random improvement，避免只优化一个有利指标。由于这些策略是看过当前结果后提出的，新的比较必须明确标为 follow-up validation。

## 11. 可复现记录

- 正式 Slurm job：`269752`
- Protocol：`cxr_real_experiment/dual_reference_cl_detection_protocol_20260803.md`
- Program：`cxr_real_experiment/dual_reference_cl_detection_benchmark.py`
- Runner：`cxr_real_experiment/run_dual_reference_cl_detection_benchmark.sh`
- Formal output：`/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/dual_reference_cl_detection_benchmark/20260803_frozen4seed_exact_top20`
- Formal uncertainty：2,000 subject-cluster bootstrap replicates
- Matched random comparator：10,000 replicates per reference/estimator
- 完整性：selection/evaluation markers、input/output hashes、expected row counts 和 final verification 全部通过；formal stderr 为空。

参考数据来源：[MIMIC-CXR-JPG 2.1](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)，[Med-PaLM 2 MIMIC-CXR annotations](https://physionet.org/content/med-palm2-mimic-cxr/1.0.0/)。
