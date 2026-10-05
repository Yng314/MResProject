# VinDr direction-stratified detector ARS 验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-12
- Verification Status: ANALYZED
- Version Label: vindr_direction_stratified_detector_validation_v1
- Result Root: `/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_direction_stratified_detector/20260812_exploratory_v1`

## Answer First

实验支持以下两个不同层次的结论：

1. **OOF/CL 与 frozen-feature neighbourhood evidence 确实互补。** OOF/CL 主要找到 observed-positive errors；SimiFeat-style k=50 能找到大量 OOF/CL 漏掉的 observed-negative errors。
2. **目前还没有可靠的 outcome-blind budget allocator。** 双队列在两种方向错误数量相等时表现更好，但当错误主要集中在一个方向时，给另一个队列保留预算会降低总体 error recall。

因此现在不应把 hybrid 直接替换为默认 detector。更准确的定位是：

> OOF/CL 保留为总错误检出的默认方法；SimiFeat-style observed-negative queue 是一个具有明确成本的 missing-positive coverage guardrail。

## Experiment Design

- 复用已完成的 VinDr MobileNet direction-sensitivity benchmark：3,000 images、六 labels、六 seeds。
- 60 个 frozen runs：clean，以及 10%/20%/30% 下的 balanced、false-positive-only 和 false-negative-only corruption。
- 不重新训练模型，不调用 LLM。
- 所有 OOF、ALC 和 SimiFeat scores 在 private reference 解封前写入并 hash。
- 总 review budget 与每个场景的 injected-error count 相同，平均约 373 entries/run。
- observed-positive queue 使用 OOF/CL 排序；observed-negative queue 使用 SimiFeat-style k=50 排序。
- fixed quota curve 是 post-hoc diagnostic；两个自动规则分别根据 OOF alternative-probability mass 和 SimiFeat disagreement mass 分配预算。

## Primary Results

### Balanced directional errors

| Method | Observed-negative budget | Total recall | `0 -> 1` recall | `1 -> 0` recall | Mean errors found |
|---|---:|---:|---:|---:|---:|
| OOF global | 2.1% selected entries | 0.2187 | 0.4326 | 0.0047 | 92.3 |
| SimiFeat global | 97.7% selected entries | 0.2310 | 0.0427 | 0.4194 | 91.9 |
| Hybrid, SimiFeat-mass allocation | 45.2% | **0.2611** | 0.2537 | 0.2685 | **105.1** |
| Hybrid, fixed 50/50 | 50.0% | 0.2597 | 0.2365 | 0.2828 | 104.3 |

SimiFeat-mass hybrid 相对 OOF global：

- total recall：`+0.04245`，即相对提高约 19.4%；
- 平均每个 run 多找到约 12.8 个错误；
- 六个 seeds 的三-rate平均差均为正，exact sign-flip `p=0.03125`。

但这是 post-hoc exploratory comparison。方法由前一实验结果启发，并在同一个 benchmark family 上评价，因此该 p-value 只能描述六-seed一致性，不能当作独立确认性显著性。

### Direction-specific regimes

| Regime | OOF global recall | SimiFeat-mass hybrid recall | Difference | Seed direction |
|---|---:|---:|---:|---:|
| Balanced | 0.2187 | 0.2611 | +0.0424 | 6/6 positive |
| False-negative-only | 0.0196 | 0.1949 | +0.1753 | 6/6 positive |
| False-positive-only | 0.3041 | 0.1693 | -0.1348 | 6/6 negative |

这说明 hybrid 的收益来自更均衡的方向覆盖，并不是对任意 noise mechanism 都提高总检出率。

## Quota Trade-off

balanced corruption 下：

- 0% observed-negative quota：`0 -> 1 / 1 -> 0 = 0.445 / 0.000`；
- 20% quota：`0.363 / 0.148`，total recall `0.255`；
- 40% quota：`0.278 / 0.245`，post-hoc最高 total recall `0.262`；
- 50% quota：`0.237 / 0.283`，两方向最均衡；
- 100% quota：`0.000 / 0.425`。

![Direction quota trade-off](/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_direction_stratified_detector/20260812_exploratory_v1/evaluation/direction_quota_tradeoff.png)

因此 quota 不是普通超参数，而是一个明确的目标权衡：优化 total errors found 与保证 missing-positive coverage 会给出不同选择。

## Automatic Allocation Did Not Solve the Problem

SimiFeat-mass rule 在 balanced、FP-only 和 FN-only 场景中平均分配给 observed-negative queue 的预算分别约为 `45.2%/45.5%/44.1%`。它几乎不能判断当前错误主要来自哪个方向。

OOF-mass rule 同样缺少适应性，对三个 regimes 均分配约 `64%-66%` 给 observed-negative queue。它在 FN-only 中提升更大，但在 FP-only 中损失也更大。

所以当前 evidence 支持“两个 detectors 互补”，尚不支持“我们已经能根据未标注数据自动决定两个队列的最优预算”。

## Global Symmetric-noise Sensitivity

作为额外的 post-hoc sensitivity，在之前 exact global symmetric-entry corruption 中，错误方向由原始标签 prevalence 决定，约 94.8% 是 `0 -> 1`、5.2% 是 `1 -> 0`。在这种高度不平衡的错误构成下：

| Noise | OOF global recall | 5% negative quota recall | Change | Recovered `1 -> 0` recall |
|---:|---:|---:|---:|---:|
| 10% | 0.6941 | 0.6681 | -0.0259 | 0.2110 |
| 20% | 0.8319 | 0.8094 | -0.0225 | 0.2948 |
| 30% | 0.8907 | 0.8613 | -0.0294 | 0.2571 |

三个 noise levels 下，5% quota 的 total recall 都在 6/6 seeds 中下降。即使只留 1% quota，也会略降 total recall，但能找回约 7%-9% 的 `1 -> 0` errors。

这再次说明：如果唯一目标是最大化总体错误数，且真实错误构成与这个受控场景相似，OOF global 更好；如果 missing positives 具有更高风险，则可以用小 quota 换取 coverage，但必须公开报告其成本。

## Relation to Existing Expert Evidence

Med-PaLM 498-entry hard-case panel 的 127 个专家确认 issues 中：

- 79 个是 current positive、expert negative；
- 48 个是 current negative、expert positive。

这说明在真实专家复核的 hard-case cohort 中，两种方向都具有实际数量，不能只关注总体占优方向。但这个 panel 预先选择了 Med-PaLM 与 CheXpert disagreements，方向比例不能估计完整 MIMIC-CXR population 的 error prevalence。

## ARS Statistical and Fallacy Audit

- Coverage: 11/11 statistical fallacy types checked.

| Item | Assessment |
|---|---|
| Effect size | balanced total recall +0.0424，且 missing-positive recall 从 0.0047 增至 0.2685；具有实际意义。 |
| Seed consistency | balanced/FN-only 6/6 positive，FP-only 6/6 negative；机制方向稳定。 |
| Multiple comparisons | quota curve、三 regimes 和两个 mass rules 均为 exploratory family；nominal p-values 未作 confirmatory multiplicity claim。 |
| Base-rate neglect | 是主要风险；overall recall 会隐藏错误方向构成和 minority-direction failure。 |
| Simpson's paradox | overall hybrid gain 依赖 balanced aggregation；按 regime 分层后可以看到 FP-only 中方向反转。 |
| Ecological fallacy | 未从 scenario-level aggregate 推断单个患者临床结局。 |
| Berkson's paradox | Med-PaLM panel 是预筛 hard cases，仅用于说明现实相关性，不估计 population prevalence。 |
| Collider bias | detector scores 在 private outcomes 前冻结，未按成功检测结果选择评价对象。 |
| Regression to mean | corruption 在 detector scoring 前独立生成，不是从极端 scores 中挑选。 |
| Survivorship bias | 60/60 score runs、54/54 noisy evaluations 全部保留。 |
| Look-elsewhere effect | fixed-quota最佳点只标为 post-hoc diagnostic。 |
| Garden of forking paths | 主方向实验已有 locked inputs；本 hybrid 是看过前序结果后的 exploratory extension。 |
| Correlation != causation | controlled detection improvement 不证明 MIMIC downstream AUROC 会改善。 |
| Reverse causality | blind scores 先于 private truth evaluation。 |

## Decision

### 现在可以保留的结论

- SimiFeat-style frozen-feature evidence 不是 OOF/CL 的全局替代品，但能为 observed-negative entries 提供实质性的独立补充。
- 方向分层能防止 overall metric 掩盖 `1 -> 0` 几乎零 recall 的问题。
- 如果研究目标包含 missing-positive coverage，应把它作为单独 guardrail 指标和复核队列报告。

### 现在不应做的事

- 不应把 40% 或 50% quota 写成已经验证的最优参数。
- 不应声称自动 mass allocation 已经学会判断真实错误方向。
- 不应以 balanced benchmark 的 gain 代替 MIMIC-CXR population validation。

### 下一项确认实验

需要先确定方法目标：

1. 若目标是最大化 total errors found，继续使用 OOF global，并把方向 recall 作为 limitation/guardrail；
2. 若目标包含最低 missing-positive recall，则预先固定一个小的 negative-queue quota 或最低 recall constraint；
3. 使用新 corruption draws 或未参与本次设计的 held-out runs 做确认；
4. 若希望自动分配预算，需要引入能估计 directional error prevalence 的独立方法，例如完整 SimiFeat transition-matrix estimation 或外部 expert-calibrated direction prior。当前两个简单 mass rules 不够。

## Reproducibility

- Unit tests: 2/2 passed.
- Blind scoring: 60/60 runs complete.
- Private evaluation: 54 noisy runs, 918 method-run rows.
- Hash/row/outcome-column verification: passed.
- Independent rerun: not performed.
- Verdict: `ANALYZED`.
