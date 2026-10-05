# VinDr detector alternatives 与 OOF AUM 的 ARS 验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-12
- Verification Status: ANALYZED
- Version Label: detector_alternatives_aum_validation_v1
- Sources:
  - `vindr_detector_alternatives/20260812_v1`
  - `vindr_oof_aum_screen/20260812_v1`

## 总体结论

当前证据支持继续把 **OOF/CL-compatible label incompatibility** 作为全局复核排序的默认方法。它在六个训练 seeds、10%/20%/30% 三种已知噪声水平下，都能在相同复核数量内找到远多于随机抽样的真实错误。

三个候选方向的结论不同：

1. **ALC 是小幅排序优化，不是独立 detector。** 它使用同一组 OOF probabilities，能轻微改善全局 AUPRC，但在预设的 matched-budget recall 上几乎不增加找回的错误数。
2. **当前 OOF AUM-style 实现没有通过两-seed screening gate。** 两个 seeds 的 AUPRC 和 matched-budget recall 都低于 OOF/CL，因此不应按现有实现扩展到六 seeds。
3. **SimiFeat-style 的全局汇总方式失败，但 feature evidence 本身并非无效。** 事后分层分析显示，它在 observed-label strata 内有很强的排序信号，尤其能补足 OOF/CL 几乎找不到的 `1 -> 0` 错误。其全局表现差，主要与当前 class-conditional percentile 无法保留各 strata 的错误基率有关。

因此，最有价值的下一步不是继续扩大当前 AUM，而是预先固定一个 **direction-stratified detector**：对 observed positive 和 observed negative entries 分开排序与分配复核预算，再用新的 corruption draws 做确认性验证。

## 1. 预设主分析：OOF/CL 是否能降低复核成本

Primary endpoint 是在复核数量等于真实错误数量时的 error recall。

| 注入噪声 | 复核数量 | OOF/CL 找到的错误 | OOF/CL recall | 随机期望 recall | 相对随机 enrichment |
|---:|---:|---:|---:|---:|---:|
| 10% | 1,800 | 1,249.3 | 0.6941 | 0.10 | 6.94x |
| 20% | 3,600 | 2,994.7 | 0.8319 | 0.20 | 4.16x |
| 30% | 5,400 | 4,809.7 | 0.8907 | 0.30 | 2.97x |

这些结果直接支持一个范围明确的结论：在这个 controlled VinDr benchmark 中，OOF/CL 能够把真实错误显著富集到有限复核队列前部，因此比随机或全量复核更具成本效率。

该结论的边界是：这里使用已知的 symmetric entry flips，不能单独证明它对真实 MIMIC-CXR pseudo-label errors、报告与影像不一致或其他系统性噪声具有相同检出率。

## 2. ALC：改善的是全局排序，不是主要复核终点

| 注入噪声 | OOF/CL AUPRC | ALC AUPRC | ALC - OOF | seeds 方向 | exact p | matched-recall 增量 |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 0.6831 | 0.6928 | +0.0098 | 6/6 | 0.03125 | +0.0094（+17.0 errors） |
| 20% | 0.8323 | 0.8382 | +0.0059 | 5/6 | 0.06250 | +0.0003（+1.2 errors） |
| 30% | 0.8797 | 0.8843 | +0.0046 | 6/6 | 0.03125 | 0（+0 errors） |

解释：

- ALC 和 OOF/CL 来自同一组 OOF probabilities，所以 ALC 没有引入独立影像证据或训练动态证据。
- 两者 matched-budget selected sets 的重合比例在 10%/20%/30% 噪声下分别为 95.1%、99.86% 和 100%。
- per-label matched recall 几乎完全相同。10% 噪声下的少量收益主要来自跨 finding 的复核名额重新分配，而不是某个 finding 内出现了新的检测能力。
- ALC 在更小预算下偶有实用收益，例如 10% 噪声、5% review budget 时平均多找到 14.5 个错误，20% 噪声、10% budget 时平均多找到 20.2 个错误；但这种收益没有跨所有预算稳定出现。

统计边界：AUPRC 的 `p=0.03125` 是 secondary endpoint 的 nominal p-value。若把三个 noise-level AUPRC 对比作为一个 family 做 Holm correction，三个 adjusted p-values 均为 0.09375。Primary matched-budget recall 的 p-values 为 0.25、1.0、1.0。因此不应写成“ALC 显著优于 OOF/CL”；更准确的结论是“ALC 提供了小幅、预算相关的排序敏感性改进”。

## 3. SimiFeat-style：全局失败来自校准，分层信号值得保留

预设全局结果确实不支持当前实现：在 20% 噪声下，SimiFeat k=50 的全局 AUPRC 只有 0.2627，而 OOF/CL 为 0.8323；matched-budget recall 分别为 0.2619 和 0.8319。固定 50/50 rank fusion 也降到 AUPRC 0.6517、recall 0.5753。

但是，当前 SimiFeat adaptation 先在每个 finding x observed-label stratum 内转成 percentile，再把所有 strata 全局混排。这样会抹掉 strata 之间非常不同的错误基率。20% 噪声下的探索性分层结果为：

| 候选队列 | 真错误 / 队列大小 | 队列错误率 | OOF/CL AUPRC | SimiFeat k=50 AUPRC |
|---|---:|---:|---:|---:|
| observed label = 1，检测 `0 -> 1` | 3,414 / 4,160 | 82.1% | 0.8550 | 0.9429 |
| observed label = 0，检测 `1 -> 0` | 186 / 13,840 | 1.34% | 0.0190 | 0.2073 |

在当前全局 matched budget 中，OOF/CL 找到约 2,995 个 `0 -> 1` 错误，但 `1 -> 0` 为 0；SimiFeat k=50 找到约 803 个 `0 -> 1` 和 140 个 `1 -> 0`。这说明：

- OOF/CL 很适合作为总体错误量最大的 observed-positive queue detector；
- feature-neighbour evidence 对少数 `1 -> 0` 方向提供了 OOF/CL 没有的补充信号；
- 简单平均两个全局 ranks 会稀释 OOF/CL 的强项，不能解决跨-stratum budget calibration。

这部分是看过真值后的 post-hoc mechanism analysis，只能用于生成下一项预注册实验，不能当作确认性结果。它也不能被表述为完整 SimiFeat 的复现，因为当前实验只是基于 frozen XRV features 的 neighbourhood adaptation，没有复现原方法全部 transition-matrix estimation 与迭代步骤。

## 4. OOF AUM-style：当前版本不值得直接扩展

| 方法 | 两-seed mean AUPRC | 两-seed mean matched recall |
|---|---:|---:|
| ALC | 0.8423 | 0.8339 |
| OOF/CL | 0.8342 | 0.8339 |
| OOF AUM-style | 0.8237 | 0.8258 |

AUM 相对 OOF/CL：

- seed 13：AUPRC -0.0045，recall -0.0122；
- seed 211：AUPRC -0.0166，recall -0.0039。

所以预先固定的 gate `do_not_expand_without_method_revision` 是合理的。两个 seeds 只适合作为 screening，不支持一般性地声称“AUM 无效”。

一个可能的机制是，四个 folds 的 best validation epoch 位于 epoch 2--6，但当前 mean margin 一直平均到 epoch 12--16，也就是每个 fold 都包含 best epoch 后 10 个 epochs；末期 validation loss 已明显恶化。这可能稀释早期训练动态，但属于事后解释。若未来重做，必须把“只平均到 best epoch”“固定 early window”或“标准 training-example AUM”作为新的、预先锁定的方法，而不能把它们与当前 screen 混为一项结果。

## 5. ARS 统计审计

### Statistical Findings

| Finding | Test / evidence | Interpretation | Confidence |
|---|---|---|---|
| OOF/CL 相对随机复核 | 六 seeds；matched-budget recall enrichment 2.97x--6.94x | controlled detection efficiency 的效果量大且方向稳定 | SOLID（限当前 benchmark） |
| ALC global AUPRC | exact paired sign-flip；nominal p 0.03125/0.0625/0.03125 | 数值改善小，且不是 primary endpoint | CAUTION |
| ALC matched-budget recall | exact p 0.25/1.0/1.0 | 没有证据表明它优于 OOF/CL，观察到的增量也很小 | SOLID descriptive |
| 当前 SimiFeat global ranking | 六 seeds、三 noise levels 均明显低于 OOF/CL | 当前全局 aggregation 不适合直接替代 OOF/CL | SOLID（限本 adaptation） |
| SimiFeat stratum-level signal | post-hoc ground-truth stratification | 有补充 `1 -> 0` 信号，需新实验确认 | CAUTION / exploratory |
| 当前 OOF AUM-style screen | 两个固定 seeds 均低于 OOF/CL | 未通过扩展 gate | SOLID screening decision；非一般性结论 |

### Multiple comparisons

正式 contrast 表包含 45 个 exact tests（5 个候选方法 x 3 个 noise levels x 3 个 metrics），未预设统一 multiplicity correction。ALC 的 nominal AUPRC p-values 不能替代 primary endpoint，也不应单独作为确认性显著结果。SimiFeat 的负向差异幅度很大且六 seeds 同方向，因此对“当前全局实现明显较差”的描述并不依赖临界 p-value。

### Fallacy Scan

- Coverage: 11/11 checked.

| Fallacy | Status | 本实验中的判断 |
|---|---|---|
| Simpson's paradox | NOTE | 未发现经典方向反转，但 global aggregation 隐藏了 observed-label strata 的相反能力结构。 |
| Ecological fallacy | CLEAR | 没有从 seed-level aggregate 推断单个 entry 的因果性质。 |
| Berkson's paradox | CLEAR | 没有在筛选后的样本中推断变量相关性；3,000 张 VinDr、六个 findings 和受控噪声仍构成外部有效性边界。 |
| Collider bias | CLEAR | 没有加入会同时受 detector 与 error outcome 影响的调整变量。 |
| Base-rate neglect | CAUTION | 20% 噪声中 `0 -> 1` 有 3,414 个、`1 -> 0` 只有 186 个；只报 overall recall 会掩盖后者 0 recall。 |
| Regression to the mean | CLEAR | 不是按极端 baseline 选择对象的 pre/post 设计。 |
| Survivorship bias | CLEAR | 六个 detector seeds 全部保留；AUM 两 seeds 是预设 screen，不是失败后筛选。 |
| Look-elsewhere effect | CAUTION | 45 个 contrasts 未统一校正；secondary p-values 应按 exploratory 解释。 |
| Garden of forking paths | NOTE | 主 benchmark 有 outcome-blind locked protocol；direction-stratified 分析是在看过结果后提出。 |
| Correlation != causation | CAUTION | controlled detection 结果不能自动证明 MIMIC-CXR 下游 AUROC 会稳定改善。 |
| Reverse causality | CLEAR | injected-error truth 在 scoring 后才解封，不存在反向因果解释。 |

## 6. 建议的下一项确认实验

### Research question

在相同总 review budget 下，direction-stratified ranking 能否保持 OOF/CL 的总体检出率，同时恢复对 `1 -> 0` minority errors 的检出？

### Locked candidate

1. observed label = 1：使用 OOF/CL 或 ALC 排序，主要寻找 `0 -> 1`；
2. observed label = 0：使用 SimiFeat k=50 或完整 SimiFeat-R 排序，主要寻找 `1 -> 0`；
3. 两个队列的预算只能通过 outcome-blind 的 estimated directional error mass 分配，例如 confident-joint / transition-matrix estimate，不能读取 injected truth；
4. 总 review budget 与 OOF/CL baseline 完全相同。

### Metrics

- Primary：total errors found at fixed total review budget；
- Guardrail：`1 -> 0` recall；
- Secondary：`0 -> 1` recall、macro directional recall、每 1,000 次 review 找到的 errors、各 finding AUPRC；
- 必须同时报告两个方向的错误基率。

由于该方向是在查看现有真值后产生的，现有六 seeds 只能用于 exploratory development。若要作为确认性证据，应使用新的 corruption draws，或者先固定规则再在未参与设计的 held-out runs 上评价。

## 7. 可以用于论文的主张边界

可以说：

> 在一个具有已知 entry-level corruption 的 controlled chest X-ray benchmark 中，OOF/CL ranking 在同等复核预算下将真实错误富集到随机抽样的约 3--7 倍。ALC 对整体排序有小幅影响，但没有稳定提高预设的 matched-budget recall。探索性分层分析进一步显示，frozen-feature neighbourhood evidence 可能为少数错误方向提供互补信号。

暂时不要说：

- ALC statistically outperforms confident learning；
- SimiFeat does not work；
- AUM is ineffective in general；
- CL detects both flip directions equally well；
- 这些 detector 结果已经证明 MIMIC-CXR downstream performance 会稳定改善。

## Primary literature context

- Bernhardt et al. (2022), Active label cleaning for improved dataset quality under resource constraints: https://doi.org/10.1038/s41467-022-28818-3
- Zhu et al. (2022), Detecting Corrupted Labels Without Training a Model to Predict: https://proceedings.mlr.press/v162/zhu22a.html
- Pleiss et al. (2020), Identifying Mislabeled Data using the Area Under the Margin Ranking: https://proceedings.neurips.cc/paper/2020/hash/c6102b3727b2a7d8b1bb6981147081ef-Abstract.html

## Reproducibility status

- Artifact integrity: formal completion markers、hashes、outcome-blind score files 与 private evaluation separation 均已检查。
- Independent rerun: 本次 ARS validation 未重新训练模型。
- Verdict: `ANALYZED`; artifact chain verified, independent reproducibility not re-tested.
