# VinDr 噪声方向与 DQS Calibration 正式验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-04
- Verification Status: ANALYZED
- Version Label: vindr_noise_direction_sensitivity_validation_v1
- Source Protocol: `vindr_noise_direction_sensitivity_protocol_20260804.md`
- Formal Result Root: `/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_noise_direction_sensitivity/20260804_v2/evaluation_formal_v1`
- Sensitivity Program SHA-256: `eabe258dd8ce0b860d458c451cd4064514bc1b7a8fa76679347b6c532602390e`
- Frozen Feature SHA-256: `d361d521e681ea8fdff94ba83c206680123032eb247a5b84ffd77d393a895327`
- Blind Manifest SHA-256: `a17fba9c02c8a8c9428c1cffa2afee3261980e3c59a4905a68be1bbb8d0da6c4`

## Answer First

这个实验对“confident learning 能不能找出真实错误”给出的答案是：

**能显著优于随机地优先找到错误，但能力强烈依赖错误方向，不能说成对所有错误都可靠。**

- 在全部 `54` 个 noisy scenario-seed runs 中，CL-first AUPRC 均高于错误
  prevalence，CL hard enrichment 均大于 `1`，true-error-count budget 下的
  recall 均高于随机期望。
- 对错误增加的阳性，即 clean `0` 被改成 noisy `1`，hard recall 在
  `10%/20%/30%` noise 下分别为 `0.871/0.877/0.863`。
- 对遗漏阳性，即 clean `1` 被改成 noisy `0`，hard recall 分别只有
  `0.298/0.156/0.064`，且噪声越强越差。
- 六个 seed 的平均方向差全部大于 `0.20`，exact sign-flip `p=0.03125`；
  预设的 directional-asymmetry gate 通过。

这个实验同时否定了 raw DQS 是通用 dataset-quality estimator 的强表述。
在 false-negative-only 场景中，真实质量下降时 raw DQS 反而升高；它在当前
实现中更准确的解释是 **OOF model-label consistency score**，而不是独立真值。

## Study Design

- 3,000 张 VinDr consensus-reference images，六个预先冻结 labels，共
  18,000 entries/run。
- 六个 seeds：`13/42/97/123` 为 parent-overlap block，`211/307` 为新的
  prospective replication block。
- 每个 seed 十个场景：clean anchor，以及 `10%/20%/30%` 下的 balanced、
  false-positive-only 和 false-negative-only corruption。
- 同一 rate 的三种 regime 注入相同错误总数：`188/372/560`。
- 60 个正式 OOF runs 只读取 outcome-free blind manifest、noisy cohort 和冻结
  XRV features；private references 只在全部 blind runs 成功后由依赖作业读取。
- 1,000 次 image-cluster bootstrap/run 仅描述 cohort 内不确定性；主要推断以
  seed 为独立重复单位，使用六-seed 双侧 exact sign-flip test。

本研究是由 parent 20%-balanced 结果启发的 post-result sensitivity study。
原四个 seeds 不能被追溯表述为全新 replication；新 seeds 单独保留 block 标记。

## Primary Findings

### 1. Detection signal 在所有 noisy runs 中均优于随机

| Predeclared gate | Formal result | Verdict |
|---|---|---|
| CL-first AUPRC > injected-error prevalence | `54/54` noisy runs；最小 margin `+0.0765` | PASS |
| CL-hard enrichment > 1 | `54/54`；最小 `7.66x` | PASS |
| CL-first recall > random expectation at matched error-count budget | `54/54`；最小 margin `+0.0906` | PASS |

因此，CL evidence 不是随机排序信号。但“超过随机”不等于“高 recall”：最弱的
false-negative-only 场景仍会漏掉大部分真实错误。

### 2. 错误方向造成系统性差异

| Noise-rate scale | `0→1` hard recall | `1→0` hard recall | Gap |
|---:|---:|---:|---:|
| 10% | 0.871 | 0.298 | 0.574 |
| 20% | 0.877 | 0.156 | 0.721 |
| 30% | 0.863 | 0.064 | 0.799 |

- 六个 seed 的 mean gap 为 `0.677-0.721`，全部超过预设 `0.20` gate。
- 六-seed exact sign-flip `p=0.03125`，达到该样本量可取得的最小双侧值。
- parent-overlap block 的平均 gap 为 `0.700`；new-replication block 为 `0.694`。
- 六个疾病、三个 rates 的全部 18 个 label-rate gaps 都为正；最小 gap 仍为
  `0.379`。因此结果不是单一 seed 或单一疾病造成的 aggregation artifact。

最弱的遗漏阳性类别包括 Lung Opacity（跨 rates recall `0.000`）、Atelectasis
（平均 `0.020`）和 Consolidation（平均 `0.072`）。Cardiomegaly 与 Pneumonia
相对较好，但平均 recall 也只有 `0.266/0.222`。

### 3. CL-first 相对简单排序的增量证据仍需谨慎

在 true-error-count budget 下，跨九个 noisy scenarios 的 seed-level 平均：

| Contrast | Mean difference | Seed direction | Exact p | Holm p | Verdict |
|---|---:|---:|---:|---:|---|
| CL-first minus self-confidence | +0.0453 recall | 6 positive / 0 tie / 0 negative | 0.03125 | 0.0625 | 未通过预设 family-wise gate |
| CL-first minus predictive entropy | +0.3319 recall | 6 positive / 0 tie / 0 negative | 0.03125 | 0.0625 | 未通过预设 family-wise gate |

六个 seed 全部同向是可信的 descriptive evidence，但两项比较经过 Holm correction
后均为 `0.0625`，因此不能宣称预设的 incremental-value gate 已确认。

结果还存在机制异质性：CL-first 相对 self-confidence 的平均增益在 balanced、
false-negative-only 和 false-positive-only 中分别为 `+0.0434/+0.0756/+0.0168`；
predictive entropy 在 false-negative-only 场景的 recall 为 `0.268`，高于 CL-first
的 `0.178`。这说明后续 missing-positive detector 不应只复用当前 CL hard rule。

## DQS Calibration Result

### Observed trajectories

| Regime | True quality: clean → 30% | Raw DQS: clean → 30% | Spearman across four anchors | Interpretation |
|---|---:|---:|---:|---|
| balanced | 1.000 → 0.969 | 0.984 → 0.975 | `1.0` for 6/6 seeds | 方向正确但变化被明显压缩 |
| false-positive-only | 1.000 → 0.969 | 0.984 → 0.959 | `1.0` for 6/6 seeds | 单调，但普遍低估质量；MAE `0.0133-0.0140` |
| false-negative-only | 1.000 → 0.969 | 0.984 → 0.993 | `-1.0` for 5/6，`-0.8` for 1/6 | 方向反转，质量越差反而看起来越好 |

预设 DQS gate 要求每个 seed-regime 同时满足 `rho≥0.95`、`MAE≤0.01` 和严格
单调下降，正式结果未通过。

### Why it fails

false-negative-only corruption 把正标签系统性改成零。OOF heads 也使用这些
noisy labels 训练，因此会逐渐学成更少预测阳性。随着真实错误从 `188` 增加到
`560`，平均 CL issue count 反而从 clean 的约 `290` 降到 `273/207/133`，导致
`1 - issue_count/18,000` 逐渐靠近 `1`。

所以 raw DQS 并非独立于训练标签的质量测量。它能表达当前模型认为标签有多
一致，但当噪声具有系统性、并改变模型本身时，模型和错误标签可以一起变得更
“一致”。后续论文中应将它称为 model-consistency-based proxy，并同时报告已知
方向 sensitivity；不能单独用它证明数据真的更干净。

## Mechanistic Interpretation

当前 evidence-consistent explanation 是：

1. `0→1` error 让 noisy label 声称疾病存在，而影像模型仍容易输出低概率，形成
   明显冲突，因此 CL hard flag recall 稳定在约 `0.86-0.88`。
2. `1→0` error 删除正监督。错误越多，OOF model 越可能也输出低阳性概率，
   noisy negative 与模型变得一致，CL 难以识别。
3. 这不仅降低 missing-positive recall，还使基于 issue count 的 DQS 发生反向
   calibration。

受控干预和跨 seed/label 一致性支持这一解释，但本实验没有单独干预模型的外部
监督来源，因此它仍是机制推断而非最终因果分解。

## Claim Ledger

### Supported

- 在当前 VinDr controlled benchmark 中，OOF CL evidence 能在所有预设噪声场景
  中把真实错误富集到随机水平以上。
- 当前方法对 `0→1` false-positive label errors 的检测能力强且跨 rate 稳定。
- 当前方法存在显著、跨 seed、跨 label 的方向不对称。
- 新 seeds `211/307` 复现了原四 seed 的方向差异。

### Not supported

- CL 能可靠找出所有类型的 suspicious entries。
- 当前 hard rule 对 missing-positive `1→0` errors 有足够 recall。
- raw DQS 是跨 corruption mechanism 都 calibrated 的真实 dataset-quality metric。
- CL hard filtering 相对 self-confidence 的增量价值已经通过 Holm-corrected
  confirmatory threshold。
- 受控 VinDr corruption 结果可直接外推到 MIMIC report-derived natural errors、
  iterative LLM correction、下游训练收益或临床效用。

## Fallacy Scan

- **Coverage**: 11/11 statistical fallacy types checked

| Fallacy | Assessment | Handling |
|---|---|---|
| Simpson's paradox | PASS/NOTE | aggregate direction gap 在全部 labels/rates 中同向；仍同时报告 per-label results |
| Ecological fallacy | PASS | claim 和 outcome unit 均为 image-label entry，不推断患者临床结局 |
| Berkson's paradox | CAUTION | VinDr test cohort 与六个 supported labels 是选择后的 benchmark scope，不外推全部疾病/医院 |
| Collider bias | PASS/NOTE | corruption 在 OOF outcome 前冻结，未按 detection success 选择样本 |
| Base-rate neglect | PASS | 同时报告 prevalence、AUPRC、precision、recall 和 enrichment |
| Regression to the mean | PASS | errors 独立于 model score 注入，不是从极端 score 中挑选 |
| Survivorship bias | PASS | 60/60 blind runs 和统一 evaluation 全部通过，无 scenario attrition |
| Look-elsewhere effect | CAUTION | direction contrast 与两项 method family 预声明；method family 使用 Holm correction |
| Garden of forking paths | PASS/NOTE | seeds、rates、regimes、budgets、gates 在 outcome 前 protocol/hash lock；parent finding 明确标为已知 |
| Correlation ≠ causation | CAUTION | controlled corruption 支持 benchmark 内机制比较，不证明自然噪声或临床效用 |
| Reverse causality | PASS | corruption 先于 blind OOF evidence，时间顺序和数据隔离明确 |

## Reproducibility

| Component | Evidence | Status |
|---|---|---|
| Protocol and code | protocol/program SHA-256 locked before formal array | PASS |
| Blind/private isolation | dedicated outcome-free array manifest；evaluation `afterok` all blind runs | PASS |
| Scenario coverage | 6 tasks × 10 disjoint indices，exactly `0-59`；60 completion markers | PASS |
| Formal OOF | all runs produced 3,000 OOF rows and 18,000 evidence rows；stderr empty | PASS |
| Private evaluation | exact output row counts、finite checks、figures、summary、markers | PASS |
| Independent rerun | no second environment/run comparison | NOT RUN |

`Verification Status` 因此保持 `ANALYZED`，不表述为独立复现意义上的 `VERIFIED`。

## Recommended Next Step

当前最有信息量的下一步不是继续增加相同 CL runs，而是针对已确认的
missing-positive failure 做一个 **independent-view detector**：

1. 保留当前 OOF CL 作为擅长 `0→1` 的 detector。
2. 对 observed-negative entries，引入不使用同一 noisy labels 训练的第二证据源，
   例如冻结外部疾病模型、report-text evidence 或二者融合。
3. 在同一个 VinDr direction benchmark 上预注册比较：CL-only、independent-view
   only、union/hybrid；主终点为 `1→0` recall，同时约束 false-positive review burden。
4. DQS 不再只用 CL issue count；若需要单一 quality score，应先在 known-quality
   anchors 上校准，并按 corruption direction 分层验证。

## Final Interpretation

这项实验成功把原来的模糊问题拆成了两个结论。第一，confident learning 确实
能够把真实错误排到随机以上，因此 detection stage 有实际信号。第二，这个信号
不是方向无关的：它很擅长发现错误增加的阳性，却会系统性漏掉被删除的阳性，
并使 raw DQS 在该机制下反向变化。论文中应把前者作为方法有效性的受控证据，
把后者作为明确的适用边界和下一项方法改进目标。
