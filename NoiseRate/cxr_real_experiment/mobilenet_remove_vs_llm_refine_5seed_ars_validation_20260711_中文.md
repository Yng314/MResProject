## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-11
- Verification Status: ANALYZED
- Version Label: validation_v1
- Data Scope: MobileNetV3 noES50, seeds `7/13/42/97/123`, baseline + simple remove + fixed LLM refine, Loops 1-5

## Validation Report

- **Source**: completed local experiment outputs and job `259487`
- **Overall Confidence**: **CAUTION**
- **Primary metric**: held-out test study-weighted AUROC
- **Test set**: 605 studies, 12 labels

### 一句话结论

五个配对 seed 显示出清楚的 **quality-coverage tradeoff**：simple remove 在早期 loop 提升较大，但随着累计删除约 14.1% 的训练样本，后期收益明显衰减；固定 LLM refinement 基本保留样本量，并在后期取得更高的点估计。不过，LLM-refine Loop5 相对 simple-remove Loop5 的 hierarchical seed+study 95% CI 仍跨 0，且排除三个极低 minority-support 标签后，优势从 `+0.0260` 缩小到 `+0.0068`。因此当前证据支持“LLM refinement 的后期 quality-coverage 趋势更好”，但不支持“LLM refinement 已被确认普遍优于 simple remove”。

### 分析单位与可比性

用户提交了 10 个 seed-method 作业，即 5 个 simple-remove seed 和 5 个 LLM-refine seed。统计上不能把它们当成 10 个独立重复：

- 真正的重复单位是相同的 5 个训练 seed；method 和 loop 是 seed 内配对条件。
- 每个方法包含 5 个累计 loop，因此共有 50 个 cleaned models，另有 5 个共同 baseline models。
- 所有 55 份预测文件使用完全相同的 605 个 test studies、标签和 valid masks。
- Simple remove 在每个 seed 中重新运行 OOF detection，清理集合随 seed 改变。
- LLM refine 在五个 seed 中复用 XRV seed13 产生的固定 refinement tables，仅改变 MobileNet 训练随机性。
- 因此两组可以比较当前数据集版本下的模型性能，但 LLM 结果并不验证完整 detection + LLM pipeline 的跨 seed 重复性。

### Data Integrity

| Check | Result |
|---|---:|
| Prediction files checked | 55 |
| Studies per file | 605 |
| Duplicate study IDs | 0 |
| Study/label/mask alignment failures | 0 |
| Recomputed metric mismatch | 0 |
| Completed seed-loop cells | 50/50 |
| Slurm stderr for unified analysis | 0 bytes |

### Statistical Findings

#### 全 12 标签点估计

Baseline 为 `0.7231 ± 0.0193`。表中 `Delta` 均与同 seed baseline 配对计算。

| Loop | Simple remove AUROC | Remove Delta | Positive seeds | LLM refine AUROC | Refine Delta | Positive seeds | Refine - Remove |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7512 | +0.0281 | 4/5 | 0.7365 | +0.0134 | 3/5 | -0.0147 |
| 2 | 0.7499 | +0.0268 | 4/5 | 0.7429 | +0.0198 | 4/5 | -0.0070 |
| 3 | 0.7432 | +0.0201 | 5/5 | 0.7372 | +0.0141 | 4/5 | -0.0059 |
| 4 | 0.7316 | +0.0086 | 3/5 | 0.7479 | +0.0248 | 4/5 | +0.0162 |
| 5 | 0.7312 | +0.0081 | 4/5 | 0.7572 | +0.0341 | 5/5 | +0.0260 |

数据呈现 method-by-loop interaction：simple remove 在 Loops 1-2 的均值较高，LLM refine 在 Loops 4-5 的均值较高。Loop3 的 LLM-minus-remove 均值为负，但 4/5 seed 实际为正，原因是 seed97 的单个大负差 `-0.0746`；因此不能只看聚合均值。

#### Seed 与 test-study 不确定性

`Seed t-CI` 使用 5 个 paired seed delta；`conditional CI` 在固定这 5 个 seed 时，对共同的 test studies 做配对 bootstrap；`hierarchical CI` 同时重采样 seed 和共同 test studies。由于只有 5 个 seed，hierarchical CI 是更保守但仍较粗糙的主要不确定性参考。

| Comparison | Delta | Seed t-CI | paired dz | Exact sign-flip p | Conditional study CI | Hierarchical seed+study CI |
|---|---:|---:|---:|---:|---:|---:|
| Remove L3 - baseline | +0.0201 | [0.0084, 0.0318] | 2.13 | 0.0625 | [-0.0030, 0.0410] | [-0.0064, 0.0484] |
| Refine L5 - baseline | +0.0341 | [0.0025, 0.0657] | 1.34 | 0.0625 | [0.0112, 0.0545] | [-0.0005, 0.0783] |
| Refine L4 - Remove L4 | +0.0162 | [0.0019, 0.0305] | 1.41 | 0.0625 | [-0.0047, 0.0351] | [-0.0121, 0.0397] |
| Refine L5 - Remove L5 | +0.0260 | [0.0076, 0.0444] | 1.76 | 0.0625 | [-0.0157, 0.0685] | [-0.0210, 0.0786] |

解释：

- Seed-level effect sizes 很大，但 n=5 时无法可靠检验分布假设。
- 5 个 paired seeds 的双侧 exact sign-flip p 最小只能达到 `0.0625`。
- 对每个 comparison family 的 5 个 loop 做 Holm correction 后，没有 seed-exact 结果通过 `0.05`。
- 全标签 Refine L5 vs baseline 的 conditional study-bootstrap 在 5-loop Holm correction 后仍为 `p=0.0100`，但 hierarchical CI 跨 0，说明该证据主要支持“固定现有 seeds 和训练结果时的 test-set improvement”，尚不足以推广到新的训练/pipeline seeds。

### Rare-label Sensitivity

三个标签的 minority support 小于 5：Atelectasis 为 `212 positive / 2 negative`，Lung Lesion 为 `51/2`，Pleural Other 为 `22/1`。它们的 AUROC 极易受少数样本影响，但当前 weighted AUROC 按 valid count 加权，因此仍可能得到较大权重。

排除这三个标签后，baseline 变为 `0.7407 ± 0.0209`：

| Loop | Remove Delta | Positive seeds | Refine Delta | Positive seeds | Refine - Remove | Direct positive seeds |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | +0.0090 | 3/5 | +0.0031 | 3/5 | -0.0059 | 2/5 |
| 2 | +0.0068 | 3/5 | +0.0079 | 3/5 | +0.0011 | 2/5 |
| 3 | +0.0130 | 3/5 | +0.0063 | 3/5 | -0.0067 | 2/5 |
| 4 | +0.0071 | 3/5 | +0.0129 | 4/5 | +0.0058 | 3/5 |
| 5 | +0.0094 | 3/5 | +0.0162 | 3/5 | +0.0068 | 5/5 |

Support-filtered Refine L5 - Remove L5 的 seed t-CI 为 `[0.0005, 0.0131]`，但 conditional study CI 为 `[-0.0090, 0.0238]`，hierarchical CI 为 `[-0.0152, 0.0267]`。因此同方向的 5/5 seed 值得关注，但不能消除 test-set sampling uncertainty。

### Per-label Findings at Loop5

| Label | Positive / Negative | Refine - Remove | Positive seeds | Interpretation |
|---|---:|---:|---:|---|
| Pleural Other | 22 / 1 | +0.2000 | 4/5 | 极低支持，不可稳定解释 |
| Atelectasis | 212 / 2 | +0.1965 | 4/5 | 极低 minority support，不可稳定解释 |
| Consolidation | 70 / 21 | +0.0940 | 5/5 | 最清楚的稳定正向标签 |
| Pneumothorax | 38 / 163 | +0.0497 | 4/5 | 较可信的正向标签 |
| Edema | 165 / 87 | -0.0286 | 0/5 | 稳定负向标签 |
| Enlarged Cardiomediastinum | 66 / 27 | -0.0475 | 1/5 | 多数 seed 负向 |
| Fracture | 37 / 7 | -0.0587 | 1/5 | 支持仍较低且负向 |
| Lung Lesion | 51 / 2 | -0.1510 | 1/5 | 极低支持，不可稳定解释 |

LLM refinement 不是对所有 pathology 一致改善。总体优势同时包含稳定标签上的真实信号候选和极低支持标签上的大幅波动。

### Coverage and Denominator

| Loop | Remove retained samples | Remove retention | Mean cumulative removed | Refine retained samples | Relabel entries | Mask entries |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 228,559 | 96.15% | 9,158 | 237,717 | 1,753 | 595 |
| 2 | 220,903 | 92.93% | 16,814 | 237,717 | 2,842 | 1,281 |
| 3 | 214,477 | 90.22% | 23,240 | 237,717 | 3,389 | 1,806 |
| 4 | 208,927 | 87.89% | 28,790 | 237,717 | 3,689 | 2,070 |
| 5 | 204,175 | 85.89% | 33,542 | 237,717 | 4,007 | 2,215 |

Simple remove 的后期指标下降与 coverage 下降同时发生，但当前设计不能单独证明哪个被删除样本造成了下降。LLM refine 保持 sample denominator 不变，但并非完全不改变分母：entry masks 会小幅减少有效 label-entry denominator。因此准确说法是“sample coverage 保持 100%，entry coverage 轻微下降”，而不是“分母完全不变”。

从点估计看，Refine L5 在 100% sample retention 下达到 `0.7572`，高于 Remove L1 在 96.15% retention 下的 `0.7512`。但两者配对差仅 `+0.0061`、3/5 seed 正向，seed t-CI `[-0.0216, 0.0338]`；这可以描述为更好的 point-estimate Pareto position，不能描述为已确认的性能优势。

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Small n | 只有 5 个 paired training seeds；exact test 分辨率不足 | 所有 seed-level claims |
| Multiple comparisons | 3 个 comparison families × 5 loops × 2 support scopes，另有 12-label exploration | best-loop 和 per-label claims |
| Post-hoc loop selection | Loop5 是看到结果后才成为最强 LLM 点 | “Loop5 最优”只能是 exploratory |
| Shared test set | 五个 seed 使用同一 605-study test set，不能当成五份独立测试集 | bootstrap design |
| Rare-label instability | 3 个标签 minority support 为 1-2，但 valid-count weighting 仍给予权重 | full weighted AUROC |
| Unequal randomness scope | Remove 每 seed 重做 detection；Refine 使用固定 XRV-seed13 tables | method-level reproducibility |
| Unequal intervention | 同一 loop 不代表相同删除/重标/掩码数量或相同成本 | direct method comparison |
| No adjudicated ground truth | Held-out AUROC 验证 downstream utility，不验证每个 refinement 的临床正确性 | label-quality claim |

### Fallacy Scan

- **Coverage**: **11/11 fallacy types checked**

| Fallacy | Severity | Finding | Required interpretation control |
|---|---|---|---|
| Simpson's Paradox | CAUTION | 无严格 Simpson reversal，但 Loop3 aggregate 为负而 4/5 seed 为正，受 seed97 大负值主导 | 同时展示 mean、每 seed 方向和轨迹 |
| Ecological Fallacy | CAUTION | 总体 weighted AUROC 不能推出每个 pathology、每份报告或每次 relabel 都正确 | 保留 per-label 与人工 audit |
| Berkson's Paradox | CAUTION | 605-study expert-labeled MIMIC subset 与 AP/PA 过滤属于选择后的样本 | 不外推到未筛选人群或外部医院 |
| Collider Bias | NOTE | 当前比较没有额外协变量控制，未发现明确 collider；但 inclusion filters 仍限制适用范围 | 记录 test-set inclusion rule |
| Base Rate Neglect | CAUTION | minority support 1-2 的标签产生大 AUROC 波动 | 必须并列报告 positive/negative support 和 sensitivity |
| Regression to the Mean | CAUTION | 每轮选择 OOF disagreement 极端样本，后续变化可能混入极端值回归 | 使用 baseline、固定 loop 和独立审查 |
| Survivorship Bias | RED_FLAG | Remove L5 只保留 85.89% 样本；剩余数据更“干净”不等于原数据被修复 | DQS/cleanliness 必须与 coverage 同报 |
| Look-Elsewhere Effect | CAUTION | 多 loop、多 contrast、多 label 后选择最好结果 | 主结论不能只报 best loop；使用 Holm 与完整曲线 |
| Garden of Forking Paths | CAUTION | top20%、5 loops、noES50、weighted metric、support threshold 均未预注册 | 当前结果标为 exploratory；后续确认实验预先锁定 |
| Correlation != Causation | CAUTION | 训练数据操作是受控的，但只能说明该配置下模型表现变化，不能证明临床正确性或普遍因果效应 | 限定到当前模型、数据和 pipeline |
| Reverse Causality | CAUTION | OOF disagreement 可能来自模型偏差/困难病例，而不一定来自错误标签 | 需要人工 adjudication 或独立证据确认 label error |

### Reproducibility

- **Method**: five-seed stochastic robustness analysis + independent metric recomputation; no exact training rerun
- **Verdict**: **PARTIALLY_REPRODUCIBLE**

| Component | Status | Evidence / limitation |
|---|---|---|
| Result completeness | VERIFIED | 50/50 cleaned seed-loop cells and 5 baselines present |
| Prediction alignment | VERIFIED | 55 files share the same 605 studies, labels, and masks |
| Metric recomputation | VERIFIED | Recomputed weighted AUROC matches stored summaries exactly |
| Training-seed robustness | PARTIAL | Five seeds available, but uncertainty remains wide |
| Simple-remove pipeline variation | PARTIAL | Detection and training rerun by seed, but only five seeds |
| Full LLM pipeline variation | NOT VERIFIED | Refinement tables fixed from XRV seed13; LLM/API and detection were not rerun |
| Exact rerun reproducibility | NOT VERIFIED | No identical-seed end-to-end rerun was performed |

### What the Evidence Supports

1. Simple remove provides an early-loop performance signal, but later removal gives diminishing returns while sample retention falls from 96.15% to 85.89%.
2. Fixed LLM refinement preserves sample coverage and produces a stronger late-loop trajectory; under the full 12-label metric, Loop5 is positive against baseline in all five seeds.
3. At the same Loop5, LLM refinement has a higher point estimate than simple remove in all five seeds, but study and hierarchical CIs still include zero.
4. Rare-label sensitivity materially weakens the headline effect; the support-filtered direct advantage is only `+0.0068`.
5. The strongest defensible interpretation is a promising quality-coverage tradeoff, not confirmed universal superiority or proof that the refined labels are correct.

### Required Next Validation

1. Treat the current result as exploratory and pre-specify one primary endpoint before any confirmation run; Loop5 is defensible as the natural cumulative-refinement endpoint, but this must be declared before seeing new data.
2. Generate refinement tables from additional independent detection seeds, rather than only retraining MobileNet on the fixed XRV-seed13 table.
3. Use support-aware primary reporting and keep the full 12-label metric as a sensitivity result, not the only headline metric.
4. Perform targeted manual adjudication for Consolidation, Pneumothorax, Edema, Fracture, and the three minority-support labels.
5. Preserve all loops and coverage counts in slides; do not report only the best method-loop cell.

### Output Files

- Unified result directory: `/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_noes50_clean_3seed/20260707_123320/evaluation_20260711_remove_vs_refine_5seed_ars`
- DQS/noise-rate/coverage companion report: `cxr_real_experiment/mobilenet_remove_vs_llm_refine_5seed_data_quality_ars_20260711_中文.md`
- Main statistics: `seed_paired_statistics.csv`, `paired_bootstrap_statistics.csv`, `score_summary.csv`
- Sensitivity and audit: `test_label_support.csv`, `per_label_deltas.csv`, `coverage_summary.csv`, `posthoc_method_comparisons.csv`
- Figures: `five_seed_method_curve.png`, `refine_vs_remove_hierarchical_ci.png`, `performance_vs_sample_retention.png`
