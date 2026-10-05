# VinDr Known-GT Confident-Learning 正式验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-04
- Verification Status: ANALYZED
- Version Label: vindr_known_gt_cl_validation_v1
- Source Protocol: `vindr_known_gt_cl_protocol_20260804.md`
- Formal Result Root: `/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_known_gt_cl/20260804_v1/evaluation_formal_v1`
- Program SHA-256: `f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83`

## Validation Report

- **Source**: VinDr-CXR 3,000-image controlled known-error benchmark
- **Overall Confidence**: **CAUTION**
- **Primary within-benchmark claim**: supported
- **General real-world claim**: not established by this experiment alone

## Answer First

这个实验支持一个限定清楚的结论：在六个预先选定且支持度充分的
VinDr consensus labels 上，使用冻结影像特征训练的 OOF label-incompatibility
evidence 能够明显富集人工注入的真实错误。四个 seed 的方向完全一致，四项
预设 primary gates 全部通过。

但结果也暴露了一个重要限制：CL 对错误阳性（clean `0` 被改为 noisy `1`）
非常敏感，平均 recall 为 `0.9107`；对遗漏阳性（clean `1` 被改为 noisy `0`）
的平均 recall 只有 `0.1772`。因此，当前证据不能被表述为“CL 对所有标签错误
同样有效”。

## Experimental Integrity

- 3,000 个 VinDr test DICOM、3,000 行 consensus labels 和 3,002 个官方
  checksum targets 已通过完整校验。
- 所有 DICOM 经分层 smoke 后确定性转换为 lossless PNG-224；全量转换逐张
  reopen、hash、shape 和 variance 检查通过。
- 每个 seed 含 18,000 个 entries 和 372 个已知注入错误，错误位置只存在于
  private reference。
- 四个 blind OOF jobs 只接收 noisy cohort 和 outcome-blind frozen XRV features；
  private evaluation 仅在四个 blind markers 全部完成后启动。
- 四个正式 blind jobs、统一 evaluation、结果 postflight 和 benchmark verifier
  全部通过，stderr 均为空。
- 当前报告没有进行第二次独立 rerun，因此 `Verification Status` 保持
  `ANALYZED`，不写成独立重现意义上的 `VERIFIED`。

## Statistical Findings

| Finding | Result | Interpretation | Confidence |
|---|---|---|---|
| CL-first continuous detection | AUPRC `0.5002-0.5174`，error prevalence `0.02067` | 四个 seed 均远高于随机基准；支持受控错误排序能力 | SOLID within benchmark |
| CL hard flags | precision `0.4826-0.5215`；recall `0.5591-0.6048`；enrichment `23.35x-25.24x` | hard pool 中约一半 entries 是真实注入错误，远高于 `2.07%` base rate | SOLID within benchmark |
| Error-count budget | 372-entry budget 下 CL-first 平均找到 `195.75/372`，recall `0.5262`；random 平均 `7.68/372` | 同工作量下，CL-first 明显优于随机 review | SOLID within benchmark |
| CL-first vs self-confidence | recall `0.5262` vs `0.4657`；平均差 `+0.0605`；四 seed 全部同向 | hard filter 在接近 hard-pool 大小的预算附近提供约 6.1 percentage points 增益 | CAUTION |
| Seed-level inference | exact sign-flip `p=0.125`；Holm `p=0.25` | 四个 seed 的最小双侧 exact p-value 就是 `0.125`，不能宣称传统 `p<0.05` significance | CAUTION |
| Hierarchical bootstrap contrast | CL-first minus self-confidence `0.0580`，95% CI `[0.0365, 0.0795]` | 图像层重采样支持正差；不能替代只有四个独立 seed 的 exact test | CAUTION |
| Directional sensitivity | `0→1` recall `0.9107`；`1→0` recall `0.1772` | 方法主要擅长发现错误增加的阳性，发现遗漏阳性的能力较弱 | SOLID descriptive finding |
| Per-label consistency | 六类平均 recall `0.4338-0.6673`；enrichment `14.10x-35.48x` | 所有类别均有富集，但 Lung Opacity 和 Atelectasis recall 较低 | CAUTION |
| OOF model signal | clean-reference AUROC `0.8564-0.9637`；noisy-label AUROC `0.7673-0.8583` | OOF estimator 有足够影像信号，但不同类别能力仍有差异 | CAUTION |
| DQS agreement | true quality `0.97933`；raw DQS `0.97539-0.97678`；bias `-0.00256` 至 `-0.00394` | 当前单个 quality level 上 DQS 接近真值且略保守；尚不能证明跨质量水平 calibration | CAUTION |

## Budget-Specific Interpretation

CL-first 与 global self-confidence 并不是在所有预算下都不同：

| Review budget | CL-first mean recall | Self-confidence mean recall | Difference |
|---|---:|---:|---:|
| 1% | 0.3280 | 0.3280 | 0.0000 |
| 2% | 0.5128 | 0.4624 | +0.0504 |
| 5% | 0.6492 | 0.6492 | 0.0000 |
| 10% | 0.8495 | 0.8495 | 0.0000 |
| True-error count (2.07%) | 0.5262 | 0.4657 | +0.0605 |

这说明 CL hard filter 的增量价值集中在 review budget 接近 hard-pool 大小的区域。
在更小或更大的预算下，当前 CL-first 排序与 self-confidence 产生了相同的选择
结果。因此不能把整体错误富集全部归因于 hard filter；label-aware OOF
self-confidence 本身已经贡献了大部分排序能力。

## Directional Failure Analysis

每个标签都注入相同数量的 `0→1` 和 `1→0` errors，因此方向差异不是由两类
错误数量不平衡造成的。

- `0→1` 表示 clean negative 被污染为 noisy positive。模型若仍预测 negative，
  noisy label 与 OOF evidence 直接冲突，因此较容易被检测。
- `1→0` 表示 clean positive 被污染为 noisy negative。六个任务均为低 prevalence
  findings；减少阳性训练样本后，OOF model 可能也输出较低 positive probability，
  因而不会把 noisy negative 标成明显冲突。

这个机制解释目前属于 evidence-consistent inference，不是已经完成的因果证明。
下一项实验必须把两种 corruption directions 分开预注册和评价。

## DQS Interpretation Boundary

当前 benchmark 只提供一个全局真实质量点：

`true quality = 1 - 372 / 18,000 = 0.97933`。

四个 seed 的 raw DQS 均接近该值，但 DQS 实际估计的 issue 数为 `418-443`，
高于真实错误数 372，因此 DQS 略微低估数据质量。这个结果说明 DQS 在当前设置
下没有因 denominator shrinkage 产生虚假改善，也说明其数量级合理；但一个点
不能用于验证 calibration slope、monotonicity 或不同 noise mechanisms 下的稳定性。

## Warnings

| Type | Detail | Affected claim |
|---|---|---|
| Four-seed ceiling | 双侧 exact sign-flip test 最小只能到 `0.125` | CL hard filter 相对 self-confidence 的 seed-level generalization |
| Synthetic corruption | 当前错误是随机、受控且 prevalence-preserving，不等同于 report-derived natural noise | 向 MIMIC-CXR 真实伪标签错误外推 |
| Direction asymmetry | `1→0` recall 明显低于 `0→1` | “CL 能发现各种错误” |
| Single calibration point | 只有 `true quality=0.97933` | DQS 是普遍 calibrated dataset-quality metric |
| Supported-label scope | 只包含六个预先选择、support ≥84 的 labels | 极低 prevalence 或未映射 findings |
| Consensus reference | 五位 radiologist consensus 是高质量 benchmark reference，但不是不可错的临床真值 | “ground truth” 的绝对表述 |
| Image-level grouping | 公开 headers 没有稳定 patient identifier，folds 以 image/study unit 建立 | 潜在同患者相关性的完全排除 |
| Budget-local CL gain | CL-first 与 self-confidence 在 1%、5%、10% budget 下完全相同 | hard filter 在所有 review budgets 都有增益 |
| No independent rerun | 当前只完成一次 hash-locked formal execution | 独立 reproducibility claim |

## Fallacy Scan

- **Coverage**: 11/11 statistical fallacy types checked

| Fallacy | Severity | Assessment | Required handling |
|---|---|---|---|
| Simpson's paradox | NOTE | 所有六类 enrichment 均 >1，没有 aggregate/per-label direction reversal；但两个 flip directions 的效果量高度异质 | 同时报告 aggregate、per-label 和 per-direction results |
| Ecological fallacy | NOTE | unit of analysis 与 claim 均为 image-label entry，没有用 group-level result 推断个人临床结果 | 保持 entry-level claim |
| Berkson's paradox | CAUTION | VinDr test cohort 和六个 supported labels 是选择后的 benchmark scope | 不外推为全部医院、全部 findings 或 MIMIC population |
| Collider bias | NOTE | 分析未按 OOF score、CL selection 或 outcome-conditioned covariate做回归控制，未发现明显 collider conditioning | 后续 sensitivity design 继续保持 outcome-blind selection |
| Base-rate neglect | PASS | 已报告 `2.067%` prevalence、precision、recall、AUPRC 和 enrichment | 汇报时必须保留 prevalence，不只说 accuracy |
| Regression to the mean | PASS | errors 在 OOF outcome 前随机注入，不是从 extreme model scores 中挑选 | 后续 injection 不得按当前 detection score 选择错误位置 |
| Survivorship bias | PASS | 3,000 images、18,000 entries 和四个 seeds 无 attrition，所有正式任务通过 | 保留 row-count 与 marker checks |
| Look-elsewhere effect | CAUTION | primary gates 已预声明；per-label、per-direction 和多个 budgets 属 secondary family | 主要结论只依赖 locked gates，secondary results 明确标注并做 family correction |
| Garden of forking paths | NOTE | protocol、seed、labels、noise rate、comparators 和 endpoints 在 outcome 前 hash-lock | 新 noise-rate/direction study 必须另建 pre-outcome protocol，不能改写本实验 |
| Correlation ≠ causation | CAUTION | controlled injection 支持在该 benchmark 内评价 detection，但不能证明真实临床标签噪声机制或 downstream utility | 使用“detects injected errors”，不使用“improves real clinical labels” |
| Reverse causality | PASS | injected-error locations 在 selection 前冻结，temporal direction 清楚 | 不适用于当前 controlled intervention |

## Reproducibility

- **Method**: hash-locked formal execution with deterministic seeds, immutable inputs,
  outcome isolation, completion markers and strict postflight; no second independent rerun
- **Verdict**: **CANNOT_VERIFY independently**

| Component | Evidence | Status |
|---|---|---|
| Source data | 3,002 official SHA-256 checks and image-label mapping | PASS |
| Image preprocessing | stratified smoke, contact-sheet review, 3,000-image hash/reopen audit | PASS |
| Blind/private isolation | separate cohort/reference files and dependency-gated evaluation | PASS |
| Four-seed OOF | all array tasks completed with empty stderr and seed-local postflight | PASS |
| Statistical outputs | exact row counts, finite values, plots, summary and markers | PASS |
| Independent rerun | no second environment/run comparison | NOT RUN |

## Recommended Next Experiment

下一项实验应优先验证 **noise-direction robustness 与 DQS calibration**，而不是只
重复当前 20% balanced corruption。

### Research Questions

1. `1→0` detection weakness 是否在不同 error rates 下持续存在？
2. 在相同真实错误数量下，balanced、false-positive-only 和
   false-negative-only noise 是否产生不同 detection performance？
3. raw DQS 是否随多个已知 true-quality levels 单调变化，并保持可接受的 bias？
4. CL hard filter 相对 self-confidence 的增益是否仍只出现在 hard-pool 附近？

### Proposed Factors

- Error-rate anchors: clean `0%`，以及 `10% / 20% / 30%` positive-support scale。
- Corruption regimes with matched total error count:
  - balanced `0→1 + 1→0`；
  - `0→1` only；
  - `1→0` only。
- Seeds: 保留原四个 seeds，并预先冻结两个全新 replication seeds。
- Features: 复用已经 outcome-blind 的 frozen XRV features；不调用 LLM API。

### Primary Endpoints

- per-direction recall and AUPRC；
- macro average across the two directions；
- CL-hard enrichment；
- CL-first vs self-confidence equal-budget recall；
- DQS bias、absolute error、calibration slope/intercept 和 monotonicity；
- original four seeds 与 two-seed prospective extension 分层报告，六-seed pooled
  analysis 只作预先声明的 replication summary。

### Gate Before Execution

在看到任何新 outcome 前，先冻结 seeds、error-count matching、budget family、
multiple-comparison correction、DQS calibration model 和 failure criteria。当前四
seed 结果不能在新 protocol 中被重新标成未见过的 confirmatory evidence。

## Final Interpretation

当前实验已经回答 detection-stage 的核心问题：OOF evidence 在受控 known-error
benchmark 中不是随机信号，而是能够大幅富集真实错误。当前最重要的新科学问题
不是“再证明一次整体效果”，而是解释和验证为什么遗漏阳性 `1→0` 的识别能力
明显更弱，以及 DQS 是否能跨多个真实质量水平稳定追踪 dataset quality。
