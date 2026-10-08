# NoiseRate - Label Noise Analysis for Chest X-rays

胸部X光多标签分类的标签噪声分析工具。

> **Repository boundary:** install dependencies from `requirements.txt` after selecting a machine-appropriate PyTorch build. Generated CSV/Excel tables, logs, prediction arrays, figures, datasets, and checkpoints are not versioned; use approved external storage as recorded in the repository-level [`DATA_LAYOUT.md`](../DATA_LAYOUT.md). The historical output paths below describe program behavior and do not mean those outputs belong in Git.

## 📁 文件结构

```
NoiseRate/
├── validate_torchxray.py          # TorchXRayVision完整流程（推理+噪声分析+AUROC）
├── train_validate_kfold.py        # K-fold训练和验证（GT和Pseudo）
├── explore_gt_data.py             # 数据探索脚本
├── cxr_real_experiment/           # 正式 MIMIC-CXR-JPG 实验（baseline / cleaned / entry-cleaned）
│
├── utils/                          # 工具函数
│   ├── noise_analysis.py           # Cleanlab噪声分析函数
│   ├── auroc_metrics.py            # AUROC计算函数
│   ├── label_utils.py              # 标签处理工具
│   ├── xrv_utils.py                # TorchXRayVision工具
│   └── training_utils.py           # 训练工具
│
├── models/                         # 模型定义
│   └── lightweight_cnn.py          # 轻量级CNN (391K参数)
│
├── config/                         # 配置文件
│   └── unified_config.yaml         # 统一配置文件
│
├── tests/                          # 单元测试
│   ├── test_label_utils.py
│   └── test_xrv_utils.py
│
└── results/                        # 输出结果目录
```

`cxr_real_experiment/` 当前包含：

- `cxr_real_noise_validation_smoke.py`
  - 正式 MIMIC-CXR-JPG 数据上的 K-fold / OOF / cleanlab 烟测与全量噪声分析
- `cxr_real_full_train_eval_cleanlab.py`
  - 全训练集 DenseNet 训练、GT test 评测、再对 train 生成 cleanlab 结果
  - 现已支持通过 metadata 文件按 `ViewPosition` 过滤，例如 `AP/PA`
- `run_baseline_full_train_test_cleanlab.sh`
  - baseline 训练脚本（现配置：`epochs=100`, `patience=10`, `restore best`）
- `run_cleaned_full_train_test_cleanlab.sh`
  - sample-level cleaned 重训脚本
- `run_entry_cleaned_full_train_test_cleanlab.sh`
  - entry-level cleaned 重训脚本
- `run_appa_pipeline_full_train.sh`
  - 单个 Slurm 作业串行跑完 `AP/PA baseline -> AP/PA sample-cleaned -> AP/PA entry-cleaned`
- `cxr_real_full_train_eval_cleanlab_xrv12.py`
  - `TorchXRayVision` 12类正式实验脚本
  - 支持 `xrv_densenet121_direct` 和 `xrv_densenet121_linearhead`
- `run_appa_xrv12_pipeline_full_train.sh`
  - 单个 Slurm 作业串行跑完 `AP/PA + XRV direct 12类` 的三阶段流程
- `run_appa_xrv12_linearhead_pipeline_full_train.sh`
  - 单个 Slurm 作业串行跑完 `AP/PA + XRV encoder + our head 12类` 的三阶段流程
- `select_topk_issue_samples.py`
  - 从 sample-level cleanlab 结果中选取 top-k / top-percentage 可疑样本
- `select_topk_issue_entries.py`
  - 从 entry-level cleanlab 结果中选取 top-k / top-percentage 可疑标签位
- `run_appa_densenet_topk_sweep.sh`
  - `AP/PA + DenseNet 14类` 的 sample-level top-k sweep（5% / 10% / 20%）
- `run_appa_xrv12_linearhead_topk_sweep.sh`
  - `AP/PA + XRV encoder + our head 12类` 的 sample-level top-k sweep（5% / 10% / 20%）
- `run_appa_densenet_entry_topk_sweep.sh`
  - `AP/PA + DenseNet 14类` 的 entry-level top-k sweep（5% / 10% / 20%）
- `run_appa_xrv12_linearhead_entry_topk_sweep.sh`
  - `AP/PA + XRV encoder + our head 12类` 的 entry-level top-k sweep（5% / 10% / 20%）
- `run_xrv_iterative_sample20_branch.sh`
  - `AP/PA + XRV encoder + our head 12类` 的 iterative sample20 loop 单分支 runner，支持 `remove_only` 和 `llm_refine`
- `run_xrv_iterative_sample20_unseen_branch.sh` / `select_topk_unreviewed_issue_samples.py`
  - 支持按全历史排除已成功 review 的 entry；full-pool 模式每轮 review 当前 CL issue pool 的全部新 entries，API 失败不会被计为已 review
- `submit_xrv_iterative_sample20_dual_loop.sh`
  - 同时提交 `remove_only` 与 `llm_refine` 两条 sample20 loop 分支，用 Slurm dependency 串起每轮作业
- `summarize_xrv_loop_metrics.py`
  - 每轮训练后汇总 `image macro`、`study macro`、`study weighted` 和累计清洗数量到 `loop_metrics.csv`
- `build_llm_refinement_audit.py`
  - 汇总 LLM refinement 对 suspicious entries 的实际动作，包括 `keep/relabel/mask`、relabel 方向、test set 分布和 per-label metric delta
- `run_sample05_llm_refinement_audit.sh`
  - 在 Slurm 作业中生成 `sample top5%` 的 LLM refinement audit 表，不在登录节点跑数据统计
- `evaluate_mobilenet_cleaning_methods_5seed.py`
  - 统一比较五 seed baseline、simple remove 和 fixed LLM refine，输出 seed 配对统计、shared-study 与 hierarchical bootstrap、support sensitivity、per-label delta 和 coverage
- `run_mobilenet_cleaning_methods_evaluation.sh`
  - 在 Slurm 中运行上述统一评估，避免在登录节点执行较长 bootstrap
- `evaluate_mobilenet_data_quality_5seed.py`
  - 重算五 seed 的 OOF DQS、正确/旧版 noise-rate 分母、coverage-adjusted DQS、selection stability、per-label distribution shift，以及 held-out AP/Brier/NLL
- `run_mobilenet_data_quality_5seed.sh`
  - 在 Slurm 中运行上述数据质量补充评估，并生成 CSV、metadata 和四张比较图
- `validate_synthetic_dqs_adjustment.py`
  - 用已知标签真值的轻量 synthetic experiment 验证 raw DQS 与 coverage-adjusted DQS 的 matching targets 和 OOF-evidence sensitivity
- `run_synthetic_dqs_validation.sh`
  - 在 Slurm 中运行 10-seed synthetic DQS validation，并保留邮件通知、CSV、JSON 和 figures
- `medpalm_external_entry_benchmark.py` / `run_medpalm_external_entry_benchmark.sh`
  - 把 Med-PaLM disagreement entries 作为外部提供的 suspicious entries，在不泄漏 Med-PaLM 答案、医生 reference 或 OOF probability 的条件下复用 U-Ones entry-level reviewer 逻辑，并用专家 binary reference 直接评价 accuracy、coverage、adjusted correct mass、relabel precision/recall 与 harmful flip
- `medpalm_external_entry_benchmark_protocol_20260730.md`
  - 在 API outcome 前锁定 498-entry scope、盲评输入、binary reference、mask-as-abstention 记分、subject-cluster bootstrap 和解释边界
- `medpalm_model_review_bakeoff.py` / `run_medpalm_model_review_bakeoff.sh`
  - 从上述 benchmark 固定抽取 100 个不同 subject 的平衡配对样本（50 keep / 50 relabel），复用存档 GPT-5.4 结果并通过 Responses API 盲测 GPT-5.6-Luna，比较 exact action accuracy、keep accuracy、relabel recall、harmful flip、mask、配对 McNemar 检验和实际 API usage；输出位于 `medpalm_model_review_bakeoff/20260820_gpt54_vs_gpt56luna_n100`
- `medpalm_luna_effort_bakeoff.py` / `run_medpalm_luna_effort_bakeoff.sh`
  - 在同一锁定 100-entry 样本上显式比较 Luna `medium` 与 `max` reasoning effort，并与存档 GPT-5.4 actions 配对；每档先执行单条 canary，代理无可用 Luna distributor 时不会启动并发批次
- `medpalm_terra_default_bakeoff.py` / `run_medpalm_terra_default_bakeoff.sh`
  - 在同一锁定样本上运行不传 `reasoning` 参数的 GPT-5.6-Terra 默认行为，与 GPT-5.4 做逐条配对并记录 action quality、mask、token、延迟及服务端返回的默认元数据
- `medpalm_cl_detection_benchmark.py` / `run_medpalm_cl_detection_benchmark.sh`
  - 用四个冻结的未清理 MobileNetV3 baseline 在完整 official test AP/PA pool 上生成 held-out probabilities；先盲化计算 per-finding CL ranks，再连接498条 Med-PaLM 专家 reference，评价 AUPRC、top-20 enrichment、hard flags、seed consistency、subject bootstrap 与 synthetic positive control，不运行 iterative loops 或 LLM API
- `medpalm_cl_detection_protocol_20260803.md`
  - 预先锁定 CL detection research question、outcome isolation、primary gate、supported-label sensitivity 与只适用于 externally selected hard cases 的解释边界
- `dual_reference_cl_detection_benchmark.py` / `run_dual_reference_cl_detection_benchmark.sh`
  - 复用冻结的四-seed/ensemble CL scores，在不读取专家答案的 `select` 阶段重现实际 study-level hard-pool/top-20 policy，再分别用 MIMIC-CXR-JPG 2.1 radiologist labels 和 Med-PaLM hard cases 评价连续分数、hard flags、exact policy、subject-cluster bootstrap 与 matched random prioritization
- `dual_reference_cl_detection_protocol_20260803.md` / `dual_reference_cl_detection_validation_20260803_zh.md`
  - 锁定双参考集 outcome isolation、主次 reference、精确 selection policy 和解释边界，并记录正式结果：CL detection 在两组 reference 中均富集真实错误，但 whole-study-mean top-20 prioritization 未在主要 MIMIC 2.1 panel 上优于 hard-pool 内随机选择
- `cl_prioritization_followup_benchmark.py` / `run_cl_prioritization_followup_benchmark.sh`
  - 在同一冻结 CL-hard pool 上 outcome-blind 比较 whole-study mean、worst-entry、hard-entry-mean 和 direct-entry；study policies 匹配 study budget，direct-entry 匹配 expanded-entry workload，并用双 reference、paired subject bootstrap 和 unit-matched random policy 分离旧聚合损失与 hard-pool 内排序能力
- `cl_prioritization_followup_protocol_20260803.md` / `cl_prioritization_followup_validation_20260803_zh.md`
  - 锁定 post-result follow-up 的三种候选策略、预算与 decision rule；正式结果显示 direct-entry 在相同 186-entry budget 下把 MIMIC 2.1 issue capture 从 6 提升到 16，但 matched-random `p=0.0595` 未通过预设 `<0.05` gate
- `cl_vs_uncertainty_ablation.py` / `run_cl_vs_uncertainty_ablation.sh`
  - 在完全相同的 entry budget 下，outcome-blind 比较 CL hard + direct rank、全局 self-confidence、finding 内 percentile、predictive entropy 和 seed disagreement，再连接 MIMIC 2.1 与 Med-PaLM private references 做 subject bootstrap 和全池随机对照
- `cl_vs_uncertainty_ablation_protocol_20260803.md` / `cl_vs_uncertainty_ablation_validation_20260803_zh.md`
  - 锁定 CL hard filter 的 incremental-value gate；正式结果确认 label-aware OOF evidence 明显优于随机，但 CL hard filter 在当前预算下没有显示超越直接 self-confidence 排序的额外价值
- `download_vindr_test_parallel.py` / `run_download_vindr_test_parallel.sh`
  - 通过受保护的 node-local PhysioNet `.netrc` 四路并发续传 VinDr-CXR 3,000-image consensus test cohort，只保留 image-level detection benchmark 所需文件，并用官方 SHA-256、image-label 一一对应和完成标记做最终核验
- `preprocess_vindr_dicoms.py` / `run_vindr_preprocess_smoke.sh` / `run_vindr_preprocess_full.sh`
  - 把已验证的 VinDr DICOM 经 modality、VOI/window、MONOCHROME inversion、中心裁剪后确定性转换为 lossless PNG-224；先覆盖全部 transfer syntax、photometric、bit-depth 和 window 状态的分层 smoke，再执行 3,000 张全量转换与逐文件 hash/readability 验收
- `vindr_known_gt_cl_benchmark.py` / `vindr_known_gt_cl_protocol_20260804.md`
  - 在六个支持充分的 VinDr consensus labels 上注入 prevalence-preserving bidirectional known errors，物理隔离 blind noisy cohort 与 private reference，并比较 nested-OOF CL-first、self-confidence、entropy 和 matched random 的真实错误检测能力
- `run_vindr_known_gt_prepare_features.sh` / `run_vindr_known_gt_oof_smoke.sh` / `run_vindr_known_gt_oof_formal_array.sh` / `run_vindr_known_gt_evaluate_formal.sh`
  - 依次完成 outcome-blind frozen XRV feature extraction、四折 smoke、四 seed 正式 blind OOF array 与 dependency-gated private evaluation；正式评估包含同预算 endpoints、10,000 次 random replicates、1,000 次 image/seed bootstrap、exact seed tests 和完整 postflight
- `vindr_noise_direction_sensitivity.py` / `vindr_noise_direction_sensitivity_protocol_20260804.md` / `vindr_noise_direction_sensitivity_validation_20260804_zh.md`
  - 在 clean、10%/20%/30% 和 matched-count balanced/false-positive-only/false-negative-only 场景中，以六 seeds、60 个 outcome-blind OOF runs 检验错误方向差异、CL-first 增量价值和 raw DQS calibration；正式结果确认 CL detection 强烈偏向 `0→1` errors，并发现 DQS 在系统性 `1→0` noise 下反向变化
- `run_vindr_noise_direction_prepare.sh` / `run_vindr_noise_direction_smoke.sh` / `run_vindr_noise_direction_oof_formal_array.sh` / `run_vindr_noise_direction_evaluate_formal.sh`
  - 生成独立 outcome-free blind array manifest，以 6 scheduler tasks × 10 runs 执行完整 60-run OOF grid，并在所有 blind markers 成功后运行 1,000 次 image-cluster bootstrap、六-seed exact/Holm tests、calibration、figures 与 verification
- `vindr_mobilenet_oof.py` / `prepare_vindr_mobilenet_replay.py` / `compare_vindr_mobilenet_frozen_xrv.py`
  - 在不重生成任何 folds 或 corruption 的前提下，把同一 60-run VinDr grid 改用原项目 scratch MobileNetV3-small 训练环境生成 nested OOF evidence，并以六-seed paired FN-recall、clean-reference AUROC、direction gap 和 DQS calibration 判断 frozen-XRV classifier 是否解释 `1→0` 检出弱点
- `run_vindr_mobilenet_direction_smoke.sh` / `run_vindr_mobilenet_direction_formal_array.sh` / `run_vindr_mobilenet_direction_evaluate_formal.sh`
  - 执行 hash-locked GPU smoke、6×10 formal blind array 和 afterok private evaluation；private stage 还会自动生成 MobileNet 对 frozen-XRV 的逐 seed 配对结果与 decision gates
- `vindr_direction_stratified_detector.py` / `vindr_direction_stratified_detector_protocol_20260812.md` / `vindr_direction_stratified_detector_validation_20260812_zh.md`
  - 复用同一60-run MobileNet方向benchmark与冻结XRV features，比较OOF global、SimiFeat global及observed-positive OOF + observed-negative SimiFeat双队列；结果确认两种证据在balanced/FN场景互补，但简单outcome-blind mass allocation不能识别错误方向构成，因此hybrid当前只作为missing-positive coverage guardrail
- `alternatives_to_confident_learning_deep_review_20260812_zh.md`
  - 基于 ARS 对 2015--2026 原始文献进行项目定向深度综述，区分可直接替代 detector、方向性辅助模块、自动纠正与仅改善鲁棒训练的方法；结合 VinDr/Med-PaLM/MIMIC 结果，优先建议 PU-OOF、外部胸片视觉证据和完整 SimiFeat-R/FINE 的同预算 known-GT benchmark
- `vindr_detector_benchmark_v2.py` / `vindr_detector_benchmark_v2_protocol_20260812.md`
  - 在六个全新 corruption/training seeds 上，以相同 entry-review budget 比较 OOF/CL、Active Label Cleaning、FINE-GMM 和独立 frozen XRV finding evidence；评分阶段与 injected-error private reference 物理分离，主终点为 matched-budget true-error recall，并单独报告 `1->0` missing-positive guardrail
- `run_vindr_detector_v2_smoke.sh` / `run_vindr_detector_v2_prepare.sh` / `run_vindr_detector_v2_oof_array.sh` / `run_vindr_detector_v2_evaluate.sh` / `submit_vindr_detector_v2_chain.sh`
  - 组成 smoke-gated Slurm 链：先验证真实 XRV 特征与一条旧 run 的完整评分路径，再准备六个新 seeds、并行生成 60 个 MobileNet OOF runs，最后盲化评分、揭示真值并执行 paired exact/Holm evaluation。正式结果显示 external-XRV 在全部 regime/rate 和六个 seeds 上均提高 matched-budget recall；FINE 只在 missing-positive 检出上形成补充，ALC 未改善该 operational endpoint
- `vindr_xrv_oof_comparison.py` / `vindr_xrv_oof_comparison_protocol_20260813.md`
  - 在 detector-v2 的相同六个新 seeds 和 60 个 blind cohorts 上比较 MobileNet OOF/CL、官方 `features2()` XRV OOF/CL、Direct XRV，以及预先固定的 50/50 percentile-rank fusion；主终点锁定为 balanced 20% matched-budget recall，private reference 只在四组分数写入并哈希后读取
- `run_vindr_xrv_oof_comparison_smoke.sh` / `run_vindr_xrv_oof_comparison_array.sh` / `run_vindr_xrv_oof_comparison_evaluate.sh`
  - 以一个完整 smoke、六个最多三并发的 CPU shard 和 dependency-gated private evaluation 执行上述比较；复用既有 MobileNet 和 Direct-XRV 分数，只新增 60 个轻量 XRV linear-head OOF runs。正式主终点中 XRV OOF/CL recall 为 `0.5493`，高于 fusion `0.5273`、Direct XRV `0.4319` 和 MobileNet OOF/CL `0.2142`，六个 seeds 方向一致
- `medpalm_cl_iterative_oracle_benchmark.py` / `run_medpalm_cl_iterative_oracle_benchmark.sh`
  - 在Med-PaLM 498-entry ground-truth benchmark上运行selected-only oracle correction loops：每轮对完整official test pool重算subject-grouped OOF与CL，只选择从未review的固定100-entry budget，并比较dynamic CL、Loop1 frozen ranking和random review的cumulative issue recall与discovery-curve area
- `medpalm_cl_iterative_oracle_protocol_20260803.md`
  - 锁定single-seed cost-controlled dynamic-OOF pilot、100/100/100/100/98 review checkpoints、oracle action隔离、subject leakage防护、frozen/random对照和single-trajectory解释边界
- `vindr_iterative_oracle_cleaning.py` / `vindr_iterative_oracle_cleaning_protocol_20260805.md`
  - 从VinDr exact 20% symmetric-noise anchor出发，每轮只审阅当前未见过的CL hard issues top20%，在blind selection冻结后用consensus GT执行selected-only perfect correction，再重新训练四折MobileNet OOF；以六seeds×八loops比较dynamic CL、frozen Loop0 ranking和matched random，并同时跟踪known quality、error recall、direction residuals与raw DQS
- `run_vindr_iterative_oracle_smoke.sh` / `run_vindr_iterative_oracle_formal_array.sh` / `run_vindr_iterative_oracle_aggregate.sh`
  - 用afterok链把one-loop smoke、六-seed GPU array和private aggregate分离；正式聚合执行matched-budget AUDC、六-seed exact sign-flip tests、Holm correction及严格marker/hash/table/figure postflight
- `vindr_dqs_calibration_transfer.py` / `run_vindr_dqs_calibration_transfer.sh`
  - 在八个既有 VinDr hard-r30 seeds 中，用训练完全隔离的 600-image sentinel cohort 校准 fold-specific self-confidence thresholds，再冻结并迁移到 2,400-image action cohort 的既有 Loop 0--5 OOF states；GPU 阶段仅重放 Round 0，并以 action OOF、sentinel mean、training history 和 fold support 的 byte-exact replay 作为校准前门控
- `vindr_iterative_oracle_replication_protocol_20260806.md` / `run_vindr_iterative_replication_prepare.sh` / `run_vindr_iterative_replication_loop0_array.sh` / `run_vindr_iterative_replication_formal_array.sh` / `run_vindr_iterative_replication_aggregate.sh`
  - 在看到原六-seed结果后固定新增两个未使用seeds `509/701`，按相同20% noise、MobileNet Loop0和八轮dynamic oracle-cleaning配置执行；两seed必须完整纳入、不得按新p值继续追加，最终同时保留原六-seed confirmatory结果与八-seed post-result descriptive extension
- `reflacx_phase3_cl_pilot.py`
  - 在 REFLACX Phase 3 严格映射的六类 mini cohort 上运行 subject-grouped 4-fold OOF 与 confident learning；运行阶段只读取原始 U-Ones labels，隐藏 REFLACX reference，并在事后评价真实 disagreement detection、top-20 enrichment 和 DQS 对 reference quality 的对应关系
- `run_reflacx_phase3_cl_pilot.sh`
  - 用单个 Slurm GPU 作业串行执行 blinded cohort preparation、OOF/CL evidence generation 和 private-reference evaluation；输出 subject bootstrap 区间、per-label 结果、gate decision 和 figures
- `reflacx_report_llm_triangulation.py` / `run_reflacx_report_llm_triangulation.sh`
  - 对同一3172个REFLACX-compatible entries执行独立report-only LLM review；API看不到CheXpert、REFLACX或CL evidence，事后再统计三方一致模式、keep/relabel/mask、per-label与CL strata结果及subject-cluster区间
- `reflacx_report_llm_triangulation_protocol_20260801.md`
  - 在API outcome前锁定三方source-triangulation目标、物理盲化边界、certainty scopes、解释规则和`gpt-5.4`执行参数
- `build_evaluation_followup_quality_figures.py`
  - 从五 seed 结果生成共用 y-axis、显式 SD 与 individual-seed traces 的 entry DQS 和 strict sample issue-free figures
- `run_mobilenetv3_own_top20_llm_refine_seed.sh`
  - 运行单个 seed 的 own-OOF/own-top20 MobileNet LLM refinement；review/action 与 detection/training 统一使用 U-Ones binary target，并支持 frozen-snapshot、stage-level 与 entry-key resume checks
- `run_mobilenetv3_full_issue_pool_no_repeat_seed13.sh`
  - 锁定 seed13 的八轮 GPT-5.4 full-issue-pool pilot，复用同设置 baseline OOF，每轮只 review 尚无成功历史结果的当前 suspicious entries，并要求本轮 API errors 清零后才训练；压力测试后使用三进程分片 reviewer，把每轮总并发上限设为 10,000，并在 retry pass 中只补失败 entries
- `run_entry_level_llm_review_sharded.py`
  - 把单轮 pending entries 锁定为最多三个互斥 shard，在同一 Slurm 作业内并行调用原 entry reviewer；按 pass 保存独立 checkpoint，原子合并成功/错误 rows，并保证已成功 entry key 不会再次请求
- `run_mimic_gpt54_concurrency_benchmark.sh`
  - 在互不重叠的真实 pending entries 上比较并发 `10/20/40/80/120` 的首次请求吞吐和错误率，并把成功 responses 原子合并回正式 checkpoint
- `prepare_own_top20_binary_v5.py` / `run_prepare_own_top20_binary_v5.sh`
  - 重新生成五 seed 的完整 binary-entry expansion，并只导入 key、target、report、OOF probability 和 source identity 全等的 raw `0/1` pilot responses；不复用 action 或 outcome
- `freeze_own_top20_binary_v5_protocol.sh`
  - 在 corrected v5 actions/outcomes 前冻结执行代码、protocol 和 SHA-256 manifest
- `evaluate_own_top20_refinement_endpoints.py`
  - 对 own-top20 五 seed Loop5/8 计算预锁定 contrasts、exact/paired/Holm p-values、hierarchical seed+study CI 与 held-out proper scores
- `evaluate_own_top20_refinement_quality.py`
  - 计算 own-top20 frozen/dynamic OOF DQS、coverage、strict sample issue-free rate 和 LLM-action support
- `run_own_top20_followup_evaluation.sh`
  - 在五个 seed 的 Loop8 完成后串行运行 performance 与 quality final evaluators
- `evaluate_own_top20_refinement_endpoints_interim.py` / `evaluate_own_top20_refinement_quality_interim.py`
  - 对明确指定的 seed 子集运行独立 interim analysis；输出强制标注为 descriptive，不能替代正式五-seed inference
- `run_own_top20_interim_3seed_evaluation.sh`
  - 在 seeds `7/13/42` 的 Loop5/8 完成后使用 Slurm shard 自动生成 interim performance、quality、uncertainty 和 figures
- `run_own_top20_catchall_finalize.sh`
  - 依赖五个主作业结束后只续跑缺失的 Loop8 seed，随后自动执行预锁定 final evaluation、重建 PPTX 并验证必要输出
- `build_evaluation_followup_deck_20260713.py`
  - 从 synthetic、同尺度 real-quality figures、own-top20 作业状态和预锁定 final CSV 自动生成含 speaker notes 的 follow-up PPTX；结果未完成时统计页只显示 pending
- `next_meeting_slides_outline_中文.md` / `build_next_meeting_slides_20260730.py`
  - 以八页 canonical 文字版 PPT 为唯一内容源，生成含真实 AUROC/DQS 图和逐页 speaker notes 的 meeting deck；Page3/7 在结果未完成时保持轻量占位

## 🚀 使用方法

### 1. TorchXRayVision噪声分析

使用预训练的DenseNet121在GT数据上推理并分析噪声：

```bash
conda activate d:\workspace\MRes\.conda
python NoiseRate/validate_torchxray.py --config NoiseRate/config/unified_config.yaml
```

**输出**：
- `results/perclass_noise_rates_*.csv` - Per-class GT和Pseudo噪声率对比
  - 列: Pathology, GT_Positive, GT_Negative, GT_Issues, GT_Noise_Rate, Pseudo_Issues, Pseudo_Noise_Rate, Diff
- `results/noise_analysis_*.npz` - 详细结果（predictions和per-class issues）
- `logs/noise_validation_*.log` - 运行日志

---

### 2. K-Fold自定义模型训练和验证

训练轻量级CNN (391K参数) 在GT和Pseudo标签上：

```bash
conda activate d:\workspace\MRes\.conda
python NoiseRate/train_validate_kfold.py --config NoiseRate/config/unified_config.yaml
```

**流程**：
1. 使用GT标签训练5-fold模型
2. 生成GT的OOF predictions
3. 计算GT模型的AUROC和噪声率
4. 使用Pseudo标签训练5-fold模型
5. 生成Pseudo的OOF predictions
6. 计算Pseudo模型的AUROC和噪声率

**输出**：
- `kfold_results/oof_predictions.npy` - GT OOF predictions
- `kfold_results/cnn_gt_results_*.csv` - GT模型结果（AUROC + GT噪声率）
  - 列: Pathology, AUROC, GT_Noise_Rate
- `kfold_results/models/fold_*_gt_model.pth` - GT模型权重
- `kfold_results_pseudo/oof_predictions_pseudo.npy` - Pseudo OOF predictions
- `kfold_results_pseudo/cnn_pseudo_results_*.csv` - Pseudo模型结果（AUROC + Pseudo噪声率）
  - 列: Pathology, AUROC, Pseudo_Noise_Rate
- `kfold_results_pseudo/models/fold_*_pseudo_model.pth` - Pseudo模型权重
- 各自的log文件包含详细训练和验证信息

---

### 3. 正式 MIMIC-CXR-JPG + GT Test 实验

这套实验使用：

- 训练图像：`MedSoul/datasets/mimic-cxr-jpg-224/`
- 训练标签：`mimic-cxr-2.0.0-chexpert.csv`
- split：`mimic-cxr-2.0.0-split.csv.gz`
- metadata：`mimic-cxr-2.0.0-metadata.csv.gz`
- GT test：`mimic-cxr-2.1.0-test-set-labeled.csv`

#### baseline（推荐起点）

```bash
cd MRes
sbatch NoiseRate/cxr_real_experiment/run_baseline_full_train_test_cleanlab.sh
```

当前 baseline 配置：

- `DenseNet121 pretrained`
- `epochs=100`
- `val_fraction=0.1`
- `early_stopping_patience=10`
- `recover_best_weights=true`

输出目录示例：

- `NoiseRate/cxr_real_experiment/results_baseline_es/slurm_<jobid>/`

主要输出：

- `baseline_run_summary.csv`
- `test_image_auroc_summary.csv`
- `test_study_auroc_summary.csv`
- `train_cleanlab_sample_details.csv`
- `train_cleanlab_sample_issues_only.csv`
- `train_cleanlab_entry_issues_only.csv`

#### sample-level cleaned

删除 `train_cleanlab_sample_issues_only.csv` 中的 issue samples 后重训：

```bash
cd MRes
sbatch NoiseRate/cxr_real_experiment/run_cleaned_full_train_test_cleanlab.sh
```

输出目录示例：

- `NoiseRate/cxr_real_experiment/results_cleaned_es/slurm_<jobid>/`

#### entry-level cleaned

不删样本，只把 `train_cleanlab_entry_issues_only.csv` 中的 issue label entries 从监督里 mask 掉：

```bash
cd MRes
sbatch NoiseRate/cxr_real_experiment/run_entry_cleaned_full_train_test_cleanlab.sh
```

输出目录示例：

- `NoiseRate/cxr_real_experiment/results_entry_cleaned_es/slurm_<jobid>/`

#### AP/PA 单作业串行版本

如果想只用 `AP/PA` 图像，并且在同一个作业里顺序跑完三阶段：

- `baseline`
- `sample-level cleaned`
- `entry-level cleaned`

可以直接提交：

```bash
cd MRes
sbatch NoiseRate/cxr_real_experiment/run_appa_pipeline_full_train.sh
```

这条脚本会自动：

1. 读取 `mimic-cxr-2.0.0-metadata.csv.gz`
2. 只保留 `ViewPosition in {AP, PA}`
3. 跑第一阶段 baseline
4. 用 baseline 产出的 `train_cleanlab_sample_issues_only.csv` 跑第二阶段 sample-level cleaned
5. 用 baseline 产出的 `train_cleanlab_entry_issues_only.csv` 跑第三阶段 entry-level cleaned
6. 最后在结果根目录输出一份 `appa_pipeline_compare.csv`

输出目录示例：

- `NoiseRate/cxr_real_experiment/results_appa_pipeline/slurm_<jobid>/`
- 其中包含：
  - `01_baseline/`
  - `02_sample_cleaned/`
  - `03_entry_cleaned/`
  - `appa_pipeline_compare.csv`

#### XRV sample20 iterative loop

当前正式 iterative loop 使用 `sample-level top20%` 作为每轮清洗候选，同时跑两条分支：

- `remove_only`: 每轮 K4 OOF 后累计删除 top20% suspicious samples，再 full-train/eval
- `llm_refine`: 每轮 K4 OOF 后把 top20% suspicious samples 展开成 entry rows，经过 LLM 审核后累计 relabel/mask，再 full-train/eval

提交入口：

```bash
cd /vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment
./submit_xrv_iterative_sample20_dual_loop.sh
```

默认配置：

- `LOOP_COUNT=5`
- `TOP_FRACTION=0.20`
- `SEED=13`

运行方式：

- 每次提交只起 2 个 job：`remove_only` 和 `llm_refine`
- 每个 job 在 3 天 A30 时限内顺序跑尽可能多的 loop
- OOF、selection、LLM review、action tables 和 full train/eval 都有独立完成检查；LLM 以 entry key 跳过已成功 rows
- 下次用同一个 `OUTPUT_ROOT_OVERRIDE` 重新提交，会从第一个未完成 loop/stage 继续；`loop_metrics.csv` 按 branch/loop upsert，不重复追加

输出目录：

- `/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_xrv_iterative_sample20_dual_loop/<timestamp>/remove_only/loop_metrics.csv`
- `/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_xrv_iterative_sample20_dual_loop/<timestamp>/llm_refine/loop_metrics.csv`

#### MobileNet 五 seed 清理方法统一评估

当同一结果根目录下已经有五 seed 的 baseline、simple-remove Loops 1-5 和 fixed-LLM-refine Loops 1-5 时，提交：

```bash
cd /vol/gpudata/yz3522-llmtest/MResProject/NoiseRate
sbatch cxr_real_experiment/run_mobilenet_cleaning_methods_evaluation.sh
```

该评估会对所有 seed 使用相同的 test-study bootstrap indices，并另外报告 seed+study hierarchical uncertainty；默认同时输出完整 12-label 和 minority support >= 5 的敏感性结果。

补充计算 DQS、noise-rate denominator、coverage、selection stability 与 AP/Brier/NLL：

```bash
cd /vol/gpudata/yz3522-llmtest/MResProject/NoiseRate
sbatch cxr_real_experiment/run_mobilenet_data_quality_5seed.sh
```

其中 `entry_issue_density_all_slots` 是兼容旧结果的 `issue_count / (samples x 12 labels)`；真正以有效标签条目为分母的指标是 `entry_issue_rate_valid_entries`。`coverage_adjusted_dqs` 是项目自定义的分母诊断，不是 Cleanlab 官方指标。

#### Synthetic DQS validation 与 own-top20 refinement

轻量 known-truth DQS validation：

```bash
cd /vol/gpudata/yz3522-llmtest/MResProject/NoiseRate
sbatch cxr_real_experiment/run_synthetic_dqs_validation.sh
```

先准备并冻结 corrected binary-label v5 root。准备阶段要求每个 selected sample 至少有一个 flagged binary entry，并把 raw `-1` 按 U-Ones 映射为 binary positive：

```bash
ROOT=/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_own_top20_binary_v5_llm_refine_5seed/<timestamp>
sbatch cxr_real_experiment/run_prepare_own_top20_binary_v5.sh "$ROOT"
bash cxr_real_experiment/freeze_own_top20_binary_v5_protocol.sh "$ROOT"
```

准备作业成功且 snapshot 冻结后，五个 seed 分别提交，Slurm 会按用户 GPU 上限排队：

```bash
sbatch --job-name=mb-own-s7 cxr_real_experiment/run_mobilenetv3_own_top20_llm_refine_seed.sh 7 "$ROOT"
```

该 runner 要求 `OPENAI_MODEL=gpt-5.4`，每个 seed 使用自己的 OOF probabilities 和 top20% sample selection。Loop1 只复用 matched simple-removal run 中配置完全相同的 baseline OOF/selection；Loops2-8 会在累计 relabel/mask 后重算 seed-specific OOF。旧 `20260713_184706` root 因 raw `-1` review omission 被保留为 invalid pilot，不能用于正式 action、metric 或 outcome inference。

论文统一 downstream 评价使用独立的 MobileNetV3-small：最多 100 epochs、early-stopping patience 10，并恢复最低 validation loss 的 checkpoint。已经完成 refinement 的数据状态不需要重新执行 OOF 或 LLM review，可直接补跑：

```bash
sbatch cxr_real_experiment/run_mobilenetv3_unified_downstream_seed.sh \
  "$ROOT" <seed> 0.20 1 8 1
```

新结果写入各轮的 `downstream_mobilenet100_es10_best/`，不覆盖旧的固定 50-epoch `train_eval/`。多个 own-top20 seeds 可使用 `run_own_top20_unified_downstream_group.sh` 分组串行补跑。MIMIC-CXR full-issue-pool 的后续轮次也使用同一 downstream 配置；已经完成的旧轮次由上述 downstream-only runner 补齐。

五个 seed 都完成 Loop8 后运行预锁定 final analysis：

```bash
sbatch cxr_real_experiment/run_own_top20_followup_evaluation.sh "$ROOT"
```

如需把超时续跑、final analysis 和 slides 重建串成一个依赖后的保险作业：

```bash
sbatch --dependency=afterany:<job7>:<job13>:<job42>:<job97>:<job123> \
  cxr_real_experiment/run_own_top20_catchall_finalize.sh "$ROOT"
```

该 finalizer 会同时检查每个 seed 的 Loop8 stage markers、action tables、train outputs 和 `loop_metrics` row；事务完整的 seed 不会重复运行，任一项缺失才进入原 runner 的 stage-level/entry-key resume。

如 meeting deadline 早于五-seed completion，可在不修改正式 evaluator 的前提下提交明确标注的三-seed interim analysis：

```bash
sbatch --dependency=afterok:<job7>:<job13>:<job42> \
  cxr_real_experiment/run_own_top20_interim_3seed_evaluation.sh

# Loop 5 已完成时可立即生成 deadline backup：
sbatch cxr_real_experiment/run_own_top20_interim_3seed_evaluation.sh checkpoint
```

该分析固定使用 seeds `7/13/42`；full mode 输出到 `evaluation_interim_3seed_loop5_loop8_20260717/`，checkpoint mode 输出到 `evaluation_interim_3seed_loop5_only_20260716/`。三 seed 的双侧 exact sign-flip p-value 最小只能达到 `0.25`，因此汇报必须以 paired effect、seed direction 和明确标为不稳定的 interval 为主，并说明正式五-seed结果仍待完成。

实验协议、Loop5/8 endpoints、统计 contrasts、post-outcome semantic correction 和解释边界记录在 `cxr_real_experiment/evaluation_followup_20260713/own_top20_refinement_protocol.md`。旧 pilot 的 v1-v4 snapshots 保留历史审计；corrected root 的 `protocol_snapshot_v5_binary_label_locked/` 是正式执行版本，冻结 binary-label exporter/reviewer/action pipeline、evaluation、deck builder 和 SHA-256 checksums。v5 明确不是相对于 invalid-pilot outcome 的 outcome-blind preregistration，但 corrected actions/outcomes 在 snapshot 后才生成。

旧 sample diagnostic 曾把 zero-valid rows 自动算成 issue-free。修正版只使用至少有一个 valid target 的 evaluable samples，并用单独 Slurm job 重算旧五-seed anchors、figures 和 deck：

```bash
sbatch cxr_real_experiment/run_corrected_sample_quality_followup.sh
```

生成当前 follow-up slides；五个 Loop8 和 final analysis 完成后运行同一命令即可自动填入预锁定统计结果：

```bash
/vol/gpudata/yz3522-llmtest/venv/bin/python \
  cxr_real_experiment/build_evaluation_followup_deck_20260713.py
```

输出为 `cxr_real_experiment/evaluation_followup_deck_20260713/evaluation_followup_20260713.pptx`。主讲页包含 synthetic validation、entry/sample denominator audit、own-top20 protocol 与 hierarchical uncertainty；sample anchors 只从带 zero-valid policy 的 metadata 读取，上次未展示的旧 hierarchical 结果和 proper scoring rules 明确放在 backup。

---

## 📊 主要功能和噪声分析逻辑

### **重要**: 噪声分析逻辑说明

不同脚本的噪声分析目的不同：

#### 1. **validate_torchxray.py** - 预训练模型标签质量对比
- 使用TorchXRayVision预训练模型的predictions
- **同时分析GT noise和Pseudo noise**
- **目的**: 对比GT和Pseudo标签质量（公平比较，因为模型未用它们训练）

#### 2. **train_validate_kfold.py** - 自训练模型性能评估

**CNN-GT模型**:
- 使用GT标签训练
- **只分析GT noise**
- **目的**: 评估模型在自己训练标签上的噪声检测能力

**CNN-Pseudo模型**:
- 使用Pseudo标签训练
- **只分析Pseudo noise**
- **目的**: 评估模型在自己训练标签上的噪声检测能力

---

## 🔧 工具模块


- `run_cleanlab_multilabel_analysis()` - 多标签cleanlab噪声检测
- `analyze_per_class_noise()` - 每个类别的噪声率分析
- `compare_noise_results()` - GT vs Pseudo对比

### AUROC Metrics (`utils/auroc_metrics.py`)

- `calculate_per_class_auroc()` - 计算per-class AUROC
- `log_auroc_results()` - 格式化输出AUROC结果

### Label Utilities (`utils/label_utils.py`)

- `filter_by_view_position()` - 过滤影像视角
- `handle_uncertain_labels()` - 处理不确定标签
- `convert_to_multilabel_format()` - 转换为cleanlab格式

---

## 🧠 正式实验标签口径

当前 `cxr_real_experiment/` 这一套正式实验，训练和 cleanlab 统一使用二值口径：

- `1 -> 1`
- `-1 -> 1`
- `0 -> 0`
- `NaN/null -> mask`

也就是：

- `positive + uncertain -> positive`
- `negative -> negative`
- `missing -> ignore`

GT test 也使用同样的二值化口径。

额外说明：

- GT test 文件里使用 `Airspace Opacity`
- 代码里会映射到 `Lung Opacity`
- 当前正式实验支持两种口径：
  - 全视角：直接使用能和 `split.csv` 对上的全部图像
  - `AP/PA`：额外读取 `mimic-cxr-2.0.0-metadata.csv.gz` 并按 `ViewPosition` 过滤
- `AP/PA` 过滤已在正式流程中接通，训练与 GT test 两侧都会一致应用

---

## ⚙️ 配置

### 统一配置文件 (`config/unified_config.yaml`)

不再使用分散的配置文件，而是使用统一的 `config/unified_config.yaml` 管理所有设定。

主要包含三个部分：
1. **`common`**: 公用设置（数据路径、预处理等）
2. **`torchxray`**: `validate_torchxray.py` 专用设置
3. **`kfold`**: `train_validate_kfold.py` 专用设置（包含 `common`, `gt`, `pseudo` 子部分）

```yaml
common:
  data:
    gt_metadata_path: "..."
    gt_image_base_path: "..."
    allowed_view_positions: ['AP', 'PA']

torchxray:
  model:
    name: "densenet121-res224-all"
    
kfold:
  common:
    kfold_settings:
      n_splits: 5
    model:
      name: "LightweightCNN"
  gt:
    data:
      use_pseudo_labels: false
  pseudo:
    data:
      use_pseudo_labels: true
```

---

## 🧪 运行测试

```bash
cd NoiseRate/tests
python -m pytest test_label_utils.py
python -m pytest test_xrv_utils.py
```

---

## 📈 实验结果说明

### 噪声率对比方法

**标签质量对比** (validate_torchxray.py):
- 使用预训练模型predictions
- 同时计算GT noise和Pseudo noise
- **目的**: 评估两种标签本身的质量差异

**模型性能评估** (train_validate_kfold.py):
- CNN-GT: 用GT训练 → 只计算GT noise
- CNN-Pseudo: 用Pseudo训练 → 只计算Pseudo noise
- **目的**: 评估模型在各自训练标签上的噪声检测性能

### 典型结果

#### TorchXRayVision (预训练模型)
- **平均AUROC**: ~0.65
- **GT噪声率**: ~8% (per-class)
- **Pseudo噪声率**: ~9% (per-class)
- **结论**: Pseudo标签噪声略高于GT

#### LightweightCNN (从头训练)
- **CNN-GT平均AUROC**: ~0.59
- **CNN-GT的GT噪声率**: ~19% (per-class)
- **CNN-Pseudo平均AUROC**: ~0.60
- **CNN-Pseudo的Pseudo噪声率**: ~19% (per-class)
- **结论**: 自训练模型检测到更多噪声，无数据泄露

### 正式 MIMIC-CXR-JPG + GT Test（2026-04 实验）

GT test 使用：

- 原始 labeled CSV：`687 studies`
- 实际可评测：`685 studies / 1029 images`

#### 全视角 baseline early-stopping

目录：

- `cxr_real_experiment/results_baseline_es/slurm_234377/`

结果：

- `best_epoch = 11`
- `best_val_loss = 0.2901`
- `test_image_macro_auroc_binary = 0.7812`
- `test_study_macro_auroc_binary = 0.7962`
- `train_cleanlab_sample_issue_rate = 0.1506`
- `train_cleanlab_entry_issue_rate = 0.0124`

#### 全视角 sample-level cleaned early-stopping

目录：

- `cxr_real_experiment/results_cleaned_es/slurm_234888/`

结果：

- 删除 `55,491` 个 issue samples
- `best_epoch = 9`
- `best_val_loss = 0.1536`
- `test_image_macro_auroc_binary = 0.7826`
- `test_study_macro_auroc_binary = 0.7891`

结论：

- image-level 略高于 baseline
- study-level 低于 baseline

#### 全视角 entry-level cleaned early-stopping

目录：

- `cxr_real_experiment/results_entry_cleaned_es/slurm_235417/`

结果：

- 保留全部 `368,562` 个样本
- mask 掉 `63,811` 个 issue label entries
- `best_epoch = 7`
- `best_val_loss = 0.1441`
- `test_image_macro_auroc_binary = 0.7799`
- `test_study_macro_auroc_binary = 0.7791`

结论：

- test 表现低于 baseline
- 当前这版 entry-level 屏蔽没有带来收益

#### 全视角当前结论

在目前这套设置下：

1. `baseline_es` 是 study-level 指标最好的方案
2. `sample-level cleaned` 略微提升 image-level，但降低 study-level
3. `entry-level cleaned` 进一步降低了 study-level

因此下一步更值得尝试的是：

- 不做全量清洗
- 改成只清洗 top-k / top-percentage 最可疑样本
- 或者重新评估 `-1 (uncertain)` 的处理策略

#### AP/PA 版本状态

`AP/PA` 版本已经通过正式小烟测，过滤规模如下：

- train split：`368,960 -> 237,972` 张图像
- test split 候选图像：`5,176 -> 3,414`
- 与 GT merge 后可评测：`676 images / 605 studies`

目前 `AP/PA` 的单作业串行脚本已经提供，结果会写到：

- `cxr_real_experiment/results_appa_pipeline/slurm_<jobid>/`

建议读取：

- `01_baseline/baseline_run_summary.csv`
- `02_sample_cleaned/baseline_run_summary.csv`
- `03_entry_cleaned/baseline_run_summary.csv`
- `appa_pipeline_compare.csv`

#### AP/PA + 自己 DenseNet（14类）

目录：

- `cxr_real_experiment/results_appa_pipeline/slurm_236304/`

三阶段结果：

| 方案 | Study Macro AUROC | Image Macro AUROC |
|---|---:|---:|
| baseline | 0.7701 | 0.7637 |
| full sample-cleaned | 0.7797 | 0.7767 |
| full entry-cleaned | 0.7528 | 0.7489 |

结论：

- `sample-level cleaned` 有提升
- `entry-level cleaned` 明显不如 baseline

#### AP/PA + XRV encoder + our head（12类）

目录：

- `cxr_real_experiment/results_appa_xrv12_linearhead_pipeline/slurm_236337/`

三阶段结果：

| 方案 | Study Macro AUROC | Image Macro AUROC |
|---|---:|---:|
| baseline | 0.8158 | 0.8095 |
| full sample-cleaned | 0.8127 | 0.8043 |
| full entry-cleaned | 0.8114 | 0.8028 |

结论：

- baseline 是这条 12类线里最好的全量方案
- 全量 sample / entry cleaning 都略微降低了 test 指标

#### AP/PA + XRV direct decoder（12类）

目录：

- `cxr_real_experiment/results_appa_xrv12_pipeline/slurm_236482/`

三阶段结果：

| 方案 | Study Macro AUROC | Image Macro AUROC |
|---|---:|---:|
| baseline | 0.6658 | 0.6638 |
| full sample-cleaned | 0.6370 | 0.6365 |
| full entry-cleaned | 0.6340 | 0.6341 |

结论：

- `XRV direct decoder` 明显弱于 `XRV encoder + our head`
- 因此后续实验默认优先保留 `linear-head` 路线

#### AP/PA + sample-level top-k cleaning

说明：

- 这里的 top-k 指从 baseline 的 `train_cleanlab_sample_issues_only.csv`
- 只选最可疑的前 `5% / 10% / 20%` issue samples 删除后重训

##### 自己 DenseNet（14类）

目录：

- `cxr_real_experiment/results_appa_densenet_topk/slurm_236927/`

结果：

| 方案 | Study Macro AUROC | Image Macro AUROC |
|---|---:|---:|
| top05 sample | 0.7677 | 0.7574 |
| top10 sample | 0.8226 | 0.8123 |
| top20 sample | 0.8174 | 0.8117 |

结论：

- `top10 sample` 是当前这条线的最佳配置
- 它优于 baseline、full sample-cleaned 和 full entry-cleaned

##### XRV encoder + our head（12类）

目录：

- `cxr_real_experiment/results_appa_xrv12_linearhead_topk/slurm_236928/`

结果：

| 方案 | Study Macro AUROC | Image Macro AUROC |
|---|---:|---:|
| top05 sample | 0.8143 | 0.8073 |
| top10 sample | 0.8384 | 0.8272 |
| top20 sample | 0.8386 | 0.8342 |

结论：

- `top20 sample` 略优于 `top10 sample`
- 两者都明显优于 baseline 和全量 cleaned

#### AP/PA + entry-level top-k cleaning

说明：

- 这里的 top-k 指从 baseline 的 `train_cleanlab_entry_issues_only.csv`
- 只选最可疑的前 `5% / 10% / 20%` issue label entries 做 mask

##### 自己 DenseNet（14类）

目录：

- `cxr_real_experiment/results_appa_densenet_entry_topk/slurm_237240/`

结果：

| 方案 | Study Macro AUROC | Image Macro AUROC |
|---|---:|---:|
| top05 entry | 0.8162 | 0.8134 |
| top10 entry | 0.8141 | 0.8059 |
| top20 entry | 0.7952 | 0.7892 |

结论：

- `top05 entry` 最好
- `entry-level top-k` 明显优于全量 entry-clean
- 但仍略逊于 `top10 sample`

##### XRV encoder + our head（12类）

目录：

- `cxr_real_experiment/results_appa_xrv12_linearhead_entry_topk/slurm_237241/`

结果：

| 方案 | Study Macro AUROC | Image Macro AUROC |
|---|---:|---:|
| top05 entry | 0.8243 | 0.8213 |
| top10 entry | 0.7848 | 0.7753 |
| top20 entry | 0.7879 | 0.7820 |

结论：

- `top05 entry` 略高于 baseline
- 但明显不如 `top10/top20 sample`

#### 当前阶段总结合

当前最值得保留的两条配置是：

1. `AP/PA + 自己 DenseNet + 14类 + top10 sample cleaning`
2. `AP/PA + XRV encoder + our head + 12类 + top20 sample cleaning`

整体结论：

- `cleanlab-based cleaning` 是有效的
- 但不是“全量 cleaning 越多越好”
- 当前结果最支持的是：`温和的 sample-level top-k cleaning`
- `entry-level top-k` 也比全量 entry-clean 更稳，但整体还是弱于最优的 sample-level top-k

---

## 📝 依赖

- Python 3.8+
- PyTorch
- torchxrayvision
- cleanlab
- scikit-learn
- pandas
- numpy
- PyYAML
- python-pptx 1.0.2（仅用于生成 evaluation slides）

---

## 🔧 开发说明

- 所有脚本使用英文注释和变量名
- 配置和文档使用中文
- 代码风格遵循PEP 8
- 新功能需要添加单元测试
