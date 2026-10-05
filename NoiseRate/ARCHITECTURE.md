# NoiseRate Architecture

## 模块职责

### 数据与模型

- `cxr_real_experiment/cxr_real_oof_cleanlab_smoke.py`：对训练 split 生成 K-fold OOF probabilities 和 sample/entry confident-learning evidence。
- `cxr_real_experiment/cxr_real_full_train_eval_cleanlab_xrv12.py`：应用 sample exclusion、entry relabel/mask，训练 XRV 或 lightweight backbone，并在固定 GT test set 输出 image/study predictions。
- `cxr_real_experiment/select_topk_issue_samples.py`：从 sample issue ranking 中选择固定 top fraction。
- `cxr_real_experiment/export_sample_topk_entries_for_llm.py`：按 U-Ones binary target 把 selected samples 展开为 suspicious disease entries，并硬性验证 100% selected-sample coverage。
- `cxr_real_experiment/run_entry_level_llm_review.py`：把 post-binarization label 作为 decision target，按 entry key 调用 mock/real LLM review，并以 append-only success rows 支持中断续跑。
- `cxr_real_experiment/run_entry_level_llm_review_sharded.py`：把单轮 pending review 锁定为多个互斥本机 process shards，在总并发预算内并行运行基础 reviewer，并以 pass-level manifest/marker 原子合并 successes 与 retryable errors。
- `cxr_real_experiment/build_llm_refinement_tables.py`：按 binary target 把 LLM decisions 转成 relabel/mask tables，同时保留 raw label provenance。

### Iterative orchestration

- `cxr_real_experiment/run_xrv_iterative_sample20_branch.sh`：串行执行 OOF -> selection -> optional LLM -> action tables -> full train/eval -> loop summary；每个 stage 独立检查完成状态。
- `cxr_real_experiment/run_xrv_iterative_sample20_unseen_branch.sh` / `select_topk_unreviewed_issue_samples.py`：在既有 loop transaction 上增加全历史 entry 去重；可选择以成功 `results.csv` 而非所有尝试作为 review history，并在 full-pool 模式保留 API 失败 entry 的重试资格。
- `cxr_real_experiment/summarize_xrv_loop_metrics.py`：按 `(branch, loop_id)` 原子 upsert 每轮性能、issue rate 和累计 action counts。
- `cxr_real_experiment/submit_xrv_iterative_sample20_dual_loop.sh`：提交 XRV remove/refine 双分支。
- `cxr_real_experiment/run_mobilenetv3_own_top20_llm_refine_seed.sh`：锁定单个 MobileNet seed 的 own-top20 Loop1-8 protocol、matched Loop1 OOF reuse 和 API configuration。
- `cxr_real_experiment/run_mobilenetv3_full_issue_pool_no_repeat_seed13.sh`：锁定 seed13、八轮、GPT-5.4 的 MIMIC full-current-issue-pool pilot，复用 matched baseline OOF，使用三 process/总并发上限10,000的 resumable review，并强制每轮 errors 清零后再训练。
- `cxr_real_experiment/run_mimic_gpt54_concurrency_benchmark.sh`：从 active checkpoint 锁定互不重叠的 untouched entries，分档测量 GPT-5.4 并发吞吐/错误率，并将成功结果与 active errors 以 entry key 原子协调后合并。
- `cxr_real_experiment/prepare_own_top20_binary_v5.py`：原子化建立 corrected v5 root、全量 binary expansion 和严格 raw `0/1` response-reuse manifest，不导入 pilot actions/outcomes。
- `cxr_real_experiment/freeze_own_top20_binary_v5_protocol.sh`：冻结 v5 execution/reporting files 与 SHA-256 manifest，供 seed runner 和 finalizer 启动时核验。
- `cxr_real_experiment/test_binary_label_refinement_pipeline.py`：回归测试 raw `-1/0/1` projection、prompt target、flip/mask/keep semantics 和 coverage failure gate。
- `cxr_real_experiment/run_own_top20_catchall_finalize.sh`：在五个主作业结束后串行补齐缺失 seed，并把预锁定 evaluation、deck rebuild 和 required-output verification 组成最终事务。

### Evaluation

- `cxr_real_experiment/evaluate_mobilenet_cleaning_methods_5seed.py`：评估旧 fixed-action baseline/remove/refine Loops1-5，包括 paired 和 hierarchical uncertainty。
- `cxr_real_experiment/evaluate_mobilenet_data_quality_5seed.py`：计算旧五-seed DQS、evaluable-sample coverage、selection stability 和 held-out proper scores；zero-valid rows 不进入 sample issue-free 分母。
- `cxr_real_experiment/validate_synthetic_dqs_adjustment.py`：用已知真值检验 raw/adjusted DQS 的 matching targets 和 evidence-model sensitivity。
- `cxr_real_experiment/build_evaluation_followup_quality_figures.py`：生成同尺度 entry DQS 和 strict sample issue-free diagnostics。
- `cxr_real_experiment/evaluate_own_top20_refinement_endpoints.py`：对 own-top20 Loop5/8 执行预锁定 performance contrasts、p-values 与 hierarchical bootstrap。
- `cxr_real_experiment/evaluate_own_top20_refinement_quality.py`：对 own-top20 actions 计算 frozen post-action、dynamic pre-action 和 sample-level quality diagnostics。
- `cxr_real_experiment/run_own_top20_followup_evaluation.sh`：在五 seed Loop8 完成后统一调用两个 own-top20 evaluators。
- `cxr_real_experiment/evaluate_own_top20_refinement_endpoints_interim.py`：在不修改 frozen five-seed evaluator 的前提下，对显式 seed 子集生成带 interim 标签的 performance effects、seed directions 与 uncertainty。
- `cxr_real_experiment/evaluate_own_top20_refinement_quality_interim.py`：为同一 interim seed 子集生成 model-based DQS、coverage、sample diagnostics 和 action support。
- `cxr_real_experiment/run_own_top20_interim_3seed_evaluation.sh`：验证 seeds `7/13/42` 的 Loop5/8 transactions 后，在 Slurm shard 上调用两个 interim evaluators 并核验输出标签与 seed set。
- `cxr_real_experiment/run_corrected_sample_quality_followup.sh`：在 Slurm shard 上重算修正版旧五-seed sample anchors、figures 与 deck。
- `cxr_real_experiment/medpalm_external_entry_benchmark.py`：把专家裁决的 Med-PaLM disagreement entries 与当前 CheXpert/U-Ones 口径对齐，隔离 blinded API input 与 private reference，并计算 direct-GT reviewer endpoints 和 subject-cluster intervals。
- `cxr_real_experiment/run_medpalm_external_entry_benchmark.sh`：校验 outcome-blind protocol/code hash 后，在无完整 GPU 请求的 Slurm 作业中串行执行 prepare、resumable review、verification 和 evaluation。
- `cxr_real_experiment/medpalm_cl_detection_benchmark.py`：复用四个冻结 baseline 对完整 official test pool 做 held-out inference，物理分离 blind CL scoring 与 private Med-PaLM reference join，并计算 detection gate、subject bootstrap、positive control 和 secondary CL-to-LLM结果。
- `cxr_real_experiment/run_medpalm_cl_detection_benchmark.sh`：以可恢复的 inference -> blind scoring -> private evaluation 三阶段 Slurm transaction 执行 CL detection benchmark，并锁定主脚本、协议及两个依赖模块的 SHA-256。
- `cxr_real_experiment/dual_reference_cl_detection_benchmark.py`：把冻结的四-seed/ensemble CL scores 拆成 reference-blind exact-policy selection 和 private dual-reference evaluation，分别计算 continuous detection、hard-flag enrichment、study-level top-20 policy、matched-random prioritization、subject bootstrap 及 per-label heterogeneity。
- `cxr_real_experiment/run_dual_reference_cl_detection_benchmark.sh`：hash-lock protocol/program/frozen scores，以 `select -> evaluate -> verify` 三阶段 shard transaction 运行 MIMIC-CXR 2.1 与 Med-PaLM 双参考集正式验证。
- `cxr_real_experiment/cl_prioritization_followup_benchmark.py`：在冻结 CL-hard candidate pool 上生成 whole-study/worst-entry/hard-entry-mean/direct-entry 四种 blind rankings，按 study 或 entry workload 匹配预算，并在 private stage 计算双 reference metrics、paired subject bootstrap、matched-random 和预锁定 decision gates。
- `cxr_real_experiment/run_cl_prioritization_followup_benchmark.sh`：锁定 follow-up program/protocol/helper hashes，以 `select -> evaluate -> verify` shard transaction 运行候选优先级策略验证，不训练模型或调用 LLM。
- `cxr_real_experiment/cl_vs_uncertainty_ablation.py`：在双 reference outcomes 隔离下生成同 entry-budget 的 CL hard、self-confidence、per-label percentile、entropy 和 seed-disagreement rankings，并计算 reference metrics、paired subject bootstrap、full-pool random 和锁定的 incremental-CL gate。
- `cxr_real_experiment/run_cl_vs_uncertainty_ablation.sh`：锁定 ablation program/protocol/helper hashes，以 `select -> evaluate -> verify` shard transaction 运行，不训练模型或调用 LLM。
- `cxr_real_experiment/download_vindr_test_parallel.py`：解析 authenticated PhysioNet test listing，以四个 Wget workers 原子续传 VinDr DICOM/labels/metadata，并验证 3,000 个 unique image-label pairs 和官方 SHA-256。
- `cxr_real_experiment/run_download_vindr_test_parallel.sh`：把 mode-600 home credential 移入 node-local private `.netrc` 后立即删除 home copy，在 Slurm shard 上运行长时并发下载并在退出时清除 node-local credential。
- `cxr_real_experiment/preprocess_vindr_dicoms.py`：审计 VinDr DICOM headers，并按 modality -> VOI/window -> MONOCHROME inversion -> center crop -> PNG-224 的确定性路径转换影像、记录逐文件 provenance 与 hash。
- `cxr_real_experiment/run_vindr_preprocess_smoke.sh` / `run_vindr_preprocess_full.sh`：先运行覆盖全部 audited marginal levels 的 20-image smoke 和 contact sheet，再对 3,000 张影像执行全量转换、逐张 reopen/hash 和 source-label ID postflight。
- `cxr_real_experiment/vindr_known_gt_cl_benchmark.py`：在 outcome-isolated VinDr consensus cohort 上执行 controlled error injection、frozen XRV feature extraction、nested four-fold OOF/CL evidence、private known-error evaluation、matched random、bootstrap、exact contrasts 和 verification。
- `cxr_real_experiment/vindr_known_gt_cl_protocol_20260804.md`：在任何正式 OOF outcome 前锁定六标签 scope、双向保 prevalence 的 20% positive-support corruption、四 seeds、比较器、endpoints、decision gates 和解释边界。
- `cxr_real_experiment/run_vindr_known_gt_prepare_features.sh` / `run_vindr_known_gt_oof_smoke.sh` / `run_vindr_known_gt_oof_formal_array.sh` / `run_vindr_known_gt_evaluate_formal.sh`：把 prepare/features、smoke、四-seed blind array 和 dependency-gated private evaluation 拆成 hash-locked Slurm transactions。
- `cxr_real_experiment/vindr_noise_direction_sensitivity.py`：在复用冻结 VinDr features 与 OOF engine 的前提下生成 clean/多噪声率/三方向机制的 matched-error grid，并负责 private detection、direction contrasts、DQS calibration、bootstrap、figures 与 verification。
- `cxr_real_experiment/run_vindr_noise_direction_prepare.sh` / `run_vindr_noise_direction_smoke.sh` / `run_vindr_noise_direction_oof_formal_array.sh` / `run_vindr_noise_direction_evaluate_formal.sh`：把 sensitivity prepare、高风险 smoke、60-run outcome-blind OOF 和 afterok private evaluation 拆成 hash-locked Slurm transactions；正式 array 只读取独立 blind manifest。
- `cxr_real_experiment/prepare_vindr_mobilenet_replay.py`：复制锁定 manifests，并为 60 个 verified prepared inputs 建立只读链接，使 MobileNet 与 frozen-XRV 使用逐 entry 相同的 folds 和 corruption。
- `cxr_real_experiment/vindr_mobilenet_oof.py`：以一通道、六输出 scratch MobileNetV3-small、原项目 XRV 预处理和 unweighted BCE 训练 nested four-fold OOF probabilities，再生成与 frozen-XRV benchmark 同 schema 的 CL evidence。
- `cxr_real_experiment/compare_vindr_mobilenet_frozen_xrv.py`：对两种 OOF model 的 60-run private evaluation 做六-seed paired analysis，并执行预锁定 FN-recall improvement、clean-reference AUROC、directionality 和 calibration gates。
- `cxr_real_experiment/run_vindr_mobilenet_direction_smoke.sh` / `run_vindr_mobilenet_direction_formal_array.sh` / `run_vindr_mobilenet_direction_evaluate_formal.sh`：负责 MobileNet smoke、60-run blind GPU array、afterok private evaluation 和 paired comparison 的 hash-locked Slurm transactions。
- `cxr_real_experiment/vindr_direction_stratified_detector.py`：在已冻结的60-run MobileNet direction grid上先盲化生成OOF/ALC/SimiFeat scores，再以固定及outcome-blind mass quotas组合observed-positive OOF队列与observed-negative SimiFeat队列，输出方向recall trade-off、paired contrasts和完整性验证。
- `cxr_real_experiment/vindr_detector_benchmark_v2.py`：从 outcome-blind OOF evidence 与 frozen XRV image evidence 生成 OOF/CL、ALC、FINE-GMM、external-XRV 四组 entry scores；所有 score 文件落盘并哈希后才连接 private reference，计算同预算 recall、方向 guardrail、ranking metrics、overlap 和 paired exact/Holm contrasts。
- `cxr_real_experiment/run_vindr_detector_v2_smoke.sh` / `run_vindr_detector_v2_prepare.sh` / `run_vindr_detector_v2_oof_array.sh` / `run_vindr_detector_v2_evaluate.sh`：以 `smoke -> prepare -> 60-run OOF array -> private evaluate` 的 `afterok` 链执行 detector v2；正式阶段固定六个未使用 seeds，最多并行占用三张 GPU。
- `cxr_real_experiment/vindr_xrv_oof_comparison.py`：盲化连接既有 MobileNet/Direct-XRV scores 与新 XRV-feature OOF/CL evidence，并生成固定 50/50 global-percentile fusion；score hashes 完成后才计算四方法同预算 known-error metrics 和三项 primary paired contrasts。
- `cxr_real_experiment/run_vindr_xrv_oof_comparison_smoke.sh` / `run_vindr_xrv_oof_comparison_array.sh` / `run_vindr_xrv_oof_comparison_evaluate.sh`：复用 detector-v2 六-seed inputs 和官方 XRV `features2()` payload，以 CPU shard 完成 smoke、60-run linear OOF array 与 private evaluation，不重复训练 MobileNet 或提取图像特征。
- `cxr_real_experiment/vindr_iterative_oracle_cleaning.py`：维护VinDr六-seed迭代oracle-cleaning状态机，按当前CL hard pool选取未review top20%、冻结blind selection、执行selected-only GT restore，并生成dynamic/frozen/random matched-budget trajectories、DQS correspondence及exact/Holm contrasts。
- `cxr_real_experiment/run_vindr_iterative_oracle_smoke.sh` / `run_vindr_iterative_oracle_formal_array.sh` / `run_vindr_iterative_oracle_aggregate.sh`：把one-loop GPU smoke、六个独立八-loop GPU tasks和dependency-gated private aggregate拆成hash-locked Slurm transactions。
- `cxr_real_experiment/vindr_iterative_oracle_replication_protocol_20260806.md` / `run_vindr_iterative_replication_prepare.sh` / `run_vindr_iterative_replication_loop0_array.sh` / `run_vindr_iterative_replication_formal_array.sh` / `run_vindr_iterative_replication_aggregate.sh`：固定新增seeds `509/701`，复用同一噪声生成、MobileNet OOF与八轮oracle-cleaning引擎，并把原六-seed与新增两-seed结果分别保存后再生成带post-result标记的八-seed描述性聚合。
- `cxr_real_experiment/medpalm_cl_iterative_oracle_benchmark.py`：按subject-grouped folds对official test pool生成可恢复的dynamic OOF，先冻结未review entry selection，再对selected-only entries接入Med-PaLM reference作oracle correction，并评价dynamic/frozen/random discovery trajectories。
- `cxr_real_experiment/run_medpalm_cl_iterative_oracle_benchmark.sh`：串行执行preflight、五轮blind OOF selection、selected-only private state update与最终trajectory evaluation；fold和loop均可恢复，所有直接及传递依赖以SHA-256锁定。
- `cxr_real_experiment/medpalm_model_review_bakeoff.py` / `run_medpalm_model_review_bakeoff.sh`：从不可泄漏的 Med-PaLM private reference 中固定生成 100-subject 平衡样本，仅把 report/current-label 输入发送给 GPT-5.6-Luna Responses API，并与同 keys 的存档 GPT-5.4 actions 做配对质量和 token-usage 比较；原 498-entry 正式 benchmark 保持不变。
- `cxr_real_experiment/medpalm_luna_effort_bakeoff.py` / `run_medpalm_luna_effort_bakeoff.sh`：复用上述锁定样本和 GPT-5.4 actions，分别显式请求 Luna `medium`/`max` reasoning effort，核验返回档位并比较 action quality、配对 McNemar、token、延迟和成本；单条 canary 隔离代理 distributor 不可用故障。
- `cxr_real_experiment/medpalm_terra_default_bakeoff.py` / `run_medpalm_terra_default_bakeoff.sh`：复用相同 100-entry 锁定样本，明确省略 `reasoning` 字段运行 GPT-5.6-Terra，记录服务端默认元数据并与 GPT-5.4 actions 做配对质量、McNemar、usage 和延迟比较。
- `cxr_real_experiment/reflacx_phase3_cl_pilot.py`：构建 REFLACX Phase 3 六类 blinded mini cohort，在 subject-grouped folds 中生成 MobileNetV3 OOF/CL evidence，并只在 evidence 固定后连接 private radiologist reference 计算 detection、DQS correspondence 和 subject bootstrap。
- `cxr_real_experiment/run_reflacx_phase3_cl_pilot.sh`：在单个 Slurm GPU 作业中执行 REFLACX prepare、blind run 与 private evaluation，保留阶段 manifest、hash、gate decision、tables 和 figures。
- `cxr_real_experiment/reflacx_report_llm_triangulation.py`：把同一REFLACX兼容cohort拆成report+finding-only API输入与CheXpert/REFLACX/CL private comparison，以独立LLM report label派生keep/relabel/mask并完成三方模式、类别、CL strata及subject-bootstrap评价。
- `cxr_real_experiment/run_reflacx_report_llm_triangulation.sh`：用一个scheduler shard运行checksum-locked prepare、resumable `gpt-5.4` review、zero-error verify和private triangulation evaluation，不请求完整训练GPU。

### Presentation and state

- `cxr_real_experiment/evaluation_followup_20260713/own_top20_refinement_protocol.md`：记录原始 locked endpoints、v5 post-outcome semantic correction、reuse boundary、参数和解释边界。
- `cxr_real_experiment/evaluation_followup_slides_20260713_中文.md`：下次 meeting 的 slide-by-slide answer-first structure。
- `cxr_real_experiment/build_evaluation_followup_deck_20260713.py`：把已完成 evidence、实时 own-top20 状态和预锁定 final CSV 组装为含 speaker notes、来源与 postflight 的 11 页 PPTX；final CSV 不存在时不生成推测结果。
- `cxr_real_experiment/next_meeting_slides_outline_中文.md`：当前下一次 meeting 的 canonical 八页文字版 PPT，集中维护页面内容、结果解释和中文讲稿。
- `cxr_real_experiment/build_next_meeting_slides_20260730.py`：读取 canonical outline 的讲稿并结合 Loop4-8、Loop1-15 与 DQS 原始结果生成八页 PPTX、图表资产、speaker notes 和 package preflight；未完成的 Page3/7 只生成占位页。
- `cxr_real_experiment/cluely_evaluation_followup_background_20260713_中文.md`：meeting assistant 使用的事实与定义上下文。
- `ASSIGNMENT.md`：Slurm submission、status 和 completion log。
- `CONTEXT.md`：最近十个阶段状态与关键决定。

## 调用关系

```text
MIMIC-CXR labels/images/reports
  -> prepare_own_top20_binary_v5.py (Loop1 source/reuse audit)
  -> cxr_real_oof_cleanlab_smoke.py
  -> select_topk_issue_samples.py
  -> export_sample_topk_entries_for_llm.py
  -> run_entry_level_llm_review_sharded.py
     -> run_entry_level_llm_review.py (per-process shards)
  -> build_llm_refinement_tables.py
  -> cxr_real_full_train_eval_cleanlab_xrv12.py
  -> summarize_xrv_loop_metrics.py
  -> endpoint + quality evaluators
  -> build_evaluation_followup_deck_20260713.py
  -> slide figures / PPTX / meeting material

next_meeting_slides_outline_中文.md
  -> build_next_meeting_slides_20260730.py
  -> next_meeting_slides_20260730/assets + PPTX + delivery_manifest

VinDr verified DICOM + consensus labels
  -> preprocess_vindr_dicoms.py (stratified smoke -> immutable PNG-224)
  -> vindr_known_gt_cl_benchmark.py prepare (blind noisy cohorts + private references)
  -> frozen XRV feature extraction (outcome blind)
  -> four-seed nested OOF array (blind evidence only)
  -> dependency-gated private evaluation
  -> known-error detection tables / bootstrap / figures / verification marker

VinDr frozen features + consensus labels
  -> vindr_noise_direction_sensitivity.py prepare
     -> private scenario manifest/reference
     -> outcome-free blind_run_manifest.csv + 60 noisy cohorts
  -> six-task Slurm array (ten blind OOF runs per task; no private outcomes)
  -> afterok private sensitivity evaluation
  -> direction recall / equal-budget contrasts / DQS calibration / verification marker

VinDr locked 60-run manifests + verified PNG-224
  -> prepare_vindr_mobilenet_replay.py (exact input replay links)
  -> vindr_mobilenet_oof.py (scratch MobileNet nested four-fold OOF)
  -> six-task Slurm array (ten blind runs per task; no private outcomes)
  -> afterok private sensitivity evaluation
  -> compare_vindr_mobilenet_frozen_xrv.py
  -> paired model-limitation decision gates

VinDr exact 20% noise cohorts + Loop0 MobileNet OOF
  -> vindr_iterative_oracle_cleaning.py select (unreviewed current CL hard pool only)
  -> selected-only consensus GT restore
  -> vindr_mobilenet_oof.py (updated blind cohort)
  -> repeat for eight loops across six seed tasks
  -> afterok aggregate: dynamic vs frozen vs random + quality/DQS trajectories

VinDr hard-r30 action/sentinel splits + completed full-pool trajectories
  -> vindr_dqs_calibration_transfer.py score (32 Round-0 fold replays on A16)
  -> byte-exact legacy replay gate + fold-specific sentinel predictions
  -> matched hard-r30 sentinel calibration and 32-threshold hash freeze
  -> separate evaluate process reads action truth and tests Loop 0--5 DQS transfer
  -> run_vindr_dqs_calibration_transfer.sh publishes only completed score/calibration/evaluation directories

Fixed replication seeds 509/701
  -> prepare the same four quality anchors
  -> train new Loop0 MobileNet OOF per seed
  -> run the unchanged eight-loop oracle-cleaning state machine
  -> combine with the original six seeds as post-result descriptive evidence
```

`run_xrv_iterative_sample20_branch.sh` owns the loop transaction. A loop enters `loop_metrics.csv` only after required OOF, action and train/eval outputs exist. Continuations reconstruct cumulative state from the last committed loop and reuse unfinished stage outputs.

`run_own_top20_catchall_finalize.sh` owns the experiment-level completion transaction. It skips a seed only when Loop8 stage markers, action tables, train outputs and the loop-metrics row all exist; otherwise it resumes that seed. It exposes success only after both evaluator output families and deck postflight exist.

`run_own_top20_interim_3seed_evaluation.sh` is deadline-driven reporting support rather than the formal completion transaction. Its default mode reads the same pre-specified Loop5/8 endpoints for seeds `7/13/42`; `checkpoint` mode produces a Loop5-only backup before Loop8 completes. Both write to isolated interim output directories, use a shard instead of a training GPU, and label all inferential outputs as descriptive so the five-seed finalizer remains authoritative.

## 关键设计决定

- **OOF evidence only**：cleaning candidates use out-of-fold probabilities to avoid in-sample confidence leakage。
- **REFLACX reference isolation**：radiologist labels and certainty are stored outside the blind cohort and are unavailable to model training, confident-learning selection and DQS calculation; they are joined only by the evaluation stage after evidence files are complete。
- **Three-source diagnostic blinding**：REFLACX report audit不向LLM显示当前CheXpert label，避免label anchoring；LLM只抽取report-supported positive/negative/ambiguous，之后才与CheXpert及REFLACX比较，因此结果用于source triangulation而非专家真值裁决。
- **Two evidence modes**：frozen initial OOF isolates action effects；dynamic iterative OOF describes each loop's current detector state。
- **Denominator transparency**：raw DQS describes current valid entries；coverage-adjusted DQS is a custom original-denominator diagnostic。
- **Sample metric denominator**：`sample_issue_free_rate` 只在至少一个 valid target 的 evaluable samples 上计算；zero-valid rows 是 unevaluable，不是 issue-free；该指标仍是 custom diagnostic，不是 official sample DQS。
- **Independent own-top20 seeds**：每个 seed 产生自己的 OOF probabilities、selection 和 LLM actions；不再把 fixed XRV seed13 tables 当成 full-pipeline repeatability。
- **One binary label basis**：detection、LLM review、action 和 training 都使用 U-Ones post-binarization target；raw `-1` 是 binary positive，raw value 只作 provenance；selected-sample expansion coverage 必须是 100%。
- **Transparent v5 correction**：旧 pilot 在部分 Loop1 outcomes 后因 label-semantics mismatch 作废；v5 沿用原先锁定的 Loop5/8 和 contrasts，但明确标为 post-outcome corrected validation，而不是 outcome-blind preregistration。
- **Restricted response reuse**：只复用 raw `0/1` 且 entry key、binary target、report、OOF probability、label/source identity 全等的 Loop1 responses；pilot action/training/outcome 不复用，raw `-1` 新审。
- **Resume without duplicate API review**：success rows以 `pool_row_id::label_index` 为 key；Slurm continuation 只补 pending/error entries。
- **Conservative response-conflict gate**：LLM 的 `mismatch_type` 与 `recommended_action` 不一致时保留 raw response，但不执行硬 relabel；该 entry 进入 mask table，并在 action-support output 单独计数。此 deviation 在任何 own-top20 action/training/outcome 前冻结于 v2 snapshot。
- **Inference hierarchy**：先报告 effect size、seed direction 和 hierarchical seed+study CI，再把 exact/paired/Holm p-values 作为 supporting evidence。
- **Scope-matched estimands**：all-label 与 support>=5 sensitivity 各自计算 point estimate 和 bootstrap interval；bootstrap tail mass 仅作描述，formal p-values 来自 paired seed tests。
- **No provisional endpoint values**：follow-up deck 在预锁定 CSV 不存在时只显示 pending；同一构建脚本在 final analysis 后读取 CSV 重建，避免运行中改 comparison 或填入推测数字。
- **External-reference module isolation**：Med-PaLM 只提供 externally flagged entry，不向 reviewer 泄漏其预测或医生 reference；无对应 OOF evidence 时移除 probability，保留 target/current label/report 和 keep/relabel/mask 行为，并把结论限制为 externally selected hard cases 上的 correction-stage validity。
- **External CL detection isolation**：Med-PaLM detection先在完整 official test pool 上冻结四-seed held-out probabilities、CL scores与hash，之后才读取专家 reference；因此测试的是外部 hard-case cohort 内的排序能力，不把该 cohort 的precision/recall外推到MIMIC-CXR总体。
- **Detection/prioritization separation**：双参考集 benchmark 先用 continuous score 和 hard-flag enrichment 检验 CL detection，再用同一 CL-hard study pool 内的 matched-random comparator 单独检验 whole-study-mean top-20 prioritization；第二阶段失败不被误写为第一阶段 detection 失败。
- **Prioritization budget matching**：study-level follow-up policies 固定相同 selected-study budget，direct-entry 固定旧 policy 的 expanded-entry workload；selected studies、selected entries、reference coverage、issue capture 和 precision 必须一起报告，避免以更高 review cost 换取更多已知 issues 后误称排序更好。
- **CL-versus-score separation**：同 entry-budget ablation 把 CL hard filter 与 label-aware self-confidence、per-label normalization、label-agnostic entropy 和 seed disagreement 分开；当前预算下 hard-filtered direct rank 与 global self-confidence 几乎重合，因此只能把真实错误富集归因于 OOF label-incompatibility signal，不能归因于 hard filter 的独立增益。
- **VinDr known-GT outcome isolation**：consensus labels 只在 prepare 阶段生成独立 private reference；正式 OOF tasks 仅接收 noisy blind cohort 和 label-blind frozen features，四 seed evidence 全部完成后 private evaluator 才能启动。该受控实验可验证 injected-error detection 和 DQS calibration，但不把 synthetic corruption 等同于 MIMIC report-derived noise，也不把 consensus reference 描述为绝对临床真值。
- **VinDr sensitivity manifest isolation**：array mapping 使用单独的 `blind_run_manifest.csv`，只保留 condition、seed 与相对输入/输出路径；injected counts、true quality、private hashes 和 clean reference 仅存在于 private manifest。原先混合 mapping/outcome 字段的 preformal v1 在任何正式 run 前作废并保留 invalid marker，正式证据只来自 v2。
- **DQS is mechanism-sensitive**：raw DQS 是由同一 noisy-label OOF model 产生的 model-consistency proxy，不再被描述为独立真值。多质量锚点结果显示 balanced noise 下单调但压缩、false-positive-only 下偏保守，而 false-negative-only 下随真实质量下降反向升高；任何 dataset-quality claim 必须按 error mechanism 分层校准。
- **DQS calibration-transfer isolation**：fold-specific thresholds 只在训练完全隔离、与 action 初始 hard-error rate 匹配的 sentinel cohort 上按 known correct mass 校准；threshold table 写入并 hash 后，独立 evaluate process 才读取 action trajectories。该阈值估计总体质量，不被描述为 entry-level correctness classifier，也不直接外推到 MIMIC-CXR。
- **Nested support-gated early stopping**：每个 VinDr outer fold 的 epoch selection 只使用 outer-train 内的 deterministic inner split；若任一目标在 inner train/validation 中少于三个正例或负例，则按预定 seed sequence 选择第一个满足支持度的 split，并记录实际 seed，不读取 injected-error detection outcome。
- **Paired MobileNet replay**：MobileNet ablation 不重新抽 folds 或 corruption，而是逐路径复用已验证的 60 个 blind/private inputs；唯一目标变化是 OOF model/training environment，因此 `1→0` recall 差异按 seed 配对解释，并以 clean-reference AUROC 防止用整体更弱的模型制造表面差异。
- **Selected-only oracle loop**：dynamic benchmark每轮只在blind selection及hash固定后读取被选entries的专家reference；下一轮可使用这些已review corrections，但不能读取未review outcomes。Subject-grouped folds阻止同一患者跨OOF train/validation，且该official test pool从此仅作development benchmark，不再支持untouched held-out performance claims。
- **VinDr iterative oracle upper bound**：每轮先从当前未review CL hard pool中冻结top20% blind selection，再只把这些entries恢复为consensus GT；不删除或mask数据，固定18,000-entry denominator，已review entries不能重入。该实验验证CL detection/selection在perfect correction下能否逐轮提高known quality，不验证LLM本身，也不把symmetric corruption当作自然report-label noise。
- **Detector confirmation before pipeline adoption**：新 detector 必须先在全新 VinDr corruption draws 上以完全相同的 review budget 与 OOF/CL 配对比较；blind score publication 先于 private truth join。FINE 只作为其 singular-alignment/GMM 核心的 binary multi-label adaptation，PU detector 因 balanced/false-positive corruption 破坏 reliable-positive 假设而留给单独 missing-positive protocol。
- **Fixed XRV fusion comparison**：XRV OOF 与 Direct XRV 共用同一 pretrained encoder，不能写成两个独立模型的证据；融合只比较 noisy-label-adapted OOF head 与 frozen direct classifier 两种 scoring views。为避免 outcome-driven tuning，正式 private evaluation 前固定为各 run 全局百分位排名的 50/50 平均。
- **Fixed post-result replication extension**：新增seeds `509/701`在任何新outcome产生前一次性固定，两个结果无论方向均须纳入，且不允许依据更新后的p值继续追加seed。原六-seed分析保留为主要锁定结果；合并八-seed统计明确标为post-result descriptive evidence，避免optional stopping造成的显著性夸大。
