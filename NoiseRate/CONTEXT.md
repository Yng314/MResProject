## 2026-10-05 — Repository publication boundary audit
- 当前正在做什么：已盘点 `MResProject` 的 GitHub、GPU Data 工作树和 Bitbucket 结果归档，并建立代码、协议、结果、生成物、环境文件和机器链接的分类边界。
- 上次停在哪个位置：已确认远端是公开仓库，现有公开历史含逐病例字段；289 个 source/protocol/docs 已在提交 `5331f88` 中推送，21 个本地笔记已移入 `local_notes/`，未删除结果或重写历史。
- 关键决定和原因：先冻结安全的 source/document allowlist，再把内部笔记移入本地 `local_notes/`；整理提交 `5331f88` 与笔记边界提交 `89fb6c8` 均已推送，大型结果继续留在 `/vol/bitbucket/yz3522/NoiseRate_results_archive`。

## 2026-08-28 — Five-seed MIMIC calibration completed and verified
- 当前正在做什么：MIMIC-CXR calibration-transfer finalizer `279601` 已完成，正式五-seed结果已通过官方 postflight 与独立数值、行数和哈希审计，并交付论文更新。
- 上次停在哪个位置：radiologist reference set 的 Calibrated DQS 在 Round 0/1/5 为 `0.9485/0.9627/0.9639`，Med-PaLM sensitivity 在 Round 0/5 为 `0.8593/0.8888`；原 DQS 的 `0.9277/0.9600/0.9618` 不变。
- 关键决定和原因：正式汇总替代三-seed preview，论文只需更新集中 calibration 数值与综合图的 calibration 曲线；coverage、review workload、AUROC 和论证结构均继续使用已验证的五-seed结果。

## 2026-08-28 — MIMIC calibration status rechecked for thesis integration
- 当前正在做什么：已于 09:04 BST 复核 formal calibration root 的完成标记，并将当前可用的 calibration-transfer summary 接入论文图表与集中数值宏。
- 上次停在哪个位置：score completion markers 仍只覆盖 seeds `7/42/123`，seeds13/97 与正式 evaluation summary 尚未完成；论文图源会在正式 summary 发布后自动优先读取。
- 关键决定和原因：论文当前采用可整体替换的工作数值与 final-first 图源，不在正文暴露运行进度；待正式 summary 到位后只更新集中数值并重绘，不改变论证结构。

## 2026-08-28 — MIMIC calibration preview and live jobs audited
- 当前正在做什么：已按 Academic Research Suite 的证据强度口径核对 MIMIC-CXR 三-seed calibration preview、两类 calibration anchors、replay 与 archived DQS 口径及当前 scorer 日志。
- 上次停在哪个位置：seeds `7/42/123` 已完成，seed13 正在 fold2、seed97 正在 fold3，正式 finalizer 尚未启动；radiologist-panel 三-seed Calibrated DQS 为 `0.9491 -> 0.9647`，Med-PaLM hard-case sensitivity 为 `0.8624 -> 0.8929`。
- 关键决定和原因：三-seed结果只能作明确标注的 preliminary sensitivity estimate；主文可用预先指定的 radiologist-panel anchor，但必须保留 Med-PaLM anchor-dependence 敏感性结果，且不得把无完整训练集真值下的数值锚定写成已证实更准确。

## 2026-08-28 — Three-seed MIMIC calibration preview generated
- 当前正在做什么：已用完成的 seeds `7/42/123` 独立重放锁定的 MIMIC calibration-transfer 算法并生成校准前后 DQS 预览图；seeds13/97 和正式 finalizer 仍在等待完成。
- 上次停在哪个位置：radiologist-panel anchor 下，三-seed均值从 Round 0 的 DQS `0.9274`／Calibrated DQS `0.9491` 变化到 Round 5 的 `0.9620`／`0.9647`；preview 保存在 `mimic_dqs_calibration_transfer/preview_completed_7_42_123_20260828T0535`。
- 关键决定和原因：预览的校准前后两条曲线使用完全相同的三个 seeds，避免把论文五-seed均值与部分校准结果直接混比；该结果只用于提前观察形态，不替代正式五-seed finalizer 输出。

## 2026-08-24 — VinDr DQS proxy reanalysis completed
- 当前正在做什么：已复用 `20260805_v1` 中 24 组已完成 VinDr-CXR MobileNet OOF evidence，在 CPU 上复算四个已知质量状态及四类固定干预的 raw／coverage-adjusted DQS，并生成论文 Figure 5.3 与机器可读 CSV／QA。
- 上次停在哪个位置：六 seed、18,000 entries、五个 720-entry 累积步骤均通过输入、join、方向性和独立数字审计；未重新训练模型、未占用 GPU，也未修改 archive。
- 关键决定和原因：将 DQS 解释为 model-dependent trend indicator；count-matched pseudo-random removal 使用固定 SHA-256 顺序，使控制 cohort 在 paired seeds 间一致且不依赖真值或 OOF 分数。

## 2026-08-24 — Seven remaining VinDr MobileNet seeds started on A16
- 当前正在做什么：A16 serial job `278151` 正在 `gpuvm36` 依次运行 `13007/17011/19001/23003/27011/31013/37003`，结束后会在同一 allocation 内生成八-seed aggregate；当前已从 seed13007 正常开始且 stderr 为空。
- 上次停在哪个位置：为释放全局第三张 GPU，刚启动约五分钟的 MIMIC seed123 job `278133` 已取消，已落盘 review checkpoint 保留；A30 seed42/97 继续运行。
- 关键决定和原因：当前最高优先级是完整跑完 VinDr A16 multi-seed 与 aggregate；在 `278151` 完成前不恢复 MIMIC seed123、不提交剩余 MIMIC seeds 或任何新的 GPU 作业。单个连续 allocation 会跨 seeds 保持 A16，不给用户的其他作业留下插队窗口。

## 2026-08-24 — A16 VinDr MobileNet pilot completed
- 当前正在做什么：A16 job `278114` 已正常完成 seed11003 的五轮 VinDr MobileNet full-issue-pool pilot；当前没有任何 A16 后续作业排队。
- 上次停在哪个位置：one-shot 后 known quality 为 `0.84653`，继续四轮后达到 `0.98583`；相对 one-shot，action/sentinel AUROC 分别增加 `0.02982/0.12457`，stderr 为空。
- 关键决定和原因：七个剩余 seeds 的 array runner 已存在但尚未提交，因此当前只得到单 seed pilot；是否扩展需由用户明确决定，不与正在运行的 MIMIC A30 multi-seed 作业混淆。

## 2026-08-24 — Seed13 completed; seed123 queued
- 当前正在做什么：seed42/97 jobs `277707/277708` 继续在 `dipper` 运行；seed13 释放槽位后已提交 seed123 job `278133`，当前因资源排队，预计今天约 `20:18 BST` 启动。
- 上次停在哪个位置：seed13 job `277538` 已正常完成全部八轮，stderr 为空；Loop8 study-weighted AUROC 为 `0.779444`，高于 matched baseline `0.728102`，累计 `14,801` relabel 和 `20,688` mask。
- 关键决定和原因：继续按固定 seeds `42/97/123/314/2718` 滚动占用最多三张 A30；当前无 seed13 续跑需要，剩余 `314/2718` 待后续槽位释放时提交。

## 2026-08-23 — Three-GPU six-seed extension started
- 当前正在做什么：seed13 job `277538` 在 `clapper` 继续运行，新增 seed42 job `277707` 与 seed97 job `277708` 已在 `dipper` 启动，三张 A30 正同时运行 full issue-pool、unseen-only、GPT-5.4 八轮流程。
- 上次停在哪个位置：两个新作业均通过 baseline OOF 与协议校验并进入 Loop1 LLM review；seed42/97 分别有 `53,977/53,264` 条待复核 entries，stderr 为空。
- 关键决定和原因：固定新增 seeds 为 `42/97/123/314/2718`，所有结果均保留；为立即三卡并行，未启动的保险作业 `277539/277540` 已取消，剩余 `123/314/2718` 将在 GPU 槽位释放后滚动提交。
