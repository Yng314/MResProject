## 2026-10-08 — Source-only Git cleanup
- 当前正在做什么：源码边界清理已推送到 `MResProject/main`；99 份结果/临时文件退出当前 Git 树，15 份合成 toy CSV/PNG 已复制到 Bitbucket 并逐文件验 hash。
- 上次停在哪个位置：远端双分支镜像备份已保存在父工作区 `.repo_backups/MResProject_remote_before_rewrite_20261008.git`；现正准备重写 `main` 历史并删除旧工作区分支。
- 关键决定和原因：按用户授权清理旧病例级文件历史并删除 `llmtest-workspace` 分支；MIMIC/REFLACX 原始数据和临床派生结果仍须使用获准存储，不因用户要求忽略限制而迁移。

## 2026-10-07 — Public workspace snapshot

当前正在做什么：实验项目的八路径源码整理已提交推送，完整 llmtest 工作区在同一公开仓库的独立 `llmtest-workspace` 分支发布，主分支继续保存实验项目。
上次停在哪个位置：三个历史 Slurm 入口已随 `eb798b8` 发布，工作树与远端一致；工作区快照采用当前文件树并记录源提交，本次同步更新入口说明。
近期的关键决定和原因：现有连接可推送文件但不能新建仓库或改可见性，因此使用已公开仓库的独立分支完成整个源码工作区公开，两个原仓库历史及所有本地数据保持不变。

## 2026-10-07 — Whole-workspace source publication

当前正在做什么：按作者公开发布整个 llmtest 源码工作区的决定，将 MedSoul 三个原先被通配规则忽略的 Slurm 入口纳入 Git，并记录它们保留的旧路径。
上次停在哪个位置：三个 shell 文件内容不变，新增源码及文档已准备作为本次普通提交发布；实验结果、数据集和本地病例文件继续留在原处，真实提交及远端核验由根工作区恢复记录保存。
近期的关键决定和原因：根仓库用子模块连接实验与论文仓库以保留独立历史，公开源代码和汇总材料，外部大数据用位置清单记录，不改程序、不跑新作业、不重写历史。

## 2026-10-07 — Case-level publication boundary

当前正在做什么：本次整理提交为 4 份病例级 CSV 和 6 份会议稿加入精确忽略规则并解除 Git 跟踪，原始文件仍在相同路径且有经过 SHA-256 核对的恢复副本。
上次停在哪个位置：36 个保留的 CSV 表头与已跟踪 Markdown 已通过本次标识字段检查，聚合表与实验源码继续跟踪；提交只包含 Git／文档边界调整，父工作区保存真实提交与远端验证记录。
近期的关键决定和原因：让本地分析可继续读取原件，同时让新 Git 树不再携带这批病例材料；不移动输入路径、不改代码、不删除原件，旧历史仍需单独决定是否处理。

## 2026-10-05 — Repository publication boundary audit
- 当前正在做什么：已盘点 `MResProject` 的 GitHub、GPU Data 工作树和 Bitbucket 结果归档，并建立代码、协议、结果、生成物、环境文件和机器链接的分类边界。
- 上次停在哪个位置：已确认远端是公开仓库，现有公开历史含逐病例字段；289 个 source/protocol/docs 已在提交 `5331f88` 中推送，21 个本地笔记已移入 `local_notes/`，未删除结果或重写历史。
- 关键决定和原因：先冻结安全的 source/document allowlist，再把内部笔记移入本地 `local_notes/`；整理提交、笔记边界提交和文档记录均已推送，大型结果继续留在 `/vol/bitbucket/yz3522/NoiseRate_results_archive`。

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
