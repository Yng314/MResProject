# Med-PaLM 动态 OOF 迭代检测验证

## 一句话结论

在 498 个 Med-PaLM hard-case entries 中，动态重训 OOF 的 confident-learning
排序在整体发现效率上略优于固定使用第一轮排序（AUDC `0.6769` 对 `0.6642`），
但优势不大，也不是每个 review budget 都更好。更强、更稳定的结论是：无论是否
动态重训，CL 都能比随机审阅更早找到专家确认的错误；第一批审阅约 20% entries
时已经找到 `68/127` 个错误（`53.5%` recall），而随机审阅同期平均只能找到约
`20.1%`。

## 实验回答什么

本实验隔离验证 upstream detection 的迭代问题：修正一批已发现错误并重新训练 OOF
模型后，更新后的 CL evidence 是否能比第一轮固定排序更早找到剩余错误。

- benchmark：498 entries，其中 127 个专家确认错误、371 个正确。
- development pool：official test AP/PA pool，共 3,414 images、3,050 studies；本实验
  将它作为开发 benchmark，因此不把它再当作独立 held-out performance test。
- 模型：MobileNetV3，seed `13`，4-fold subject-grouped OOF；每轮最多 100 epochs，
  patience 10，并恢复最佳权重。
- review budget：五轮依次审阅 `100/100/100/100/98` 个从未审阅过的 entries。
- 动态条件：每轮只把已选 entry 按专家答案作 oracle correction，然后重新训练 OOF、
  重新计算 CL 排序。
- 固定条件：始终沿用 Loop 1 的盲态 CL 排序，在相同累计 review budget 下计算发现率。
- 随机条件：10,000 次不放回随机排序，用作工作量基线。
- 专家答案只在每轮 blind selection 固定并写入 manifest 后连接。

## 发现曲线

| 累计审阅 | 动态 CL 找到错误 | 动态 recall | 固定 CL 找到错误 | 固定 recall | 动态减固定 |
|---:|---:|---:|---:|---:|---:|
| 100 / 498（20.1%） | 68 / 127 | 53.5% | 68 / 127 | 53.5% | 0 |
| 200 / 498（40.2%） | 87 / 127 | 68.5% | 81 / 127 | 63.8% | +6 errors |
| 300 / 498（60.2%） | 101 / 127 | 79.5% | 98 / 127 | 77.2% | +3 errors |
| 400 / 498（80.3%） | 111 / 127 | 87.4% | 112 / 127 | 88.2% | -1 error |
| 498 / 498（100%） | 127 / 127 | 100% | 127 / 127 | 100% | 0 |

随机审阅在相同五个 checkpoints 的平均 recall 约为 `20.1%`、`40.2%`、`60.3%`、
`80.3%` 和 `100%`。第一批 100 次审阅中，动态与固定 CL 均找到 68 个错误，远高于
随机预期约 25 个。第一轮两条 CL 曲线完全相同是设计所致：动态更新尚未发生。

## 动态重训是否必要

以整条发现曲线下面积 AUDC 衡量：

- dynamic CL：`0.6769`；
- frozen CL：`0.6642`；
- dynamic minus frozen：`+0.0127`；
- random mean：`0.5000`，95% simulation interval `[0.4574, 0.5426]`。

因此，动态重训在本次 seed 中有小幅总体收益，主要出现在累计审阅 40% 和 60% 时；
到 80% 时固定排序反而多找到一个错误。结果不支持“动态 OOF 每一轮都会持续优于
固定排序”的强结论，但支持“动态更新可能略微提高中等 review budget 下的发现效率”。

## 为什么没有提前找完全部错误

动态 CL 在审阅 80.3% entries 后仍剩 16 个错误，只有审阅全部 498 entries 后才达到
127/127。最后达到 100% 是因为所有 entries 最终都被检查，是算术必然，不是 CL 在有限
预算下找齐全部错误的证据。

CL 的离散 hard flags 也很快耗尽：五轮选中的 hard flags 数分别为 `50`、`13`、`0`、
`2`、`0`，其中第四轮的 2 个都不是真实错误。因此后期发现主要依赖连续 suspiciousness
排序，而不是新的 hard flags。每批新增真实错误从第一轮的 68 个降到 `19/14/10/16`，
说明随着高优先级错误被移除，后续 review precision 自然下降。

## 可以和不可以声称什么

当前可以声称：在这个外部 adjudicated hard-case benchmark 内，CL 明显优于随机审阅；
用 oracle correction 更新标签并重训 OOF 后，整条发现曲线比固定第一轮排序略好。

当前不可以声称：动态重训在每个 budget 都更好、能在有限预算内找齐全部错误、结果已
跨 seed 重复，或该 precision/recall 可直接外推到完整 MIMIC-CXR population。本实验使用
oracle correction，验证的是 CL detection，不是 LLM correction 或完整 end-to-end loop。

## Reproducibility

- Protocol: `cxr_real_experiment/medpalm_cl_iterative_oracle_protocol_20260803.md`
- Runner: `cxr_real_experiment/run_medpalm_cl_iterative_oracle_benchmark.sh`
- Job: `269716`, completed successfully with empty stderr.
- Canonical outputs: `/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/medpalm_cl_iterative_oracle_benchmark/20260803_seed13_dynamic5loop`
- Main table: `iterative_detection_trajectory.csv`
- Summary: `iterative_detection_summary.json`
- Figure: `iterative_detection_trajectory.png`
