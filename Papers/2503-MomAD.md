---
title: "Don't Shake the Wheel: Momentum-Aware Planning in End-to-End Autonomous Driving"
authors: ["Ziying Song", "Caiyan Jia", "Lin Liu", "Hongyu Pan", "Yongchang Zhang", "Junming Wang", "Xingyu Zhang", "Shaoqing Xu", "Lei Yang", "Yadan Luo"]
institute: ["Beijing Jiaotong University", "Horizon Robotics", "University of Macau", "Tsinghua University", "The University of Queensland"]
date_publish: 2025-03
venue: arXiv Preprint
tags: [imitation-learning, spatial-memory]
url: https://arxiv.org/abs/2503.03125
cite_key: song2025momad
arxiv_id: "2503.03125"
doi: "10.48550/arXiv.2503.03125"
code: https://github.com/adept-thu/MomAD
rating: 3
content_scope: full-text
verification_status: source-checked
date_added: 2026-11-26
---
## Summary

端到端规划普遍采用 one-shot 多模态轨迹预测：每一帧独立生成候选、按分数取最高，缺少时间维度的约束，于是相邻帧会选到不同机动，表现为方向盘抖动。MomAD 把"动量"引入规划，拆成两层：**trajectory momentum** 用 Topological Trajectory Matching（TTM）在候选集中挑出与上一帧轨迹 Hausdorff 距离最小的一条，保证路径连续；**perception momentum** 用 Momentum Planning Interactor（MPI）把被选中的 planning query 与上一帧的多模态 planning query 做 cross-attention（历史 query 先经分数门控 + LSTM 演化），重建更长的时域上下文。感知侧另加一个训练期的高斯扰动 + 轻量 encoder-decoder 去噪模块，缓解稀疏实例特征的抖动。论文同时提出 Trajectory Prediction Consistency（TPC）指标量化跨帧一致性。在 Turning-nuScenes 上 6 s 预测把碰撞率降低 26%、TPC 降低 0.97 m（33.45%）；Bench2Drive 闭环成功率 18.11%，相对 VAD 多模态变体提升 16.3%。

## Problem & Motivation

- 确定性回归（UniAD、VAD）在存在多种合理机动时只能给出一个折中解，缺乏动作多样性，风险高。
- 多模态规划（VADv2、SparseDrive）解决了多样性，但仍然是 **one-shot 且只依赖当前感知帧**：遮挡或关键线索丢失时，多模态质量下降；更关键的是**没有跨帧约束**，相邻帧的最高分候选可能属于不同机动，导致控制不稳、方向抖动与振荡。
- 现有工作只处理"时序实例特征"，没有处理"时序规划一致性"——感知的时序建模不等于规划输出的时序一致。
- 作者的观察是：人类驾驶有惯性，当前决策受上一时刻路径约束；因此规划应当同时具备轨迹惯性与感知惯性，而不是每帧重新做一次独立选择。
- 顺带的问题：主流指标（L2、碰撞率）无法反映轨迹抖动，缺少一个能量化"跨帧一致性"的度量，因此论文自建 TPC 指标与 Turning-nuScenes 子集。

## Method

**0. 总体框架**

+ 感知沿用 SparseDrive 的稀疏范式：多视角图像编码后，在 anchor box / polyline 周围采样 keypoint，经检测-跟踪与在线建图得到实例特征 $\mathbf{F}_t^{\mathrm{ins}}$；规划侧在其上接 TTM 与 MPI，最后经 PlanHead 输出多模态轨迹与分数。

+ 训练分两阶段（与 SparseDrive 一致）：Stage 1 只训稀疏感知（检测、跟踪、在线建图）；Stage 2 解冻感知、联合训练运动预测与规划，共 10 epoch、batch 48、lr $3\times10^{-4}$。

**1. Topological Trajectory Matching（TTM，§3.1）**

+ 候选集 $\mathcal{T}_t=\{T_t^k\}_{k=1}^{K}$，$K=6\times3$：三种导航命令（左/右/直行）各 6 条 proposal；每条含 $N_t$ 个 waypoint（6 或 12 个，对应 3 s / 6 s 时域）。

+ 历史轨迹与当前候选生成于不同时刻的 ego 坐标系，先统一到 $t-1$ 帧：

$$
T_t^k \leftarrow \mathbf{R}_{t-1}^{-1}\left(T_t^k-\mathbf{\Gamma}_{t-1}\right),\quad \forall k\in[K]
$$

+ 逐点欧氏距离只衡量局部接近度，在转弯或点密度不均时不可靠；因此用双向 Hausdorff 距离衡量"最坏情况下的整体错位"：

$$
d_1=\sup_{p\in T_t^k}\inf_{h\in T_{t-1}}\lVert p-h\rVert,\qquad d_2=\sup_{h\in T_{t-1}}\inf_{p\in T_t^k}\lVert h-p\rVert,\qquad d_H=\max(d_1,d_2)
$$

+ 取距离最小的候选作为"惯性一致"的一条：

$$
k^{*}=\arg\min_{k\in[K]} d_H\left(T_t^k,T_{t-1}\right)
$$

+ 这一步只约束"不要突然换机动"，并不直接决定最终输出——被选中的轨迹在 MPI 里被重新编码、重新打分。

**2. Momentum Planning Interactor（MPI，§3.2）**

+ 被选中的 query $\mathbf{Q}_t^{p*}$ 仍然只基于当前感知，遮挡下可能偏；MPI 用上一帧的多模态 planning query $\mathbf{Q}_{t-1}^{p}\in\mathbb{R}^{K\times D_q}$ 与其分数 $\mathbf{S}_{t-1}^{p}$ 来补时域上下文。

+ 历史 query 先按分数门控再做 LSTM 演化，模拟 planning 意图随时间的漂移：

$$
\mathbf{Q}_{t-1}^{p'}=\mathrm{LSTM}\left(\sigma(\mathbf{S}_{t-1}^{p})\circ\mathrm{MLP}(\mathbf{Q}_{t-1}^{p})\right)
$$

+ 其中 $\sigma$ 是 sigmoid，$\circ$ 是逐元素乘；LSTM 把"上一帧哪些机动被看好"这一信息压进 query，而不是简单拼接。

+ 当前 query 作 Q、演化后的历史 query 作 K/V 做 cross-attention，得到增强 query，再与实例特征一起送 PlanHead：

$$
\tilde{\mathbf{Q}}_t^{p*}=\mathrm{Attention}\left(\mathbf{Q}_t^{p*},\mathbf{Q}_{t-1}^{p'},\mathbf{Q}_{t-1}^{p'}\right),\qquad \tilde{\mathcal{T}}_t,\tilde{\mathbf{S}}_t=\mathrm{PlanHead}\left(\tilde{\mathbf{Q}}_t^{p*},\mathbf{F}_t^{\mathrm{ins}}\right)
$$

+ 最终仍按多模态分数取最高的一条，但此时每条候选都已吸收历史信息，因此"分数最高的那条"在跨帧上更稳定。

**3. Robust Instance Denoising via Perturbation（§3.3）**

+ 规划高度依赖检测与地图的实例特征，而检测器不稳、地图动态变化都会引入噪声；训练时对实例特征加受控高斯扰动（噪声强度 $\mathrm{NS}$），再用一个轻量 encoder-decoder transformer block 学去噪。

+ 推理时不加噪，模型只需对实例特征的正常波动保持鲁棒；噪声强度是核心超参，默认 $\mathrm{NS}=0.1$。

**4. 新指标 TPC（§4.1）**

+ 把当前预测轨迹变换到历史帧坐标系后，统计两条轨迹在重叠时域上的差异：

$$
\mathrm{TPC}=\frac{1}{N_T}\sqrt{\sum_{i=1}^{N}\left(\left(T_t^{\mathrm{Pred}}-T_{t-1}^{\mathrm{Pred}}\right)^{2}\cdot T^{\mathrm{Mask}}_{\mathrm{GT}}\right)}
$$

+ $N_T$ 为验证集 GT 轨迹总数，$T^{\mathrm{Mask}}_{\mathrm{GT}}$ 只统计两条轨迹时间重叠的部分；该指标是数据集级的统计量，不是单样本指标。

**5. Turning-nuScenes 子集（附录 A.2）**

+ nuScenes 以直行为主，会掩盖时序不一致；作者按"未来 3 s 内 x 方向位移是否超过阈值 $\varepsilon=25$"筛选转弯样本，得到 17 个 scene / 680 个样本的验证子集，包含路口、T 字口、环岛、交通岛与巷道转弯。

+ 这个子集只用于验证，不参与训练；规模很小，这是后面要打折的地方。

**6. 实现细节**

| 项目 | 设置 |
|:--|:--|
| Backbone | ResNet50，输入 256×704 |
| 检测范围 / 建图范围 | 半径 55 m / 纵向 60 m × 横向 30 m |
| 实例记忆队列 | $H=3$ 帧；运动模态数 $K_m=6$ |
| 规划候选 | $6\times3=18$ 条（3 命令 × 6 proposal），6 / 12 waypoint |
| 训练 | 两阶段（感知 → 全任务联合），Stage 2 共 10 epoch，batch 48，lr $3\times10^{-4}$ |
| 硬件 | 8×RTX 4090 24GB |
| 推理速度 | 7.8 FPS（SparseDrive 为 9.0 FPS） |

## Pipeline

![MomAD 整体架构](https://arxiv.org/html/2503.03125v3/mainV19.png)

*多视角图像先经稀疏感知编码为实例特征（含训练期的扰动与去噪模块），再进入动量规划：TTM 用 Hausdorff 距离从 18 条候选里选出与上一帧最一致的一条，MPI 把它的 planning query 与上一帧的多模态 query 做 cross-attention 得到增强 query，最后由 PlanHead 重新生成多模态轨迹与分数并取最高分输出。*

1. **感知**：多视角图像 → 稀疏实例特征 $\mathbf{F}_t^{\mathrm{ins}}$（检测框、跟踪 ID、地图 polyline 的 keypoint 特征）。
2. **训练期去噪**：对实例特征加高斯噪声，经轻量 encoder-decoder 重建干净特征；推理时该扰动关闭。
3. **生成候选**：按三种导航命令各生成 6 条多模态轨迹，共 18 条。
4. **轨迹对齐**：把当前候选变换到 $t-1$ 帧坐标系，算与上一帧轨迹的双向 Hausdorff 距离。
5. **动量选择**：取距离最小的候选 $T_t^{k^*}$，把它对应的 planning query 作为当前 query。
6. **历史交互**：上一帧 18 条 query 经分数门控 + LSTM 演化后作为 K/V，与当前 query 做 cross-attention。
7. **重规划与输出**：增强 query 与实例特征一起过 PlanHead，重新生成轨迹与分数，取最高分作为最终规划；同时用 TPC 统计跨帧一致性。

## Key Results

**1) nuScenes 验证集规划（Table 1，VAD 协议那一组）**

| Method | L2 Avg (m)↓ | Col. Avg (%)↓ | TPC Avg (m)↓ | FPS↑ |
|:--|--:|--:|--:|--:|
| VAD | 0.72 | 0.21 | 0.64 | – |
| SparseDrive | 0.61 | **0.08** | 0.57 | **9.0** |
| UniAD | 0.73 | 0.61 | 0.68 | 1.8 |
| **MomAD** | **0.60** | 0.09 | **0.54** | 7.8 |

+ L2 只比 SparseDrive 好 0.01 m，TPC 好 0.03 m；**碰撞率反而从 0.08% 升到 0.09%**，说明这套动量约束主要改善平滑性，不是安全余量。
+ 在 UniAD 评测协议那一组（Table 1 上半部分），MomAD 为 0.98 / 0.30 / 0.80，对应 SparseDrive 1.01 / 0.32 / 0.85，方向一致但幅度同样很小。

**2) Turning-nuScenes（Table 2）**

| Method | L2 Avg (m)↓ | Col. Avg (%)↓ | TPC Avg (m)↓ |
|:--|--:|--:|--:|
| SparseDrive | 0.86 | 0.40 | 0.79 |
| **MomAD** | **0.76** | **0.32** | **0.63** |

+ 转弯场景上三项都有改善：L2 −0.10 m、碰撞率 −0.08 个百分点、TPC −0.16 m，说明"多模态最高分在转弯时频繁切换"确实是 SparseDrive 的一个弱点。
+ 但该子集只有 17 个 scene / 680 样本，单项指标的分母很小，没有报方差。

**3) 6 s 长时域预测（Table 3，Turning-nuScenes）**

| Method | L2 6s (m)↓ | Col. 6s (%)↓ | TPC 6s (m)↓ |
|:--|--:|--:|--:|
| SparseDrive | 3.36 | 2.57 | 2.90 |
| **MomAD** | **2.51** | **2.31** | **1.93** |

+ 相对 SparseDrive：L2 −25.3%、碰撞率 −0.26 个百分点（论文表述为碰撞率降低 26%）、TPC −0.97 m（−32.45%）。
+ 增益随时域变长而放大（4 s 的 TPC 只降 0.17 m，6 s 降 0.97 m），符合"历史信息对远端点影响更大"的直觉。

**4) Bench2Drive 闭环（Table 4）**

| Method | Avg. L2↓ | DS↑ | SR (%)↑ | Effi↑ | Comf↑ |
|:--|--:|--:|--:|--:|--:|
| VAD | 0.91 | 42.35 | 15.00 | 157.94 | 46.01 |
| VAD$_{\mathrm{mmt}}$* | 0.89 | 42.87 | 15.91 | 158.12 | 47.22 |
| SparseDrive* | 0.87 | 44.54 | 16.71 | 170.21 | 48.63 |
| MomAD (Euclidean) | 0.84 | 46.12 | 17.45 | 173.35 | 50.98 |
| **MomAD** | **0.82** | **47.91** | **18.11** | **174.91** | **51.20** |

+ 相对 VAD 多模态变体 SR +16.3%（15.91 → 18.11），相对 SparseDrive 复现 SR +8.4%；DS 也从 44.54 提到 47.91。
+ 表中带 * 的 SparseDrive 是作者复现值（16.71），与论文原报数字不同，因此跨表比较要小心。

**5) 消融：扰动强度与历史帧数（Table 6、7）**

| 维度 | 设置 | 结果 |
|:--|:--|:--|
| 噪声强度 NS | 0 / 0.05 / **0.1** / 0.2 / 0.3 | TPC 0.55 / 0.54 / **0.54**；mAP 0.417 / 0.421 / **0.423**；NS=0.2 起 mAP 与 TPC 同时变差（0.420→0.412、TPC 0.54→0.61） |
| 历史帧数 $t$ | 1（无历史）/ 2（前 1 帧）/ 3（前 2 帧） | TPC 0.81 / **0.63** / 0.66；L2 0.88 / **0.76** / 0.78 |

+ 噪声强度是一个窄窗口的超参：0.1 最好，0.2 起检测与规划同时退化，说明"加噪去噪"不是越强越好。
+ 历史只用**前一帧**最好，融合两帧反而变差，作者归因于更早特征引入不确定性——也就是说这里的"动量"实际只有一步记忆。

**6) 消融：MPI 内部结构（Table 8）**

+ 直接用加法把历史 query 加到当前 query（Add）就能把 TPC 从 0.79 降到 0.67，说明收益主要来自"用了历史"而不是复杂结构。
+ 换成 LSTM + cross-attention 的 Query Mixer 后进一步到 0.63、L2 0.76，结构仍有增益但幅度有限。

**7) 感知与鲁棒性（Table 5、附录 A.5）**

+ 检测 mAP 0.423（+0.5）、NDS 0.531（+0.6）；跟踪 AMOTA 39.1（+0.5）；在线建图 mAP 55.9（+0.8）；运动预测 minADE 0.61 / minFDE 0.98。
+ 在 nuScenes-C 损坏集上，平均相对 SparseDrive 提升：检测 22.9%、跟踪 27.1%、建图 25.1%、运动 24.2%、规划 40.0%。

## Strengths & Weaknesses

**Strengths**

- 问题切得具体：把"多模态规划的最高分在相邻帧之间跳动"单独拎出来，用轨迹惯性与感知惯性两个机制分别对应"输出层"与"query 层"，比笼统说"要时序建模"清楚。
- TTM 的度量选择有理由：逐点欧氏距离只看局部，Hausdorff 取双向最坏偏移，能捕捉整体路径错位；这个替换在转弯场景（点密度不均）下更有意义。
- MPI 的 LSTM 门控设计合理：先用上一帧分数对 query 加权，再让 LSTM 演化，相当于把"上一帧看好哪些机动"作为先验压进 query，而不是简单拼接历史。
- 新指标 TPC 填补了一个真实空白：L2 与碰撞率都看不出抖动，TPC 直接把跨帧差异量化，且与主结论方向一致（转弯、长时域上差距放大）。
- 消融诚实地报告了负结果：历史帧用两帧反而比一帧差、噪声强度 0.2 起全面退化，这两条都削弱了"更多记忆更好"的叙事，但作者照实写了。
- 有闭环验证：Bench2Drive 上 SR 18.11% 高于 SparseDrive 复现值 16.71%，不是只报开环。

**Weaknesses**

- **主表增益接近噪声**：nuScenes 上 L2 0.60 vs 0.61、TPC 0.54 vs 0.57，而**碰撞率 0.09% 比 SparseDrive 的 0.08% 更差**。论文用 TPC 的改善支撑"更稳定"，但稳定性提升没有转化为安全指标，也没有任何 seed / 方差。
- **"动量"实际只有一帧记忆**：Table 7 显示 $t=2$（前一帧）最好，$t=3$（前两帧）反而变差。这与"long-horizon"的表述有张力——机制上它是单步惯性，不是长时记忆；作者归因于历史不确定性，但没有进一步验证。
- **TTM 与最终选择脱钩**：TTM 只负责挑一条"最连续"的轨迹交给 MPI，最终输出仍按多模态分数选。因此"保证连续"并没有硬约束，只是改变了 query 的初始化；论文没有报告被选中轨迹与最终输出轨迹不一致的比例，也没有做"直接用 Hausdorff 选轨迹输出"的对照。
- **TPC 作为优化目标有方向性问题**：它奖励"与上一帧相似"，而保持上一帧轨迹在机动切换（例如前车急刹、路口需要改道）时恰恰是错的。论文没有在"必须改变决策"的场景上单独统计 TPC，因此该指标可能把"该变不变"也算作好。
- **Turning-nuScenes 太小**：17 个 scene、680 样本，且筛选规则只用 x 方向位移阈值 25，没有按机动类型分层；在这个规模上 0.08 个百分点的碰撞率差异不足以支撑强结论。
- **与 SparseDrive 的比较不完全同源**：Bench2Drive 表里 SparseDrive 是作者复现（16.71），而原文报的是另一数字；开环表里部分行用官方 checkpoint 重评（†/*），协议混用，跨表读数容易误判。
- **去噪噪声强度是窄窗口超参**：0.1 最优、0.2 起退化，但论文没有解释这个窗口与检测器噪声水平的关系，换数据集或换检测器时需要重新扫。
- **感知增益与规划增益的归因未拆开**：ED 模块同时提升了检测、跟踪、建图、运动与规划，最终规划提升里有多少来自"感知变好"、多少来自"动量规划"，消融没有做"固定感知、只换规划"的对照。
- **没有真实闭环或跨城市验证**：Bench2Drive 是 CARLA 仿真，nuScenes 是开环；结论是否成立于真实交互场景未知。

## Notes

+ **论文定位**：不是新的感知或新的生成头，而是给"多模态候选 + 最高分选择"这个范式补一条**跨帧一致性约束**。它属于 memory 在规划里的最轻量形态——不建外部记忆库、不做检索，只把上一帧的 planning query 当作隐式记忆，一步传递。

+ **可迁移点**：
  + **Hausdorff 匹配作为轨迹连续性先验**：比逐点 L2 更能反映"整体路径是否换了机动"，且不需要训练；任何"每帧独立选候选"的规划器都可以加这一层做后处理或 query 初始化。
  + **分数门控 + LSTM 的历史 query 演化**：把"上一帧哪些候选被看好"编码进 query，比直接 concat 历史特征更省参数，也更容易与现有 PlanHead 对接。
  + **训练期加噪 + 轻量去噪模块**：对稀疏实例特征这类"检测器输出"的输入，加噪训练可以显著提升损坏集鲁棒性（nuScenes-C 上规划 +40%），代价只是训练期多一个分支。
  + **TPC 这类跨帧一致性指标**应当与安全指标一起报：只看 L2 会掩盖抖动，只看 TPC 又会掩盖"该变不变"，两者必须联合解读。

+ **待跟进**：
  - **多帧记忆的边界**：为什么两帧历史反而更差？是特征不确定性累积，还是 LSTM 在短序列上过拟合？需要做 2/3/4 帧的逐帧增益分析，而不只是总表。
  - **TTM 与输出的耦合**：把 TTM 选出的轨迹直接输出、或对最终输出加大于某一阈值的 Hausdorff 惩罚，看 TPC 与碰撞率如何权衡。
  - **TPC 的反例场景**：在必须改道/急刹的场景上单独统计 TPC，确认这个指标不会奖励"错误的惯性"。
  - **与检索式记忆的关系**：MomAD 用的是单步 planning-state 记忆；外部场景记忆（街景先验、失败案例库）是另一条线，两者的组合（短期状态惯性 + 长期场景记忆）目前还没有工作做过。
  - **闭环失败分析**：Bench2Drive 上 SR 只有 18.11%，需要按场景类型（cut-in、环岛、无保护左转）拆解，判断动量约束在哪些交互下反而有害。
