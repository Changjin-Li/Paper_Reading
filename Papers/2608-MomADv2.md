---
title: "MomADv2: Reliable Temporal Memory for End-to-End Autonomous Driving"
authors: ["Ziying Song", "Shengkai Zhang", "Lin Liu", "Peiliang Wu", "Lei Yang", "Dongyang Xu", "Bin Sun", "Li Wang", "Shaoqing Xu", "Caiyan Jia", "Yadan Luo"]
institute: ["Yanshan University", "Beijing Jiaotong University", "Nanyang Technological University", "Tsinghua University", "China Automotive Technology and Research Center", "Beijing Institute of Technology", "University of Macau", "The University of Queensland"]
date_publish: 2026-08
venue: arXiv Preprint
tags: [imitation-learning, spatial-memory]
url: https://arxiv.org/abs/2608.23405
cite_key: song2026momadv2
arxiv_id: "2608.23405"
doi: "10.48550/arXiv.2608.23405"
code:
rating: 3
content_scope: full-text
verification_status: source-checked
date_added: 2026-11-26
---
## Summary

MomAD 把上一帧的 planning query 无条件传给当前帧，这在场景突变、时序断裂或**导航命令切换**时会把无效意图带进来。MomADv2 把"是否复用历史"做成一个显式的可靠性判定：维护一个以（命令，时序连续性 token）为键的记忆库，只保留命令一致、时序连续且非空的条目；再用按时间偏移对齐的轨迹距离在每个候选上找到历史对应项，按相似度与时间衰减加权，最后用一个可靠性门控把 SSM 输出的残差裁剪后注入当前 query。轨迹侧另加 Flow-Matching Trajectory Residual Refiner（FM-Ref）：把 planner 输出当作流匹配的起点、专家轨迹当作终点，学一个条件残差速度场，推理只需 2 步 Euler，且用有界门控保证只做修正、不替换 anchor-based planner 的输出。nuScenes 6 s 时域下平均碰撞率从 MomAD 的 0.90% 降到 0.76%（−15.6%）、平均 L2 1.21 m；Bench2Drive 闭环 SR 从 18.11% 提到 24.24%；NAVSIM v1 PDMS 89.9、v2 navtest EPDMS 87.9、navhard 39.5。

## Problem & Motivation

- 长时域规划要求跨帧的规划状态一致，因此 temporal memory 是必需品；MomAD 证明了用历史 planning query 做 momentum 能降低模式不稳定与轨迹抖动。
- 但历史并不总是可靠：动态场景变化、时序不连续（丢帧、跳 scene）、以及**高层导航命令切换**时，历史意图与当前意图不一致，直接复用会产生负迁移。
- 现有做法要么整段复用历史 query，要么只做时序特征建模，缺少"这条历史还该不该用"的判定；这是论文主张的核心缺口。
- 另一个问题是误差累积：anchor-based planner 的候选来自离散 anchor 附近的偏移，长时域上会累积局部偏差，而扩散式生成器虽然能细化，却会破坏 anchor planner 的稳定性。
- 因此需要两个机制：**可靠的历史筛选**（决定读什么）与**受约束的残差细化**（决定改多少）。

## Method

**0. 总体框架**

+ 感知与规划沿用稀疏范式：多视角图像 → 稀疏 scene representation → anchor-based 运动/规划解码器，输出多模态运动预测与初始 ego 候选。
+ 在其上接两个模块：SSM-Q 负责筛选并增强历史 planning query，FM-Ref 负责在候选轨迹上做残差修正；前者读记忆，后者不读记忆。
+ 训练分两阶段：Stage I 按 MomAD 原配方训出完整模型；Stage II 冻结 backbone、感知、运动预测与大部分规划头，只训新增的 SSM 编码器、门控、残差头与流匹配头（10 epoch，lr $3\times10^{-6}$，8×RTX 4090）。

**1. 记忆库的构造与写入策略（附录 A.5）**

+ 记忆条目以（命令，时序连续性 token）为复合键，值存**增强前**的原始 query 与 baseline 候选轨迹：

$$
\mathcal{M}_{t}=\left\{\chi_{t-k}\mapsto\left(\mathbf{Q}^{\mathrm{raw}}_{t-k},\ \mathbf{Y}^{\mathrm{base,cmd}}_{t-k},\ c_{t-k}\right)\right\}_{k=1}^{K}
$$

+ $c_{t-k}\in\{\text{left},\text{straight},\text{right}\}$ 是高层命令，$\chi_{t-k}$ 是帧级连续性标识（只用做索引与硬掩码，不作为特征输入）。
+ 关键设计是**read-before-write**：每帧先读历史、出结果，再把本帧的原始 query 与 baseline 轨迹写入；SSM 增强后的 query 与 flow 修正后的轨迹都不回写。这避免了"修正被反复叠加"的自激漂移。
+ 记忆是 FIFO，容量 $K=4$；每个 sequence 开始、跨 scene、无有效前驱或帧间隔异常时整体重置。

**2. 可靠历史检索（§3.2）**

+ 三类有效性同时满足才允许使用：命令一致、时序连续、历史 query 非空：

$$
m_{t,k}=\mathbb{I}\left[c_{t}=c_{t-k},\ \chi_{t}^{(-k)}=\chi_{t-k},\ \mathbf{Q}^{\mathrm{raw}}_{t-k}\neq\mathbf{0}\right]
$$

+ 命令过滤是硬约束：当前是左转时，只允许检索历史帧中的左转候选，直行/右转候选完全不可见。
+ 同一命令下的候选顺序会随场景变化而漂移，因此不能按固定索引对齐；对每个当前候选 $j$，在该命令的历史候选里找轨迹最一致的一条：

$$
h^{*}_{t,k,j}=\arg\min_{h\in\{1,\dots,M_{c}\}}D_{\mathrm{shift}}\left(\mathbf{Y}^{\mathrm{base,cmd}}_{t-k,h},\ \mathbf{Y}^{\mathrm{base,cmd}}_{t,j}\right)
$$

+ $D_{\mathrm{shift}}$ 是**带时间偏移的轨迹距离**：历史轨迹预测的起点早 $k$ 帧，先丢掉已经流逝的 $s_k=\mathrm{round}(k\Delta_{\mathrm{frame}}/\Delta_{\mathrm{traj}})$ 个 waypoint，再把剩余点变换到当前 ego 坐标系，只在重叠时域上算平均距离；非整数偏移用线性插值。没有这一步，"同一意图"会被误判为不一致。
+ 对应的历史 query 按全局索引取出：$\widehat{\mathbf{Q}}^{\mathrm{hist}}_{t,k,j}=\mathbf{Q}^{\mathrm{raw}}_{t-k}[\mathrm{idx}(c_t,h^*)]$。

**3. 可靠性加权的时序聚合（§3.2）**

+ 历史 query 按有效性、轨迹相似度与时间衰减加权求和：

$$
\bar{\mathbf{Q}}^{\mathrm{hist}}_{t,j}=\sum_{k=1}^{K}w_{t,k,j}\widehat{\mathbf{Q}}^{\mathrm{hist}}_{t,k,j},\qquad w_{t,k,j}=\frac{\phi_{t,k,j}}{\sum_{k'}\phi_{t,k',j}+\epsilon}
$$

$$
\phi_{t,k,j}=m_{t,k}\exp\left(-\frac{d_{t,k,j}}{\alpha}\right)\exp\left(-\frac{k}{\beta}\right)
$$

+ $d_{t,k,j}$ 是对齐后的轨迹距离，$\alpha$ 控制"轨迹差多少就不信"，$\beta$ 控制"多旧就不信"；两个指数项分别对应空间与时间两个维度的衰减。
+ 另有一个可靠性门控，防止"只有一条历史有效"时被单条噪声主导：

$$
r_{t,j}=r^{\mathrm{cnt}}_{t}\cdot r^{\mathrm{dist}}_{t,j},\qquad r^{\mathrm{cnt}}_{t}=\min\left(\frac{\sum_k m_{t,k}}{K_0},1\right),\qquad r^{\mathrm{dist}}_{t,j}=\sigma\left(\frac{d_0-\min_k d_{t,k,j}}{\gamma_d}\right)
$$

+ $r^{\mathrm{cnt}}$ 要求有效历史条数达到 $K_0$ 才给满权重，$r^{\mathrm{dist}}$ 在最近邻轨迹距离小于阈值 $d_0$ 时才打开；两者相乘意味着"历史够多且至少有一条足够像"才会真正影响当前规划。

**4. Selective SSM 编码与残差注入（§3.2）**

+ 把对齐后的历史 query 按时间从远到近与当前 query 拼成序列，送 selective state-space 编码器（LN → causal depthwise conv → 输入相关的 selective scan → SiLU 门控 → 残差），取最后一个 token 作为当前意图：

$$
\mathbf{Z}^{0}_{t,j}=\mathrm{Concat}\left(\widehat{\mathbf{Q}}^{\mathrm{hist}}_{t,K,j},\dots,\widehat{\mathbf{Q}}^{\mathrm{hist}}_{t,1,j},\mathbf{Q}_{t,j}\right),\qquad \mathbf{H}^{\mathrm{cur}}_{t,j}=\mathrm{SSMEncoder}\left(\mathbf{Z}^{0}_{t,j}\right)[-1]
$$

+ 注入方式是**残差而非替换**，并做范数裁剪：

$$
\Delta\mathbf{Q}_{t,j}=\tanh\left(\mathbf{H}^{\mathrm{cur}}_{t,j}-\mathbf{Q}_{t,j}\right),\qquad \mathbf{U}_{t,j}=\mathrm{Clip}\left(g_{t,j}\Delta\mathbf{Q}_{t,j},\ \rho\lVert\mathbf{Q}_{t,j}\rVert_2\right),\qquad \widetilde{\mathbf{Q}}_{t,j}=\mathbf{Q}_{t,j}+\mathbf{U}_{t,j}
$$

+ $g_{t,j}$ 汇总有效历史指示、全局残差门、可靠性门、候选级门与残差尺度；$\rho$ 限制单次扰动幅度。这样即使检索到错误历史，最坏情况也只是有限的 query 偏移。

**5. Flow-Matching 轨迹残差修正（§3.3）**

+ 与从噪声生成轨迹的扩散/流匹配不同，这里把 planner 精炼后的轨迹当起点、专家轨迹当终点，学一个条件残差速度场。先把位移式轨迹累加成绝对轨迹并归一化：

$$
\overline{\mathbf{Y}}^{\mathrm{ref}}_{t}=\mathrm{Cumsum}\left(\mathbf{Y}^{\mathrm{ref}}_{t}\right),\qquad \mathbf{x}_{0}=\frac{\overline{\mathbf{Y}}^{\mathrm{ref}}_{t}}{s},\qquad \mathbf{x}_{1}=\frac{\mathrm{Cumsum}\left(\mathbf{Y}^{\star}_{t}\right)}{s}
$$

+ 在两端之间做线性插值，目标速度就是残差位移：

$$
\mathbf{x}_{\tau}=(1-\tau)\mathbf{x}_{0}+\tau\mathbf{x}_{1},\qquad \mathbf{v}^{\star}=\mathbf{x}_{1}-\mathbf{x}_{0},\qquad \mathcal{L}_{\mathrm{FM}}=\left\|f_{\theta}\left(\widetilde{\mathbf{Q}}_{t},\mathbf{x}_{\tau},\tau\right)-\mathbf{v}^{\star}\right\|_2^2
$$

+ 速度场以增强 query、当前轨迹状态与流时间为条件，先各自投影拼接、过 MLP + LayerNorm + Transformer encoder，再由 MLP 头输出；planner 侧输入做 stop-gradient，避免辅助目标扰动主规划器。
+ 推理从 $\mathbf{x}_0$ 出发做 $S=2$ 步 Euler 积分，再用**有界门控**控制修正强度：

$$
\mathbf{x}_{k+1}=\mathbf{x}_{k}+\frac{1}{S}f_{\theta}\left(\widetilde{\mathbf{Q}}_{t},\mathbf{x}_{k},\tau_{k}\right),\qquad \eta=\eta_{\min}+\eta_{\max}\sigma\left(g_{\mathrm{flow}}\right)
$$

$$
\overline{\mathbf{Y}}^{\mathrm{flow}}_{t}=s\left[\mathbf{x}_{0}+\eta\left(\mathbf{x}_{S}-\mathbf{x}_{0}\right)\right]
$$

+ $\eta$ 的上界保证输出不会离 planner 候选太远——这是"保留 anchor-based 稳定性"的具体实现；最后再用差分把绝对轨迹转回位移式。
+ 总损失：

$$
\mathcal{L}=\mathcal{L}_{\mathrm{plan}}+\mathcal{L}_{\mathrm{SSM}}+\lambda_{\mathrm{flow}}\mathcal{L}_{\mathrm{flow}}+\lambda_{\mathrm{FM}}\mathcal{L}_{\mathrm{FM}},\qquad \mathcal{L}_{\mathrm{flow}}=\lambda_{\Delta}\ell_{\mathrm{reg}}\left(\mathbf{Y}^{\mathrm{flow}}_{t},\mathbf{Y}^{\star}_{t}\right)+\lambda_{\mathrm{cum}}\ell_{\mathrm{reg}}\left(\overline{\mathbf{Y}}^{\mathrm{flow}}_{t},\overline{\mathbf{Y}}^{\star}_{t}\right)
$$

+ 位移式与累积式两套监督同时使用，前者约束逐点位置、后者约束整体形状。

**6. 实现细节**

| 项目 | 设置 |
|:--|:--|
| nuScenes / Adv-nuSc | ResNet-50，256×704，检测半径 55 m，建图 60×30 m，6 个运动模态 |
| Bench2Drive | ResNet-50，6 层 decoder，640×352，900 agent / 100 map / 480 planning query |
| NAVSIM | 沿用 TransFuser 的感知与 ResNet-34 backbone |
| 记忆容量 | $K=4$（FIFO），跨 scene / 丢帧 / 无前驱时重置 |
| 流匹配推理 | $S=2$ 步 Euler；Heun 在同样设置下更差（PDMS 89.0 vs 89.9，FPS 38 vs 43） |
| 训练 | 8×RTX 4090；nuScenes 20 epoch（约 10.3 h）、Bench2Drive 2 epoch（约 49.1 h）、NAVSIM 100 epoch（约 3.0 h）；batch 48 / 64 |
| Stage II | 10 epoch，lr $3\times10^{-6}$，BN 保持 eval 模式 |

## Pipeline

![MomADv2 整体架构](https://arxiv.org/html/2608.23405v2/main_mambav4.png)

*多视角图像编码为检测与地图表示，anchor-based 解码器产出初始 planning query 与命令相关的候选轨迹；SSM-Q 依据命令、时序连续性、轨迹相似度与 buffer 状态筛出可靠历史，按时间顺序送 selective SSM 编码并以有界残差注入当前 query；FM-Ref 再把精炼后的轨迹当作流匹配起点，学一个指向专家轨迹的残差速度场，经 2 步 Euler 与有界门控输出最终轨迹。*

1. **感知与初始规划**：稀疏 scene representation → anchor-based 头输出多模态运动预测与按命令分组的 ego 候选。
2. **读记忆**：取最近 $K=4$ 帧条目，按"命令一致 + 时序连续 + query 非空"做硬过滤。
3. **轨迹对齐检索**：对每个当前候选，丢弃已流逝 waypoint、变换坐标系后算 $D_{\mathrm{shift}}$，找历史最一致的一条并取出其原始 query。
4. **加权聚合**：按相似度与时间衰减算权重，再乘上"有效条数"与"最近邻距离"两个可靠性门。
5. **SSM 编码与注入**：历史 + 当前 query 组成短序列，经 selective SSM 取末位状态，残差经 tanh 与范数裁剪后加回当前 query。
6. **规划精炼**：增强 query 送规划头，输出分类、轨迹回归与规划状态。
7. **残差细化**：轨迹累加成绝对形式并归一化，从 $\mathbf{x}_0$ 出发做 2 步 Euler，得到 $\mathbf{x}_S$ 后按有界 $\eta$ 与原轨迹混合，再转回位移式作为最终输出。
8. **写回**：只把本帧的原始 query 与 baseline 轨迹写入记忆，增强/修正结果不写回。

## Key Results

**1) nuScenes 6 s 长时域规划（Table 1）**

| Method | L2 Avg (m)↓ | Col. Avg (%)↓ | L2 6s (m)↓ | Col. 6s (%)↓ |
|:--|--:|--:|--:|--:|
| UniAD | 1.70 | 1.06 | 3.07 | 2.51 |
| SparseDrive | 1.59 | 0.97 | 2.95 | 2.33 |
| MomAD | 1.42 | 0.90 | 2.45 | 2.13 |
| LAW | 1.46 | 0.96 | 2.61 | 2.31 |
| DIVER | 1.37 | 0.87 | 2.49 | 2.11 |
| **MomADv2** | **1.21** | **0.76** | **2.40** | **2.03** |

+ 相对最强 prior（DIVER）平均 L2 好 11.7%、平均碰撞率好 11.6%；相对 MomAD 的 0.90% → 0.76% 即摘要中的 −15.6%。
+ 增益主要来自前 3 s（1 s 碰撞率 0.00%），6 s 端点的改善幅度明显收窄（2.13 → 2.03），说明记忆能压住中段抖动，但对最远端的误差累积帮助有限。

**2) Bench2Drive 闭环（Table 2）**

| Method | Avg. L2↓ | DS↑ | SR (%)↑ | Effi↑ | Comf↑ |
|:--|--:|--:|--:|--:|--:|
| MomAD | 0.82 | 47.91 | 18.11 | 174.91 | 51.20 |
| DIVER | 1.13 | 47.95 | 19.47 | 164.66 | 51.28 |
| **MomADv2** | **0.76** | **52.32** | **24.24** | 169.09 | **55.11** |
| MomADv2*（专家特征蒸馏） | 0.77 | 78.82 | 46.50 | 83.41 | 31.23 |

+ 不带蒸馏时 SR 24.24%，相对 MomAD +6.13 个百分点；DS 52.32。
+ 带 * 的 78.82 / 46.50 是加了训练期专家特征蒸馏的版本，与上面几行不是同一训练配方，不能直接当作方法本身的增益。

**3) NAVSIM 三套协议**

| Split | NC↑ | DAC↑ | TTC↑ | EP↑ | 总分 |
|:--|--:|--:|--:|--:|--:|
| NAVSIM v1 navtest（PDMS） | 98.7 | 96.8 | 95.4 | 84.5 | **89.9** |
| NAVSIM v2 navhard（EPDMS，Stage-2） | 85.9 | 87.0 | 83.8 | 77.2 | **39.5** |
| NAVSIM v2 navtest（EPDMS） | 98.3 | 97.4 | 97.8 | 89.6 | **87.9** |

+ v1 上 PDMS 89.9 与 DriveSuprim 并列表内最高，NC / TTC 为表内最高；v2 navtest 上 EPDMS 87.9 为表内最高，DDC、EP、EC 三项第一。
+ navhard 的 39.5 明显高于 TransFuser（23.1）、DiffusionDrive（24.2）与 GuideFlow（27.1），但这两套结果分别用 ResNet-34（NAVSIM）与 ResNet-50（nuScenes）训练，跨表不可比。

**4) 模块消融（Table 6、12）**

| SSM-Q | FM-Ref | PDMS↑（NAVSIM v1） |
|:--:|:--:|--:|
| × | × | 84.0 |
| ✓ | × | 86.9 |
| ✓ | ✓ | **89.9** |

+ 单加 SSM-Q 涨 2.9 PDMS，再加 FM-Ref 涨 3.0，两个模块的贡献接近，作者据此主张互补。
+ 附录 Table 12 在 nuScenes 上给出同方向的结果：加 FM-Ref 后 4/5/6 s 的 L2 为 1.32 / 1.82 / 2.40 m、碰撞率 0.72% / 1.33% / 2.03%，但论文只给了"逐项都不差"的定性描述，没有列未加 FM-Ref 的同一套数字。
+ FM-Ref 是一个额外参数量的回归式残差头，"涨 3.0"里有多少来自 flow 参数化、多少来自"多了一个细化模块"，没有做"残差 MLP 无 flow"的对照。

**5) 记忆策略消融（Table 7，nuScenes，均带 FM-Ref）**

| 记忆策略 | 平均碰撞率 (%)↓ |
|:--|--:|
| 无历史记忆 | 0.92 |
| 朴素历史融合（naive） | 1.01 |
| 命令感知记忆 | 0.89 |
| + 时序连续性过滤 | 0.83 |
| + 轨迹级对齐 | 0.79 |
| **完整选择性记忆** | **0.76** |

+ **最有信息量的一行是"朴素融合 1.01 比无记忆 0.92 更差"**：无条件复用历史会带来负迁移，这正是论文动机的直接证据。
+ 之后每一步只降 0.03–0.06 个百分点，绝对量很小，且没有任何方差；在 0.76–1.01 这个量级上，单次运行的稳定性未知。

**6) 记忆长度与序列建模（Table 8、13）**

| 维度 | 取值 → 结果 |
|:--|:--|
| 记忆长度 $K$ | 1 → 88.2；2 → 89.1；**4 → 89.9**；6 → 88.5；8 → 87.7（PDMS） |
| 时序建模 | Concat+MLP 84.2 / 51.0 FPS；GRU 86.4 / 41.9；LSTM 86.7 / 39.6；Transformer 87.2 / 33.0；Vanilla SSM 87.0 / 43.6；**Selective SSM 87.9 / 43.0**（EPDMS / FPS） |

+ $K=4$ 最好、再长就退化：说明"可靠记忆"的有效窗口很短，这与标题里的 long-horizon 有张力——机制上它记的是约 2 s 的 planning state。
+ Selective SSM 比 Transformer 高 0.7 EPDMS 且快 30%（43.0 vs 33.0 FPS），这是论文选择 SSM 而非注意力的主要实证依据。

**7) 其他结果**

+ 轨迹一致性（附录，TPC）：4 s 1.04 m（−12.6%）、5 s 1.32 m（−9.0%）、6 s 1.61 m（与最好 baseline 持平），平均 −6.6%。
+ 3 s 短时域（附录）：平均碰撞率 0.07%（MomAD 0.09%），1 s 零碰撞，平均 L2 0.54 m。
+ 求解器（Table 14）：Euler 89.9 PDMS / 43 FPS，优于 Heun 的 89.0 / 38——2 步 Euler 已够用，再加步数只是增加延迟。

## Strengths & Weaknesses

**Strengths**

- 问题选得准：把"历史什么时候不可信"拆成可实现的三个判据（命令一致、时序连续、轨迹相似），而不是笼统地说"要做选择性记忆"。
- **朴素融合 1.01 差于无记忆 0.92** 这一行是全文最强证据：它直接证明"无条件复用历史会负迁移"，也让后续每一层过滤的增益有了参照。
- 时间偏移对齐处理得细：丢掉已流逝 waypoint、变换 ego 坐标系、非整数偏移做插值，避免了"同一意图因时间原点不同被误判为不一致"这个常见坑。
- 门控体系完整且各有明确失效模式：有效条数门管"历史不够"，距离门管"没有一条像"，范数裁剪管"即使检索错也只偏移有限量"。
- FM-Ref 的定位清楚：不是从噪声生成轨迹，而是从 planner 输出做残差运输；有界门控 $\eta$ 保证输出不会离开 anchor 候选太远，这个"只修正、不替换"的设计对工程部署友好。
- 选型有代价证据：Selective SSM 比 Transformer 高 0.7 EPDMS、快 30%；Euler 优于 Heun。这两组对比让架构与求解器的选择不是凭直觉。
- 做了三套 benchmark（nuScenes 开环、Bench2Drive 闭环、NAVSIM 伪闭环），比只报单一指标更完整。

**Weaknesses**

- **"long-horizon"与技术机制不匹配**：$K=4$ 最好、$K=6/8$ 退化，记忆实际覆盖约 2 s 的 planning state；论文标题与动机都指向长时域，但没有解释为什么更长的历史反而有害，也没有做逐帧贡献分析。
- **增益集中在很小的绝对量**：nuScenes 平均碰撞率 0.90 → 0.76 个百分点，Table 7 各档之间只差 0.03–0.06；全文无 seed、无标准差、无置信区间，这些差异是否超出训练噪声无法判断。
- **依赖预定义命令**：记忆检索以 $c_t\in\{\text{left},\text{straight},\text{right}\}$ 为硬键，命令本身错了（或没有导航输入）时，整条检索链路失效；作者在 limitation 里承认，但这是方法可用性的核心前提。
- **FM-Ref 的归因不清**：消融只做了"有无 FM-Ref"，没有"同等参数的残差 MLP 但不做 flow"的对照；因此 3.0 PDMS 的增益无法区分"flow 参数化的好处"与"多一个细化头的好处"。
- **Table 7 的所有变体都带 FM-Ref**：记忆策略的比较是在同一个强细化头上做的，若换成不带 FM-Ref 的配置，过滤各层的相对重要性可能不同。
- **Bench2Drive 的 78.82 与 24.24 是两个不同配方**：带专家特征蒸馏的版本 DS/SR 高得多，但 Comfort 只有 31.23，与不带蒸馏版本的 55.11 差距巨大；论文把它们放在同一张表里，容易被读成同一方法的两个数字。
- **跨 benchmark 的 backbone 不一致**：nuScenes 用 ResNet-50、NAVSIM 沿用 TransFuser 的 ResNet-34，三套协议的数字不可横向比较；"NAVSIM v1 89.9 并列最高"只在表内成立，同期 PlanWAM 报 93.8、RiSE 报 94.2。
- **推理开销只给了相对 FPS**：Table 13 是记忆模块的吞吐（43.0 vs 51.0 FPS），但没有给完整模型（含 FM-Ref 的 2 步 Euler）相对 MomAD 的延迟与显存；对一个"实时规划"方法，这是关键缺失。
- **记忆库不跨 scene**：跨 scene 即重置，因此方法只处理短期状态惯性，无法利用"同一路段的历史经验"；这与外部场景记忆（街景先验、失败案例库）是两条不同的线，论文没有讨论二者关系。
- **失败模式未统计**：作者提到 baseline 候选不准时对齐会失效，但没有给出"检索被全部拒绝"的比例，也没有按场景类型拆解闭环失败。

## Notes

+ **论文定位**：不是新的感知或生成架构，而是把 MomAD 的"单步 planning-state 记忆"升级成**带可靠性判定的短期记忆**。它属于 memory 在规划里最轻的一档：不建外部库、不做语义检索，只在最近 4 帧的 planning query 上做筛选与状态空间演化。

+ **可迁移点**：
  + **"先判断该不该用，再决定怎么融合"** 是比"设计更强的融合模块"更根本的一步；Table 7 里朴素融合差于无记忆，说明任何 memory 模块都应当先做有效性过滤。
  + **时间偏移对齐**是跨帧轨迹比较的必要预处理：丢掉已流逝 waypoint、变换坐标系、非整数偏移插值，这套操作可以直接复用到任何"跨帧候选一致性"或"跨帧损失"里。
  + **有效条数门 + 距离门 + 范数裁剪**三层保护可以套到任何"外部信号注入主干"的模块上：条数门防单条噪声主导，距离门防语义不符，裁剪防最坏情况失控。
  + **有界残差细化**（flow 学残差、$\eta$ 限幅、只修正不替换）是一个适合工程部署的折中：既拿到生成式细化的收益，又不破坏 anchor planner 的稳定性。
  + **Selective SSM 用于短序列**（$K\le4$）在精度与吞吐上都优于 Transformer（87.9 vs 87.2 EPDMS、43.0 vs 33.0 FPS），这个结论对"短历史 + 实时"的场景有直接参考价值。

+ **待跟进**：
  - **为什么 $K>4$ 会退化**：是旧 query 的语义漂移、还是可靠性门在长窗口下误判？需要逐帧贡献分析（分别只用第 1/2/3/4 帧历史）。
  - **命令错误的鲁棒性**：给命令注入随机翻转，统计记忆检索被拒绝的比例与碰撞率变化，判断方法对导航输入的依赖强度。
  - **FM-Ref 的归因**：加一组"同参数量的残差 MLP（无 flow、无 Euler）"对照，把"生成式参数化"的净收益从"多一个细化头"里剥离出来。
  - **完整推理开销**：报出含 SSM-Q + 2 步 Euler 的端到端延迟、显存与吞吐，而不只是记忆模块的 FPS。
  - **与外部记忆的结合**：短期 planning-state 记忆（本文）与长期场景记忆（街景先验、失败案例检索）能否叠在同一条规划管线里，目前没有工作验证。
  - **闭环失败拆解**：按 cut-in、无保护左转、环岛等场景统计 SR，看"可靠记忆"在哪些交互下反而保守过度。
