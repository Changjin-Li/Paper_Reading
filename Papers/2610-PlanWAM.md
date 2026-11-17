---
title: "PlanWAM: Planning-Shaped Future Representations for End-to-End Autonomous Driving"
authors: ["Jinchang Xu", "Hongda Yu", "Fengwei Dong", "Wenhui Huang", "Xi Wei", "Yongzhi Liu", "Sunan Zhang", "Jirao Wang", "Chen Lv", "Bingbing Li", "Guodong Yin", "Weichao Zhuang"]
institute: ["Southeast University", "NIO", "Nanyang Technological University", "Tongji University"]
date_publish: 2026-10
venue: arXiv Preprint
tags: [world-model, imitation-learning]
url: https://arxiv.org/abs/2610.11382
arxiv_id: "2610.11382"
doi: "10.48550/arXiv.2610.11382"
code:
rating: 4
content_scope: full-text
date_added: 2026-11-17
---

## Summary

现有驾驶 world model 都在研究"怎么预测未来"和"怎么用未来"，却很少有人问**哪种未来表示对规划真正有用**——用像素重建或 DINO/JEPA/BEV 特征定义的未来潜空间，与轨迹决策目标之间存在错配。PlanWAM 把未来建模重述为 **planning-oriented predictive representation learning**：训练时用一个只看得到 GT 未来帧的 privileged posterior，把未来潜表示直接挂在轨迹生成与轨迹选择的损失上，使它被规划目标"塑形"；再用 Hindsight-to-Foresight Distillation 把这份 privileged 未来潜表示蒸馏给只看历史的 prior，推理时只保留 prior。历史侧用 Temporal Register Pyramid 按时间远近分配 query 预算（越近越多），压成 128-token 的紧凑历史表示；预测出的 256-token 未来潜表示与历史拼成 384-token planning memory，同时驱动 64 条候选轨迹的生成与打分。**NAVSIM-v1 navtest 93.8 PDMS、NAVSIM-v2 90.9 EPDMS、HUGSIM 零样本闭环 38.7 HD-Score / 53.7 RC**。

## Problem & Motivation

- 端到端规划器通常直接建模 $p(\tau_{1:T}\mid v_{t-n:t},c_{t-n:t})$，只用当前和历史观测决策，缺乏对场景未来演化的预判；world model 的价值就在于插入一个未来状态表示 $z_t^+$，把过程分解为 $p(z_t^+\mid v,c)\,p(\tau\mid v,c,z_t^+)$。
- 但这条分解式把真正难的问题留空了：**$z_t^+$ 应该怎么定义、怎么构造，才对规划最有利？** 论文主张这是比"预测得更准"更基础的问题。
- 现有三代做法各有缺口：①像素/视频/稠密场景状态重建（Epona、PWM）要建模大量与规划弱相关的视觉细节，生成代价高；②潜空间预测（LAW、Latent-WAM）把未来表示当作预测目标或辅助监督，对规划的影响是间接的；③future-conditioned planning（DriveFuture、WorldDrive）把预测的未来潜显式喂进 planner，但表示本身仍由重建或通用图像潜目标定义。
- 关键错配：**能重建场景的未来表示，不一定保留了影响轨迹决策的信息**。未来表示的定义权在重建目标或通用预测目标手里，而不在规划目标手里。
- 由此提出重述：world model 不仅要预测未来，还要学习**哪些未来信息值得被预测**。规划定义"什么未来信息重要"，未来预测反过来决定"哪些历史信息该被保留"，被塑造出的未来表示再反过来改善规划。

## Method

**0. 总体框架**

- 三个模块串成一条链：历史压缩 $h_t=E_\eta(v_{t-n:t},c_{t-n:t})$ → 未来潜表示（训练时 privileged posterior $P_\phi$，推理时 history-only prior $P_\theta$）→ 以 $[h_t;\,\hat z_t^+]$ 为 planning memory 的轨迹生成与选择。
- posterior 只在训练阶段存在，它做的事情是**定义未来潜空间的坐标系**：因为 $z_t^+$ 唯一的监督来自 $L_{\text{plan}}+L_{\text{score}}$，潜变量的有限容量被迫只保留影响轨迹生成与打分的未来信息，这就是 "planning-shaped" 的含义。
- 推理阶段 posterior 与未来帧全部移除，规划被写成 $p(\tau_{1:T},\hat z_t^+\mid v,c)=p_\theta(\hat z_t^+\mid h_t)\,p_\theta(\tau_{1:T}\mid h_t,\hat z_t^+)$。

**1. Temporal Register Pyramid：近端优先的历史压缩（§3.2）**

- 共享视觉编码器对每个相机帧提取 16 个 scene registers：$x_i^k=F_{\text{vis}}(v_i^k)$，$r_i^k=F_{\text{reg}}(x_i^k)$；ego state 与导航信息经 MLP 编码为 $e_i=F_{\text{ego}}(c_i)$。4 帧 × 4 相机 × 16 registers = 256 个原始 token。
- 把历史按时间距离划成 $L=4$ 个层级 $\{G_l\}$，每层分配不同的可学习 query 预算：越近的层 query 越多，越远的层压得越狠。默认从远到近为 $[4,4,8,16]$ 个 query/相机，输出 $4\times(4+4+8+16)=128$ 个历史 token。
- 每一层的 query 先注入结构信息与当前场景上下文，再做 Q-Former 压缩：

$$
\tilde Q^{(l)}=Q^{(l)}+E_{\text{time}}^{(l)}+E_{\text{cam}}^{(l)}+E_{\text{level}}^{(l)}+E_{\text{ego}}^{(l)}+\bar r_t,\qquad h_t^{(l)}=\text{QFormer}\big(Q=\tilde Q^{(l)},\,K=V=R^{(l)}\big)
$$

- 这里 $R^{(l)}$ 是该层级所有相机-帧的 registers 拼接，$\bar r_t$ 是当前时刻场景 registers 聚合出的共享上下文；式中的 ego 注入让 query 能按自车状态自适应地选择历史证据。
- **刻意不加历史重建目标**：$h_t$ 只由后续的 $L_{\text{latent}}$ 与 $L_{\text{task}}$ 联合优化，因此被保留的是"对预测未来和规划有用"的动态线索，而不是能复原画面的纹理。

**2. Planning-shaped future posterior：让规划损失塑形未来潜空间（§3.3）**

- 未来帧走同一套 scene register 提取，加未来时间嵌入后排成 query，而不是用随机初始化的可学习 query：

$$
f_t^+=\text{Concat}_{i,k}\big[r_i^k+E_{\text{time}}^{\text{fut}}(i-t)\big],\qquad i\in\{t+1,\dots,t+T\}
$$

- posterior world model 以 GT 未来 registers 为 Q、历史表示为 K/V，聚合出未来潜表示：$z_t^+=P_\phi(h_t,f_t^+)=\text{WM}_\phi^{\text{post}}(Q=f_t^+,K=V=h_t)$。默认 4 个未来时刻（1–4 s）× 4 相机 × 16 registers = 256 个 token。
- 未来潜表示**不接任何重建损失**，而是加 type embedding 后与历史拼接成 planning context，同时送进轨迹生成器与轨迹打分器：

$$
m_t^+=\text{Concat}\big(h_t+E_{\text{type}}^{\text{hist}},\; z_t^++E_{\text{type}}^{\text{fut}}\big),\qquad \tau_{1:T}^+=G_{\text{plan}}(m_t^+),\quad s^+=G_{\text{score}}(\tau_{1:T}^+,m_t^+)
$$

- 监督就是 $L_{\text{post}}=L_{\text{plan}}+L_{\text{score}}$。规划与打分的梯度可以穿过 planner / scorer 回传到 $z_t^+$，因此潜空间被"规划目标塑形"——这也是该方法与用重建目标定义未来表示的本质区别。

**3. Hindsight-to-Foresight Distillation：把 privileged 未来蒸馏成可部署的 foresight（§3.4）**

- prior 与 posterior 共享 world-model 结构，但 query 换成可学习的 future queries，并配上与 posterior **完全同构**的时间 / 相机 / register-slot 嵌入，保证 token 一一对应：

$$
Q_{i,k,j}^{\text{prior}}=Q_{0,i,k,j}+E_{\text{time}}^{\text{prior}}(i-t)+E_{\text{cam}}(k)+E_{\text{slot}}(j),\qquad \hat z_t^+=P_\theta(h_t)=\text{WM}_\theta^{\text{prior}}(Q=Q^{\text{prior}},K=V=h_t)
$$

- 蒸馏损失是逐 token 的归一化 Frobenius 距离，posterior 一侧 stop-gradient，posterior 作为冻结 teacher，只让 prior 去逼近：

$$
L_{\text{latent}}=\frac{1}{N_z d}\big\|\hat z_t^+-\text{sg}(z_t^+)\big\|_F^2
$$

- 蒸馏的可行性依赖 token 对齐：prior 的 256 个 query 与 posterior 的 256 个未来 slot 在时间-相机-register 三个维度上按同一顺序排列，所以 $L_{\text{latent}}$ 是一次逐位置比较，而不是需要额外匹配器的集合对齐。
- 得到 $\hat z_t^+$ 后走与 posterior 相同的拼接路径（$m_t^-$），进入同一个 planner / scorer，构成 Foresight-Conditioned Planning。

**4. 轨迹生成器与打分器（附录 A.3–A.4）**

- 生成器：以 64 个 planner query 为 Q，对 384-token planning memory 做 4 层 cross-attention 解码，再由 MLP（256→1024→24）解成 8 个 waypoint 的 $(x,y,\psi)$，得到 $K=64$ 条候选。
- 打分器：候选轨迹编码为 Q，对同一份 planning memory 做 4 层解码并注入 ego status token，六个 MLP 头分别输出 NC / DAC / TTC / EP / DDC / comfort 六个子分数。
- NAVSIM-v1 的选轨规则沿用 PDMS 的乘性结构，DDC 参与训练但在该排序中权重为零：

$$
\log S_k=\log p_k^{\text{NC}}+\log p_k^{\text{DAC}}+\log 5p_k^{\text{TTC}}+5p_k^{\text{EP}}+2p_k^{\text{C}},\qquad k^*=\arg\max_k\,\log S_k
$$

- 打分路径上候选轨迹的几何被 detach，打分损失只更新打分器，不回传轨迹解码器。

**5. 两阶段训练（§3.5、附录 B）**

- Stage 1：只训 posterior 与共享规划模块，目标是 $L_{\text{task}}=L_{\text{plan}}+L_{\text{score}}+\lambda_{\text{reg}}L_{\text{reg}}$，其中 $L_{\text{reg}}$ 为两个防塌缩项——token 范数项 $L_{\text{norm}}$ 惩罚范数趋零，多样性项 $L_{\text{div}}$ 惩罚 slot 之间余弦相似度过高，权重均为 $10^{-4}$。
- $L_{\text{plan}}$ 是 best-of-K 的 L1 回归：每个样本只有离 GT 最近的候选拿到直接监督；$L_{\text{score}}$ 用官方 proposal simulator 给出的六项子分数做 BCEWithLogits，NC/DDC 的 0.5 中间值按保守负类处理，TTC 未定义值做掩码而非当负样本。
- Stage 2：冻结 posterior teacher，用其 checkpoint 初始化 prior 中结构兼容的参数（prior 的 query slot 仍是独立参数），以更低学习率优化 $L_{\text{train}}=L_{\text{task}}+\lambda_{\text{latent}}L_{\text{latent}}$，$\lambda_{\text{latent}}=0.05$；蒸馏损失在 FP32 下计算。
- 关键点：Stage 2 里 planner 与 scorer **不能冻结**——它们是为 posterior 潜空间学到的权重，prior 必须为自己的潜表示重新学下游读取方式（附录 D.4 用消融证实）。

**6. 实现细节**

| 项 | 设置 |
|:--|:--|
| 输入 | 4 帧历史 × 4 相机（前/左/右/后），1148×672 |
| 训练期未来观测 | {1,2,3,4} s，同四个相机，仅 posterior teacher 可见 |
| 视觉 backbone | DINOv2 ViT-S，base 冻结 + rank-32 LoRA，每帧 16 个 scene registers |
| 历史表示 | $h_t$ 128 token（[4,4,8,16] query/相机） |
| 未来潜表示 | $N_z=256$ token，宽度 $d=256$ |
| Planning memory | 128 + 256 = 384 token |
| 候选轨迹 | $K=64$，8 个 waypoint、0.5 s 间隔、4 s 时域 |
| Transformer 配置 | TRP 2 层 / posterior 4 层 / prior 4 层 / planner 4 层 / scorer 4 层，宽度 256、FFN 1024、4 头（planner/scorer 为 1 头） |
| 优化 | AdamW，两阶段各 50 epoch，warm-up 10% 后 cosine 衰减；peak lr $1\times10^{-4}$ / $1\times10^{-5}$ |
| 算力 | 16×A800，8 样本/卡，global batch 128 |
| 损失权重 | $\lambda_{\text{norm}}=\lambda_{\text{div}}=10^{-4}$，$\lambda_{\text{latent}}=0.05$ |

## Pipeline

![PlanWAM 整体框架](https://arxiv.org/html/2610.11382v1/figs/framework1.png)

*历史多视角帧经 Temporal Register Pyramid 按时间远近压缩成紧凑历史表示；训练时 privileged posterior 把 GT 未来帧的 scene registers 当作 query，聚合出未来潜表示，并由轨迹生成与轨迹选择损失塑形；Hindsight-to-Foresight Distillation 把这份潜表示逐 token 蒸馏给只看历史的 prior；推理时仅保留 prior，预测出的未来潜与历史拼成 planning memory，驱动 64 条候选轨迹的生成与打分选轨。*

1. **编码**：4 帧 × 4 相机图像经冻结的 DINOv2 ViT-S（LoRA）得到 256 个 scene registers；ego 状态与导航经 MLP 编码。
2. **历史压缩**：按时间距离分 4 层，query 预算 [4,4,8,16]/相机，注入时间、相机、层级、ego 与当前场景上下文后做 Q-Former 压缩，得到 128-token $h_t$。
3. **训练分支 A（Stage 1）**：GT 未来 4 帧的 registers 加未来时间嵌入形成 256 个 query，对 $h_t$ 做 cross-attention 得 $z_t^+$；$[h_t;z_t^+]$ 送 planner 与 scorer，损失直接塑形 $z_t^+$。
4. **训练分支 B（Stage 2）**：posterior 冻结为 teacher；prior 用同布局的可学习 query 从 $h_t$ 预测 $\hat z_t^+$，逐 token 对齐 teacher（stop-gradient），同时用规划与打分损失继续训练 planner / scorer / TRP / LoRA。
5. **推理**：丢弃 posterior 与未来帧，只算 $h_t$ → prior → $\hat z_t^+$ → 384-token planning memory → 64 条候选 → 六项子分数 → 按 $\log S$ 选一条轨迹。
6. **闭环部署**：NAVSIM 训练完的模型不做任何 HUGSIM 适配，直接在 nuScenes / KITTI-360 / Waymo / PandaSet 场景上闭环滚动。

## Key Results

**1) NAVSIM-v1 navtest（Table 2，PDMS）**

| Method | NC↑ | DAC↑ | TTC↑ | C↑ | EP↑ | PDMS↑ |
|:--|--:|--:|--:|--:|--:|--:|
| Human Agent | 100 | 100 | 100 | 99.9 | 87.5 | 94.8 |
| iPad | 98.6 | 98.3 | 94.9 | 100 | 88.0 | 91.7 |
| DrivoR | 98.9 | 98.3 | 96.2 | 100 | 89.1 | 93.1 |
| DriveFuture（world-model 最强） | 98.8 | 99.1 | 95.4 | 100 | 84.2 | 90.7 |
| **PlanWAM** | 98.9 | 98.8 | 96.3 | 100 | **91.0** | **93.8** |

- 相比 DrivoR +0.7，相比最强 world-model planner DriveFuture +3.1；但离 Human 94.8 仍有 1.0 的差距。
- 优势主要来自 EP（91.0，唯一超过 Human 87.5 的强项）与 TTC；DAC 98.8 仍低于 DriveFuture 的 99.1。

**2) NAVSIM-v2 navtest（Table 1，EPDMS）**

| Method | NC↑ | DAC↑ | DDC↑ | TTC↑ | EP↑ | EPDMS↑ |
|:--|--:|--:|--:|--:|--:|--:|
| Human Agent | 100.0 | 100.0 | 99.8 | 100.0 | 87.4 | 90.3 |
| MeanFuser | 98.3 | 97.2 | 99.6 | 97.4 | 87.6 | 89.5 |
| Latent-WAM | 98.1 | 97.3 | 99.6 | 97.3 | 87.7 | 89.3 |
| DriveFuture | 98.8 | 99.1 | 99.6 | 98.4 | 86.6 | 89.9 |
| **PlanWAM** | **99.0** | 99.2 | 99.8 | 98.5 | **90.8** | **90.9** |

- 首次在学习方法中超过 Human Agent（90.9 vs 90.3），NC / TTC 两项在学习方法中最高。
- 但 EC（Extended Comfort）只有 70.5，低于表内绝大多数方法（仅高于 DriveVLA-W0 的 58.9），说明时间一致性上的收益没有跟上安全指标。

**3) HUGSIM 零样本闭环（Table 3，RC / HD-Score）**

| Method | RC Avg.↑ | HD-Score Avg.↑ |
|:--|--:|--:|
| UniAD | 40.6 | 28.9 |
| LTF | 41.4 | 24.8 |
| Latent-WAM | 45.9 | 28.9 |
| **PlanWAM** | **53.7** | **38.7** |

- 零样本跨 KITTI-360 / PandaSet / nuScenes / Waymo 四个域，无任何模拟器微调；RC 比最强 baseline 高 7.8、HD-Score 高 9.8。
- 增益集中在 Hard / Extreme：HD-Score 35.6 / 22.0，说明预见性收益主要体现在需要提前决策的困难场景；Easy 只有 75.3 RC，落后于 Latent-WAM 的 84.2。

**4) 组件与设计消融（Table 4–8）**

| 设置 | PDMS↑ |
|:--|--:|
| 逐步加 TRP → H2F → FCP | 93.1 → 93.4 → 93.5 → **93.8** |
| 未来表示：无未来 / 通用特征重建 / planning-shaped | 93.4 / 93.5 / **93.8** |
| 未来潜用法：仅辅助监督 / 仅生成 / 仅选择 / 生成+选择 | 93.5 / 93.5 / 93.7 / **93.8** |
| 历史分配：[8,8,8,8] / [16,8,4,4] / **[4,4,8,16]** / [2,4,8,16] | 93.4 / 92.9 / **93.8** / 93.6 |
| 蒸馏：prior-only / +posterior init / +H2F / **init+H2F** | 93.4 / 93.5 / 93.7 / **93.8** |

- **最关键的一行是"通用特征重建 93.5 vs planning-shaped 93.8"**：同样大小的未来潜、同样的蒸馏流程，只换监督目标，说明增益来自"被规划目标塑形"而不是"多了一个预测任务"。
- 未来潜的用法比"有没有"更重要：仅当辅助监督或仅做生成条件都停在 93.5，用于**选轨**才能到 93.7——收益主要发生在候选比较环节。
- 历史 token 数不是瓶颈：把预算扩到 256 token 反而只有 93.6，[4,4,8,16] 的 128 token 最好，且 342 ms 延迟低于 256-token 的 380 ms。
- 蒸馏把 prior 与 posterior 的平均余弦相似度从 0.22（仅 init 0.46）拉到 0.93，对应 PDMS 93.7 → 93.8；$L_{\text{latent}}$ 单独加到 0.96 相似度反而掉到 93.0（附录 D.4），说明"对齐得更紧"不等于"规划得更好"。
- Oracle 行：直接使用 posterior（看得见 GT 未来）是 95.0 PDMS，与部署的 93.8 之间有 1.2 点差距，这就是"hindsight → foresight"这一步的信息损失上限。

**5) 失败类型分析（§4.4）**

- PDMS=0 的失败场景从 314 降到 237，其中**轨迹选择错误减少 87 例**，是最大来源，与"未来潜主要用于选轨"的消融结论一致。
- 安全相关失败大幅下降：NC −62.3%、TTC −52.4%、DDC −66.7%；但 DAC 失败略有上升，作者解释为收益集中在动态场景理解与决策，而非静态可行驶区域判断。

## Strengths & Weaknesses

**Strengths**

- **问题提法本身是贡献**：把"未来表示该怎么定义"从重建目标和通用预测目标手里拿回来，交给规划损失来定义。这个提法解释得通——既然下游只需要可比较的未来信息，就没有必要让潜空间承担重建任务。
- **训练-推理解耦干净**：privileged posterior 只在训练期存在，部署时整条未来观测链路删除；与需要推理时想象未来的 WAM 相比，这套方案没有额外的推理期生成代价。
- **token 级对齐让蒸馏变简单**：prior 与 posterior 共用时间-相机-slot 的 query 布局，$L_{\text{latent}}$ 是逐位置比较，不需要 projector、匹配器或对抗训练。这是"表示迁移"里少见的低成本实现。
- **消融直击主张**：Table 5 用同规模未来潜、同蒸馏流程只换监督目标（93.5 vs 93.8），Table 6 换使用位置（93.5/93.5/93.7/93.8），两个消融合起来把"planning-shaped"和"用于选轨"这两件事分别隔离出来。
- **诚实地报告了对齐-性能的非单调关系**：相似度 0.95 时 PDMS 掉到 93.7，附录 D.4 进一步显示只对齐不规划只有 93.0。这个反例比增益本身更有信息量，说明蒸馏目标只是手段、规划损失才是约束。
- **给了 oracle 上界**：posterior 直接上 95.0，读者可以据此判断蒸馏还差多少，而不是只能看一个没有参照的绝对分数。
- **零样本闭环不是同分布刷分**：HUGSIM 覆盖四个数据源、四档难度、436 个场景，Hard/Extreme 上的增益比 Easy 更明显，符合"预见性在提前决策场景更有用"的预期。

**Weaknesses**

- **绝对增益小，且集中在 1 个点以内**：93.1（base）→ 93.8（full），整套 TRP + H2F + FCP 只换来 0.7 PDMS；相对 DrivoR 的 +0.7 与 DriveFuture 的 +3.1 中，有多少来自 world model 而非 backbone / 训练配方（DINOv2 + LoRA、64 候选、六项打分器）无法区分，缺少同 harness 下"同等算力、无未来分支"的严格对照。
- **"planning-shaped" 与"多任务正则"难以彻底分离**：Table 5 的 generic future latent 用 $L_{\text{feature}}$ 重建未来 registers，但重建任务的难度、梯度尺度与 $L_{\text{plan}}+L_{\text{score}}$ 并不可比；0.3 点差距也可能来自优化难度的差异，而不是表示内容本身的差异。更干净的对照应是"未来潜 + 一个与规划无关但同等难度的判别目标"。
- **未来潜的可解释性没有验证**：论文声称 $z_t^+$ 优先编码"影响决策的未来信息"，但没有任何探针实验（例如线性探测 $z_t^+$ 是否编码了前车速度、TTC、可行驶区域）来支撑这一点。失败分析只能间接说明它改善了选轨，不能说明它编码了什么、丢掉了什么。
- **全部结果是单次运行、无方差**：HUGSIM 只有 436 个场景、NAVSIM 上的增益多在 0.1–0.3 点量级，论文没有报告 seed 数、标准差或置信区间，无法判断 93.5 → 93.8 这类差异是否超出训练噪声；这一点在组件消融里尤其关键，因为四个 ID 的差距只有 0.1–0.3。
- **Easy 档反而落后**：HUGSIM Easy 上 RC 75.3 低于 Latent-WAM 的 84.2，与"零样本通用提升"的叙事有张力；论文没有解释为何简单场景不受益，可能的推测是规划风格偏保守，但未被验证。
- **选轨依赖仿真器规则标签**：$L_{\text{score}}$ 的监督来自官方 proposal simulator 的六项子分数，EP 等指标本身是规则定义的；这让方法在一定程度上"拟合评估器"，在 NAVSIM 之外（尤其真实道路）的迁移性未知。零样本 HUGSIM 只能部分缓解这一质疑，因为 HUGSIM 的 HD-Score 仍复用同一套 NC/DAC/TTC/Comfort 结构。
- **蒸馏上限被固定**：posterior 一旦在 Stage 1 定型就不再更新，prior 只能逼近它的 95.0，无法在 Stage 2 中通过规划损失反过来改进 teacher。作者没有讨论 joint / iterative 蒸馏是否能缩小这 1.2 点差距。
- **计算成本不低但未与基线对齐**：16×A800、两阶段各 50 epoch、DINOv2 ViT-S + 4 层 ×4 个 Transformer；论文只报了推理延迟（342 ms @A800），没有给出与 DriveFuture / DrivoR 的训练成本与显存对比，"可部署"的论据只有延迟一项。
- **超参敏感性只覆盖两处**：只报了 $\lambda_{\text{latent}}$ 与 token 分配，TRP 的层数 $L$、未来时域 $T$、register 数（16/相机）都没有敏感性分析，而 16 registers 直接决定 $N_z=256$ 的容量。
- **定性证据弱**：正文只有一组与 DrivoR 的对比图和四组 HUGSIM 滚动图，没有反事实案例（如同场景下关闭未来分支后在哪里选错），无法直接展示"预见性改变了哪个决策"。

## Notes

+ **论文定位**：不是新的候选生成器，也不是更好的视频预测器，而是**重新定义了 world model 中"未来表示"的监督来源**——从重建/预测目标换成规划目标，用 privileged posterior 提供坐标系、用蒸馏把它变成可部署的因果预测。核心 insight 是：未来表示的价值不在于还原度，而在于它与下游决策损失是否对齐。
+ **可迁移点**：
  - **"目标塑形 + 特权蒸馏"是通用的表示学习配方**：任何"训练时拿得到、部署时拿不到"的信息（未来帧、仿真器真值、专家中间状态）都可以按"特权分支 → 挂下游损失 → 冻结 teacher → token 对齐蒸馏到学生"四步走，且不需要额外匹配器。
  - **token 布局同构是对齐蒸馏的关键工程**：prior 与 posterior 的 query 在时间/相机/slot 三维上按同一顺序排列，比学一个 projector 更稳定；这个技巧可直接迁移到任何 teacher-student latent 迁移。
  - **按时间远近分配 token 预算**优于均匀分配和逆向分配（Table 7），且更少的 token（128）比更多（256）更好——在固定容量下，分配方式比容量大小更决定信息内容。
  - **未来信息最有价值的落点是选轨**：生成端收益有限（93.5），选择端收益明显（93.7），与失败分析中"选择错误减少 87 例"互证。对任何候选式规划器，给打分器而不是生成器注入未来信息可能更划算。
  - **oracle 上界应成为此类工作的标配报告**：posterior 95.0 与 prior 93.8 的差距，把"表示本身有多好"与"蒸馏损失了多少"分开，比单一部署分数更有诊断价值。
+ **待跟进**：
  - 补一个探针实验：用线性探测检查 $\hat z_t^+$ 是否编码前车相对速度、TTC、可行驶区域边界，验证失败类型变化（NC/TTC 大降、DAC 微升）背后的表示内容。
  - 把 Stage 2 改成 teacher 也参与微调（或用 prior 的规划损失反向更新 posterior），检验那 1.2 点 oracle gap 是否可缩小。
  - 用同一个 DINOv2 backbone 与 64 候选打分器，跑一个"无未来分支、但同样两阶段训练"的 control，把 TRP / H2F 的净增益从 backbone 与训练配方中剥离出来。
  - 检查 $L_{\text{score}}$ 对仿真器规则的依赖：把 EP 从排序公式中替换为学习式的 progress 预测后，零样本 HUGSIM 的 RC 是否仍保持 53.7。
