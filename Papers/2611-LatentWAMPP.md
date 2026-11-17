---
title: "Latent-WAM++: Shaping World and Action Representations for End-to-End Autonomous Driving"
authors: ["Yichen Zhang", "Yupeng Zheng", "Linbo Wang", "Pengxuan Yang", "Zebin Xing", "Junli Wang", "Shiwei Li", "Kangwei Liu", "Fan Ren", "Qichao Zhang", "Yifeng Pan", "Ji Tao", "Dongbin Zhao"]
institute: ["Institute of Automation, Chinese Academy of Sciences", "Chongqing Changan Technology Co., Ltd."]
date_publish: 2026-11
venue: arXiv Preprint
tags: [world-model, flow-matching]
url: https://arxiv.org/abs/2611.10632
arxiv_id: "2611.10632"
doi: "10.48550/arXiv.2611.10632"
code:
rating: 4
content_scope: full-text
date_added: 2026-11-17
---

## Summary

现有 WAM 大多只监督"预测的未来状态"或"最终动作"，夹在中间的 **world representation** 与 **action representation** 只被这些输出目标间接约束。Latent-WAM++ 的主张是**给每种表示的监督匹配它在规划中的角色**：`WoRC` 用 scene-ego 未来潜预测 + SIGReg 分布正则 + 冻结几何先验，得到"可预测、不塌缩、带几何"的世界表示（并去掉 EMA target encoder）；`TrajMF` 直接预测干净轨迹、但在 Improved MeanFlow 的诱导速度空间施加监督，一步采样出 $K_p$ 条多模态 proposal；`VARA` 把这些 proposal 的 pre-output 特征与 stop-gradient 的当前视觉语义做非对称 cosine 对齐。**NAVSIM v2 90.1 EPDMS，HUGSIM 零样本 47.0 RC / 30.1 HD-Score，端到端训练 planner 190 GPU-hours，对 Latent-WAM 的 1,600 降低 88.1%**。

## Problem & Motivation

- WAM 的信息通路可以写成三段：观测 → **world representation**（哪些场景信息可用）→ **action representation**（如何翻译成轨迹）→ 轨迹。因此规划性能同时取决于两个表示本身以及它们的交互方式。
- 早期工作只监督预测的未来状态或最终动作，中间表示是输出导向目标的副产物，没有被直接塑形。
- 第一个缺陷：**联合训练 encoder 与 latent predictor 时没有 anti-collapse 机制**，模型可以靠把所有场景映射到相近 token 来降低预测损失（low-variance 解），表示虽然"可预测"但没有区分度。Drive-JEPA / Latent-WAM 用 EMA target encoder 来挡，代价是多一份目标编码器。
- 第二个缺陷：**world conditioning 只让 action decoder "看得到"场景信息，没有在表示层面施加对齐**。两个表示空间可以任意漂移，无法保证 action representation 保留了 world model 编码的规划相关信息。
- 由此把问题统一为：world 表示应当可预测、非退化、几何感知；action 表示应当直接参数化轨迹；二者的对应关系要被显式鼓励，但**不强求共享同一语义**（这是 VARA 采用非对称 cosine 而非逐元素对齐的原因）。

## Method

**0. 总体框架**

- 三个组件各管一段：`WoRC` 造世界表示 $\mathcal{W}_{t_{\text{cur}}}$ → `TrajMF` 以 $\mathcal{W}$ 为条件生成并聚合轨迹，其 pre-output 特征 $H_a$ 定义 action representation → `VARA` 在 $H_a$ 与当前视觉 token 之间加一条表示级正则。
- 训练时额外接两个**只在训练期存在**的分支：动态世界模型 DWM（预测未来潜 + ego 状态）与冻结几何 teacher（patch 级蒸馏）、冻结 occupancy 提取器（当前帧几何上下文）。
- 推理只用模态编码器与 TrajMF，输入仅为相机图像与 ego 状态。

**1. WoRC：世界表示构造（§3.2）**

- **场景编码**：每个视角、每一帧广播 $N_q$ 个可学习 query，与 patch token 拼接后过 DINOv2 编码器与 projector：

$$
[F_v,\widehat X]=\mathcal{P}_v\big(\mathcal{E}_v([Q_v;X])\big),\qquad F_v\in\mathbb{R}^{T\times N_v\times D},\ N_v=VN_q
$$

- 其中 $\widehat X$ 是投影后的 patch 特征，专门留给几何蒸馏用；默认 3 个视角 × 16 query = 48 个视觉 token，$D=256$。
- **几何蒸馏**：DINOv2 特征缺少度量空间线索，所以用冻结的 WorldMirror teacher 在 patch 层面做 stop-gradient 的余弦蒸馏：

$$
\mathcal{L}_{\text{geo}}=1-\cos\big(\mathrm{LN}(\mathcal{P}_g(\widehat X)),\ \mathrm{sg}(\mathrm{LN}(\mathcal{E}_{\text{geo}}(I)))\big)
$$

- 这里 $\mathcal{P}_g$ 把视觉 patch 特征投到 teacher 的特征空间；蒸馏只训练 $\mathcal{P}_g$ 与视觉侧，teacher 分支不产生可训练参数。
- **ego 与 occupancy 线索**：ego 状态经编码器得到 $F_e=\mathcal{E}_e(S_{\text{ego}})$；当前帧由冻结 occupancy 提取器给出占据栅格，栅格单元经 MLP 投成 token，再由视觉 token 做 cross-attention 取用：

$$
F_{\text{occ}}^{t_{\text{cur}}}=\mathcal{A}_{\text{occ}}\big(F_v^{t_{\text{cur}}},O_{t_{\text{cur}}},O_{t_{\text{cur}}}\big)
$$

- planning 时的世界表示是三者拼接，且**只有当前帧**进入 planner：

$$
\mathcal{W}_{t_{\text{cur}}}=[F_e^{t_{\text{cur}}};F_v^{t_{\text{cur}}};F_{\text{occ}}^{t_{\text{cur}}}]\in\mathbb{R}^{(1+2N_v)\times D}
$$

- ego token 提供导航意图与运动学、视觉 token 提供语义、occupancy token 通过 query 选择提供度量几何——三种信息在同一序列里以不同来源区分。
- **时间潜预测（DWM）**：把 $Z_t=[F_e^t;F_v^t]$ 送进带 3D rotary positional encoding 的 Transformer decoder，位置编码覆盖时间、相机视角与 query identity，掩码允许帧内双向、跨帧因果（frame-wise teacher forcing）：

$$
\widehat Z_{2:T}=\mathcal{D}_{\text{dwm}}(Z_{1:T-1}),\qquad \mathcal{L}_{\text{wm}}=\mathrm{MSE}(\widehat Z_{2:T},Z_{2:T})
$$

- ego 侧另加命令分类与速度/加速度回归：$\mathcal{L}_{\text{ego}}=\mathcal{L}_{\text{cmd}}^{\text{ce}}+\mathrm{MSE}(\hat v,v)+\mathrm{MSE}(\hat a,a)$，合起来 $\mathcal{L}_{\text{dwm}}=\mathcal{L}_{\text{wm}}+\mathcal{L}_{\text{ego}}$。
- **分布稳定化（SIGReg）**：对每个 token identity $n$，跨 batch 估计经验特征函数，并与标准高斯匹配：

$$
\widehat\varphi_{tnj}(\omega_\ell)=\frac{1}{B}\sum_{b=1}^{B}\exp\big(\mathrm{i}\omega_\ell r_j^{\top}f_{b,t,n}\big),\qquad \mathcal{L}_{\text{sigreg}}=\frac{B}{TN_vJ}\sum_{t,n,j,\ell}\widetilde w_\ell\,\big|\widehat\varphi_{tnj}(\omega_\ell)-e^{-\omega_\ell^2/2}\big|^2
$$

- $r_j$ 是随机单位方向、$\omega_\ell$ 是积分频率、$\widetilde w_\ell$ 是积分权重。这个正则**没有 target encoder**：它约束的是"同一 token 在不同场景下的分布"，因此不需要额外的 EMA 网络就能避免低方差解。

**2. TrajMF：轨迹 MeanFlow 规划器（§3.3）**

- 先把世界表示压成规划条件：$N_a$ 个可学习 action query 对 $\mathcal{W}$ 做 cross-attention 得 $C_{t_{\text{cur}}}$，条件 $c=(\mathcal{W},C)$。
- 把专家轨迹归一化成相对轨迹 $x=\mathcal{R}(\tau)$，噪声 $\epsilon$ 从 $K_p$ 分量高斯混合先验 $p$ 采样，线性流路径与目标速度为：

$$
z_t=(1-t)x+t\epsilon,\qquad v_t=\frac{\mathrm{d}z_t}{\mathrm{d}t}=\epsilon-x
$$

- **关键设计是"预测 $x$、监督 $v$"**：网络直接输出干净轨迹 $\widehat x_\theta=\mathcal{F}_\theta(z_t,r,t,c)$，再换算成诱导平均速度：

$$
u_\theta=\frac{z_t-\widehat x_\theta}{t}\ (t>0),\qquad (\widehat v_\theta,\cdot)=\text{JVP}\big[u_\theta(\cdot\mid c);(z_t,r,t);(v_{\text{ins}},0,1)\big]
$$

- JVP 沿诱导轨迹方向求方向导数，避免显式构造 Jacobian；瞬时速度用一阶展开构造，导数项 stop-gradient：

$$
\widehat v_\theta=u_\theta+(t-r)\,\mathrm{sg}(\dot u_\theta),\qquad \mathcal{L}_{\text{trajmf}}=\|\widehat v_\theta-v_t\|_1
$$

- 采样阶段不需要任何速度换算：从 $p$ 的每个分量各取一个噪声，$\mathcal{F}_\theta(\{\epsilon^k\},0,1,c)$ **一次前向**得到 $K_p$ 条 proposal，再由 refiner $\mathcal{D}_{\text{refine}}$ 条件聚合为最终轨迹，$\mathcal{L}_{\text{traj}}=\|\widehat x-x\|_1$。
- **action representation 的定义**：每条 proposal 在输出头之前的特征 $h_a^k$ 被收集成 $H_a=[h_a^1;\dots;h_a^{K_p}]\in\mathbb{R}^{K_p\times D}$。选它而不是最终轨迹，是因为它是"直接支撑轨迹预测"的最后一层表示，也是 VARA 要正则的接口。
- 与 iMF 的差别只在预测目标：iMF 预测平均速度 $u$，TrajMF 预测轨迹 $x$，二者都在速度空间监督；论文的说法是在低维轨迹空间两者都能work，但预测 $x$ 收敛更快（见 Key Results 的迭代数对比）。

**3. VARA：视觉-动作表示对齐（§3.4）**

- 把 proposal 特征投影后平均，得到动作侧表示；视觉侧取当前帧 token 平均并 stop-gradient：

$$
z_a=\frac{1}{K_p}\sum_{k=1}^{K_p}\mathcal{P}_a(h_a^k),\qquad z_v=\mathrm{sg}\Big(\frac{1}{N_v}\sum_{n=1}^{N_v}F_v^{t_{\text{cur}},n}\Big),\qquad \mathcal{L}_{\text{align}}=1-\cos(z_a,z_v)
$$

- 三个设计点：①用**余弦**而不是 MSE，只要求语义一致、不要求逐元素相等；②梯度只经动作侧回传，视觉侧被 $\mathrm{sg}$ 挡住，避免把视觉表示拉向动作表示（非对称正则）；③对齐的是"规划前的动作特征"而不是输出轨迹，等于给 decoder 内部加了一个语义锚。

**4. 训练与推理（§3.5）**

- 损失按"世界 / 动作 / 对齐"分组：

$$
\mathcal{L}_{\text{world}}=\lambda_1(\mathcal{L}_{\text{geo}}+\mathcal{L}_{\text{dwm}}+\mathcal{L}_{\text{sigreg}}),\qquad \mathcal{L}_{\text{action}}=\lambda_2\mathcal{L}_{\text{trajmf}}+\lambda_3\mathcal{L}_{\text{traj}},\qquad \mathcal{L}=\mathcal{L}_{\text{world}}+\mathcal{L}_{\text{action}}+\lambda_4\mathcal{L}_{\text{align}}
$$

- 权重 $\lambda_1=0.2$、$\lambda_2=20.0$、$\lambda_3=7.0$、$\lambda_4=2.0$；推理时只保留模态编码器与 TrajMF，几何 teacher、DWM、SIGReg、VARA 全部不参与。

**5. 实现细节**

| 项 | 设置 |
|:--|:--|
| 视觉编码器 | DINOv2-Base，3 路相机，224×448，每视角 16 个 scene query |
| 特征维度 | 所有潜特征统一投到 $D=256$ |
| 几何分支 | 冻结 WorldMirror-based teacher（仅训练期蒸馏） |
| occupancy | 冻结提取器，仅用当前帧；NAVSIM-Occ 标注来自 NuPlan-Occ |
| 训练 | Adam，25 epoch，batch 128，lr $10^{-4}$ warm-up 10% 后 cosine 退火到 $10^{-6}$ |
| 算力 | 16×H20，端到端训练约 12 h（≈190 GPU-hours） |
| 评测 | NAVSIM v2 navtest（EPDMS）+ HUGSIM 零样本（436 场景，无微调） |

## Pipeline

![Latent-WAM++ 整体框架](https://arxiv.org/html/2611.10632v1/overview.png)

*WoRC 用 DINOv2 视觉 token、ego token、occupancy token 与冻结几何蒸馏构造世界表示，并由 DWM 做 scene-ego 未来潜预测、SIGReg 稳定分布；TrajMF 以世界表示为条件，一次前向从高斯混合先验生成多条轨迹 proposal 并聚合；VARA 把 proposal 的输出前特征与当前视觉语义做 stop-gradient 的余弦对齐。训练期的 DWM、几何 teacher 与 VARA 在推理时都不保留。*

1. **编码**：3 路相机图像 + 可学习 query 过 DINOv2-B，得视觉 token $F_v$；ego 状态编码为 $F_e$；当前帧 occupancy 栅格经 MLP 成 token $O$。
2. **世界表示**：视觉 token cross-attend occupancy 得 $F_{\text{occ}}$，与 $F_e,F_v$ 拼成 $\mathcal{W}_{t_{\text{cur}}}$（只含当前帧）。
3. **训练期世界监督**：DWM 用因果掩码预测未来帧的 $[F_e;F_v]$；SIGReg 约束同一 token 跨场景分布；冻结几何 teacher 在 patch 级蒸馏。
4. **条件化**：$N_a$ 个 action query 对 $\mathcal{W}$ 做 cross-attention 得 $C$，与 $\mathcal{W}$ 一起构成条件 $c$。
5. **一步生成**：从 $K_p$ 分量高斯混合先验各取一个噪声，$\mathcal{F}_\theta(\cdot,0,1,c)$ 一次前向输出 $K_p$ 条 proposal，收集 $H_a$ 作为动作表示。
6. **聚合与对齐**：refiner 条件聚合 proposal 得最终轨迹；训练时 VARA 用 $H_a$ 的均值与 stop-gradient 视觉均值算 cosine 对齐损失。
7. **推理**：只跑 1–2、4–6，DWM / SIGReg / 几何 teacher / VARA 全部移除；输出一条轨迹。

## Key Results

**1) NAVSIM v2 navtest（Table 1，EPDMS）**

| Method | NC↑ | DAC↑ | EP↑ | TTC↑ | EC↑ | EPDMS↑ |
|:--|--:|--:|--:|--:|--:|--:|
| DiffusionDrive | 98.2 | 95.9 | 87.5 | 97.3 | 87.7 | 84.5 |
| World4Drive | 97.8 | 96.3 | 88.3 | 97.1 | 53.9 | 84.8 |
| Drive-JEPA | 98.4 | 98.6 | 88.4 | 97.8 | 84.8 | 87.8 |
| MeanFuser | 98.3 | 97.2 | 87.6 | 97.4 | 88.2 | 89.5 |
| Latent-WAM | 98.1 | 97.3 | 87.7 | 97.3 | 87.3 | 89.3 |
| **Latent-WAM++** | **98.5** | 97.5 | 87.8 | **98.3** | **89.0** | **90.1** |

- 比基座 Latent-WAM 高 0.8，比 MeanFuser 高 0.6；除 DAC 与 EP 外，其余子项均为最优或并列最优。
- EC 从 87.3 提到 89.0 是最大的单项跨度，对应"轨迹更平滑、相邻帧更一致"这类时间一致性收益。
- EP 87.8 与 DAC 97.5 都不是最优，说明增益集中在安全与舒适维度，而非进度与可行驶区域合规。

**2) HUGSIM 零样本闭环（Table 2）**

| Method | RC Avg.↑ | HD-Score Avg.↑ | RC(E/H/X) | HD(H/X) |
|:--|--:|--:|:--|:--|
| MeanFuser | 43.9 | 28.4 | 76.7 / 37.9 / 28.4 | 21.5 / 11.0 |
| Latent-WAM | 45.9 | 28.9 | 84.2 / 30.6 / 35.5 | 12.2 / 18.1 |
| **Latent-WAM++** | **47.0** | **30.1** | 80.1 / **38.7** / 32.0 | **20.5** / 13.9 |

- 加权平均 RC / HD 分别比 Latent-WAM 高 1.1 / 1.2；Hard 档提升最大（RC 30.6→38.7，HD 12.2→20.5）。
- 但并非全档领先：Easy 的 RC 80.1 低于 Latent-WAM 的 84.2，Extreme 的 RC 32.0 / HD 13.9 也低于 35.5 / 18.1，呈现"中等与困难档受益、两端退步"的形态。

**3) 统一组件消融（Table 3，NAVSIM v2）**

| ID | 配置 | EPDMS↑ | Δ |
|:--|:--|--:|--:|
| 1 | MLP planner baseline | 88.2 | – |
| 2 | + DWM（无稳定化） | 88.0 | −0.2 |
| 3 | + EMA target encoder | 88.6 | +0.4 |
| 4 | + SIGReg（去掉 EMA） | 88.8 | +0.6 |
| 5 | + occupancy | 89.1 | +0.9 |
| 6 | + TrajMF（替换 MLP） | 89.5 | +1.3 |
| 7 | + VARA | 89.8 | +1.6 |
| 8 | + 几何蒸馏 | 90.1 | +1.9 |

- **ID 2 是全表最有信息量的一行**：只加未来潜预测、不做分布约束，反而比 baseline 低 0.2，说明"世界模型本身不产生收益，产生收益的是被正则约束住的预测表示"。
- SIGReg 以 88.8 超过 EMA 的 88.6，同时省掉 86.6M 参数的目标编码器——这是"去掉 teacher 网络不牺牲性能"的直接证据。
- 各组件增益都很小（0.2–0.4），且是顺序累加，交互项未知。

**4) 动作解码器对比（Table 4，同一训练配方）**

| Decoder | Pred. | Loss 空间 | EPDMS↑ | 达到该分数的迭代数↓ |
|:--|:--|:--|--:|--:|
| MLP | $x$ | – | 89.1 | 16,160 |
| iMF | $u$ | $v$ | 89.4 | 16,160 |
| **TrajMF** | $x$ | $v$ | **89.5** | **8,080** |

- 精度上 TrajMF 只比 iMF 高 0.1，真正差别是一半的迭代数就到同一水平——这是"预测干净轨迹 + 速度空间监督"的实际卖点。
- 但迭代数不等于算力，论文没有给 wall-clock 或 FLOPs 的对比（见 Weaknesses）。

**5) 表示层面的证据（§4.4、Figure 3/5）**

- 未加稳定化时，同一 token identity 跨场景的特征标准差均值 0.088、余弦相似度 0.973（几乎退化成常量）；EMA 部分缓解；SIGReg 后变为 0.900 / 0.017。这组统计把"塌缩"从口号变成了可测现象。
- VARA 的可视化用冻结 DINOv2 的 CLS 特征做 $K=8$ 聚类上色，动作表示在加 VARA 后邻域更一致；对应 EPDMS 89.5 → 89.8。
- **成本主张**：同硬件、同 data-loading 下，端到端训练 planner 从 Latent-WAM 的 1,600 GPU-hours 降到 190（−88.1%），对应 16×H20 上约 12 小时。

## Strengths & Weaknesses

**Strengths**

- **"为每种表示匹配其规划角色"是一个可操作的框架**，而不只是口号：world 表示对应"可预测 + 非退化 + 几何"，action 表示对应"直接参数化轨迹"，接口对应"非对称语义一致"。三个组件都能从这句话推出来。
- **SIGReg 替代 EMA 是干净的结构简化**：省掉 86.6M 参数的目标编码器，性能仍从 88.6 提到 88.8；并且论文给了标准差 0.088→0.900、余弦相似度 0.973→0.017 这种可直接复核的量化证据，比"我们观察到表示更好"有力得多。
- **TrajMF 的"预测 $x$、监督 $v$"是一个值得记住的工程技巧**：采样时不需要任何速度换算，一步出多模态 proposal；训练时又能享受 velocity-space 目标的稳定性。
- **消融包含了负结果**：ID 2 的 −0.2 明确写出"加世界模型但不加稳定化会更差"，这类信息比只报最优配置更有价值，也解释了为什么必须引入 SIGReg。
- **VARA 的非对称设计有明确理由**：只用 stop-gradient 视觉目标、只回传动作侧，避免两个表示互相拉扯；用 cosine 而非 MSE 避免强迫逐元素对应，这与"不强求共享语义"的论点一致。
- **成本主张有对照口径**：明确写了"同硬件、同 data-loading"，并给出 GPU-hours 而非只给 epoch 数，比多数论文的"训练更快"更可核查。
- **零样本闭环做了跨难度拆分**：四档分别报 RC 与 HD-Score，读者能看到收益来自 Hard 档而不是平均掩盖。

**Weaknesses**

- **增益幅度小、且叠加式消融无法隔离交互**：90.1 对基座 89.3 只有 +0.8，对 MeanFuser 89.5 只有 +0.6；Table 3 是顺序累加的 8 行，每行只加一个组件，既没有单独移除某项的 leave-one-out，也没有 seed / 方差，0.2–0.4 的单步增益无法排除训练噪声。
- **"190 vs 1,600 GPU-hours"的口径没有拆开**：两个模型的 backbone、训练 epoch、planner 结构与采样步数都不同。省下的算力究竟来自一步采样、更小的 DINOv2-Base、还是 25 epoch 的短训练，论文没有做"固定 backbone / 固定 epoch 只换采样步数"的对照，−88.1% 这个数字容易被读成"方法本身的效率"。
- **TrajMF 相对 iMF 的精度增益只有 0.1**：真正的差异是迭代数减半，但没有 wall-clock、FLOPs 或每步耗时；在 batch 128、16 卡的设置下，迭代数受 lr schedule 与数据加载影响，不能直接换算成加速比。
- **HUGSIM 结果并非全面领先，但正文只讲平均值**：Easy RC 80.1（Latent-WAM 84.2）、Extreme RC 32.0（35.5）、Extreme HD 13.9（18.1）都退步，论文没有讨论"为什么两端变差"。一个合理但未验证的猜测是 TrajMF 的混合先验在极端场景下覆盖不足，但缺少按场景类型的失败统计。
- **occupancy 分支的输入来源含糊**：正文说"所有输入来自相机与 ego 状态，不需要额外传感器"，但 occupancy 提取器是冻结的，其训练依赖 NuPlan-Occ 的标注（附录 A.3），提取器本身的输入模态（图像 / LiDAR / 深度）没有交代。如果它隐式用到了 LiDAR 或标注深度，"仅相机"的宣称就打折扣，复现也无从下手。
- **VARA 的机制证据偏弱**：+0.3 EPDMS 只配了一张聚类可视化；没有做"对齐到视觉 CLS / 对齐到 world 表示 / 双向对齐 / 换成 MSE"的对照，也没有探针说明动作特征到底继承了什么视觉语义（语义分类？几何？运动？）。
- **超参缺乏敏感性**：$\lambda_2=20$ 与 $\lambda_3=7$ 的量级差异、$\lambda_4=2.0$、SIGReg 的方向数 $J$ 与频率网格都没有消融；考虑到各组件增益本身只有 0.2–0.4，超参选择对结论的影响可能不小于组件本身。
- **"SOTA" 的对照集合有时间窗问题**：论文报 90.1 EPDMS 并称 state-of-the-art，但同期 PlanWAM（2610.11382）在同一 benchmark 上报 90.9 EPDMS，基座 Latent-WAM 的 89.3 也在其表内。baseline 集合没有覆盖这批工作，因此"SOTA"只对论文所选比较集成立。
- **几何蒸馏的收益未与 occupancy 分离**：Table 3 中 occupancy（+0.3）与几何蒸馏（+0.3）都是"加几何信息"，但一个是推理期可见的 token、一个是训练期蒸馏；两者是否在提供同一份信息、能否只留一个，论文没有做交叉消融。

## Notes

+ **论文定位**：不是新的 WAM 架构，而是**对 WAM 中间表示的监督重新分工**——world 侧负责"可预测、不塌缩、带几何"，action 侧负责"直接参数化轨迹"，接口侧负责"语义一致但不必同构"。核心 insight 是：中间表示的性能不是靠更大模型或更多输出监督堆出来的，而是靠"监督信号是否匹配该表示在规划中的用途"。
+ **可迁移点**：
  - **SIGReg 作为 EMA-free 的 anti-collapse 手段**：任何"encoder + 未来预测器"联合训练的自监督世界模型都可以直接换上，代价是每个 token identity 上的一次跨 batch 特征函数估计，省掉一整份 target encoder。
  - **预测干净样本、在诱导速度空间监督**：在低维动作空间里，这种组合可以既保留一步采样的便利，又获得 velocity-space 目标的稳定梯度；对 flow / MeanFlow 类策略是可复用的训练配方。
  - **非对称表示对齐**：只对下游侧（action）加约束、上游侧 stop-gradient，可以避免两个表示空间互相拉扯；cosine 而非 MSE 则避免把"语义一致"写成"逐元素相等"。
  - **"role-matched supervision" 的提问方式**：先问每个中间表示在最终任务里承担什么职责，再决定监督信号，这比"给每个模块都加一个辅助 loss"更有结构。
+ **待跟进**：
  - 用固定 backbone、固定 epoch、只改采样步数的对照，重算 190 vs 1,600 GPU-hours，把"一步生成"的净收益从"更小模型 + 更短训练"里剥离出来。
  - 补齐 occupancy 提取器的输入模态与训练数据，确认"仅相机输入"是否成立；若不成立，需要重做 zero-shot 的部署假设。
  - 对 VARA 做四组对照（对齐到 CLS、对齐到 world token、双向、MSE），并用线性探针检查 $H_a$ 是否编码了前车距离 / TTC / 车道几何，判断它到底传递了什么语义。
  - 统计 HUGSIM Easy 与 Extreme 两档的失败类型，验证"混合先验覆盖不足"这一猜测，或找出 TrajMF 在极端场景退步的真实原因。
