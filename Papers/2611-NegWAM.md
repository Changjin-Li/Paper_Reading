---
title: "Reinforcing End-to-End Autonomous Driving Safety via Negative-Aware Planning in a Unified World-Action Model"
authors: ["Zihao Zhang", "Xinyu Zhang", "Bing Zeng", "Xingyu Chen", "Zhenning Li", "Bin Yu", "Ziyuan Pu"]
institute: ["Southeast University", "Shenzhen Loop Area Institute", "Guangxi University", "University of Macau", "Beihang University"]
date_publish: 2026-11
venue: arXiv Preprint
tags: [world-model, imitation-learning]
url: https://arxiv.org/abs/2611.08111
arxiv_id: "2611.08111"
doi: "10.48550/arXiv.2611.08111"
code: https://github.com/Zizizi-hao/NegWAM
rating: 4
content_scope: full-text
date_added: 2026-11-17
---

## Summary

WAM 的动作分支仍由专家轨迹 imitation 主导，因此**"离专家很近但实际不安全"的那类轨迹从未被显式排除**——更强的世界建模并不自动产生安全边界。NegWAM 把这类近专家失败轨迹当作后训练信号，构造方式是从 GTRS 的 16K 轨迹词表里挑出与专家 ADE 在 $[0.3,1.0)$ m、且用 NAVSIM v2 指标缓存判定为 0 EPDMS 的候选，形成场景专属负样本池；这些负样本同时以两种方式进入统一的 world-action DiT：作为**排斥锚**（distance-adaptive hinge 把一步解码轨迹推离），以及作为**scene-conditioned rider tokens**（与专家轨迹并列成组、只看当前观测视觉 token，再用共享 compatibility head 做排序）。rider 分支与兼容头在推理时整体删除，不增加部署开销。**NAVSIM v2 navhard 40.4 EPDMS（相对最强 scorer-free WAM baseline +4.1%）、navtest 93.3 EPDMS（超过 BeyondDrive 0.4 分）、HUGSIM 零样本 45.1 RC / 30.6 HD-Score，推理延迟只增加 6.5 ms**。

## Problem & Motivation

- 端到端策略从大规模专家示范里学，routine 场景表现好，但危险行为与失败案例在真实数据里极其稀缺，imitation 学不到显式安全边界。
- WAM 通过预测未来世界状态引入动力学先验，可它的**动作分支仍只在专家轨迹上做 flow matching**：这条监督告诉模型"该模仿什么"，不告诉它"哪些近专家替代方案必须被拒绝"。
- 后训练需要额外的监督来源，现有两条路都有缺口：①闭环环境（仿真器或世界模型）受 sim-to-real gap、构造成本与重建伪影限制；②外部 scorer（规则或学习 reward）受手工规则、特权信息与 evaluator bias 限制。
- 作者的主张是换一个信息源：**不安全行为比"完整的好行为谱"更容易刻画**，近专家失败轨迹恰好标出了轨迹空间中局部不可行的区域。
- 与既有 hard-negative 工作（BeyondDrive 用 repulsive distance loss 正则 planner 输出）的区别在于：NegWAM 同时后训练**单步动作流**与**共享的 world-action 表示**，负样本既约束生成、也约束"场景-轨迹"兼容性判断。

## Method

**0. 总体框架（两阶段）**

- Stage 1：从 Wan2.2-TI2V-5B 视频基座出发，按 DriveWAM 的统一 world-action flow matching 范式在 navtrain 上预训练 100K 步，得到共享 base checkpoint（所有后训练变体都从它初始化，以控制初始化差异）。
- Stage 2：只做 10K 步安全后训练，损失为 $\mathcal{L}_{\text{post}}=\mathcal{L}_{\text{FM}}+\lambda_{\text{rep}}\mathcal{L}_{\text{rep}}+\lambda_{\text{compat}}\mathcal{L}_{\text{compat}}$；原始 flow-matching 项始终保留，用来维持世界动力学与专家驾驶先验。
- 推理：删除 rider 组与 compatibility head，只走原始 world-action 生成路径。

**1. 统一 world-action 模型的输入与流匹配（§III-A）**

- 一个训练样本包含当前图像 $\mathcal{I}^{\text{obs}}$、未来视觉目标 $\mathcal{I}^{\text{fut}}$、专家轨迹 $\tau^+$，以及导航命令与 ego 状态（含 4 个历史 ego pose）。
- 冻结的因果视频 VAE 把观测帧与未来帧编码成干净视觉流 $\mathbf{x}_0^v$；动作 tokenizer 把专家未来 8 个 $\Delta$-pose 归一化后拼上一个零条件 chunk，得到 $\mathbf{x}_0^a=E_a(\tau^+)$。损失掩码排除观测视觉前缀与条件动作 chunk。
- 对每个模态独立采样噪声与 scheduler 索引，经 SNR shift 得到 $t^m$：

$$
\mathbf{x}_t^m=(1-t^m)\mathbf{x}_0^m+t^m\boldsymbol\epsilon^m,\qquad \mathbf{v}^m=\frac{\partial\mathbf{x}_t^m}{\partial t^m}=\boldsymbol\epsilon^m-\mathbf{x}_0^m
$$

- 共享 DiT 接收四段序列，其中两个 $\mathbf{x}_c^m$ 是因果 teacher-forcing 条件流：

$$
\mathcal{S}_{\mathbf{t}}=[\,\mathbf{x}_t^v\,\|\,\mathbf{x}_c^v\,\|\,\mathbf{x}_t^a\,\|\,\mathbf{x}_c^a\,]
$$

- 原始 block-causal 掩码 $\mathcal{A}_{\text{base}}$ 允许噪声 token 看到之前的条件 chunk 与同一时间块内的其他噪声 token，条件 token 保持因果可见性；视频条件流以 0.5 概率加噪、其观测前缀保持干净，动作条件流不加噪（零初始 chunk + 之前的专家动作 chunk）。

$$
\mathcal{L}_{\text{FM}}=\sum_{m\in\{v,a\}}\mathbb{E}\Big[w_m(t^m)\frac{\lVert \mathbf{M}^m\odot(\hat{\mathbf{v}}_\theta^m-\mathbf{v}^m)\rVert_2^2}{\lVert\mathbf{M}^m\rVert_1}\Big]
$$

- 一步干净估计与解码轨迹是负样本监督的作用对象：

$$
\hat{\mathbf{x}}_0^a=\mathbf{x}_t^a-t^a\hat{\mathbf{v}}_\theta^a,\qquad \hat\tau=D_a(\hat{\mathbf{x}}_0^a)
$$

- 其中 $D_a$ 取未来动作 chunk、反归一化 $\Delta$-pose、再累积成 ego 坐标系下的位姿序列。

**2. 近专家负样本构造（§III-B1、Algorithm 1）**

- **不从策略自身采样**：作者做了一个诊断实验，同一场景采样 8 条轨迹的平均两两 ADE 只有 0.02 m，说明在强场景条件下策略采样多样性极低，挖不出有意义的失败样本。
- 改从 GTRS 的 16K 稠密轨迹词表取候选：每条候选是 4 s、40 个 ego 位姿；用算子 $S$ 每 5 个取 1 个（0.5 s 间隔）得到 8 步轨迹，与专家轨迹对齐：

$$
a_{i,j}=\text{ADE}(\tau_{i,j},\tau_i^+)=\frac{1}{T}\sum_{t=1}^{T}\lVert \mathbf{p}_{i,j}^t-\mathbf{p}_i^{+,t}\rVert_2
$$

- 离线挖掘分两步筛：先保留 $a_{i,j}<1.0$ m 的候选，再用原始 40 步轨迹、场景的 NAVSIM v2 指标缓存与 log-replay 交通，保留 **EPDMS = 0** 的失败候选，形成场景专属池 $\mathcal{H}_i$。
- 每次训练读取时，把池内轨迹降到 8 步、重算 ADE，只保留 $a_{i,j}\ge0.3$ m 的候选，最终范围 $[0.3,1.0)$；随机无放回抽至多 $K=16$ 条，零填充并用 validity mask 区分。掩码与索引细节：缓存保留 40 个 0.1 s 位姿，8 步训练轨迹取索引 $\{4,9,\dots,39\}$。
- 没有有效负样本的样本只贡献 flow-matching 损失；EPDMS 只在离线挖掘阶段被查询，训练循环内不再调用评估器。
- 统计：**85.8% 的训练样本至少有一条有效 hard negative**；全部有效负样本到专家的中位 ADE 为 0.80 m，其中约 9% 落在 $[0.3,0.5]$ m。

**3. Noise-aware 排斥锚（§III-B2）**

- 负样本在轨迹空间被当作固定排斥锚。设 $d_{i,k}$ 为一步解码轨迹 $\hat\tau_i$ 与第 $k$ 条负样本的 ADE，$a_{i,k}$ 为该负样本到专家的 ADE；固定 margin 无法适配不同距离的负样本（太小则不激活、太大则与 imitation 冲突），因此用**距离自适应 margin**：

$$
\ell_{i,k}^{\text{rep}}=\big[\alpha\, a_{i,k}-d_{i,k}\big]_+,\qquad 0<\alpha\le1
$$

- $\alpha=0.9$。这个设计有个自洽性质：在专家轨迹处 $d_{i,k}=a_{i,k}$，所以只要 $\alpha\le1$，排斥项在专家轨迹上恒为 0，不会把模型从专家行为上推开。
- 由于高噪声下的一步估计不可靠，按 $1-t_i^a$ 加权并在有效负样本上平均：

$$
\mathcal{L}_{\text{rep}}=\mathbb{E}_{i\sim\mathcal{B}}\Big[(1-t_i^a)\frac{\sum_k M_{i,k}^{-}[\alpha a_{i,k}-d_{i,k}]_+}{\max(1,\sum_k M_{i,k}^{-})}\Big]
$$

- 对 $\hat{\mathbf{x}}_0^a=\mathbf{x}_t^a-t^a\hat{\mathbf{v}}_\theta^a$ 求导会引入一个 $t^a$ 因子，与权重相乘得到 $t^a(1-t^a)$ 的前置系数，在 $t=0$ 与 $t=1$ 两端都归零——这是"只在中等噪声区间施加排斥"的机制来源。
- 排斥复用原始动作输出，不需要额外一次 backbone 前向。

**4. Scene-conditioned rider tokens（§III-B3）**

- 同样的失败轨迹还定义了"场景-轨迹"比较：专家与每条负样本各成一个辅助 rider 组 $\mathcal{R}_g$，复用共享动作嵌入与同一套 RoPE 网格；每组 = 一个零条件 chunk + 一个 8 步轨迹 chunk = 16 个动作 token。rider 是干净的（不加噪、$t=0$）。
- 关键在可见性：rider 能看当前观测视觉 token 与自身组，但**原始序列看不到 rider**，因此原前向路径在固定权重下完全不变：

$$
\mathcal{A}(p,q)=\begin{cases}\mathcal{A}_{\text{base}}(p,q),&p,q\in\mathcal{S}_{\mathbf{t}},\\ 1,&p\in\mathcal{R}_g,\ q\in\mathbf{x}_c^{v,\text{obs}}\cup\mathcal{R}_g,\\ 0,&\text{otherwise}\end{cases}
$$

- 专家与负样本的 tokenization、位置编码、可见性完全一致，因此打分输入之间的结构差异被压到最小——分数差只能来自轨迹内容与场景匹配度。
- 每组 16 个 token 的输出特征平均后，过共享 compatibility head $h_\phi$（线性投到 256 → SiLU → 标量）。**最后一层权重与偏置零初始化**，使初始时专家与负样本分数相同、初始兼容性梯度为 0，避免后训练一开始就把 backbone 带偏；随着头部输出权重更新，这条梯度通路才逐渐打开。
- 排序目标是把专家排在所有有效负样本之上：

$$
\mathcal{L}_{\text{compat}}=-\mathbb{E}_{\mathcal{B}_+}\log\frac{\exp(s_i^+/\tau_s)}{Z_i},\qquad Z_i=\exp(s_i^+/\tau_s)+\sum_k M_{i,k}^{-}\exp(s_{i,k}^{-}/\tau_s)
$$

- $\tau_s=0.1$；$\mathcal{B}_+$ 为有有效负样本的样本集合，为空时该项取 0。梯度经共享 attention 传播，等于给 backbone 增加了一个显式的场景-轨迹排序目标。

**5. 实现细节**

| 项 | 设置 |
|:--|:--|
| 输入 | 1 路前视相机 256×448 + 4 个历史 ego pose |
| 动作目标 | 未来 4 s、8 步 $\Delta$-pose，按 1%/99% 分位归一化 |
| 视觉主干 | Wan2.2-TI2V-5B（统一 world-action DiT），冻结因果视频 VAE |
| 未来视觉目标 | 4 帧，序列索引 {1,3,5,7}（缓存索引 {2,4,6,8}） |
| SNR shift | 视频 5.0 / 动作 1.0；去噪步数 3 / 10 |
| 预训练 | 100K 步，lr $1\times10^{-5}$，30K/70K/90K 各降半，4×H100，22 h（88 GPU-hours） |
| 后训练 | 10K 步，lr $1\times10^{-6}$，5K 降半，4×H100，2 h（8 GPU-hours） |
| 损失权重 | $\lambda_{\text{rep}}=100$，$\lambda_{\text{compat}}=0.1$；$\alpha=0.9$，$\tau_s=0.1$ |
| 优化器 | AdamW，$\beta=(0.9,0.95)$，weight decay 0.1，每卡 batch 1、全局 4 |
| 负样本 | GTRS 16K 词表，ADE 范围 $[0.3,1.0)$ m，每次最多 $K=16$ 条 |

## Pipeline

![NegWAM 的两阶段框架](https://arxiv.org/html/2611.08111v1/Figures/figs/figure_02_methods.jpg)

*Stage 1 把预训练视频 DiT 适配成统一 world-action 模型，用 flow matching 同时学未来视觉流与专家动作流；Stage 2 用场景专属的 hard-negative 轨迹，一路作为排斥锚作用于一步解码轨迹，一路作为 rider token 组与专家轨迹并列、通过共享 attention 读当前观测并输出兼容性分数；rider 分支与兼容头只在后训练存在。*

1. **离线挖掘**：对每个训练场景，用 GTRS 16K 词表算候选到专家的 ADE，先按 $<1.0$ m 过滤，再用 NAVSIM v2 指标缓存评估 40 步原轨迹，保留 0 EPDMS 的失败候选进场景池。
2. **训练取负**：每次读取该场景池，降到 8 步、重算 ADE，保留 $[0.3,1.0)$ m 的候选，无放回抽 ≤16 条并零填充 + validity mask。
3. **Stage 1 前向**：当前帧经 VAE 得视觉条件流，动作 token 与未来视觉 token 拼成四段序列进共享 DiT，用 block-causal 掩码预测两路速度场，最小化 $\mathcal{L}_{\text{FM}}$。
4. **一步解码**：用 $\hat{\mathbf{x}}_0^a=\mathbf{x}_t^a-t^a\hat{\mathbf{v}}_\theta^a$ 得到干净动作，再由 $D_a$ 累积成 8 步轨迹。
5. **排斥**：用该轨迹与每条负样本算 ADE，按 $[\alpha a-d]_+$ 与 $t^a(1-t^a)$ 权重反向传播，把预测推离负样本邻域。
6. **兼容性学习**：专家与负样本各成 16-token rider 组，只 attend 当前观测视觉 token 与自身组；每组特征平均后过零初始化的兼容头，用 softmax 排序损失更新共享 backbone。
7. **推理**：删除 rider 组与兼容头，视觉流 3 次 Euler、动作流 10 次 Euler 更新，输出 8 步轨迹；不加载负样本池、不调用评估器。

## Key Results

**1) NAVSIM v2 navhard（Table I，两阶段 EPDMS）**

| Method | Sensor | EPDMS↑ |
|:--|:--|--:|
| PDM-Closed | – | 56.6 |
| SpanVLA | MV | 40.1 |
| EponaV2 | SV | 36.1 |
| DriveWAM | SV | 37.7 |
| BeyondDrive | SV | 38.8 |
| **NegWAM** | SV | **40.4** |

- 比 BeyondDrive 高 1.6 分（相对 +4.1%），作者据此主张在"不使用外部轨迹 scorer 的 WAM"里最强；PDM-Closed 56.6 这类规则模型仍远高，说明该 split 上学习方法整体还有很大空间。
- 增益主要来自 DAC（S2 从 DriveWAM 的 75.8 升到 78.0）；但 NC 79.5→79.2、TTC 75.7→75.3、DDC 87.8→86.4 都略降，说明总分提升并非所有安全子项同时改善。

**2) NAVSIM v2 navtest（Table II）**

| Method | Sensor | NC↑ | DAC↑ | EPDMS↑ |
|:--|:--|--:|--:|--:|
| Human agent | – | 100.0 | 100.0 | 97.9 |
| DiffusionDrive | MV | 98.2 | 95.9 | 84.5 |
| DriveVLA-W0 | SV | 98.5 | 99.1 | 86.1 |
| Latent-WAM | MV | 98.1 | 97.3 | 89.3 |
| DriveFuture | MV | 98.8 | 99.1 | 89.9 |
| DriveWAM | SV | 98.4 | 96.7 | 92.2 |
| BeyondDrive | SV | 98.2 | 97.7 | 92.9 |
| **NegWAM** | SV | 98.1 | **98.4** | **93.3** |

- navtest 上只比 BeyondDrive 高 0.4 分，但 DAC 从 97.7 提到 98.4，高于 BeyondDrive 与 DriveWAM（96.7），仍低于 DriveVLA-W0 / DriveFuture 的 99.1；NC 98.1 略低于 DriveWAM 的 98.4。
- 与 navhard 的 +1.6 对照可以看出：这套后训练在**困难 split 上的收益远大于常规 split**，符合"补的是近专家安全边界"的定位。

**3) HUGSIM 零样本（Table III）**

| Method | RC Overall↑ | HD-Score Overall↑ | RC E/M/H/X | HD E/M/H/X |
|:--|--:|--:|:--|:--|
| DriveWAM | 43.6 | 27.2 | 76.6 / 33.8 / 33.9 / 30.2 | 64.0 / 16.1 / 16.1 / 12.6 |
| UniAD | 40.6 | 28.9 | 58.6 / 41.2 / 40.4 / 26.0 | 48.7 / 29.5 / 27.3 / 14.3 |
| **NegWAM** | **45.1** | **30.6** | 90.3 / 32.8 / 31.2 / 26.1 | 81.7 / 15.8 / 14.2 / 10.8 |

- 相对 DriveWAM，RC +3.4%、HD-Score +12.5%，但**增益全部集中在 Easy 档**（RC 76.6→90.3、HD 64.0→81.7）。
- Medium / Hard / Extreme 三档全部小幅退步：HD 16.1→15.8、16.1→14.2、12.6→10.8。这与"困难场景更安全"的叙事有张力，作者只写了"transfer varying across scenario difficulty"。

**4) 组件消融（Table IV，navhard）**

| 后训练 | Repulsive anchors | Rider tokens | EPDMS↑ |
|:--:|:--:|:--:|--:|
| × | × | × | 37.7 |
| ✓ | × | × | 38.2 |
| ✓ | ✓ | × | 39.1 |
| ✓ | × | ✓ | 38.8 |
| ✓ | ✓ | ✓ | **40.4** |

- **matched post-training control 38.2 是关键对照**：同样多训 10K 步只涨 0.5，说明 40.4 不是"多训练"带来的。
- 单独加排斥 +0.9、单独加 rider +0.6；两者合并后总增益 2.7（37.7→40.4），高于两者单独增益之和 1.5，作者据此主张两种机制不是冗余。
- 场景-轨迹兼容性的定性证据：三组场景-轨迹配对的匹配分数都为正（1.11 / 0.32 / 1.58），所有错配为负。

**5) 超参与训练预算（Table V、Figure 5/6）**

| 维度 | 取值 → EPDMS |
|:--|:--|
| 学习率 | $5\times10^{-7}$ → 39.5；$1\times10^{-6}$ → **40.4**；$2\times10^{-6}$ → 39.1 |
| $\lambda_{\text{rep}}$ | 50 → 39.0；100 → **40.4**；150 → 39.8 |
| $\lambda_{\text{compat}}$ | 0.05 → 39.3；0.1 → **40.4**；0.2 → 39.1 |

- 所有测试设置落在 39.0–40.4，说明对中等幅度的超参变化不敏感；但从 40.4 到 39.1 的落差也说明默认值确实是被挑出来的。
- 后训练步数从 10K 到 30K 一直优于 matched baseline（三 seed 均值±标准差），navtest 上 5K–30K 同样稳定领先。

**6) 推理开销（Table VI）**

| Method | 延迟 (ms)↓ | FPS↑ | 模型大小 (MB)↓ | 峰值显存 (MB)↓ | EPDMS↑ |
|:--|--:|--:|--:|--:|--:|
| DriveWAM | 871.35 | 1.15 | 6241.81 | 14029.08 | 37.7 |
| **NegWAM** | 877.86 | 1.14 | 6241.81 | 14029.08 | **40.4** |

- 延迟只涨 6.51 ms（0.75%），模型大小与峰值显存完全相同——因为 rider 组与兼容头在推理时被整体删除，收益被"烧进"了共享 backbone。

## Strengths & Weaknesses

**Strengths**

- **监督信号的选择有依据**：不从策略采样（诊断实验给出 8 条轨迹两两 ADE 仅 0.02 m），而是从固定轨迹词表里挖近专家失败，避开了"强场景条件下采样多样性低"这个实际障碍。这个诊断本身就是一个可复用的教训。
- **"近专家"这个区间的设定很关键**：$[0.3,1.0)$ m 既保证负样本与专家足够接近（属于"看起来合理但危险"），又不至于变成明显的坏轨迹；同时用 $[\alpha a-d]_+$ 的 hinge 保证在专家轨迹处损失恒为零，不会破坏 imitation。
- **两个组件作用在不同层面且互相补充**：排斥锚只约束输出轨迹，rider token 通过共享 attention 把负样本送进表示层、约束"场景-轨迹"兼容性。消融显示两者叠加的总增益超过各自单独增益之和，这是"表示层监督不是轨迹层正则的重复"的直接证据。
- **零初始化兼容头是一个细节但重要的设计**：初始分数相同、初始梯度为零，避免后训练开始阶段用随机排序把 backbone 带偏；这条梯度通路是"逐渐打开"的。
- **matched post-training control 做得对**：38.2 这一行把"多训 10K 步"的收益单独测出来，否则无法说明 40.4 来自负样本而非额外优化。
- **推理零开销的论证完整**：不只是口头说"删除分支"，而是给出延迟（+6.5 ms）、FPS、模型大小、峰值显存四项，模型大小与显存完全相同这一点尤其能说明 rider 分支确实不参与推理。
- **诚实报告了失败案例与适用范围**：附录明确列出两个 NegWAM 比 DriveWAM 更差的场景（未跟随左侧超车、转弯净空不足），并在 A-D 里承认负样本覆盖受词表限制、标签继承评估器假设、ADE 不等于交互风险。这比只报成功案例可信。

**Weaknesses**

- **负样本标签完全依赖离线评估器**：0 EPDMS 是一个二值标签，把碰撞、驶出可行驶区域、方向违规等不同严重程度混为一类；两个目标都继承这套假设。换一个评估器或换一个 benchmark 的指标缓存，负样本池会整体改变，方法的核心输入不具备跨基准稳定性。
- **"几何接近"被当成"风险接近"**：ADE 只度量平面位置差异，不考虑时间、相对速度与道路几何；两条路径相似但一条在当前车流下会撞、另一条不会，这种情况 ADE 分辨不出来。作者在附录里承认，但正文的主结论仍建立在"近专家 = 高危"这一近似上。
- **词表覆盖决定安全边界**：负样本只能来自 GTRS 的 16K 词表，词表之外的稀有机动（非常规避让、复杂交互）得不到直接负监督。这意味着方法学到的是"词表内的失败模式"，而不是一般意义上的安全边界。
- **HUGSIM 的增益与主张不匹配**：RC/HD 的提升全部来自 Easy 档，Medium/Hard/Extreme 三档都退步（Extreme HD 12.6→10.8）。如果安全后训练真的改善了困难场景，应该在这里也看到收益；论文只写"transfer varying"，没有解释为什么极端档反而更差。
- **navtest 的 +0.4 在噪声量级**：93.3 vs 92.9，而论文只给了后训练步数的三 seed 曲线，没有给最终 checkpoint 的标准差；navhard 的 +1.6 相对更稳（但同样没有报 CI）。
- **navhard 的分项变化不自洽**：DAC 提升 2.2，但 NC 与 TTC 都略降（79.5→79.2、75.7→75.3），DDC 也从 87.8 降到 86.4。作者用总 EPDMS 说明安全改善，但"总安全"里最直接的碰撞项并没有变好，这与"学习安全边界"的表述存在张力。
- **排斥项与兼容项的分工没有被完全隔离**：消融只做了"有无"，没有做"只对专家-负样本做排序、不做排斥"或"把负样本换成正样本对照"这类更强的控制；因此残余增益有多少来自对比学习的一般正则化效应，无法排除。
- **$\lambda_{\text{rep}}=100$ 与 $\lambda_{\text{compat}}=0.1$ 相差三个数量级**：论文给了敏感性表，但没有说明这两个量级差异的来源（是否因为两个损失的梯度尺度本身就差三个数量级），复现时需要重新扫参。
- **闭环证据只有定性**：HUGSIM 只展示两段 rollout 图，没有按场景类型统计成功/失败；唯一的定量闭环结果就是 Table III 的四档均值，样本量（436 场景）下 Medium/Hard 的 0.3–1.9 分差异是否显著未知。

## Notes

+ **论文定位**：不是新的 WAM 架构，而是**给统一 world-action 模型加一条"负样本后训练"通道**。核心 insight 是：专家示范只说明"什么可以做"，近专家失败轨迹才标出"什么看起来可以做但不能做"；而且这类负样本可以同时约束输出轨迹（排斥）与共享表示（兼容性排序），后者是它与既有 hard-negative planner 正则的本质区别。
+ **可迁移点**：
  - **"从固定动作词表挖负样本、用离线评估器打二值标签"是一个便宜且可复用的后训练数据管线**：不需要在线仿真、不需要 reward model，只需要一个能批量评估轨迹的指标缓存。
  - **距离自适应 hinge $[\alpha a-d]_+$**：用负样本到专家的距离 $a$ 自动设定 margin，既保证专家处损失为零，又能让不同远近的负样本各自激活；任何"推离坏样本"的正则都可以借用。
  - **通过可见性掩码把辅助 token 组接入冻结前向路径**：rider 只看当前观测、原始序列看不到 rider，因此主干行为在初始化时完全不变，辅助目标只通过共享 attention 的权重更新生效。这是"加辅助分支而不破坏原模型"的通用做法。
  - **辅助头零初始化**：让新目标在训练开始时梯度为零、随头部学习逐渐介入，避免后训练早期用随机信号污染 backbone。
  - **必须报 matched post-training control**：只多训同样的步数能涨多少，是把"负监督的贡献"从"多训练的贡献"里分离出来的最低要求。
+ **待跟进**：
  - 换一个评估器（例如 HUGSIM 的 HD-Score 或规则化的驾驶方向检查）重建负样本池，检验增益是否依赖 NAVSIM v2 指标缓存本身。
  - 把 ADE 换成带时间的风险度量（TTC、相对速度加权的距离），看能否修复 HUGSIM 极端档的退步。
  - 补一组"把负样本替换成随机正样本"的对比学习对照，确认兼容性分支的增益来自负样本的语义，而不是额外的对比正则。
  - 统计 HUGSIM 四档的场景级失败类型，解释为什么 Easy 档大幅提升、Extreme 档反而下降。
