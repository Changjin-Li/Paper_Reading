---
title: "Bridging Scene Generation and Planning: Driving with World Model via Unifying Vision and Motion Representation"
authors: ["Xingtai Gui", "Meijie Zhang", "Tianyi Yan", "Wencheng Han", "Jiahao Gong", "Feiyang Tan", "Cheng-zhong Xu", "Jianbing Shen"]
institute: ["University of Macau", "Afari Intelligent Drive"]
date_publish: 2026-03
venue: arXiv Preprint
tags: [world-model, imitation-learning]
url: https://arxiv.org/abs/2603.14948
arxiv_id: "2603.14948"
doi: "10.48550/arXiv.2603.14948"
code: https://github.com/TabGuigui/WorldDrive
cite_key: gui2026worlddrive
rating: 3
content_scope: full-text
verification_status: source-checked
date_added: 2026-10-18
---

## Summary

`WorldDrive` 试图消除"生成未来"和"决定未来"之间的表征割裂：先训练一个 **Trajectory-aware Driving World Model (TA-DWM)**，以轨迹词表作为运动条件生成可控未来视频；再把冻结的视觉与轨迹编码器**继承**给多模态 planner；最后用 **Future-aware Rewarder (FAR)** 在训练期从冻结世界模型蒸馏未来 latent，推理期无需生成视频即可对候选轨迹重新打分。在 `NAVSIM` navtest 上以单前视相机取得 **88.1 PDMS**（非 † 设置下 camera-only 方法中最高，与多视角 `DiffusionDrive` 持平），全 navtrain 训练后升至 **89.0**（表中非 oracle 最高）；`NAVSIM-v2` navhard **34.9 EPDMS**；`nuScenes` 开环 **L2 0.68 m / CR 0.38%**。生成侧 `nuScenes` val 256×512 取得 **FID 12.8 / FVD 131.7**，单卡 `A800` 规划推理 **53 ms**。

## Problem & Motivation

- **场景生成与规划的表征目标不一致**：现有 driving world model 主要优化视觉重建，运动表征没有被设计成"可被 planner 共享、可继承"的形式；planner 则独立地做动作回归。两者特征空间不对齐，planner 无法真正利用世界模型学到的动力学与运动先验。
- **现有两种集成范式各有硬伤**（Figure 1）：
  - (a) 先生成未来场景、再交给下游 planner：生成高保真视频的计算开销和延迟对实时规划不可接受。
  - (b) 完全在 latent world model 内规划：效率高，但牺牲了可解释的场景仿真与显式的视觉验证能力。
- **论文的核心主张**：**能生成未来的 latent，应该就是用来决定未来的 latent**。实现路径不是把两个模型简单拼接，而是让运动意图以结构化方式进入生成模型，再把生成模型训练好的编码器原样交给 planner。
- **由此派生三个设计目标**：动作可控的未来生成、可继承的视觉/运动表征、低延迟的前瞻评估。

## Method

**0) 总体框架（Figure 2）**

系统分两个耦合阶段：Phase 1 训练 TA-DWM 做条件视频生成；Phase 2 冻结其视觉与轨迹编码器，训练多模态 planner 与 FAR。FAR 中的未来场景生成器只在训练期使用，推理期被蒸馏出的轻量 decoder 替代。

**1) TA-DWM：把轨迹变成结构化运动条件（§3.1）**

- 历史观测 $x \in \mathbb{R}^{T \times 3 \times H \times W}$ 先经**冻结的 3D Causal VAE** 编码，再通过一个可训练的轻量 **visual adapter** 对齐到驾驶域，得到时空视觉 latent：

  $$
  f = E_{\text{vis}}(x), \qquad f \in \mathbb{R}^{C \times H' \times W'}.
  $$

- 运动侧不使用单条专家轨迹直接做 condition，而是先构造**轨迹词表** $V \in \mathbb{R}^{N \times F \times 3}$（对大量驾驶日志做 K-means，$N=256$）。给定专家轨迹 $Y$，取 top-$K$ 最近锚点 $V_K$，再用残差编码细粒度调整：

  $$
  c = E_a(V_K) + E_o(Y - V_K), \qquad c \in \mathbb{R}^{K \times C},
  $$

  其中 $E_a$ 为 anchor encoder，$E_o$ 为 offset encoder。这样运动条件同时包含"粗意图"与"精细修正"。

- 未来帧 $\bar{x}$ 由同一 VAE 编码为目标 latent $z_0 = E_{\text{vis}}(\bar{x})$，加噪得 $z_t$；**TA-DiT** 在历史 latent $f$、扩散步 $t$ 与运动嵌入 $c$ 的条件下预测噪声：

  $$
  \mathcal{L}_{\text{world}} = \mathbb{E}_{z_0,t,\epsilon}\left[\left\|\epsilon - \epsilon_\theta(z_t; f, t, c)\right\|_2^2\right].
  $$

- 实现上，轨迹嵌入通过 **Motion AdaLN** 注入 transformer block，作用类似 CogVideoX 中的 text prompt；历史 latent 通过 **Vision AdaLN** 注入。未来 latent 目标由历史帧与未来帧的 latent 沿时间轴拼接构成。

**2) Representation inheritance 与多模态 planner（§3.2）**

- Planner 直接继承 TA-DWM 的 **visual adapter、visual encoder 与 multi-modal trajectory encoder**，并在 Phase 2 全程冻结。这是论文所称的 representation inheritance：planner 一出生就处在被场景生成任务预对齐过的特征空间里。
- Ego status（速度、加速度、驾驶命令）经轻量 MLP 编码为 $e \in \mathbb{R}^{1 \times C}$。
- 规划被建模为**集合预测**：全部预定义锚点嵌入 $Q_a = E_a(V)$ 作为 queries，历史视觉 latent 与 ego embedding 拼接后作为 key/value：

  $$
  Q_p = D_{\text{plan}}\!\left(Q_a,\ [f, e],\ [f, e]\right),
  $$

  其中 $D_{\text{plan}}$ 是 planning transformer decoder。每个 query 因而对应一个运动基元及其环境上下文。
- 沿用 WoTE 的打分范式，设置 imitation reward head、simulation reward head 与 offset regression head；按 combined score 取 top-$K$ 锚点，再用预测 offset 精修，得到多模态候选集 $\hat{V}_K$。

**3) Future-aware Rewarder：把前瞻压进轻量 scorer（§3.3）**

- 对每个候选轨迹都跑完整扩散采样不可行，因此 FAR 只在**训练期**调用冻结的 TA-DiT 生成未来 latent $z_k$，并学习一组可学习的 **Future Scene Queries** $Q_s \in \mathbb{R}^{M \times C}$ 作为信息瓶颈：

  $$
  \hat{z}_k = D_{\text{scene}}\!\left(Q_s,\ [f, c_k],\ [f, c_k]\right),
  $$

  其中 $c_k$ 是候选轨迹 $k$ 的运动嵌入。随后用轨迹嵌入查询蒸馏出的未来特征：

  $$
  h_f^k = D_{\text{future}}\!\left(c_k,\ \hat{z}_k,\ \hat{z}_k\right),
  $$

  再由 MLP 映射为标量 reward $r_k$，推理时取 $\arg\max_k r_k$。
- 对齐损失让轻量 decoder 逼近世界模型生成的未来 latent：

  $$
  \mathcal{L}_{\text{align}} = \mathbb{E}_k\left[\left\|\hat{z}_k - \mathrm{SG}(z_k)\right\|_2^2\right],
  $$

  $\mathrm{SG}(\cdot)$ 为 stop-gradient。
- 选择本身用 Bradley-Terry 偏好排序学习：

  $$
  \mathcal{L}_{\text{reward}} = -\mathbb{E}_{(v_{\text{pos}}, v_{\text{neg}}) \sim \hat{V}_K}\left[\log \sigma\!\left(r_{\text{pos}} - r_{\text{neg}}\right)\right].
  $$
- 偏好对构造（§6.3）：每个场景先取 planner 的 **top-16** 候选；其中 planner score 最高的 top-1 作为正样本，simulation score 最低的 3 条作为 hard negatives，再从剩余池随机取 3 条增加多样性。

**4) 训练目标与实现细节**

- Planner 损失 $\mathcal{L}_{\text{plan}}$ 由三项组成：imitation reward 的 cross-entropy、simulation reward 的 binary cross-entropy、positive anchor 的 offset L1 回归。FAR 总目标为 $\mathcal{L}_{\text{align}} + \mathcal{L}_{\text{reward}}$，训练时冻结 planner。
- 关键实现设置：

| 项目 | 设置 |
|:-----|:-----|
| 生成基座 | 3D VAE 与 DiT 均从 `CogVideoX` 初始化 |
| 轨迹词表 | K-means，256 个 anchors，follow WoTE |
| Phase 1 预训练 | nuPlan 200k iters → nuScenes 100k iters；256×512；8 历史帧 → 17 未来帧；10 Hz；恒定 lr $1\times10^{-4}$ |
| NAVSIM 适配 | 100k iters；512×1024；4 历史帧（2 s）→ 9 未来帧（4.5 s）；lr $5\times10^{-5}$；padding 满足 VAE 空间约束 |
| Phase 1 硬件 | 16×A100，batch size 32 |
| Planner | 50 epochs；8×RTX 3090；batch size 256；cosine schedule，peak lr $6\times10^{-4}$；编码器冻结 |
| FAR | 10 epochs；lr $3\times10^{-4}$；PDMS 作为 preference oracle |
| 推理 | 单卡 A800，batch size 1，$K=5$，不生成未来视频 |

## Pipeline

![WorldDrive 整体框架](https://arxiv.org/html/2603.14948v1/main_fig_v29.png)

1. **离线 Phase 1-a**：在 nuPlan 上以 8 帧历史预测 17 帧未来，训练 TA-DWM 的 visual adapter、trajectory encoder 与 TA-DiT；3D VAE 保持冻结。
2. **离线 Phase 1-b**：在 nuScenes 上继续微调 100k iters，得到论文报告生成指标所用的模型。
3. **NAVSIM 适配**：提高分辨率到 512×1024，改为 2 s 历史 → 4.5 s 未来，继续训练 TA-DWM。
4. **Planner 训练**：继承并冻结适配后的 visual adapter / visual encoder / trajectory encoder，在 navtrain 上训练 50 epochs；输出 top-$K$ 候选及 offset 精修结果。
5. **FAR 训练**：冻结 planner，对每个场景取 top-16 候选，用冻结 TA-DWM 生成对应未来 latent，构造正/负偏好对，训练未来场景 decoder、future-aware decoder 与 reward MLP。
6. **推理**：前视图像 → 冻结 VAE + adapter 得 $f$；ego status → MLP 得 $e$；全部 anchor queries 做 cross-attention → 打分取 top-5 → offset 精修 → FAR 对 5 条候选打 reward → 输出最高分轨迹。需要可视化时，才把选中轨迹送回 TA-DiT 生成未来视频。

## Key Results

**1) NAVSIM navtest（Table 1）**

| Method | Sensors | WM | NC | DAC | TTC | Comf. | EP | PDMS |
|:--|:--|:--:|--:|--:|--:|--:|--:|--:|
| TransFuser | MV & L | × | 97.7 | 92.8 | 92.8 | 100.0 | 79.2 | 84.0 |
| DiffusionDrive | MV & L | × | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| WoTE | MV & L | ✓ | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| ImagiDrive | SV | ✓ | 98.6 | 96.2 | 94.5 | 100.0 | 80.5 | 87.4 |
| Epona | SV | ✓ | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| **WorldDrive** | **SV** | ✓ | 98.4 | 96.2 | 95.1 | 100.0 | 81.9 | **88.1** |
| PWM† | SV | ✓ | 98.6 | 95.9 | 95.4 | 100.0 | 81.8 | 88.1 |
| DriveVLA-W0† | SV | ✓ | 98.7 | 96.2 | 95.5 | 100.0 | 82.2 | 88.4 |
| **WorldDrive†** | **SV** | ✓ | 98.4 | 96.8 | 95.2 | 100.0 | 83.3 | **89.0** |
| WorldDrive‡ | SV | ✓ | 99.3 | 98.8 | 97.9 | 100.0 | 88.3 | 93.6 |

- †：用完整 navtrain 训练；‡：6 条候选中用 oracle 选最优，仅用于探测上界，推理不使用。
- 单目、仅相机条件下 88.1 与多视角 + LiDAR 的 `DiffusionDrive` 持平，且在非 † 设置中高于 `ImagiDrive`（87.4）与 `Epona`（86.2）；但 `WoTE`（88.3）与 `DriveVLA-W0†`（88.4）仍更高。完整 navtrain 后 89.0 超过 `PWM†`（88.1）与 `DriveVLA-W0†`（88.4）。

**2) NAVSIM-v2 navhard（Table 2）**

| Method | Sensors | Stage | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
|:--|:--|:--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| LTF | MV | S1 | 96.2 | 79.5 | 99.1 | 99.5 | 84.1 | 95.1 | 94.2 | 97.5 | 79.1 | 23.1 |
| LTF | MV | S2 | 77.7 | 70.2 | 84.2 | 98.0 | 85.1 | 75.6 | 45.4 | 95.7 | 75.9 | — |
| DiffusionDrive | MV | S1 | 96.8 | 86.0 | 98.8 | 99.3 | 84.0 | 95.8 | 96.7 | 97.6 | 79.6 | 27.5 |
| DiffusionDrive | MV | S2 | 80.1 | 72.8 | 84.4 | 98.4 | 85.9 | 76.6 | 46.4 | 96.3 | 72.8 | — |
| **WorldDrive** | **SV** | S1 | 97.3 | 89.1 | 97.6 | 99.7 | 60.5 | 96.8 | 87.7 | 93.1 | 60.0 | **34.9** |
| **WorldDrive** | **SV** | S2 | 91.4 | 82.0 | 91.0 | 98.5 | 53.1 | 90.6 | 52.3 | 93.3 | 62.8 | — |

- EPDMS 34.9 明显高于 LTF 23.1 与 DiffusionDrive 27.5，且 NC/DAC/DDC/TTC 在 S1、S2 两个 stage 都更强。
- 但 EP 只有 60.5/53.1，远低于两个基线的约 84–86；EC 也只有 60.0/62.8。EPDMS 的领先并非所有子项都领先。

**3) nuScenes val 开环（Table 3）**

| Method | Sensors | L2 3s ↓ | L2 Avg. ↓ | CR 3s ↓ | CR Avg. ↓ |
|:--|:--|--:|--:|--:|--:|
| SparseDrive | V | 0.96 | 0.61 | 0.18 | 0.08 |
| World4Drive | V | 0.81 | 0.50 | 0.33 | 0.16 |
| Drive-WM | V | 1.20 | 0.80 | 0.48 | 0.26 |
| Epona | SV | 1.98 | 1.25 | 0.85 | 0.36 |
| **WorldDrive** | **SV** | **0.68** | **0.42** | 0.38 | 0.16 |

- L2 最低，但 CR 并非最低：`SparseDrive` 的 0.18/0.08 明显更好。

**4) 预训练消融（Table 4）**

| VAE Pretrain | TA-DWM Vision | TA-DWM Motion | NC | DAC | TTC | EP | PDMS |
|:--:|:--:|:--:|--:|--:|--:|--:|--:|
| × | × | × | 69.0 | 57.6 | 57.2 | 28.7 | 31.4 |
| ✓ | × | × | 97.6 | 93.6 | 93.2 | 79.5 | 84.9 |
| ✓ | ✓ | × | 97.9 | 94.1 | 93.3 | 80.7 | 85.8 |
| ✓ | ✓ | ✓ | 98.4 | 95.1 | 94.7 | 80.4 | 86.9 |

- 从 31.4 到 84.9 的 **+53.5 PDMS** 来自 CogVideoX 3D VAE 初始化；TA-DWM 的视觉与运动编码器继承分别只再加 **+0.9** 与 **+1.1**。

**5) FAR 消融（Table 5 与 Table 8）**

| Traj feat. | Future feat. | NC | DAC | TTC | EP | PDMS |
|:--:|:--:|--:|--:|--:|--:|--:|
| × | × | 98.4 | 95.1 | 94.7 | 80.4 | 86.9 |
| ✓ | × | 98.3 | 95.3 | 94.3 | 81.2 | 87.0 |
| ✓ | ✓ | 98.4 | 96.2 | 95.1 | 81.9 | 88.1 |

- 只用 trajectory feature 仅 +0.1；加入 future feature 后 +1.2。
- 候选数 $K=1/3/5/10$ 对应 PDMS 86.9/87.9/88.1/87.6，默认 $K=5$。

**6) 推理延迟（Table 6）与生成质量（Table 7）**

| Method | NC | DAC | TTC | EP | PDMS | Latency |
|:--|--:|--:|--:|--:|--:|--:|
| PWM† | 98.0 | 95.1 | 94.1 | 82.4 | 87.3 | 570 ms |
| PWM | 98.6 | 95.9 | 95.4 | 81.8 | 88.1 | 850 ms |
| **WorldDrive** | 98.4 | 96.8 | 95.2 | 83.3 | **89.0** | **53 ms** |

- 53 ms 为单卡 A800、batch size 1、$K=5$；其中 encoders 17.9 ms、planner 18.9 ms、FAR 16.2 ms。

| Method | Resolution | FID ↓ | FVD ↓ |
|:--|:--:|--:|--:|
| DriveDreamer | 128×192 | 52.6 | 452.0 |
| WoVoGen | 256×448 | 27.6 | 417.7 |
| GenAD | 256×448 | 15.4 | 184.0 |
| Drive-WM | 192×384 | 15.2 | **122.7** |
| Vista | 480×832 | 18.2 | 158.0 |
| Driverse | 480×832 | 20.1 | 143.5 |
| **WorldDrive** | 256×512 | **12.8** | 131.7 |

- FID 最低；FVD 131.7 高于 `Drive-WM` 的 122.7，而后者分辨率更低（192×384），两者不能直接横比。

**Critical reading**

- **增益归因**：真正的大头是"用 CogVideoX VAE 初始化"（+53.5），"统一视觉与运动表征"只贡献 +2.0（+0.9 视觉、+1.1 运动）。论文的叙事重心在后者，但消融数字更支持"强视频先验 + 轻量继承"这一解释。无预训练时 31.4 PDMS 也说明该 planner 极度依赖冻结的预训练编码器，本身并不能从零学起。
- **FAR 的评测泄漏风险**：FAR 的偏好监督来自 NAVSIM 的 simulation/PDMS 分数，而最终评测也是 PDMS。+1.2 的增益可能部分来自把评测规则蒸馏进 reward model，而不是真正利用了未来 latent 的动力学信息。要证明前瞻有用，需要与"只用仿真分数训练的同容量 reranker"做对照，论文没有提供这个 baseline。
- **候选池与选择器的差距**：best-of-6 oracle 93.6 vs FAR 88.1，差 5.5 PDMS。这既证明候选集多模态，也说明**选择才是瓶颈**；FAR 只回收了其中一小部分。论文把它当作上界分析，但没有进一步分析 FAR 在哪些场景选错。
- **navhard 的领先结构不均衡**：EP 60.5/53.1 远低于 LTF 与 DiffusionDrive 的 84–86，EC 也只有 60 左右；EPDMS 更高主要来自 NC/DAC/DDC/TTC 等安全与合规项。需要确认 EPDMS 的加权方式，以及这是否是"用慢速/保守换安全"。
- **开环不是全面领先**：nuScenes L2 最低，但 CR 0.38/0.16 不及 SparseDrive 的 0.18/0.08；论文措辞是 "highly competitive"，与数字一致。
- **可控性证据偏弱**：motion sensitivity 只用"latent cosine similarity 随轨迹几何距离下降"来证明，没有报告生成视频中自车轨迹与条件轨迹的几何误差（如 ADE），也没有验证生成场景中他车/车道几何是否物理一致。
- **单目约束**：TA-DWM 以单前视视频训练，planner 也只能用单前视输入。NAVSIM 的 SV 设置下这是公平评测，但真实部署的侧后方感知、遮挡场景未验证；论文的 "vision-only" 实际是 camera-only 且 single-view，和表中 MV 方法不是同一感知条件。
- **"closed-loop" 措辞**：NAVSIM v1 本身是非反应式 pseudo-simulation，NAVSIM-v2 才是带反应式交通的 pseudo closed-loop；论文把两者都称为 closed-loop，容易 overclaim。
- **关键细节缺失**：没有参数量、FAR 训练时 $z_k$ 的采样方式（是否为每个候选跑完整扩散、采样步数多少）、偏好数据规模；也没有生成质量对下游 planning 的因果消融（例如降低 FVD 是否提升 PDMS）。
- **† 的含义不完整**：caption 只说 † 用 full navtrain，但未说明非 † 设置具体用了哪个子集/多少数据；Table 6 中 WorldDrive 的 89.0 实际对应 † 模型，非 † 的 88.1 模型延迟没有单独报告。

## Strengths & Weaknesses

**Strengths**

- **问题表述清晰**：把"生成未来的表征"和"决定未来的表征"统一，是对 driving world model 与 planner 关系的一个干净 reformulation，而不是又一个模块堆叠。
- **运动条件设计合理**：轨迹词表 + 残差编码把连续轨迹压成"粗锚点 + 细修正"，比直接用一条轨迹做 condition 更适合多模态生成，也给 planner 提供了结构化 query 空间。
- **表征继承简单有效**：冻结编码器 + 轻量 planner，Phase 2 只需 8×3090 训练 50 epochs，工程上可复现、可迁移。
- **FAR 的 train-only 设计实用**：用未来 latent 蒸馏替代推理期视频生成，53 ms 的延迟相比 PWM 的 570–850 ms 有数量级优势，是把世界模型前瞻落到实时规划的一种可行工程解。
- **结果覆盖面较好**：同时报告 NAVSIM、NAVSIM-v2、nuScenes 与生成指标，且给出了预训练、FAR、K 值三组消融，关键数字可核对。
- **诚实报告上界**：主动给出 best-of-6 oracle 93.6 并声明推理不使用，也承认部分 comfort 指标仍具挑战。

**Weaknesses**

- **核心增益的归因不利**：统一表征只带来 +2.0 PDMS，远小于通用视频预训练的 +53.5；"representation unification" 的独立价值没有被干净地隔离出来。
- **FAR 与评测指标耦合**：用 PDMS/仿真分数构造偏好、又在 PDMS 上评测，存在 metric leakage；缺少"同容量、只用仿真分数"的 reranker 对照。
- **选择器是瓶颈**：oracle 与 FAR 相差 5.5 PDMS，说明候选质量足够但 reward model 不够好；论文没有分析失败模式。
- **navhard 的 EP/EC 明显偏弱**：安全项强、效率项弱，EPDMS 的领先可能被指标权重放大。
- **单目限制**：无法利用多视角，真实场景的侧后方与遮挡问题未覆盖。
- **生成评估不完整**：FVD 不是最低，跨分辨率比较不公平；没有报告采样步数、生成延迟与可控性几何误差。
- **方法新颖性偏工程组合**：轨迹词表、打分头、simulation reward 均沿用 WoTE，主要新意集中在 inheritance + FAR；两阶段、多冻结策略也增加了复现复杂度。

## Notes

**论文定位**

- 把 driving world model 当作**表征预训练器 + 轨迹评估器**，而不是推理时必须调用的视频模拟器；核心贡献是把"共享表征"从口号落成一条可执行的两阶段继承 pipeline。
- 更接近 "world model as pretraining + reranker" 的路线，而不是 "world model as online simulator" 的路线。

+ **可迁移点**：
  - **轨迹词表 + 残差**可作为任何视频 world model 的通用运动条件接口：粗锚点负责多模态覆盖，残差负责精度。
  - **frozen encoder inheritance** 可迁移到"大规模视频/世界模型预训练 → 下游规划/控制"的场景，显著降低下游训练成本。
  - **train-only future latent distillation + Bradley-Terry ranking** 是一种把生成模型的前瞻能力压缩进轻量 scorer 的通用模板，可复用到其他 candidate reranking 任务。
  - **top-K 候选 + hard negative + random negative** 的偏好对构造方式简单，适合候选集规模不大的场景。

+ **待跟进**：
  - 如果不用 PDMS/仿真分数、而用真实碰撞或人工偏好构造 preference，FAR 的 +1.2 还剩多少？
  - 训练 FAR 时 $z_k$ 是如何得到的：每个候选都跑完整扩散采样，还是用单步/少步近似？这部分数据构造成本没有报告。
  - EPDMS 的权重定义是什么？EP 大幅下降是否意味着策略在用"慢"换"安全"？
  - 多视角、多传感器扩展后，representation inheritance 是否仍然成立，还是单目视频预训练的表征会成为瓶颈？
  - 能否把 TA-DWM 的生成能力用于 hard case mining / data augmentation，而不仅是表征预训练与离线可视化？
