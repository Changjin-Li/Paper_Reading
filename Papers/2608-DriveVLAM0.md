---
title: "DriveVLA-M0: Failure-Aware Memory Augmentation for Autonomous Driving"
authors: ["Zebin Xing", "Yupeng Zheng", "Qiang Chen", "Linbo Wang", "Yichen Zhang", "Pengxuan Yang", "Junli Wang", "Deheng Qian", "Xiaoqing Ye", "Junyu Han", "Yifeng Pan", "Qichao Zhang", "Dongbin Zhao"]
institute: ["Institute of Automation, Chinese Academy of Sciences", "Chongqing Chang'an Technology Co., Ltd."]
date_publish: 2026-08
venue: ACM MM 2026
tags: [VLA, spatial-memory]
url: https://arxiv.org/abs/2608.10413
cite_key: xing2026drivevlam0
arxiv_id: "2608.10413"
doi: "10.1145/3767308.3835233"
code: https://github.com/ZebinX/DriveVLA-M0
rating: 4
content_scope: full-text
verification_status: source-checked
date_added: 2026-11-26
---
## Summary

VLA 驾驶模型会在相似场景上反复犯同样的错，因为训练完就没有"记住上次失败"的机制。DriveVLA-M0 建一个**离线失败案例潜记忆库**：用 oracle PDM 打分筛出低分场景，把检索键（静态地图特征 + 动态 agent 特征）、动作解码器的中间特征（语言嵌入、ego 嵌入、候选轨迹簇）与监督标签（专家轨迹 + 各候选的 oracle 子分数）一起存进去。检索键由一个专门训练的 Retrieve Model 产生，它用解耦的双 LoRA 分支分别学道路结构与 agent 布局，而不是直接用 VLM 的语言特征。推理时按地图与 agent 两路分别检索 top-k，命中相似度超过阈值才触发 **Decoupled LoRA 的 test-time training**：地图检索到的案例只调静态 LoRA 分支，agent 检索到的案例只调动态分支，各自负责可行驶区域类与碰撞类子分数。NAVSIM v1 PDMS 94.1（Base 92.3）、NAVSIM v2 navhard 47.0 EPDMS、navtest 88.9 EPDMS；TTT 反向仅 26.44 ms（全量微调 55.42 ms），记忆扩到 10K 案例可在不重训主干的情况下继续涨点。

## Problem & Motivation

- VLA 驾驶模型缺少"从过去失败中学习"的机制：同一类场景（无保护左转、遮挡路口）会持续失败，且分布漂移时没有在线适应手段。
- 现有记忆增强方法（MemoryVLA、EvoVLA 等）主要在**视觉-语言空间**做检索，而驾驶决策真正依赖两类信息：动态 agent 的运动、静态道路拓扑；语言特征能表达语义相似，却未必结构与动力学相似。
- 另一条路是失败案例后训练（SERA、BeyondDrive 等），但需要大规模重训；论文想要的是"不重训主干、在推理时针对性修正"。
- 因此两个设计目标：**结构化的检索**（按场景几何与 agent 布局检索，而不是按语义）与**失败感知**（只记模型表现差的场景）。

## Method

**0. 总体框架**

+ 离线阶段（Memory Generation）：基座模型跑训练/合成数据，oracle PDM 打分低于阈值 $\beta$ 的场景被写入潜记忆 $\mathbb{M}$。
+ 在线阶段（Inference with TTT）：Retrieve Model 抽当前帧的地图/agent 特征去检索，相似度超过 $\lambda$ 才触发 TTT，用检索到的案例标签微调 Action Decoder 的解耦 LoRA，再重新生成候选并按融合分数选轨迹。
+ 两个阶段共享同一套 Decoupled LoRA 结构，保证"检索时用的表示"与"适应时调的分支"一一对应。

**1. 基座模型（§3.1）**

+ VLM backbone 沿用 ReCogDrive 的 InternVL3（已在驾驶 QA 数据上微调）；取 LLM 最后一层特征 $h^{-1}$（约 2800×1536），用 Q-Former 式压缩到 $F_{\text{lang}}\in\mathbb{R}^{16\times256}$，压缩比约 1050×：

$$
F_{\text{lang}}=\mathrm{Transformer}\left(Q_{\text{cmp}},\ \mathrm{Linear}(h^{-1}),\ \mathrm{Linear}(h^{-1})\right)
$$

+ Action Decoder 分两段：Trajectory Head 把 ego 状态编码为 $F_{\text{ego}}$，与 $M$ 个可学习 ego query 相加后对 $F_{\text{lang}}$ 做 cross-attention，解出 $M$ 条候选轨迹 $\hat{\mathbb{T}}\in\mathbb{R}^{M\times8\times3}$；Score Head 再把候选重新编码、注入 ego 状态，输出每条候选的 $K$ 个子分数 $\hat{\mathbb{S}}\in\mathbb{R}^{M\times K}$，按聚合分数选一条。

**2. Retrieve Model（§3.2）**

+ 用轻量 DINOv2 + LoRA 做特征提取，但分成**两条解耦分支**：Map 分支学静态道路结构，Agent 分支学动态交通参与者：

$$
F_{\text{map}},F_{\text{agent}}=\mathrm{DINO}_{\text{LoRA}}(I)
$$

+ 每条分支后接 Transformer decoder 与独立头，分别预测 occupancy 栅格；监督沿用 Transfuser 的栅格范式，两项 BCE 加权：

$$
\mathcal{L}_{\text{Retrieve}}=\mathrm{BCE}\left(\hat{\mathcal{M}}_{\text{map}},\mathcal{M}_{\text{map}}\right)+\alpha\,\mathrm{BCE}\left(\hat{\mathcal{M}}_{\text{agent}},\mathcal{M}_{\text{agent}}\right)
$$

+ 论文的注意力可视化显示 Map 分支关注边界与车道线、Agent 分支关注前向车辆——这是"解耦确实学到不同东西"的直接证据。

**3. 记忆的构造（§3.3）**

+ 每个案例存三元组 $(k,x,y)$：$k=(F_{\text{map}},F_{\text{agent}})$ 是检索键；$x=(F_{\text{lang}},F_{\text{ego}},\hat{\mathbb{T}})$ 是 TTT 的输入；$y=(\tau,\mathbb{S})$ 是专家轨迹与 oracle 子分数标签。
+ 失败判定用 PDM 聚合分：$\mathcal{Q}(\hat\tau)<\beta$ 时写入；写入前做余弦相似度去重，避免外部数据把库撑爆。
+ 记忆是"案例级"的，不更新权重；扩展记忆只需追加案例，不需要重训基座——这是论文"training-free scaling"的来源。

**4. 推理期 TTT（§3.4）**

+ 当前帧过 Retrieve Model 得到两路 key，分别检索 top-$k$；触发条件是余弦相似度超过阈值：

$$
g=\begin{cases}1,&\cos(F,F^{*})>\lambda\\ 0,&\text{otherwise}\end{cases}
$$

+ 两路都没检索到就直接输出基座轨迹、跳过 TTT；否则为每个测试场景**重新初始化** LoRA，用检索到的 $(x,y)$ 做若干步监督微调：

$$
\theta\leftarrow\theta-\eta\nabla_{\theta}\mathcal{L}_{\text{TTT}}\left(\mathcal{C}_{\text{map}},\mathcal{C}_{\text{agent}}\right)
$$

+ 解耦体现在梯度与打分两侧：地图检索到的案例只更新 Map（静态）LoRA，agent 案例只更新 Agent（动态）LoRA；最终打分时，可行驶区域类子分数取静态分支、碰撞类子分数取动态分支，做路径感知的分数融合。
+ 图 4 的可视化说明机制：失败场景的候选簇集中在低分区域，注入后候选簇整体右移，模型才有高分轨迹可选——这解释了为什么"改候选质量"比"改选择规则"更有效。

**5. 实现与开销**

| 项目 | 设置 |
|:--|:--|
| 基座 | InternVL3（驾驶 QA 微调）+ Q-Former 压缩（16×256） |
| 检索模型 | DINOv2 + LoRA，双分支（Map / Agent），occupancy BCE 监督 |
| 记忆规模 | 主实验 4K 案例；扩展到 10K（SimScale 合成数据） |
| 硬件与延迟 | 单张 H20：检索 15.19 ms、前向 30.79 ms、LoRA 反向 26.44 ms（全量微调 55.42 ms） |
| 触发 | 余弦相似度阈值 $\lambda$，两路都为空则跳过 TTT |

## Pipeline

![DriveVLA-M0 整体框架](https://arxiv.org/html/2608.10413v2/main_fig.png)

*上半部分是离线 Memory Generation：基座模型推理、oracle PDM 打分，低分场景的检索键、中间特征与标签写入潜记忆；下半部分是在线推理：Retrieve Model 用解耦的地图/agent 特征检索相似失败案例，触发后分别微调静态与动态 LoRA 分支，再由 Action Decoder 重新生成候选并融合两路分数选轨迹。*

1. **离线打分**：基座模型在每个场景输出轨迹簇与分数，PDM 给出聚合分 $\mathcal{Q}$。
2. **写入记忆**：$\mathcal{Q}<\beta$ 的场景被存为 $(k,x,y)$，写入前去重。
3. **在线编码**：当前帧过 VLM 得到语言/ego/候选簇特征，同时过 Retrieve Model 得到 Map / Agent 两路 key。
4. **双路检索**：分别取 top-$k$，判断相似度是否超过 $\lambda$。
5. **触发判断**：两路都为空则直接输出基座轨迹；否则进入 TTT。
6. **解耦微调**：重新初始化 LoRA，地图案例只更静态分支、agent 案例只更动态分支。
7. **重打分与输出**：两分支各自给出子分数，按"静态管可行驶区域、动态管碰撞"的规则融合，选最终轨迹。

## Key Results

**1) NAVSIM v1 navtest（Table 1，PDMS）**

| Method | NC↑ | DAC↑ | EP↑ | TTC↑ | PDMS↑ |
|:--|--:|--:|--:|--:|--:|
| iPad | 98.6 | 98.3 | 88.0 | 94.9 | 91.7 |
| Centaur | 99.5 | 98.9 | 85.9 | 98.0 | 92.6 |
| DriveSuprim | 98.6 | 98.6 | 91.3 | 95.5 | 93.5 |
| DriveVLA-M0-Base | 99.0 | 97.7 | 89.5 | 95.0 | 92.3 |
| **DriveVLA-M0-Scale** | **99.1** | 98.1 | 90.2 | **98.5** | **94.1** |

+ Base 版（4K 记忆）92.3，已超过多数 VLA 方法；Scale 版（10K 记忆）94.1，比 Base 高 1.8，比最强经典 E2E（DriveSuprim 93.5）高 0.6。
+ 关键点：**94.1 与 92.3 的差别只来自记忆扩容**（加 SimScale 合成案例），不是新训练——这是论文"training-free scaling"的核心证据，但也意味着 94.1 里有一部分增益来自额外数据。

**2) NAVSIM v2（Table 2、3）**

| Split | NC | DAC | EP | TTC | EPDMS |
|:--|--:|--:|--:|--:|--:|
| navhard（Stage-2） | 91.1 | 89.5 | 65.5 | 89.2 | **47.0** |
| navtest | – | – | – | – | **88.9** |

+ navhard 47.0 明显高于同表经典 E2E（TransFuser 23.1、DiffusionDrive 28.9、GoalFlow 28.7），作者同时指出该 split 上 VLA 方法大多尚未报告结果，比较集不全。
+ navtest 88.9 比 GoalFlow 高 1.6，作者称 EP 与 DDC 为表内最高；这两项恰好对应"更敢往前开、方向更合规"，与 TTT 提升候选簇质量的机制一致。

**3) 检索策略消融（Table 4）**

| 检索键 | NC | DAC | EP | TTC | PDMS |
|:--|--:|--:|--:|--:|--:|
| 无记忆（base） | 98.4 | 97.1 | 87.7 | 95.2 | 91.0 |
| 语言特征 $F_{\text{lang}}$ | 98.0 | 97.3 | 88.1 | 93.9 | 90.7 |
| 仅地图 $F_{\text{map}}$ | 98.4 | 97.7 | 89.1 | 94.5 | 91.7 |
| **地图 + agent** | **98.9** | **97.7** | **89.6** | **95.0** | **92.3** |

+ **语言特征检索 90.7 反而低于不用记忆的 91.0**，这是全文最有价值的一行：在驾驶里用 VLM 语义相似度检索，会把结构不符的案例拉进来，产生负迁移。
+ 地图键单用 +0.7，加 agent 键再 +0.6；agent 键主要抬 NC 与 TTC，符合"动态分支管碰撞"的分工。

**4) 注入方式与触发阈值（Table 5、6）**

| 配置 | PDMS↑ |
|:--|--:|
| 无记忆 base | 91.0 |
| 离线在失败样本上再训 10 epoch | 91.2 |
| TTT 全量微调 Action Decoder | 92.4 |
| **TTT Decoupled LoRA** | **92.3** |

| 触发阈值 $\lambda$ | 0.70 | **0.90** | 0.95 | 0.99 |
|:--|--:|--:|--:|--:|
| PDMS | 90.4 | **91.7** | 91.4 | 89.4 |

+ 离线重训只涨 0.2，TTT 涨 1.3——差距说明"每个场景单独适应"比"把失败样本混进训练集"更有效。
+ LoRA 与全量微调只差 0.1 PDMS，但反向耗时从 55.42 ms 降到 26.44 ms，这是解耦 LoRA 的主要卖点。
+ 阈值过松（0.70）引入噪声、过紧（0.99）几乎不触发，0.99 时 89.4 甚至低于 base 91.0——说明"该不该适应"本身是性能的一部分。

**5) 效率（Table 7，单张 H20，4K 记忆）**

+ 检索 15.19 ms、前向 30.79 ms、LoRA 反向 26.44 ms（全量 55.42 ms）；触发一次 TTT 的总开销约为 72 ms。
+ 论文只报组件延迟，没有报触发比例与端到端 FPS；若大多数场景都触发，这个开销在实时性上不可忽略。

## Strengths & Weaknesses

**Strengths**

- 问题定位清楚：不是"再训一个更强的 VLA"，而是补上"从失败中检索并即时修正"这条缺失通路；记忆只存失败案例，比全量经验库更省也更聚焦。
- **语言特征检索差于无记忆（90.7 vs 91.0）** 这一负结果支撑了整个方法的设计前提：驾驶需要结构化检索键，而不是语义检索。这比只报正向消融更有说服力。
- 解耦贯穿三处且一致：检索键分 Map/Agent、LoRA 分支分静态/动态、打分子分数也按"可行驶区域 vs 碰撞"分派，形成闭环，不是一个孤立的模块设计。
- 触发机制是必要的工程约束：Table 6 显示阈值过松或过紧都掉点，说明"选择性适应"确实是设计的一部分，而不是可有可无的开关。
- 记忆扩容不需要重训：10K 案例把 PDMS 从 92.3 推到 94.1，给出了一条"靠数据而非参数"的扩展路径。
- 效率报告到组件级（检索/前向/LoRA 反向/全量反向），并给了同一硬件口径，读者可以自行估算触发比例下的总开销。

**Weaknesses**

- **94.1 与 92.3 的差异来自额外数据而非方法**：Scale 版用 SimScale 合成场景把记忆扩到 10K，论文把它与 4K 的 Base 并列在表 1，容易被读成"同方法的两个版本"；但两者记忆规模与数据来源都不同，增益中有多少来自方法、多少来自合成数据无法分离。
- **TTT 用的是带标签的记忆案例**：检索到的 $y=(\tau,\mathbb{S})$ 是专家轨迹与 oracle 子分数，相当于在测试时做有监督微调。这与通常意义上"无标签自监督 TTT"（TENT、TTT++）不是一回事，论文没有讨论标签泄漏与过拟合记忆的风险。
- **失败率与触发比例未报告**：记忆规模给了 4K / 10K，但没有说训练集里有多少场景被判为失败（$\beta$ 也未给），也没有说推理时有多少帧触发 TTT；因此"平均开销很小"的结论缺少分母。
- **检索质量没有被直接评估**：没有检索精度 / 召回、也没有"检索到的案例是否真的与当前场景同类失败"的人工核验；证据只有注意力图与最终 PDMS，链条较长。
- **结构相似性用 occupancy 监督间接定义**：Map/Agent 特征学的是栅格重建，而"结构相似"是否等价于"失败模式相似"没有验证；两个不同失败模式可能在栅格上很像。
- **消融的阈值与配置不统一**：Table 4 的 map+agent 92.3 与 Table 6 的 map-only 91.7 用了各自的 $\lambda$；主表 Base 92.3 与消融 base 91.0 也不是同一个 checkpoint，跨表读数需要小心。
- **navhard 对照集不全**：作者自己承认 VLA 方法在该 split 上报告很少，47.0 的"领先"主要相对经典 E2E，不能当作 VLA 内部的 SOTA。
- **无方差、无失败分析**：所有结果单次运行；也没有按场景类型统计 TTT 的收益与副作用（例如简单场景被误触发后是否变差）。
- **每场景重新初始化 LoRA**：没有 warm start、没有步数敏感性分析；TTT 的稳定性（不同初始化下结果是否一致）完全未讨论。

## Notes

+ **论文定位**：不是新的 VLA 架构，而是给现有 VLA 加一条**外部失败记忆 + 测试时适应**的通路。与 MomAD 的"最近 4 帧 planning-state 记忆"、PriorEye 的"离线街景视觉先验"不同，它记的是**模型自己的失败案例**，检索键是结构化场景表示，用途是推理期参数适应。

+ **可迁移点**：
  + **用结构化键替代语义键做检索**：Map/Agent 双分支 + occupancy 监督是一个便宜且可复用的实现；只要任务里有"几何 + 交互"两类因素，就可以拆成两路检索。
  + **解耦 LoRA 与分数分派一一对应**：检索到的案例类型决定更新哪条分支，分支又决定哪些子分数由它负责——这种"检索-适应-打分"三段对齐的设计，可以套到任何"多子指标打分"的规划器上。
  + **触发门控**：不是每帧都适应，只有相似度足够高才进入 TTT；Table 6 说明这个门控本身能值 2 个点以上（0.90 vs 0.99）。
  + **记忆扩容免重训**：把新失败案例追加进库即可获得增益，适合数据持续增长的部署场景；代价是检索库与延迟随规模增长。

+ **待跟进**：
  - 报告失败率、触发比例与端到端延迟分布，才能判断"平均开销很小"是否成立。
  - 把 TTT 的监督换成无标签目标（一致性、熵最小化），与当前"用记忆标签"的版本对比，区分"适应"与"有监督微调"的贡献。
  - 评估检索质量：人工标注"是否同类失败"，算检索 precision@k，并检验结构相似度与失败模式相似度的相关性。
  - 在 Base 与 Scale 之间加一个"4K 记忆 + 同样合成数据"的对照，把 94.1 的增益拆成"记忆规模"与"数据来源"两部分。
