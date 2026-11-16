---
title: "自动驾驶规划前沿：2026-08 至 2026-11 新增工作地图"
date: 2026-11-16
tags: [report, autonomous-driving, world-model, VLA, end-to-end-planning]
related:
  - "[[Topics/WorldModel-Survey]]"
  - "[[Topics/VLA-Survey]]"
---

# 自动驾驶规划前沿：2026-08 至 2026-11 新增工作地图

## 结论

近三个月的增量集中在四件事上：**候选评估独立成一个问题**、**WAM（world-action model）成为默认架构标签**、**训练范式从人类日志转向自博弈与 RL 后训练**、**评测从轨迹误差转向长程闭环与诊断性反事实**。

- **候选评估的路线之争是本轮最值得跟的冲突。** 已整理的 [[2611-RFWorld]] 主张"不要为每条候选预测未来，建一次共享记忆让候选来读"；新出的 RiSE-Drive（2611.13790）走反方向——仍然做 candidate-conditioned latent rollout，但把打分从确定性标量改成**长期后果分布 + 尾部风险排序**。两条路线都自报 SOTA，说明"是否需要候选条件化的未来"目前没有定论，而 World4Scorer（2609.36438）给了第三种答案：未来本身只在训练期用，但要用**仿真器给每条未执行候选打标签**，而不是只拟合日志里那条被执行的未来。
- **WAM 已经不是一个方法名而是一类架构。** Latent-WAM++ / SV-WAM / PlanWAM / OSWAM / NegWAM / ReWAM / DA-WAM / StyleDrive / OccWAM 全部共享同一骨架（联合去噪未来与动作），差异收窄到三处：监督哪种表示、用什么 mask 阻止动作看到未来、部署时丢掉哪一支。已整理的 [[2608-SimWAM]] 与 [[2608-WAMDiff2]] 属于同一族，本轮新增的是"把监督目标从视频改成 planning-shaped latent / occupancy / risk field"这条线。
- **训练范式出现明确分叉，且新路线的数字已经压过库内记录。** TerraTransfer++（2610.30319）完全不使用人类示范：自博弈预训练 → 蒸馏成 flow-matching teacher → 在冻结视觉 backbone 上训 adapter，报 **NAVSIM navhard 59.4 EPDMS / navtest 95.5 PDMS**，明确点名超过 DriveZero-Scale 的 57.1。这条直接把 [[2609-DriveZero]] 笔记里"57.1 是 navhard 最高"的读数变成历史值。
- **同一批论文的数字分属三套不可互比的协议**：NAVSIM v1 PDMS（TerraTransfer++ 95.5 > DriveZero 94.8 > PlanWAM 93.8）、NAVSIM v2 navtest EPDMS（NegWAM 93.3 / ViRA-Diffusion 92.3 / Latent-WAM++ 90.1）、NAVSIM v2 navhard EPDMS（TerraTransfer++ 59.4 / NegWAM 40.4 / DriveZero-Scale 57.1）。本报告不跨列排序，任何"更高"只在同列同协议内成立。
- **评测侧出现两股反向压力。** 一边是更硬的新基准（Odyssey 的 100 s 长程闭环 + SD 路线、Ego2Flow 的交通系统级影响、Bench2Drive-Speed 的速度合规专测），另一边是对排行榜本身的方法论质疑（Beyond the Leaderboard 用反事实编辑把"排名"拆成行为诊断，Breaking survivorship bias 指出驾驶世界模型系统性幻想安全未来）。
- **最值得记住的机制发现是两个"gap"**：Perceives but Doesn't Brake（2610.28184）证明推理型 VLA 的视觉表征里**已经编码了**在路径上的危险，但纵向规划被 ego 速度历史支配、对危险距离几乎不响应；STEER（2611.00718）则把过度 CoT 视为"overthinking"，用路由替代难度标签。两者共同指向"推理/感知能力与实际动作控制之间没有自动耦合"。

## 0. 覆盖范围与读法

**检索方式**：8 条 arXiv API 查询（`end-to-end autonomous driving` / `world model` / `vision-language-action + driving` / `trajectory evaluation + driving` / `closed-loop` / `reinforcement learning` / `diffusion` / `benchmark + planning`），按提交时间倒序各取 60 条，去重后 **308 条**，过滤 `published >= 2026-08-01` 得 **229 条**，其中 **4 条已有笔记**（[[2610-DriveMLP]]、[[2609-DriveZero]]、[[2608-SimWAM]]、[[2608-WAMDiff2]]），其余 225 条为本轮新增。

**核对级别**：26 篇逐条通读完整 abstract 并写入正文；其余条目只依据标题与 abstract 首句写入 §7 的补充表。**所有数字均为作者自报，未做独立复现，也未回表核对**——需要进入 survey 或 idea 的数字，应另起一轮 `paper-digest` 回原文取。

**优先级标记**：★★ 必读（能改变既有结论或定义新问题）/ ★ 值得读 / · 略读。

**与库内笔记的分工**：本报告只做格局与冲突，不复述已整理论文的方法细节。[[Topics/WorldModel-Survey]] 负责 world model 全景，[[Topics/VLA-Survey]] 负责 VLA 的 action representation × data recipe 切面；本报告是"AD 规划"这一条纵向切片，不替代两者。

---

## 1. 候选轨迹评估：从"打分"到"分布化后果"

评估这条线的共识是：候选生成已经足够多模态，瓶颈转移到"给定 N 条都合理的轨迹，怎么选"。分歧在于是否为每条候选构造它自己的未来。

| 工作 | arXiv | 机制 | 自报关键数字 | 优先级 |
|:--|:--|:--|:--|:--:|
| **RiSE-Drive** | 2611.13790 | candidate-conditioned **latent** rollout → 预测长期结果**分布** → 尾部风险 + 扰动鲁棒排序 | NAVSIM-v1/v2 + 闭环 HUGSIM 上 SOTA；含 6 张表（MM '26 正刊） | ★★ |
| **World4Scorer** | 2609.36438 | scorer 建成 trajectory-conditioned JEPA predictor；每条候选的 state 用**仿真器结果标签**监督，执行轨迹的真实未来只作 anchor（仅训练期需要） | NAVSIM-v2 SOTA；Bench2Drive 适配系统结果；同框架在 OGBench-Cube 上迁移 | ★★ |
| **RFWorld**（已整理） | 2611.05072 | 建一次共享记忆，候选位姿作空间 query 稀疏读取，推理期不解码未来 | PDMS 90.5；1024 候选下显存 8.7 vs 33.9 GB，延迟 37.5 vs 49.2 ms | — |
| **RRDrive** | 2609.21486 | risk-aware occupancy 作为稠密、时序对齐、可被轨迹 query 的表示；全局结构管生成，候选条件风险 query 管排序 | 选定轨迹 PDMS 0.951（baseline 0.937）；候选平均 PDMS +7.7%；困难场景 +30.2%；good 候选间 Spearman 0.26→0.67 | ★ |
| **Evaluation Is All You Need** | 2609.30818 | 把多模态规划的重心放在评估环节 | 仅读到标题 + 首句 | · |
| **CARE-Drive** | 2611.15360 | 把轨迹精修拆成 position 残差专家 + heading 条件生成专家，backbone 冻结、两专家交替 SFT+RFT | NAVSIM v1 88.8 PDMS（较同 backbone 未精修基线 +2.4） | ★ |
| **CLAP-AD** | 2610.36079 | path 用模仿学，**速度**用学到的多准则函数打分；首次支持测试时显式控速而不需速度条件训练 | Bench2Drive-Speed SAS 82.53（较最强基线 TFv6 +12.9）；Bench2Drive DS +28.5/+32.7 对 UniAD/VAD；NAVSIM v2 EPDMS +6.1（WACV 2027） | ★ |

**为什么这一族重要**：Three 条路线在同一个问题上正面冲突，且都给了能判别的证据。

- **RFWorld 的隐含赌注**是"规划只需要可比较的未来相关信息"，因此可以完全不做候选条件化预测。它的反面证据来自 RiSE-Drive：如果打分的难点是**延迟失效与尾部风险**，那么一条候选的失败模式恰恰取决于"这条候选把车带到哪个未来"，共享记忆里没有这个条件信号。
- **World4Scorer 指出一个更根本的数据问题**：驾驶日志只记录被执行的那条轨迹，因此"拟合日志未来"对未执行候选毫无约束。它用仿真器给**每条**候选打标签，代价是引入仿真器分布偏移；而 RFWorld 用共享记忆绕开这个问题。这两条是对同一缺陷的两种补救，谁更好取决于"仿真器标签的可信度"与"共享记忆的信息上限"，目前没有第三方对照。
- **RRDrive 的数字特别值得记**：在 register-based baseline 上，bad/poor 候选占候选集的 19.74%，而 **oracle-best 候选平均只排在第 33.9 位**。这句话把"评估是瓶颈"从直觉变成了可量化的 gap——生成器里已经有正确答案，只是排不上来。
- **CLAP-AD 把"多模态"从 path 拆到 speed**：作者的主张是驾驶的多模态性主要来自速度而非路径。如果成立，那前面所有沿轨迹空间做候选的工作都只覆盖了一个维度。这条主张目前只有它自己的 Bench2Drive-Speed 比分支撑，属于强 hypothesis、弱独立验证。

## 2. World-Action Model：把未来预测移到训练期

这一族共享同一骨架：一个视频/潜空间分支与一个动作分支联合去噪，用 mask 控制动作 token 能看到什么。已整理的 [[2608-SimWAM]] 用 isolated attention mask 给出"训练用 WM、推理删 WM"的模板；本轮新增的工作基本是在这个模板上换监督对象与换部署策略。

| 工作 | arXiv | 与 SimWAM 的差异 | 自报关键数字 |
|:--|:--|:--|:--|
| **Latent-WAM++** | 2611.10632 | 不预测未来状态/最终动作，而是**分别约束世界表示与动作表示**：SIGReg 稳定分布 + 冻结几何先验；MeanFlow 一步采样；动作特征与当前视觉语义对齐 | NAVSIM v2 90.1 EPDMS；HUGSIM 零样本 RC 47.0 / HD 30.1；训练 190 vs 1,600 GPU-hours（−88.1%） |
| **PlanWAM** | 2610.11382 | 先问"哪种未来表示对规划最有用"，用轨迹目标**塑造**未来潜表示（privileged future posterior → 蒸馏成只依赖历史的 prior） | NAVSIM v1 93.8 PDMS / v2 90.9 EPDMS；HUGSIM 零样本 38.7 HD |
| **SV-WAM** | 2609.03602 | 保留**六路环视**而不是单前视；action-centered causal mask；加可微 drivable-area 正则 | NAVSIMv2 闭环 SOTA，nuScenes 开环与零样本迁移 |
| **OSWAM** | 2610.35850 | 解决"去噪步数仍要几十步"：用 imagination token 初始化 + 特权蒸馏，做到**一步**动作生成 | 约 3× 加速，RTX 5090 上 40 ms |
| **NegWAM** | 2611.08111 | 安全后训练：构造"几何上贴近专家但带安全风险"的负轨迹作排斥锚，辅助分支推理期丢弃 | navhard 40.4 EPDMS（较最强 scorer-free WAM +4.1%）；navtest 93.3 EPDMS；代码开源 |
| **GuidedWAM** | 2611.07912 | 不动权重、在采样时注入**能量引导**，把新驾驶目标变成推理期控制通道 | 三个已发布冻结 WAM 在 navhard 上 +1.8~3.6 EPDMS，采样时间 +<10% |
| **ReWAM** | 2609.39245 | 把其他 agent 当**条件响应者**（Level-k 层级），用博弈结构建模交互 | NAVSIM SOTA，交互场景增益最大 |
| **StyleDrive** | 2609.03225 | world-model RL + 长程一致性正则 + 显式状态解耦 + GRPO 多风格 | Bench2Drive DS 88.44（+17.08）、SR 66.82（+16.58）；实车 AGV 部署 |
| **OccWAM / DA-WAM / MM-Future / WA-JEPA / Drive-HWM / DriveWLA** | 2610.17688 / 2608.19085 / 2609.20377 / 2608.20974 / 2609.03572 / 2610.17903 | 同一骨架的不同监督对象：occupancy、决策对齐潜变量、多模态联合、V-JEPA、层级潜变量、语言-世界-动作三合一 | 见 §7 |

**这一族的共同收敛与共同盲区**：

- **收敛**：几乎所有工作都把"推理期生成未来"视为浪费，只保留训练期监督；差别在于监督的是像素、视频潜变量、occupancy 还是"planning-shaped latent"。
- **共同的证据缺口**：这些论文的消融普遍只比较"有无未来监督 + 有无 WM 分支"，没有一组能分离"共享表示带来的信息"与"额外辅助任务带来的正则化"。[[2608-SimWAM]] 笔记里已经点过这一点，本轮新增工作没有填上。
- **OSWAM 与 GuidedWAM 是两条正交的降本路线**：前者压采样步数（结构侧），后者把后训练目标搬进采样器（部署侧）。两条都不需要重训，对实际部署的参考价值高于再刷一个 PDMS。

## 3. 训练范式：不再依赖人类日志

| 工作 | arXiv | 做法 | 自报关键数字 | 优先级 |
|:--|:--|:--|:--|:--:|
| **TerraTransfer++** | 2610.30319 | 向量化仿真器里自博弈预训练 → 蒸馏成 flow-matching teacher → 冻结视觉 backbone 上训 adapter；**全程不使用人工采集示范** | HUGSim 闭环超过所有已发表模仿类方法；NAVSIM navhard **59.4 EPDMS**（点名超过 DriveZero-Scale 57.1）；navtest **95.5 PDMS** | ★★ |
| **RoXDrive** | 2609.36851 | 用视频世界模型做 RL 环境，但先用 Action-Vision Faithfulness Evaluator 过滤"动作条件未忠实反映"的 rollout，再做 scene-level 闭环 RL | nuScenes 上 DiffusionDrive 安全违规 −27.6%；内部 130K 场景 + Qwen3-VL 上 −33.7%；代码开源 | ★★ |
| **OPTED** | 2609.20756 | on-policy 微调 + **免渲染** teacher | 仅读到标题 + 首句 | ★ |
| **FPD-Drive** | 2610.13232 | 用"日志只记录执行轨迹"这一点出发做 future policy distillation | 仅读到标题 + 首句 | ★ |
| **CARE-Drive** | 2611.15360 | backbone 冻结，两专家交替 SFT + RFT（见 §1） | NAVSIM v1 88.8 PDMS | ★ |

**这一节的判断**：**"要不要人类示范"正在从信仰变成成本问题。** TerraTransfer++ 的论证方式值得单独记：自博弈在向量化仿真器里能以百万步/秒产生碰撞、险情、恢复的分布，这种分布**任何驾驶日志都不含**；而驾驶日志的问题不是量不够，是安全关键尾部被系统性欠采样。这比 [[2609-DriveZero]] 的"goal-conditioned teacher 扩增监督"更彻底——DriveZero 仍然用日志初始化场景与 goal，TerraTransfer++ 连示范都不需要。

**同时它的证词要打折**：navhard 59.4 是自报数字，且 HUGSIM 上的"超过所有模仿方法"是在它自己选的对比集内；真正的独立验证需要第三方在同一协议下复现。

**RoXDrive 的观察最尖锐**：把视频世界模型当 RL 环境时，最大的风险不是不真实，而是**动作与预测未来不忠实**（action-vision mismatch）——如果模型生成了一段与所给动作无关的未来，用它算出来的奖励就是噪声。它的解法是先训练一个逆动力学评估器做过滤，只保留 action-faithful 的 rollout。这与 §1 的评估线是同一问题的两面：都在问"这条轨迹/这段未来是否可信"。

## 4. 表征与预训练：scale 什么才有效

| 工作 | arXiv | 主张 | 自报关键数字 |
|:--|:--|:--|:--|
| **ScaleDrive** | 2611.02853 | 只用**无动作**的大规模互联网驾驶视频做视频潜预测预训练，下游冻结 encoder；ego-motion / depth probe 显示随规模增强 | 跨多种规划器与闭环 benchmark 一致提升 |
| **ViRA（Do Better Visual Representations Always Lead to Better E2E AD?）** | 2610.09695 | 给 planner 无关的视觉表示对齐框架，并回答"VFM 表征什么时候有用" | 三条发现：VFM 对齐普遍有效；对齐目标的选择有影响；加辅助感知监督后五个目标的 EPDMS 极差从 2.7 压到 0.5。ViRA-Diffusion 在 NAVSIM v2 navtest **92.3 EPDMS**（较对比方法至少 +1.9） |
| **ConDenseDrive** | 2610.16908 | 知识密集 / token 稀疏的视觉 encoder | 仅标题 + 首句 |
| **Drive3R** | 2610.15112 | block-wise streaming 几何 foundation model | 仅标题 + 首句 |
| **MomADv2 / Not All History Helps** | 2608.23405 / 2608.15573 | 长程记忆：可靠的时序记忆 / 速度感知的选择性记忆 | 仅标题 + 首句 |

**ViRA 的第三条结论是这轮最反直觉的一条**：VFM 目标选得好不好，在有辅助感知监督时会被抹平（极差 2.7 → 0.5）。换句话说，**表示学习的收益可以被感知监督替代**，两者不是叠加关系。这对所有"换更强 backbone"的工作是一个约束条件——如果 planner 本身带强感知监督，换 VFM 的边际收益会被吃掉。

## 5. VLA 的推理税与推理-动作耦合

这一族的问题已经从"要不要 CoT"变成"CoT 应该在多大程度上控制动作"。

| 工作 | arXiv | 结论/机制 | 自报关键数字 | 优先级 |
|:--|:--|:--|:--|:--:|
| **Perceives but Doesn't Brake** | 2610.28184 | 提出 perception-action binding gap：危险信息**确实编码在**视觉表征里，但纵向规划由 ego 速度历史支配、对危险距离和推理文本都几乎不响应；在冻结的 flow-matching 采样器里注入 hazard-conditioned energy 即可修复 | 精确障碍几何下 footprint 侵入率 58.1% → 14.5%，制动相关闭环 battery 碰撞清零 | ★★ |
| **STEER** | 2611.00718 | 不需要场景难度标签的自适应推理：路由不确定性感知 rollout + 跨路由优势分配；把过度 CoT 当 overthinking 治理 | NAVSIM v1/v2 上与 SOTA 竞争力持平，同时给出推理效率 Pareto 前沿（NeurIPS 2026） | ★★ |
| **How Much Should a VLA Trust Its Reasoning?** | 2610.40002 | CoC-CFG：在"有推理 span"与"无推理 span"之间做 classifier-free guidance，每步权重用**被测量的动作价值**（非 bootstrap）优化 | Alpasim 闭环 262/300 场景（原策略 252）；Waymo held-out 前三阶段胜所有固定权重与 SAC/GRPO；hidden test 8.135159 vs 已发表最佳 8.079 | ★ |
| **Measuring Reasoning-Action Coupling via Counterfactuals** | 2610.30953 | 用反事实测量推理与动作的耦合度 | 仅标题 + 首句 | ★ |
| **TacitVLA** | 2610.17540 | 无推理延迟的"隐性"推理 | 仅标题 + 首句 | · |
| **Scene-Adaptive VLA** | 2610.24590 | 动态层路由降推理成本 | 仅标题 + 首句 | · |

**这三篇放在一起改变了什么**：过去一年的默认叙事是"加 CoT → 更好规划"。这一族给出的是**分段结论**——推理有用，但（a）它不一定被动作通路读取（binding gap），（b）过量会损害规划（overthinking），（c）控制它需要一个**强度**旋钮而不是开关。三篇的解法出奇一致：都在**冻结的生成器上加引导/路由**，不重训模型。这对已有 VLA 的改造成本极低，是这一轮最可迁移的工程结论。

## 6. 评测：从分数到诊断

| 工作 | arXiv | 改变的是什么 | 关键事实 |
|:--|:--|:--|:--|
| **Beyond the Leaderboard** | 2609.22582 | 排名 → 行为诊断。对少量真实帧做两次反事实编辑（去掉行人 / 夜间重打光），读规划变化 | 6 个已发布策略；246 个 NAVSIM 近行人场景中，行人位于规划路径上的格子里**只有 1.9% 是真正的避让**；中位 clearance 变化 ≤0.03 m、中位规划距离变化 ≤0.08 m；预注册的左右舵迁移实验显示部分判据可迁移 |
| **Breaking survivorship bias** | 2611.11516 | 指出驾驶世界模型的**训练数据幸存者偏差**：日志里的未来几乎都以安全收场，导致模型在碰撞将至时仍幻想良性结局；用日常视频 + 真实事故视频训 neutral world model | 更能预见迫近碰撞且误报率低；作为 imagination engine 时在困难 NAVSIM 场景上比 baseline 更可靠地选出更好轨迹 |
| **Odyssey** | 2610.06469 | 长程闭环：100 个场景各由 100 s nuPlan 日志重建；用显式 SD 路线替代方向指令；新增 SD Route Compliance / Pre-Lane Change Score / RouteDS | 现有 benchmark 只评短片段，后期后果与导航意图都被省略 |
| **Ego2Flow** | 2611.17720 | 从 ego 安全转向**交通系统级**影响（效率、稳定、能耗）；从 Argoverse 2 蒸馏 6 张 CARLA 真实城市地图并公开 | 成对真实/仿真观测下规划输出的平均位移误差最低 |
| **Bench2Drive-Speed** | 2610.36079（提出） | 速度合规专测 | CLAP-AD 在该指标 SOTA 82.53 |
| **VIPS / TrafficSignBench / CausalDriveBench / DriveHierarchy** | 2609.02462 / 2609.38463 / 2609.32157 / 2609.31814 | 协同规划伪仿真 / 交通标志合规 / 因果推理 / 从开环理解到闭环执行的诊断 | 仅标题 + 首句 |

**这一节的方法论意义**：Beyond the Leaderboard 与 Breaking survivorship bias 是两条独立的"分数不可信"论证——前者说排名不含行为信息，后者说训练数据分布让模型必然乐观。两条都指向同一件事：**NAVSIM 类非反应式协议上的 +0.5 PDMS 已经无法区分机制差异**，这与库内多篇笔记（[[2609-DriveZero]]、[[2605-CoPhy]]、[[2606-PriorEye]]）里反复出现的"0.5 点无法排除噪声"是同一结论的不同表述。

## 7. 补充表：其余值得扫一眼的近期工作

只依据标题与 abstract 首句，未展开阅读。

| 方向 | 工作（arXiv） |
|:--|:--|
| 世界模型骨架变体 | DA-WAM 决策对齐潜变量（2608.19085）· OccWAM occupancy 联合 flow matching（2610.17688）· MM-Future 多模态联合（2609.20377）· WA-JEPA 重审 V-JEPA（2608.20974）· Drive-HWM 层级世界模型（2609.03572）· DriveWLA 世界-语言-动作（2610.17903）· ForeDrive planning-relevant 潜世界模型（2609.26299）· EditWM 事件分解增量修正（2609.22317）· 4D-WAM 4D 一致性（2608.10107）· PhysWAM 物理一致（2609.37970）· V2X-WAM 协同（2609.37098）· BrainWAM 动作空间协调（2608.12854） |
| 长程与记忆 | MomADv2 可靠时序记忆（2608.23405）· Not All History Helps 速度感知选择性记忆（2608.15573）· Directional Pullback Regularization 约束潜空间非完整动力学（2610.16139）· World Action Critic 联合预见与批评（2610.22825） |
| 规划器结构 | DiffuSearch 混合规划目标对齐（2609.02252）· Quality-Axis Diffusion 沿"效用轴"生成（2610.23885）· Latent-Aligned Planning 离散推理→连续动作（2609.04070）· Beyond Planning 可微控制闭环（2610.18167）· FeasibleFlow 可行性场与轨迹联合一步传输（2609.23488） |
| VLA 效率与对齐 | Scene-Adaptive VLA 动态层路由（2610.24590）· TacitVLA 无推理延迟（2610.17540）· AUGUstus 统一生成与理解（2610.16985）· RAF-VLA 未来表示对齐（2609.17728）· Reflect-VLA 双路径在线自蒸馏（2610.13453）· FIVE-VLA 循环动作记忆（2609.18623）· FlashDrive 快速推理（2608.12932） |
| 风险与安全 | DRiF 数据驱动风险场（2609.10377）· READ 风险知情场（2609.12371）· 风险敏感不确定性决策（2609.09650）· Interaction Constrained Offline RL（2610.09763）· Dreamer-SAC 潜世界模型 off-policy（2608.10386） |
| 数据与场景生成 | 5,500 小时视频扩散 scaling law（2608.28404）· Scaling Curriculum Learning（2608.22549）· 行人均风险动作生成（2610.06171）· Radar2Plan 4D 雷达开环规划（2610.06121） |
| 综述（锚点） | **Planning-Oriented E2E AD: Architectures, Evaluation, and Emerging Paradigms**（2608.20111）· **A Survey on E2E AD Training: Data, Strategy, Platform**（2610.00926，T-ITS 录用） |

## 8. 与 notebook 的关系与下一步

**库内已覆盖**（24 篇）：本轮新增工作与之重叠的部分已在 §1–§3 逐条对照，不重复记录。三处需要留意的时间戳问题：

1. [[2609-DriveZero]] 的 navhard 57.1 已被 TerraTransfer++ 的 59.4 超过（自报），该笔记正文的"首个超过人类"类表述应限定为"截至 2026-09"。
2. [[2611-RFWorld]] 的"不做候选条件化未来"论点需要与 RiSE-Drive / World4Scorer 对读，才能看出这是一条有争议的路线而非既定方向。
3. [[2608-SimWAM]] 的 isolated mask 结论在 SV-WAM / Latent-WAM++ / PlanWAM 上被复用了三次，说明它已经稳定为模板；但三篇都没有补上"共享表示 vs 辅助正则"的分离实验。

**建议优先 digest 的 6 篇**（理由是可改变既有结论，不是数字更大）：

| 优先 | 论文 | 为什么值得单独一篇笔记 |
|:--:|:--|:--|
| 1 | TerraTransfer++（2610.30319） | 目前唯一"零人类示范"且同时压过 DriveZero-Scale 的完整训练配方，直接挑战库内 drive-zero 叙事 |
| 2 | RiSE-Drive（2611.13790） | 与 RFWorld 构成候选评估的正面路线冲突，且是 MM 正刊，值得回表核对 |
| 3 | World4Scorer（2609.36438） | 指出"日志只记录执行轨迹"这一数据缺陷并给出仿真器标签方案，是评估线的第三种答案 |
| 4 | Perceives but Doesn't Brake（2610.28184） | perception-action binding gap 是可直接迁移到 GUI/具身 agent 的机制性发现（表征里有、动作不读） |
| 5 | Beyond the Leaderboard（2609.22582） | 方法论模板：反事实编辑 + 诊断判据，可复用到其他"排行榜不可信"的领域 |
| 6 | Planning-Oriented E2E AD 综述（2608.20111） | 四轴分类（输入表示 / 规划输出 / 监督信号 / 评测协议）可作为建 canonical survey 的骨架 |

**如果要建 canonical survey**：目前 `Topics/` 没有 AD 规划专属 survey，24 篇笔记散落在 [[Topics/WorldModel-Survey]]、[[Topics/VLA-Survey]]、[[Topics/EmbodiedAI-Survey]] 里。本报告的四条路线（候选评估 / WAM / 训练范式 / 诊断式评测）可以直接长成 `Topics/AD-Planning-Survey.md` 的骨架——需要 Supervisor 确认后再建，避免和现有两份 survey 的 world-model 部分重复。

## 9. 证据边界

- **全部数字为作者自报**，来源是 arXiv abstract 与（少数）comment 字段；本报告没有回表核对任何单元格，也没有独立复现。
- **26 篇读了完整 abstract，199 篇只读了标题与首句**（§7）。§7 中条目的定位仅凭标题语义，属最低等级证据。
- **跨论文数字不可比**：NAVSIM v1 PDMS 与 v2 的 navtest / navhard EPDMS 是三套协议；表格只做同协议内比较。
- **单次运行、无方差**是这一批工作的普遍特征；0.5 EPDMS 级别的领先不应被赋予机制含义。
- **"SOTA" 均为各自论文内的对比集**，其中多数没有与 TerraTransfer++、RiSE-Drive 这类最新工作同 harness 对照。
- 本轮未做矛盾检索与负结果检索的完整走查（仅有 Beyond the Leaderboard、Breaking survivorship bias、ViRA 三条自然命中的批判性工作）；若要把本报告升格为 canonical survey，需按 `literature-survey` 的 B 组策略补跑。
