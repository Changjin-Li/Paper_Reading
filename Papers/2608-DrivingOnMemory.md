---
title: "Driving on Memory"
authors: ["Christian Löwens", "Thorben Funke", "Alexandru Paul Condurache"]
institute: ["Bosch Research", "Automated Driving, Bosch", "University of Lübeck"]
date_publish: 2026-08
venue: arXiv Preprint
tags: [imitation-learning, spatial-memory]
url: https://arxiv.org/abs/2608.31029
cite_key: lowens2026drivingonmemory
arxiv_id: "2608.31029"
doi: "10.48550/arXiv.2608.31029"
code: https://github.com/boschresearch/MemoryDrivoR
rating: 4
content_scope: full-text
verification_status: source-checked
date_added: 2026-11-26
---
## Summary

这篇不是提出更强的规划器，而是对评测做一次**压力测试**：把被评测场景的相机输入整个删掉，换成"同一位置、过去行驶记录"里的场景 token。检索到的记忆只含静态信息（道路布局、地点相关规律），拿不到当前交通状态。结果相当反直觉——在 NAVSIM v1 上 MemoryDrivoR 拿到 91.1 PDMS，接近相机版 DrivoR 的 93.7；NAVSIM v2 navhard 上 45.0 EPDMS 对 46.7。也就是说，**不观察当前场景也能接近榜首**。但在 Bench2Drive 上同样替换只得到 34.7 DS / 9.5% SR（相机版 61.0 / 30.5），RealEngine 闭环也大幅下滑。结论是 NAVSIM 的高分很大程度上可以由初始 ego 状态 + 静态场景信息解释，而闭环长路线基准仍要求对动态场景做出反应。

## Problem & Motivation

- 基准高分的默认解读是"模型看懂了当前场景"，但高分本身不能说明模型用了哪些信息；基准可能奖励了与目标能力相关、但更简单的线索。
- nuScenes 时代已经暴露过这个问题：只给 ego 状态（速度、导航命令）的 MLP 也能拿到不错的开环分数；NAVSIM / Bench2Drive 用仿真指标与更复杂操作设计来堵这个漏洞，但"堵住了多少"没有被量化。
- 论文把相机输入拆成静态与动态两部分：静态是道路结构、地点相关的常态（如某个路口通常很堵），动态是当前车辆、信号灯、天气。要问的是：**只用静态部分能拿到多少分？**
- 为此需要一个"可控地只保留静态信息"的探针，而不是再训一个无感知基线——后者无法区分"架构强"与"信息够"。
- 另一个要排除的解释是位置记忆（训练时见过测试地点），所以论文补了地理切分实验。

## Method

**0. 总体框架**

+ 基座是 DrivoR：ViT 编码单帧多视角图像为 register tokens，规划器含两个 transformer decoder（proposal 与 scoring），都对 scene tokens 做 cross-attention，取最高分轨迹。
+ 干预点在 planner 的 K/V：把相机产生的 scene tokens 替换为"从记忆库检索到的、带相对位姿编码的压缩记忆 token"，planner 结构本身完全不动。
+ 这是一个 **analytic probe**（解析探针）：模型仍然有强架构与预训练权重，只是被剥夺了当前场景的动态信息。

**1. 记忆库构建**

+ 用冻结的 DrivoR ViT 对训练集的每一帧编码，存下 scene tokens 与全局位姿：

$$
Z_i\in\mathbb{R}^{n\times c}\ \text{（n=64 个 register token）},\qquad q_i=(x_i,y_i,\psi_i)
$$

+ 只存潜 token、不存原图，因此记忆是压缩的；库覆盖 navtrain 全部样本（Bench2Drive 用更大的 Full 训练集以保证覆盖）。

**2. 检索**

+ 用当前 ego 位姿 $q_t$ 在库中做最近邻：半径 $r=20$ m 内取至多 $k=10$ 条，按距离排序；**每条轨迹（traversal）至多取一条**，避免同一段路重复采样造成冗余。
+ NAVSIM 官方不提供绝对 ego pose，论文为此补上了位姿字段才能做基于位置的检索——这一点决定了方法只能用于"有重复行驶轨迹"的数据集。

**3. 相对位姿编码与注入**

+ 记忆不是几何 warp 到当前视角，而是编码"这条记忆是在哪里被观察到的"：对记忆位姿 $q_m$ 与当前位姿 $q_t$ 计算相对变换

$$
T_{m\to t}=T_t^{-1}T_m
$$

+ 把平移按固定范围归一化、做正弦编码，再过两层 MLP 得到 $c$ 维嵌入，加到该记忆的每个 token 上；这样 planner 知道"这条路在前方 30 m、偏左 5°"这样的空间关系。

**4. 压缩与替换 K/V**

+ 每条带位姿的记忆经 2 层 resampler（Perceiver 式）压到 $\ell=8$ 个 token，再线性投影成 $M_{m,t}$；所有检索到的记忆拼接后作为两个 planner decoder 的 K/V。
+ 没有可用记忆时直接跳过对应 cross-attention 层，模型退化为只看 ego 状态——这给了论文一个干净的下界对照。
+ resampler 的作用不只是省算力：它需要学会过滤记忆里**过时的动态信息**（例如上次经过时这里停着一辆车），否则会把旧场景当成当前场景。

**5. 训练**

+ pose embedder 与 resampler 从零训练；planner 用 DrivoR 权重初始化后在记忆 token 上微调（NAVSIM 5 epoch、Bench2Drive 20 epoch），损失保持 DrivoR 原样：候选轨迹的模仿损失 + 各子分数的 BCE。
+ 关键点：**训练与推理都在记忆 token 上**，不存在"训练用相机、测试用记忆"的错配，因此测到的是"这套信息够不够"，而不是"换输入导致的分布外失败"。

**6. 实现细节**

| 项目 | 设置 |
|:--|:--|
| 基座 | DrivoR（ViT-S，64 register tokens），冻结视觉编码器 |
| 记忆表示 | 每帧 64 token + 全局位姿；注入时压到 8 token/记忆 |
| 检索 | $k=10$，半径 $r=20$ m，每条 traversal 至多 1 条 |
| ego 输入 | NAVSIM：2D 速度 + 2D 加速度 + 4D 命令 one-hot；Bench2Drive：1D 速度 + 6D 命令 |
| 训练 | NAVSIM 5 epoch；Bench2Drive 20 epoch（从 DrivoR 复现权重出发） |
| 评测 | NAVSIM v1/v2、Bench2Drive、HUGSIM-nuScenes、RealEngine |

## Pipeline

（原论文无 HTML 版，架构图见 PDF 图 2：pose-indexed memory bank 与两阶段注入流程。）

1. **建库**：训练集每帧过冻结 ViT 得 scene tokens，与绝对位姿一起入库。
2. **检索**：当前帧只需 ego 位姿（不需要相机），在 20 m 半径内取 ≤10 条不同 traversal 的记忆。
3. **位姿编码**：算 $T_t^{-1}T_m$，归一化平移 + 正弦编码 + MLP，加到记忆 token 上。
4. **压缩**：resampler 把每条记忆压到 8 token，过滤过期动态信息。
5. **注入**：拼接后的记忆作为 planner 两个 decoder 的 K/V，替换原本的相机 scene tokens。
6. **规划**：proposal decoder 生成候选、scoring decoder 打分，取最高分轨迹；无记忆时可跳过 cross-attention。
7. **审计**：在 NAVSIM / Bench2Drive / HUGSIM / RealEngine 上比较"仅有 ego 状态""ego + 记忆""ego + 相机"三档，量化静态信息的贡献。

## Key Results

**1) NAVSIM v1 navtest（Table 1，PDMS）**

| Method | 相机 | 记忆 | NC | DAC | TTC | EP | PDMS |
|:--|:--:|:--:|--:|--:|--:|--:|--:|
| Human driver | – | – | 100.0 | 100.0 | 100.0 | 87.5 | 94.8 |
| DrivoR | ✓ | × | 99.0 | 98.9 | 96.7 | 90.0 | **93.7** |
| Ego Status MLP | × | × | 93.1 | 78.3 | 84.0 | 63.2 | 66.4 |
| DrivoR（无相机） | × | × | 96.1 | 85.6 | 91.1 | 62.8 | 72.9 |
| **MemoryDrivoR** | × | ✓ | 98.3 | 98.5 | 95.3 | 86.0 | **91.1** |

+ 91.1 已经接近相机版 93.7，远高于只给 ego 状态的 66.4；DAC 几乎不降（98.5 vs 98.9），因为记忆本身就携带稳定的道路几何。
+ 最值得注意的是 NC 与 TTC 也没有崩：论文的解释是模型在记忆提示的"常态拥堵位置"提前减速，4 s 评测窗口内交互次数减少，于是碰撞自然变少——这是**保守驾驶**而非场景理解。

**2) NAVSIM v2 navhard（Table 2，EPDMS）**

| Method | 相机 | 记忆 | EPDMS |
|:--|:--:|:--:|--:|
| DrivoR | ✓ | × | 46.7 |
| ZTRS | ✓ | × | 48.1 |
| Ego Status MLP | × | × | 14.2 |
| DrivoR（无相机） | × | × | 28.3 |
| **MemoryDrivoR** | × | ✓ | **45.0** |

+ 即使 v2 有反应式交通与偏移的 Stage-2 起点，记忆版仍只比相机版低 1.7；论文称这一结果此前没有被展示过。
+ 分项上，与静态信息相关的指标（DAC、LK）接近甚至更好，而与动态相关的（NC、TLC）下降但不崩，主要代价落在 EP。

**3) 闭环基准：结论反转（Table 3、4、5）**

| Benchmark | 指标 | DrivoR（相机） | MemoryDrivoR |
|:--|:--|--:|--:|
| Bench2Drive | DS / SR | 61.0 / 30.5% | 34.7 / 9.5% |
| RealEngine | open-loop | 86.6 | **87.1** |
| RealEngine | base closed-loop | 88.2 | 72.3 |
| RealEngine | safety test | 53.8 | 28.1 |
| RealEngine | multi-agent | 63.9 | 43.4 |
| HUGSIM-nuScenes | HD-Score | 36.4 | 32.2 |

+ Bench2Drive 上记忆版仍显著优于无相机基线（11.6 DS / 2.4% SR），但离相机版差一倍；成功路线平均约 110 m / 79 s，远超 NAVSIM 的 4 s 窗口。
+ RealEngine 的 open-loop 甚至略高于相机版（87.1 vs 86.6），但一进入闭环就掉到 72.3，safety test 掉到 28.1——**开环分数与闭环能力在这里出现明显背离**。
+ HUGSIM-nuScenes 差距较小（32.2 vs 36.4），作者归因于该子集只有 12 条基本笔直路线、87 个场景，动态交互本就少。

**4) HD map 对照（Table 6）**

| 输入 | NAVSIM v1 PDMS | NAVSIM v2 EPDMS |
|:--|--:|--:|
| Ego + 相机 | 93.7 | 46.7 |
| Ego only | 72.9 | 28.3 |
| Ego + HD map | 86.6 | 40.8 |
| Ego + 记忆 | **91.1** | **45.0** |

+ HD map 能补上大部分差距（72.9 → 86.6），但仍低于记忆；作者认为记忆额外携带了"地点相关的常态"（如通常拥堵程度），这是结构化地图没有的。
+ 这一行同时说明：**结构化的静态信息本身就是很强的评测捷径**，不需要记忆这种形式。

**5) 地理切分排除"位置记忆"（Table 7）**

| 设置 | NAVSIMv2（Pittsburgh 留出） | Bench2Drive（Town11+13 留出） |
|:--|--:|--:|
| 原始训练集 | 39.0 EPDMS | 61.0 DS / 24.1% SR |
| 去掉测试地点 | **42.7** | **61.2 DS / 27.7% SR** |

+ 把测试地点从训练集中删掉后，分数没有下降反而略升，说明相机版 DrivoR 的高分不是靠"记住地理位置"，而是靠测试时仍然能看到的静态信息。
+ 这条对照很关键：它把"基准可被静态信息刷分"与"模型过拟合训练地点"两个解释分开了。

## Strengths & Weaknesses

**Strengths**

- 实验设计干净：唯一被替换的是 planner 的 K/V 来源，架构、权重初始化、训练目标、评测协议全部不变，因此 91.1 与 93.7 的差距可以直接归因于"当前场景的动态信息"。
- 三档对照（ego only / ego + memory / ego + camera）给出了连续的剂量-反应曲线，而不是只有一个"有/无感知"的对比；72.9 → 91.1 → 93.7 让"静态信息贡献了多少"变成可读的数字。
- 结论有跨基准的反证：同一个干预在 NAVSIM 上几乎不掉、在 Bench2Drive/RealEngine 上腰斩，说明这不是"方法不够好"，而是**基准对动态反应的要求不同**。
- 主动排除了最自然的替代解释：地理切分实验显示去掉测试地点后分数不降，把"位置记忆"从解释里剔除。
- HD map 对照回答了"是不是只要有静态信息就行"：地图能补上大半差距但仍不及记忆，界定了记忆的独特成分。
- 对机制的解读具体：ego 状态本身已包含"是否在刹车"的信息，记忆补的是道路上下文；短窗口下保守驾驶即可减少交互，这解释了为什么 NC/TTC 不崩而 EP 下降。
- 论文明确不 overclaim：反复说明这只说明基准可被静态信息刷分，不说明感知对真实驾驶不重要；也承认 91.1 不是该方法的上限。

**Weaknesses**

- **探针依赖"同一地点有重复行驶记录 + 高精度相对位姿"**：记忆库来自训练集同一路段的多次经过，NAVSIM 还需要额外补绝对位姿；在只有单次轨迹或定位噪声大的数据集上（如 WOD-E2E）无法直接套用。作者在 limitation 中承认。
- **记忆与相机不是信息对等的比较**：记忆库覆盖整个 navtrain，相当于把训练集的所有静态快照都给了模型，而相机只给当前帧；因此"记忆接近相机"里有一部分来自"记忆的覆盖更全"，不是纯粹的静态/动态之分。
- **结论受评测窗口长度强烈影响**：NAVSIM 只评 4 s，初始 ego 状态几乎泄漏了当前机动；Bench2Drive 从静止起步、路线长达百米，ego 状态不再有信息量。因此"NAVSIM 高分不代表会开车"这一结论成立，但"静态信息能拿多少分"的数值不能外推到其他窗口长度。
- **RealEngine 上 open-loop 高于相机版（87.1 vs 86.6）却没有解释**：可能是记忆让轨迹更平滑、更贴合静态路线，但论文只用"与 NAVSIM 结果一致"带过，没有分析这个反超。
- **Bench2Drive 的相机版是作者复现**（DrivoR* 61.0），与官方榜单其他方法不可直接比；记忆版也基于同一复现，比较公平但绝对数值参考价值有限。
- **resampler 过滤动态信息的能力没有单独验证**：论文说它会学过滤过期动态信息（引 App. A.6），但正文没有给"记忆里放一辆已离开的车"这类对照，机制解释停留在定性。
- **检索参数与记忆规模没有敏感性分析**：$k=10$、$r=20$ m、每条 traversal 一条，都是固定值；在更密或更稀的采集条件下，这些参数会直接决定"记忆够不够用"。
- **没有分析失败类型**：Bench2Drive 上 9.5% SR 具体失败在哪些场景（cut-in、环岛、行人）只给了 HighwayCutIn 的成功案例，缺少失败拆解。
- **对基准设计的建议较笼统**：提出"更长路线 + 反事实参与者"，但没有给出可操作的指标或协议草案。

## Notes

+ **论文定位**：不是记忆增强方法，而是**用记忆做基准审计**。它把"从前的行驶记录"当作静态信息的可控代理，从而在不改模型的前提下测试"当前场景的动态信息值多少分"。这一点与 MomAD / MomADv2（短期 planning-state 记忆，为性能服务）和 DriveVLA-M0（失败案例记忆 + TTT）不同：那两篇把记忆当能力，这篇把记忆当尺子。

+ **可迁移点**：
  + **"只给静态信息"是一个通用的基准探针**：任何有重复采集的场景数据都可以构造 pose-indexed memory bank，用来检验高分是否依赖当前观测。
  + **三档剂量对照**（ego only / +静态记忆 / +相机）比单一"有/无感知"对比更有解释力，能给出各信息源的边际贡献。
  + **地理切分是排除位置记忆的标准做法**：留出测试城市/城镇重训，若分数不降即可排除"背地图"解释。
  + **短窗口 + 初始 ego 状态的组合会泄漏机动意图**：任何 4 s 级评测都应检查"初始 ego 状态是否已经暗示了正确答案"，否则 NC/TTC 的高分可能来自保守减速而非场景理解。

+ **待跟进**：
  - 在 NAVSIM 上做窗口长度扫描（2 / 4 / 8 / 12 s），看记忆版的优势随窗口如何衰减，定位"ego 状态泄漏"的具体时间尺度。
  - 构造"记忆里含过期动态"的对照（同一地点、不同时刻的交通），检验 resampler 是否真的学会过滤，而不是简单平均。
  - 把同样的探针用到 plan-based 闭环基准（HUGSIM 全量、Bench2Drive 长路线）上，给出"静态信息贡献率"这一可比较的指标。
  - 若换成无重复轨迹的数据集，能否用"离线街景 + 单次采集"替代记忆库（作者提到的 spatial retrieval 变体），量化定位误差与域差对审计信号的削弱。
