---
title: "Spatial Retrieval Augmented Autonomous Driving"
authors: ["Xiaosong Jia", "Chenhe Zhang", "Yule Jiang", "Songbur Wong", "Zhiyuan Zhang", "Chen Chen", "Shaofeng Zhang", "Xuanhe Zhou", "Xue Yang", "Junchi Yan", "Yu-Gang Jiang"]
institute: ["Fudan University", "Shanghai Jiao Tong University", "Aerospace Information Research Institute, CAS", "University of Science and Technology of China"]
date_publish: 2025-12
venue: arXiv Preprint
tags: [spatial-memory, scene-understanding]
url: https://arxiv.org/abs/2512.06865
cite_key: jia2025spatialretrieval
arxiv_id: "2512.06865"
doi: "10.48550/arXiv.2512.06865"
code:
rating: 4
content_scope: full-text
verification_status: source-checked
date_added: 2026-11-26
---
## Summary

车载传感器的感知距离与视线都被限制，遮挡、逆光、夜间、雨雾下会明显退化；而人类司机可以"想起"这条路长什么样。论文把这种"回忆"做成一个新模态：按 ego 位姿从离线地理数据库（Google Maps 街景 + 卫星图）检索当前位置的图像，作为额外输入喂给模型。为了系统评估，作者用 Google Maps API 扩展 nuScenes 得到 **nuScenes-Geography**，并设计一个 model-agnostic 的 Spatial Retrieval Adapter（BEV 特征做 query、带 3D 位置编码的地理特征做 K/V 的 cross-attention），在五个任务上建立基线：在线建图、occupancy、3D 检测、端到端规划、生成式世界模型。结论是**对静态结构类任务收益大、对动态物体类任务几乎无效**：MapTRv2 mAP 68.7→78.2，FBOcc mIoU 39.11→39.74（静态类别提升更大），UniMLVG FVD 36.10→29.97；检测基本不变，规划只在夜间子集把碰撞率从 0.55% 降到 0.48%。另有一个可靠性门控，在检索缺失或错位时自动衰减地理分支，50% 数据被丢弃时性能平滑退化而非崩溃。

## Problem & Motivation

- 车载感知的物理边界：视野有限、被遮挡就看不见、夜间/雨雾信噪比骤降；下游的建图、occupancy、规划都会被连带拖累。
- 生成式世界模型还有另一个问题：ego 偏离录制轨迹后，缺乏空间锚点，容易生成漂移或幻觉，难以当闭环仿真器用。
- 人类驾驶在视野不足时依赖对道路结构的记忆；这类信息本质上是**离线、全局可获取、不受行驶时干扰**的。
- 现有检索式方法多用于视觉定位、LLM 规则理解或轨迹采样，没有把"与位置对齐的地理图像"当作感知输入直接注入模型。
- 论文的定位不是刷某个任务，而是提出并验证一个**新输入范式**，并用统一 adapter 在五个任务上给出可复现的上界/下界。

## Method

**0. 总体框架**

+ 数据流：在线传感器 $S_t$ + ego 位姿 $P_t$ → 检索函数 $\mathcal{R}$ 从地理库取对齐图像 → 与车载图像同一 backbone 编码 → 带 3D 位置编码后通过 cross-attention 融合进 BEV 特征 → 原任务头。
+ 四个在线任务（检测、建图、occupancy、规划）按帧检索最近的地理图像；世界模型是离线任务，已知未来轨迹，因此沿生成路径检索多张图作为"空间脚手架"。

**1. 空间检索的形式化（§3.1）**

+ 驾驶数据是 $\{(S_t,P_t)\}_{t=1}^{T}$，地理库是 $\{I_{\text{geo}},P_{\text{geo}}\}$（图像 + 经纬度与相机参数）；检索定义为：

$$
\text{RetrievedGeoData}_t=\mathcal{R}\left(I_t,I_{\text{geo}},P_t,P_{\text{geo}}\right)
$$

+ 实现上取"每个相机、每帧最近的一张地理图"；若 3D 距离超过阈值则返回 NONE。作者明确这是最简基线，邻域多图、任务特定检索留作未来工作。

**2. Spatial Retrieval Adapter（§3.2）**

+ 检索图用与车载相机相同的 backbone 编码得到 $F_{\text{geo}}$，再用 PETR 的方式给每个 patch 算相对当前 ego 的 3D 位置编码 $F^{\text{pos}}_{\text{geo}}$。
+ 融合是残差式 cross-attention，并以可靠性分数 $w$ 调制：

$$
F_{\text{BEV}}^{\prime}=F_{\text{BEV}}+w\cdot\text{CrossAttn}\left(F_{\text{BEV}},\ F_{\text{geo}}+F^{\text{pos}}_{\text{geo}},\ F_{\text{geo}}\right)
$$

+ 这个设计的关键性质是**不动原模型**：query 仍是原 BEV 特征，只改 K/V 来源，训练目标与任务头都不变，因此可以逐个任务插拔。

**3. 世界模型的地理注入（§3.3）**

+ 世界模型跑在服务器上、且知道未来轨迹，所以可以在生成片段的起点与终点位置各检索一张图，在 DiT 的每个 block 里加一层地理 cross-attention：

$$
F^{\prime}=F+w\cdot\text{CrossAttn}\left(F,\ F_{\text{geo}}+F^{\text{pos}}_{\text{geo}},\ F_{\text{geo}}\right)
$$

+ $F$ 是加噪 latent；起点/终点两张图提供"这段路两端长什么样"的锚点，用来抑制长时域 rollout 的场景漂移。

**4. 可靠性估计门控（§3.4）**

+ 地理数据会缺失（覆盖不到）或错位（施工改道、GPS 误差、街景拍摄高度不同），直接融合会把错误结构写进 BEV。
+ 门控用两个线索：车载与地理特征的 ZNCC 相似度、检索点到 ego 的 GPS 距离：

$$
w=\sigma\left(\text{MLP}\left([\text{ZNCC}(F_{\text{onboard}},F_{\text{geo}}),\ d_{\text{GPS}}]\right)\right)
$$

+ 训练时用二值标签监督 $w$（缺失/错位为 0，可靠为 1）；作者人工核查了 1800 个错位样本作为负例。推理时 $w$ 自动压低不可靠地理特征的权重。

**5. nuScenes-Geography 的构建（§4）**

+ 由 nuScenes map 的全局原点加 ego 位姿算出每帧的经纬度，查询 Google Maps API 取街景与卫星图。
+ 街景采样频率远低于 nuScenes 帧率，多帧会对应同一街景点；为省存储，每个街景点只下载一次，并建立"街景点 ↔ 最近帧"的映射。
+ 每点取 18 个 yaw 视角（覆盖 360°、pitch 固定 0°），拼成等距柱状全景；对每个 nuScenes 相机实例化一个**虚拟相机**（内参一致、外参由 ego 位姿与街景点偏移推出），从全景重投影出几何对齐的街景图。相比逐帧裁剪下载，存储降低 70% 以上。

**6. 实现细节**

| 项目 | 设置 |
|:--|:--|
| 地理源 | Google Maps Street View + Satellite/Maps API |
| 街景采样 | 每点 18 个 yaw 视角 → 等距柱状全景 → 虚拟相机重投影 |
| 融合位置 | BEV 特征 / DiT latent 的 cross-attention 残差分支 |
| 位置编码 | PETR 式 3D positional encoding |
| 门控监督 | 人工标注 1800 个错位样本作为负例 |
| 评测任务 | 检测（BEVDet/BEVFormer）、建图（MapTR/MapTRv2）、occupancy（FBOcc/FlashOCC）、规划（VAD）、世界模型（UniMLVG/MagicDriveDiT） |

## Pipeline

![Spatial Retrieval Adapter](https://arxiv.org/html/2512.06865v1/method_v5.png)

*检索到的地理图像与车载图像共享 backbone 编码，patch 特征加上相对 ego 的 3D 位置编码后作为 K/V，与原 BEV 特征（query）做 cross-attention，输出经可靠性分数 $w$ 加权后残差加回 BEV；世界模型分支把同一结构插在 DiT 的每个 block 里，query 换成加噪 latent，并沿生成轨迹取起点与终点的地理图。*

1. **定位**：由 ego 位姿换算出经纬度，确定当前所在路段。
2. **检索**：取每个相机最近的一张街景/卫星图；距离超阈值则返回空。
3. **编码**：地理图与车载图共用 backbone，得到地理 patch 特征并加 3D 位置编码。
4. **门控**：用 ZNCC 相似度与 GPS 距离估计可靠性 $w$。
5. **融合**：以 $w$ 加权的 cross-attention 把地理特征注入 BEV（或 DiT latent）。
6. **下游**：融合后的特征直接进入原任务头，训练目标与网络结构不变。
7. **世界模型变体**：沿已知目标轨迹取起终点地理图，注入生成过程以维持背景一致。

## Key Results

**1) 在线建图（Table 2，mAP）**

| Method | Epoch | APped | APdiv | APbound | mAP |
|:--|--:|--:|--:|--:|--:|
| MapTR | 24 | 46.3 | 51.5 | 53.1 | 50.3 |
| MapTR+Geo | 24 | 56.5 | 67.3 | 59.8 | **61.2**（+10.9） |
| MapTRv2 | 110 | 68.1 | 68.3 | 69.7 | 68.7 |
| MapTRv2+Geo | 110 | 79.8 | 77.4 | 77.3 | **78.2**（+9.5） |

+ 增益最大的是 **APdiv（可行驶区域分隔带）**：MapTR 24 epoch 时 +15.8，说明地理图确实补上了被遮挡或超出视野的道路结构。
+ 24 epoch 的短训练版本增益更大（+10.9），说明地理先验在训练不充分时更能补信息；110 epoch 时仍有 +9.5，不是"只帮欠训练模型"。

**2) Occupancy（Table 3，Occ3D-nuScenes）**

| Method | Overall mIoU | driveable | terrain | vegetation |
|:--|--:|--:|--:|--:|
| FBOcc | 39.11 | 80.07 | 55.13 | 37.53 |
| FBOcc+Geo | **39.74**（+0.63） | **82.47** | **57.70** | **38.57** |
| FlashOCC:M1 | 31.92 | 77.95 | 50.92 | 31.08 |
| FlashOCC:M1+Geo | **32.35**（+0.43） | 78.72 | 51.82 | 31.94 |

+ 总体只涨 0.4–0.6，但**静态类别的涨幅是总体的 3–5 倍**（terrain +2.57、sidewalk +2.08），与"地理图只提供背景结构"的定位一致。
+ 动态类别（车、行人）不在表内单列，作者在分析里直接说明地理模态不包含它们。

**3) 3D 检测（Table 4）**

| Method | NDS | mAP |
|:--|--:|--:|
| BEVDet | 39.41 | 30.85 |
| BEVDet+Geo | 39.43 | 30.69（−0.16） |
| BEVFormer | 51.70 | 41.60 |
| BEVFormer+Geo | 51.80 | 41.64 |

+ 基本无效，甚至 mAP 微降。这是论文最有信息量的负结果：**地理先验不能替代动态感知**，把它当"更强的感知输入"会失望。

**4) 端到端规划（Table 6，VAD）**

| Split | Method | L2 Avg↓ | Collision Avg↓ |
|:--|:--|--:|--:|
| Full | VAD | 0.64 | 0.37% |
| Full | VAD+Geo | 0.64 | 0.36% |
| Night | VAD | 0.87 | 0.55% |
| Night | VAD+Geo | 0.87 | **0.48%** |

+ L2 完全不变，说明地理图不影响轨迹回归的精度；收益只体现在**夜间碰撞率**（−0.07 个百分点，相对约 −13%）。
+ 这符合"补的是被光照破坏的道路结构"这一机制，但也说明它对常规场景几乎没有增益。

**5) 生成式世界模型（Table 5）**

| Method | FVD↓ | FID↓ |
|:--|--:|--:|
| UniMLVG | 36.10 | 5.82 |
| UniMLVG+Geo | **29.97**（−6.13） | **5.60** |
| MagicDriveDiT | 84.43 | 18.38 |
| MagicDriveDiT+Geo | **81.52** | **18.10** |

+ 五个任务里增益最明确的一项：FVD 降 6.1，说明长时域 rollout 的背景漂移被明显抑制。
+ 机制上这是"空间脚手架"最自然的用法：生成时已知未来位置，可以提前取到该位置的地理图，避免模型凭空想象背景。

**6) 消融与鲁棒性（Table 7、Fig. 11）**

+ 去掉地理图：静态 mIoU 47.86→46.66，FVD 29.97→35.42；去掉 3D 位置编码：FVD 从 29.97 升到 32.82，说明**位置编码是地理分支生效的必要条件**，否则模型不知道这张图对应哪里。
+ 去掉可靠性门控：静态 mIoU 47.86→47.65、FVD 29.97→30.95，增益虽小但一致。
+ 随机丢弃或替换 50% 的地理图时性能平滑下降，仍保留相当一部分增益；没有门控时下降更陡——门控的价值主要体现在"坏数据"场景，而不是平均性能。

## Strengths & Weaknesses

**Strengths**

- **范式清晰且可证伪**：地理图只携带静态结构，因此"静态任务受益、动态任务无效"是可以事先预测并验证的；结果（建图/occupancy/世界模型受益，检测无效）与预测一致，这比单纯刷分更有说服力。
- 数据工程做得实：等距柱状全景 + 虚拟相机重投影，把"每帧下载"变成"每街景点下载一次"，存储降 70%，这套做法可直接复用到其他城市的离线建库。
- Adapter 真正做到了 model-agnostic：五个任务、七种 baseline（MapTR/MapTRv2、FBOcc/FlashOCC、BEVDet/BEVFormer、VAD、UniMLVG/MagicDriveDiT）共用同一融合结构，只改注入位置。
- 负结果报告得清楚：检测一项 mAP −0.16，规划 L2 完全不变，论文都照实列出，并用"地理模态不含动态元素"解释，没有硬拗成全面领先。
- 可靠性门控的动机与验证配套：人工标注 1800 个错位样本做负例，再配随机丢弃/替换实验，给出了"坏数据下平滑退化"的曲线，而不是只声称鲁棒。
- 世界模型的增益最大且机制最自然：未来轨迹已知，起终点地理图就是天然的空间锚点，FVD 降 6.1 是比较硬的证据。
- 明确区分"离线检索（推理时无需联网）"与"在线 API 调用"，并在附录说明车载缓存与低覆盖场景的回退行为，工程落地性交代得比较完整。

**Weaknesses**

- **所有增益都建立在一个特定数据源上**：街景来自 Google Maps，覆盖密度、拍摄时间、季节与 nuScenes 采集时间都不一致；换城市、换供应商、或用车企自有采集数据时，增益能否复现未知。
- **地理图与车载图不是同一时刻**：街景可能是几年前拍的，道路改建、临时施工、季节变化都会让"静态信息"本身过期；论文用门控缓解，但没有量化"过期程度"与性能的关系。
- **检索只取最近一张图**：没有利用邻域多图、也没有按导航意图检索；对于路口这类需要"前方/侧向"结构的场景，单张最近图的视野可能不够，这限制了范式上限（作者自己列为 future work）。
- **规划的收益非常窄**：只在夜间子集把碰撞率从 0.55% 降到 0.48%，L2 完全不变；用 VAD 一个规划器、nuScenes 开环协议，无法说明闭环或反应式场景下是否有价值。
- **occupancy 的总体增益小且集中在静态类**：0.43–0.63 mIoU 在 Occ3D 这种指标上属于噪声量级，没有 seed 或方差；"静态类提升更大"是定性解释，没有做类别加权的显著性检验。
- **门控的监督依赖人工标注**：1800 个错位样本是人工核查的，跨数据集迁移时需要重新标注；且 ZNCC 在夜间/雨雾下本身不可靠——恰恰是地理先验最该发挥作用的场景，门控可能把有用的先验也压掉。
- **检测的负结果没有进一步挖掘**：作者说"用地理图区分前景/背景是有趣的方向"，但没有做任何尝试；对读者来说，"为什么连 background 都没帮到检测"缺少分析。
- **世界模型的对比不充分**：只与两个 baseline 比 FVD/FID，没有与"用 HD map 条件化"或"用卫星图条件化"做对照，无法判断增益来自"地理图像"还是"任何静态条件"。
- **没有真实闭环与跨城市验证**：全部实验在 nuScenes 上（含夜间子集），没有新城市、没有 CARLA/HUGSIM 闭环，部署层面的结论只能算初步。

## Notes

+ **论文定位**：不是新的规划器或世界模型，而是提出**一个输入模态 + 一套数据集 + 一组跨任务基线**。它属于 memory 在 AD 里的"离线地理记忆"分支：不做在线历史积累、不学检索策略，只按位姿取预先存好的静态画面。与 MomAD/MomADv2 的短期 planning-state 记忆、DriveVLA-M0 的失败案例记忆、Driving on Memory 的重复行驶记忆都是不同层次。

+ **可迁移点**：
  + **"静态 vs 动态"的任务分类是判断一个先验有没有用的最快方式**：先问这个模态能不能提供动态物体，再决定往哪个任务上试，可以省掉大量无效实验。
  + **等距柱状全景 + 虚拟相机重投影**是低成本的"一次采集、多视角复用"方案，适合任何需要按位姿取图的场景（街景先验、地图对齐、仿真背景）。
  + **3D 位置编码不可省**：消融显示去掉后 FVD 从 29.97 回到 32.82，几乎抵消掉地理图本身的收益——外部图像必须让模型知道"它对应哪里"。
  + **可靠性门控 + 人工负例**是处理外部数据的通用做法：先定义"不可靠"的几类模式，标一小批负例，再让模型学着压低权重；比规则阈值更稳。
  + **生成式世界模型是最适合地理先验的落点**：离线、已知未来轨迹、需要长时域背景一致性，三个条件都满足。

+ **待跟进**：
  - 把"最近一张图"换成"沿导航意图检索多张图"，看路口与匝道场景的建图/occupancy 增益能否放大。
  - 量化地理图"时效性"的影响：按街景拍摄年份分层，统计增益随年份差的衰减曲线。
  - 在闭环（HUGSIM / Bench2Drive）里验证规划收益，尤其是夜间与遮挡场景；开环 L2 不变说明不了闭环安全。
  - 与 HD map / 卫星图条件化做对照，判断世界模型的 FVD 增益来自"图像外观"还是"任意静态空间条件"。
  - 用自家车队采集的街景替换 Google Maps，检验跨数据源的可迁移性与门控的重新标定成本。
