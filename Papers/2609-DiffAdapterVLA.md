---
title: "Planning in the Backbone: DiffAdapterVLA for Native Continuous Trajectory Generation with Driving VLMs"
authors: ["Changxin Lu", "Xiaoliang Meng", "Yu Wu", "Rui Huang", "Honglin Li", "Tao Chen", "Kaixuan Zhou", "Yadong Shao"]
institute: ["School of Remote Sensing and Information Engineering, Wuhan University", "Dongfeng Research & Development Institute"]
date_publish: 2026-09
venue: arXiv preprint
tags: [VLA, diffusion-policy]
url: https://arxiv.org/abs/2609.15322
cite_key: lu2026planning
arxiv_id: "2609.15322"
doi: "10.48550/arXiv.2609.15322"
code:
rating: 3
content_scope: full-text
verification_status: source-checked
date_added: 2026-10-20
---

## Summary

现有 driving VLM 的轨迹生成几乎都在 backbone 之外完成：要么外接一个 planner 读最终层 cache，要么等 VLM 形成完整 condition 后再由一个独立 denoiser 采样。DiffAdapterVLA 把显式 trajectory token 插入 VLM 最后 K=12 层，让轨迹状态在去噪的每一步都与不同深度的 driving condition 共同前向演化；每层后接一个轻量 DiffAdapter 负责轨迹更新，condition 侧保持冻结且看不到带噪轨迹。整个 8B Cosmos-Reason2 backbone 冻结，只训练轨迹嵌入、12 个 adapter、history encoder 和 readout，占 3.05% 参数。NAVSIM navtest 上 SFT 达 **88.3 PDMS**，加 DiffGRPO 后 **90.3 PDMS**；端到端延迟 80.4 ms，比 Alpamayo-1.5 的 action expert（1003.8 ms、71.1 PDMS）快 12.5×。

## Problem & Motivation

- 预训练 driving VLM 的表征目标与连续规划目标不一致：VLM 用离散 next-token 学语义，规划要求几何精确、时序连续、动力学可行的轨迹；语义能力强并不自动等于控制能力强。
- 作者把现有接口归为两类：**loosely coupled**（VLM 语义线索交给独立 planner）与 **final-condition**（planner 只读最后一层 cache）。两者的共同缺陷是 condition 在轨迹生成开始前就已经算完，轨迹状态无法在 condition 逐层形成的过程中施加影响。
- 由此产生的具体损失是：最终层表征已经把多层、多尺度的中间信息压缩掉，denoiser 只能从一份"成品"condition 解码，无法按噪声水平选择不同深度的条件。
- 论文的主张是把 planning 放进 backbone：轨迹 token 与 condition token 在最后 K 层共享同一个 forward，condition 逐层指导轨迹，而轨迹不反写 condition。这样既不改动冻结 VLM 的语义通路，又能让轨迹 refinement 复用它的深度计算。
- 工程动机是参数效率：工业部署中 8B VLM 无法重训，只能加少量可训练模块；3.05% 的可训练比例使这条路线在成本上可行。

## Method

**0. 总体框架**

- 基座为 Alpamayo-1.5-10B，其中 Cosmos-Reason2-8B driving VLM 全程冻结；只训练线性 trajectory embedding、12 个 layer-wise DiffAdapter、history trajectory encoder 与线性 readout。
- 输入为当前前视相机、route 信息与 4 帧历史自车位姿，输出 2 Hz、8 个 future waypoint（4 秒时域）。
- 训练分两段：SFT 80 epochs（DDPM 噪声预测、100 扩散步），再从 SFT 权重做 10 epochs DiffGRPO；推理用 5 步确定性 $x_0$ 投影。

**1. History Trajectory Encoder（§4.1）**

- 把自车历史位置 $h_t^{\text{ego}}=[q_{t-T_h+1},\dots,q_t]\in\mathbb{R}^{T_h\times3}$ 转成七维描述子，同时保留绝对位置、局部平面位移、观测有效性与历史有效性统计：

$$
r_i=\left[q_i,\ \Delta q_i^{xy},\ m_i,\ \rho_t\right]
$$

- 描述子线性投影后做 mask，再用 depthwise 时序卷积捕捉局部运动变化，且不混合 hidden channel：

$$
\tilde{u}_i=u_i+\operatorname{DWConv}_3(u)_i
$$

- 用 masked mean pooling 聚合，无效位置不参与；随后两个 residual MLP 得到共享上下文向量：

$$
\bar{u}_t=\frac{\sum_{i=1}^{T_h}m_i\tilde{u}_i}{\max\left(\sum_{i=1}^{T_h}m_i,1\right)}
$$

$$
v_t=\bar{u}_t+f_{\text{temp}}\left(\bar{u}_t\right),\qquad g_t=v_t+f_{\text{out}}\left(\operatorname{LN}\left(v_t\right)\right)
$$

- $g_t\in\mathbb{R}^D$ 由所有所选层的 DiffAdapter 共用，并且每个样本只算一次、在所有去噪步之间复用。

**2. Trajectory Token 与 Backbone 注入（§4.2）**

- 带噪未来轨迹 $x_s\in\mathbb{R}^{H\times d}$ 的每个 waypoint 被映射为 VLM hidden space 中的 token，并叠加未来位置、token 类型与扩散步三种 embedding：

$$
z_{s,i}^{0}=W_{\text{traj}}x_{s,i}+e_i^{\text{pos}}+e^{\text{type}}+e_s^{\text{time}}
$$

- 前 $L-K$ 层冻结，只负责把多模态 condition 推到第 $L-K$ 层；随后把 trajectory token 追加到 condition token 之后，一起进入最后 $K$ 层。
- 每个冻结层输出联合序列后，DiffAdapter 只改写 trajectory 切片，condition 切片原样传给下一层：

$$
C^{\ell+1}=\tilde{C}^{\ell+1},\qquad Z_s^{\ell+1}=\mathcal{A}_\ell\left(\tilde{Z}_s^{\ell+1},\tilde{C}^{\ell+1},g_t,s\right)
$$

- DiffAdapter 使用 384 维 bottleneck、8 个 attention head。输入先投影到 bottleneck，扩散步 $s$ 经 sinusoidal embedding + SiLU MLP 产生 AdaLN 仿射参数与各子模块的 residual gate：

$$
u_1=u_0+\tanh\left(g_{\text{tmp}}(s)\right)\odot\operatorname{DWConv}_3\left(\operatorname{AdaLN}_{\text{tmp}}\left(u_0;s\right)\right)
$$

$$
u_2=u_1+\tanh\left(g_{\text{self}}(s)\right)\odot\operatorname{MHSA}\left(\operatorname{AdaLN}_{\text{self}}\left(u_1;s\right)\right)
$$

- DWConv3 混合相邻 waypoint，MHSA 建模整条轨迹的长程依赖；随后以 trajectory 为 query、当前层 condition 为 key/value 做 cross-attention：

$$
u_3=u_2+\operatorname{CrossAttn}\left(\operatorname{LN}(u_2),\ W_c\operatorname{LN}\left(\tilde{C}^{\ell+1}\right),\ W_c\operatorname{LN}\left(\tilde{C}^{\ell+1}\right)\right)
$$

- $g_t$ 广播到每个 trajectory token，经 gated GELU MLP 融合；之后是 gated FFN 与外层 residual。每个 adapter 因此同时具备局部运动混合、轨迹全局协调、条件读取与历史调制四种作用。
- 最后经 VLM 的 final norm 与线性 readout 预测该步噪声：

$$
\hat{\epsilon}_s=W_{\text{out}}Z_s^{L}+b_{\text{out}}
$$

**3. Condition-Preserving Asymmetric Joint Attention（§4.3）**

- 最后 $K$ 层收到 condition 与 trajectory 两段 token，二者的读权限不对称：condition query 不能看 trajectory，trajectory query 可以看全部有效 condition，trajectory 之间双向可见：

$$
M_{ij}=\begin{cases}
M_{ij}^{\text{cond}}, & i,j\in\mathcal{I}_{\text{c}}\\
-\infty, & i\in\mathcal{I}_{\text{c}},\ j\in\mathcal{I}_{\text{z}}\\
0, & i\in\mathcal{I}_{\text{z}},\ j\in\mathcal{I}_{\text{z}}\\
0, & i\in\mathcal{I}_{\text{z}},\ j\in\mathcal{I}_{\text{c}}\ \text{且}\ j\ \text{有效}\\
-\infty, & i\in\mathcal{I}_{\text{z}},\ j\in\mathcal{I}_{\text{c}}\ \text{且}\ j\ \text{无效}
\end{cases}
$$

- 这个 mask 保证冻结 VLM 的 condition 通路与原始模型完全一致，带噪轨迹不会污染语义表征；同时轨迹在每个深度都能读到该层最新的条件，而不是等最终层。

**4. 去噪与训练配置**

- 从高斯噪声出发，做 5 步等间隔的确定性 $x_0$ 投影；每步先估计 clipped clean trajectory，再缩放到下一噪声水平：

$$
\hat{x}_0^{(s)}=\operatorname{clip}\left(\frac{x_s-\sqrt{1-\bar{\alpha}_s}\,\hat{\epsilon}_s}{\sqrt{\bar{\alpha}_s}},-1,1\right)
$$

$$
x_{s'}=\sqrt{\bar{\alpha}_{s'}}\,\hat{x}_0^{(s)}
$$

- condition 在去噪过程中固定，因此 $C^{L-K}$、其 validity mask 与位置信息可以缓存并在 5 步之间复用，每步只跑最后 $K$ 层与 adapter。
- DiffGRPO 每个 condition 采 8 条轨迹做 group-relative 更新，并加 teacher-chain behavior cloning anchor（权重 0.1）与 SFT reference regularizer（权重 0.05）。

| 项目 | 设置 |
|:--|:--|
| Backbone | Alpamayo-1.5-10B，Cosmos-Reason2-8B 冻结 |
| 可训练参数 | 3.05%（K=12） |
| 注入层数 | 最后 K=12 层，每层一个 DiffAdapter |
| Adapter 容量 | 384 维 bottleneck，8 heads |
| SFT | 80 epochs，DDPM，100 扩散步 |
| RFT | DiffGRPO 10 epochs，每 condition 8 样本，BC anchor 0.1，SFT reg 0.05 |
| 优化 | 8×A800，bf16，ZeRO-2；SFT batch 16/卡（global 128）、lr 1e-5、500 warmup、cosine；RFT batch 8/卡（global 64）、lr 1e-5 cosine 到 1e-6、weight decay 1e-4 |
| 数据 | navtrain 1,192 场景；navtest 136 场景 |
| 推理 | 5 步确定性 $x_0$ 投影，condition cache 复用 |

## Pipeline

![DiffAdapterVLA 整体架构](../images/DiffAdapterVLA-architecture.png)

*冻结的早期 VLM 层编码 driving condition；trajectory token 在最后 K 层前注入，每层冻结 VLM 更新联合序列后由 DiffAdapter 只改写轨迹切片；readout 预测噪声，5 步 $x_0$ 投影得到最终轨迹。*

1. **条件编码**：前视相机、route、语言提示与 4 帧历史位姿进入冻结的 Cosmos-Reason2，前 $L-K$ 层算出 $C^{L-K}$ 并缓存。
2. **历史编码**：历史自车位姿经七维描述子、DWConv3、masked pooling 与 residual MLP 得到 $g_t$，每个样本只算一次。
3. **trajectory token 构造**：当前带噪轨迹 $x_s$ 逐 waypoint 投影，叠加位置、类型与扩散步 embedding。
4. **联合前向**：trajectory token 追加到 condition 之后进入最后 12 层；每层在 asymmetric mask 下做联合 attention。
5. **逐层轨迹更新**：该层 DiffAdapter 读取更新后的 condition、$g_t$ 与扩散步 $s$，经 AdaLN、DWConv3、MHSA、cross-attention 与 gated FFN 改写 trajectory token。
6. **噪声预测与投影**：final norm + 线性 readout 输出 $\hat{\epsilon}_s$，按确定性 $x_0$ 投影更新到下一噪声水平，重复 5 次。
7. **训练**：先 80 epochs DDPM 噪声预测，再用 DiffGRPO 做 10 epochs 轨迹级 RL。

## Key Results

**1) NAVSIM navtest 主结果（Table 1，PDMS）**

| Method | Image | LiDAR | NC | DAC | TTC | Comf. | EP | PDMS |
|:--|:--:|:--:|--:|--:|--:|--:|--:|--:|
| ReCogDrive | ✓ | | 98.1 | 94.7 | 94.2 | 100 | 80.9 | 86.5 |
| DiffusionDrive | ✓ | ✓ | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| WoTE | ✓ | ✓ | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| DP-VLA | ✓ | | 98.0 | 97.0 | 94.3 | 100 | 82.5 | 88.3 |
| AutoVLA | ✓ | | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| CoWorld-VLA | ✓ | | 99.2 | 96.8 | 96.6 | 100 | 83.6 | 89.8 |
| DriveDPO | ✓ | ✓ | 98.5 | 98.1 | 94.8 | 99.9 | 84.3 | 90.0 |
| DriveVLA-W0 | ✓ | | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| **DiffAdapterVLA-SFT** | ✓ | | 98.7 | 96.7 | 95.9 | 100 | 80.9 | **88.3** |
| **DiffAdapterVLA-RFT** | ✓ | | **98.8** | **98.1** | **96.7** | **100** | **83.0** | **90.3** |

- SFT 只用 3.05% 参数就达到 88.3，与带 LiDAR 的 WoTE、纯相机的 DP-VLA 持平；RFT 再加 2.0 到 90.3，超过 DriveDPO 0.3、DriveVLA-W0 0.1。
- 增益结构值得注意：SFT 的 EP 只有 80.9，明显低于同级方法的 82–84；RFT 把 EP 拉到 83.0、DAC 从 96.7 拉到 98.1，也就是说 2.0 的提升主要来自"更敢往前走"和"更少出可行驶区域"，而不是碰撞指标。
- 安全性并未全面领先：NC 98.8 高于 DiffusionDrive / WoTE，但低于 CoWorld-VLA 的 99.2；TTC 96.7 也低于 AutoVLA 的 98.0。Comfort 的 100 是表内上限，该项已饱和，无法再区分轨迹平滑度。

**2) 推理效率（Table 2，共享 8-GPU 协议）**

| Model | Trajectory generation | PDMS | Model ms/sample | E2E ms/sample |
|:--|:--|--:|--:|--:|
| Cosmos-Reason2 + DiT（无 LoRA） | External DiT planner | 76.4 | 40.2 | 67.5 |
| Alpamayo 1.5（stage-2） | External action expert | 71.1 | 1003.0 | 1003.8 |
| Cosmos-Reason2 + DiT（LoRA） | External DiT planner | 81.9 | 43.6 | 67.6 |
| ReCogDrive Large-IL | External diffusion planner | 86.5 | 722.5 | 739.1 |
| DriveVLA-W0（flow） | External action expert | 87.2 | 1186.8 | 1186.8 |
| OneVL AR Answer | Autoregressive action tokens | 87.5 | 1218.3 | 1258.8 |
| **DiffAdapterVLA（K=12）** | Native diffusion adapters | **88.3** | **77.4** | **80.4** |

- 相对 Alpamayo 自家 stage-2 action expert 是 12.5× 加速且 PDMS +17.2；相对 DiT+LoRA 只慢 12.8 ms（80.4 vs 67.6），但 PDMS 高 6.4。
- 80.4 ms 包含输入准备、后处理与评测接口；模型本身 77.4 ms，说明瓶颈几乎全在 backbone 前向与 5 步去噪，不在宿主侧。

**3) 组件消融（Table 3，50 epochs）**

| Asymmetric attention | History encoder | DiffAdapter | NC | TTC | EP | PDMS |
|:--:|:--:|:--:|--:|--:|--:|--:|
| ✓ | × | × | 86.1 | 74.9 | 51.9 | 58.2 |
| ✓ | × | ✓ | 98.0 | 94.2 | 78.0 | 85.3 |
| × | ✓ | ✓ | 98.5 | 96.0 | 77.8 | 85.6 |
| ✓ | ✓ | ✓ | 98.1 | 94.8 | 80.5 | 86.9 |

- 同时去掉 DiffAdapter 与 history encoder 后 PDMS 塌到 58.2，是结构性的：没有逐层轨迹更新，trajectory token 在冻结层里只是被动地被 condition 混合。
- history encoder 与 asymmetric attention 各贡献 1.6 与 1.3 PDMS，量级接近；两者都主要体现在 EP（78.0→80.5），说明它们改善的是"轨迹是否走得远"，而非安全性。
- 完整模型 NC 98.1 反而低于去掉 asymmetric 的 98.5、去掉 history 的 98.0 基本持平，安全性维度上组件贡献并不单调。

**4) 接口对照（Table 4）**

| Planning interface | NC | DAC | TTC | Comf. | EP | PDMS |
|:--|--:|--:|--:|--:|--:|--:|
| Final-cache condition only | 95.7 | 90.9 | 89.7 | 100 | 76.2 | 80.2 |
| Late-layer conditions，trajectory outside VLM | 97.6 | 94.5 | 93.3 | 100 | 79.3 | 85.2 |
| Late-layer conditions，trajectory inside VLM | 98.1 | 95.7 | 94.8 | 100 | 80.5 | 86.9 |

- 这是全文最直接支持标题的证据：仅把 final cache 换成逐层 condition 就从 80.2 升到 85.2（+5.0），再把 trajectory 放进 backbone 联合前向又 +1.7。
- 但 +5.0 的那一档仍然依赖 5 步去噪与同一套 adapter 容量，因此它衡量的是"条件深度"而非"是否在 backbone 内"；作者把两者拆开是对的，只是后者的净收益只有 1.7。

**5) 层数与去噪步数（Table 5–6）**

| K（注入层数） | 可训练参数 | EP | PDMS |
|--:|--:|--:|--:|
| 4 | 2.03% | 67.0 | 73.0 |
| 8 | 2.54% | 75.7 | 84.1 |
| 12 | 3.05% | 80.5 | 86.9 |
| 16 | 3.56% | 74.2 | 82.1 |
| 20 | 4.06% | 71.1 | 81.1 |

| 去噪步数 | PDMS | E2E ms/sample |
|--:|--:|--:|
| 1 | 1.92 | 74.1 |
| 2 | 82.63 | 68.5 |
| 3 | 84.32 | 72.1 |
| 5 | 87.00 | 80.8 |
| 10 | 87.07 | 126.8 |

- K 的最优值在 12，继续扩到 16/20 后 DAC 与 EP 同时下降（DAC 95.7→91.5→91.0，EP 80.5→74.2→71.1）；作者解释为轨迹过早进入 backbone 会干扰 condition 形成，但这只是曲线形状，未给出机制证据。
- 去噪步数从 5 增到 10 只涨 0.07 PDMS，延迟却从 80.8 涨到 126.8 ms；5 步是合理工作点。2 步已经能到 82.63，1 步完全不可用（1.92），说明该方法仍需要多步 refinement。

**6) NAVSIM v2（Table 7）**

- EPDMS 86.3，比 DriveVLA-W0 高 0.2、比 ReCogDrive 高 2.7；EP 88.2、TTC 97.7、DDC 99.1、TLC 99.8。
- 这一结果对输出做了 path-preserving temporal calibration，只调整沿路径的时间参数化、不改空间折线与终点；该后处理是否同样施加于 baseline 未说明。

## Strengths & Weaknesses

**Strengths**

- 把"接口"而不是"新 denoiser"作为研究对象，问题提得准：Table 4 用同一 backbone、同一 adapter 容量、同一采样步数只改 condition 的可见深度，直接把"final cache 是否够用"变成可测变量。
- asymmetric mask 的设计有明确的功能理由：condition 通路与冻结 VLM 完全一致，避免带噪轨迹污染语义表征；同时轨迹在每个深度都能读到最新条件。这比"把轨迹拼进序列一起 attention"更克制。
- 参数效率真实可核查：3.05% 可训练参数、80.4 ms 端到端、与 DiT+LoRA 的 67.6 ms 同量级；对工业部署而言这组数字比 PDMS 更有说服力。
- 工程细节完整：condition cache 复用、$g_t$ 每样本只算一次、5 步 $x_0$ 投影、K 的扫描、去噪步数的 quality–efficiency 曲线都给了。
- 附录的 layer-wise 动态分析（adapter 更新 RMS、cross-attention residual）至少展示了不同深度、不同噪声步的分工确实存在，而不是所有 adapter 学成同一个函数。
- 作者在 Limitations 中主动承认只建模自车、交互参与者仍是隐式的，没有把单智能体假设包装成多智能体能力。

**Weaknesses**

- **消融与主结果训练预算不一致**：Table 3/4/5 全部是 50 epochs，而主结果 SFT 是 80 epochs；同一完整配置在消融里是 86.9，在主表里是 88.3。组件增益（1.3–1.6 PDMS）是在欠训练 checkpoint 上测的，不能直接外推到最终模型。
- **只在 NAVSIM 单 benchmark 上验证**：navtest 仅 136 个场景，没有多 seed、没有置信区间；88.3 与 90.3 之间 2.0 的 RFT 增益是否超出运行方差未知。NAVSIM 本身是非反应式伪闭环，论文多处称 closed-loop，实际不含其他交通参与者对自车动作的响应。
- **RFT 的增益没有与"更长 SFT"隔离**：DiffGRPO 从 SFT checkpoint 再训 10 epochs，同时带 teacher-chain BC anchor（0.1）与 SFT reference regularizer（0.05）。缺少一个"SFT 继续训 10 epochs"的对照，无法判断 +2.0 来自 RL、来自更多梯度步，还是来自 BC anchor 的保守化。
- **交互参与者完全隐式**：显式状态只有自车未来轨迹，其他 agent 的多模态未来与不确定性都压在 condition 表征里。Table 1 的 NC/TTC 提升因此不能解释为"更好地预测了他人"，只能解释为"更保守或更贴合专家分布"。
- **12.5× 加速的参照系偏弱**：对比对象 Alpamayo stage-2 action expert 是 1003.8 ms 的异常慢基线；与真正轻量的 DiT+LoRA（67.6 ms）比，本文反而慢 19%，只是 PDMS 高 6.4。标题级的效率结论依赖于选取哪个 baseline。
- **EPDMS 86.3 含测试时后处理**：path-preserving temporal calibration 只改时间参数化，但它直接作用于被评测的输出；论文没有说明 baseline 是否也做同样校准，也没有给不做校准的 EPDMS。该分数不能与主表 PDMS 等同看待。
- **K=12 的最优性缺少机制解释**：K=16/20 下降被归因于"干扰 condition 形成"，但没有任何干预实验（例如只冻结前 12 层、或在 16 层中随机选 12 层）来区分"层数过多"与"注入了错误的层"。
- **layer-wise 动态分析是描述性的**：更新 RMS 与 cross-attention residual 只说明不同 adapter 行为不同，不能证明这些差异对最终轨迹有因果贡献；没有 ablation 按深度禁用 adapter。
- **延迟数字不可直接迁移到车端**：80.4 ms 是 A800 上 batch size 8、固定输入形状的结果；8B backbone 的显存占用、量化后的精度损失、车端芯片上的实际时延都未讨论。
- **消融表缺少方差与重复**：Table 3 中 history 与 attention 的差距只有 0.3 PDMS（85.3 vs 85.6），小于常见 seed 波动；把这 0.3 解释为"两者互补"证据不足。

## Notes

+ **论文定位**：不是新的扩散 policy，而是对"driving VLM 与连续规划之间接口"的一次系统性重排；核心 insight 是 condition 应该按深度逐步暴露给轨迹，而不是在最终层一次性交付。
+ **可迁移点**：
  - asymmetric joint attention 的"单向可见"mask 可用于任何冻结 backbone + 新增连续状态 token 的场景（VLA 的 action chunk、world model 的 latent slot），保证新增状态不污染预训练通路。
  - 把 condition hidden 缓存、只重跑最后 K 层，是把迭代式生成嵌进大 backbone 的通用工程手段；收益取决于 K 与去噪步数的乘积。
  - 七维历史描述子（绝对位置 + 平面位移 + 有效性 + 统计量）是一个很轻的历史编码基线，可直接用于其他 waypoint 预测器。
+ **待跟进**：
  - 用相同 80 epochs 重跑 Table 3/4，确认组件增益是否仍在 1–2 PDMS 量级；若显著缩小，说明部分增益来自欠训练下的优化难度而非结构。
  - 补"SFT 90 epochs"与"DiffGRPO 无 BC anchor"两个对照，分离 RL 与正则项的贡献。
  - 在 navtest 多解场景（路口、环岛）统计 RFT 前后的模态覆盖，检查 90.3 是否以牺牲多模态为代价。
  - 查 path-preserving temporal calibration 的具体实现与 baseline 是否同待遇，再决定 EPDMS 86.3 能否与 Table 2 的方法横比。
