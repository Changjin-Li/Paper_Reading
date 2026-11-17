---
title: "GuidedWAM: Training-Free Energy Guidance for World Action Models in Autonomous Driving"
authors: ["Xiyan Su", "Finn Rasmus Schäfer", "Hojun Lim", "Frank Diermeyer", "Johannes Betz", "Markus Lienkamp"]
institute: ["Technical University of Munich", "Munich Institute of Robotics and Machine Intelligence (MIRMI)"]
date_publish: 2026-11
venue: arXiv Preprint
tags: [world-model, flow-matching]
url: https://arxiv.org/abs/2611.07912
arxiv_id: "2611.07912"
doi: "10.48550/arXiv.2611.07912"
code:
rating: 3
content_scope: full-text
date_added: 2026-11-17
---

## Summary

WAM 把驾驶策略与视觉世界模型耦合在同一个生成网络里，训练结束后**驾驶目标就冻结了**：想加"窄路多留边距""行人周围更大净空""分段限速"这类新目标，只能重新收集监督并重训十亿参数模型。GuidedWAM 把能量引导做成一个**训练无关的控制通道**：在每个采样步上，用冻结模型的一步干净估计 $\hat\tau_0$ 评估可微能量，把它的梯度直接加到速度场上；由于梯度只对解析能量求、干净估计被 detach，整条链路只需要对冻结模型做前向。为了让这种 Jacobian-free 形式可用，论文给出三项稳定化措施——**噪声水平上的梯形调度窗、逐能量归一化、以首次引导步的速度范数为上界的修正裁剪**。在 NAVSIM v2 hard-split 上，用单一的相机可行驶区域能量，三个已公开 WAM 的 EPDMS 提升 **+1.8 至 +3.6**（三 seed 平均），每步采样耗时增加不到 10%。

## Problem & Motivation

- WAM 的规划目标在训练时被冻结：新目标只能靠"补监督 + 重训"获得，而且每加一个新目标就要重来一次，成本与目标数量线性增长。
- 采样期的条件化是另一条路，但它通常要求该目标在训练数据里出现过（conditioning 依赖训练时的 label），对"部署时才出现的临时目标"无效。
- 现成的 guidance 机制（classifier guidance、classifier-free guidance）都需要额外的分类器或无条件分支；能量引导（energy guidance）只需要一个可微目标函数，与模型训练解耦，是更合适的载体。
- 难点在于：精确的能量引导需要 clean estimate 对 $\tau_t$ 的 Jacobian $J_t$，而 $J_t$ 含网络 Jacobian，等价于每个采样步做一次完整反向传播——对十亿参数视频 WAM 不现实。
- 因此论文的问题是：**能否只靠前向访问，在冻结 WAM 的采样过程中注入新目标，并让它稳定工作？**

## Method

**1. 预备：flow matching 的 score-velocity 对应（§3.1）**

- 动作流 $\tau\in\mathbb{R}^{T\times d}$（自车轨迹或动作 token 序列）由条件 flow matching 采样器生成：$\tau_t=(1-t)\tau_0+t\varepsilon$，$\varepsilon\sim\mathcal{N}(0,I)$，网络学 $v_\theta(\tau_t,t,X)\approx\mathbb{E}[\varepsilon-\tau_0\mid\tau_t,X]$，采样从 $t=1$ 积分到 $t=0$。
- 引导本质是对 score $s_t=\nabla_{\tau_t}\log p_t(\tau_t\mid X)$ 的修正；在 Gaussian flow matching 路径上 score 与速度可互相换算：

$$
v(\tau_t,t)=-\frac{\tau_t+t\,s_t(\tau_t)}{1-t}
$$

- 因此一个 score 修正 $\Delta s$ 进入速度场时要乘上 $-t/(1-t)$。论文在 score 空间推导、在速度空间实现，因为公开的 WAM 采样器只暴露速度场。

**2. 一步干净估计上的能量引导（§3.2）**

- 每个采样步都能得到当前噪声下的后验均值，也就是 clean estimate：

$$
\hat\tau_0(\tau_t)=\tau_t-t\,v_\theta(\tau_t,t,X)\approx\mathbb{E}[\tau_0\mid\tau_t,X]
$$

- 驾驶目标被写成关于干净轨迹的可微能量 $\mathcal{E}:\mathbb{R}^{T\times d}\to\mathbb{R}$（到车道中心线的距离、到可行驶边界的 signed distance、到预测 agent 框的净空等）；实际使用时先把 $\hat\tau_0$ 解码成带位置与朝向的 waypoint 序列，再在几何量上解析求梯度。
- 目标分布是对原后验的能量倾斜：

$$
p_\gamma(\tau_0\mid X)\propto p(\tau_0\mid X)\,e^{-\gamma\mathcal{E}(\tau_0)}
$$

- 把倾斜写成"以辅助变量 $\mathcal{G}$ 为条件"，并令 $p(\mathcal{G}\mid\tau_0)\propto e^{-\gamma\mathcal{E}(\tau_0)}$，由 Bayes 规则得 $p(\tau_0\mid\mathcal{G},X)=p_\gamma(\tau_0\mid X)$，于是条件化后的 score 是：

$$
s_t^\gamma(\tau_t)=s_t(\tau_t)+\nabla_{\tau_t}\log\mathbb{E}_{p(\tau_0\mid\tau_t,X)}\big[e^{-\gamma\mathcal{E}(\tau_0)}\big]
$$

- 期望不可解，于是用 clean estimate 代替后验平均（与 diffusion posterior sampling 同源），log-expectation 变成 $-\gamma\mathcal{E}(\hat\tau_0)$，链式法则给出：

$$
\nabla_{\tau_t}\mathcal{E}\big(\hat\tau_0(\tau_t)\big)=J_t^{\top}\nabla_{\hat\tau_0}\mathcal{E}(\hat\tau_0),\qquad J_t=\frac{\partial\hat\tau_0}{\partial\tau_t}=I-t\,\frac{\partial v_\theta}{\partial\tau_t}
$$

- **这里做了全文最关键的一步近似**：把 $J_t$ 换成单位阵，只保留能量自身的梯度，并在 detached 的干净估计上求值（manifold-preserving guided diffusion 的做法）。换算回速度空间后得到实际执行的更新：

$$
v'=v_\theta+\lambda(t)\,\nabla_{\hat\tau_0}\mathcal{E}(\hat\tau_0),\qquad \lambda^*(t)=\gamma\frac{t}{1-t}
$$

- "无需反向传播"是这么来的：干净估计被 detach，唯一被求导的对象是解析能量对一个 $T\times d$ 张量的函数；冻结模型每步只前向一次，不保存激活。这也解释了为什么它能直接作用在已发布的 checkpoint 上。
- 代价同样明确：被删掉的 $J_t$ 原本是一个**预条件子**，把推力投影到局部合理的流形方向上；删掉后修正更容易离开流形，而 canonical 调度 $\lambda^*(t)=\gamma t/(1-t)$ 在 $t\to1$（clean estimate 最不可靠处）发散。

**3. 三项稳定化（§3.3）**

- **梯形调度窗**：canonical $\lambda^*$ 把最大修正放在高噪声端（$t\to1$），那里 clean estimate 只是未来的粗略平均；常数 $\lambda$ 则在 $t\to0$ 还在推、而那时计划已定。于是把调度截成噪声水平上的梯形窗——中段最大、两侧线性升降：修正只在干净估计变得有信息之后才起作用，末段留给冻结模型把计划拉回流形。
- **逐能量归一化**：不同能量的梯度尺度差异大，所以先按样本归一化再加权，让 $\lambda(t)$ 独占"总强度"、$w_k$ 只表达相对重要性：

$$
\Delta v=\sum_{k=1}^{K}w_k\,\frac{\nabla_{\hat\tau_0}\mathcal{E}_k(\hat\tau_0)}{\lVert\nabla_{\hat\tau_0}\mathcal{E}_k(\hat\tau_0)\rVert_2}
$$

- **速度有界修正**：修正的幅度相对模型自身速度设上界。若上界按**当前**速度测，会被"推力把下一步速度抬高、上界随之抬高"的正反馈放大（作者报告早期实验里速度范数涨了数倍、轨迹端点跑出路网一公里以上），所以参考量在第一引导步记录一次后固定：

$$
\rho=\lVert v_\theta(\tau_t)\rVert_2,\qquad \Delta v\leftarrow\min\Big(1,\frac{c\,\rho}{\lambda(t)\lVert\Delta v\rVert_2}\Big)\Delta v
$$

- 这样每一步都满足 $\lambda(t)\lVert\Delta v\rVert\le c\rho$，且比例常数 $c$ 跨模型共享——这是"强度只需按模型标定一次"的机制基础。

**4. 能量的构造（§4.2、§4.3）**

- 主实验只用一个**合法**能量：相机可行驶区域（DA）。它由前视相机回归出的 signed distance field 给出，当车体某个角点距道路边缘小于 0.5 m 时激活，因此对已经安全远离边界的场景几乎不改动（约在 15% 的已通过场景上触发）。
- 消融里加入两项需要特权信息的能量：**DD**（waypoint 朝向与车道方向相反则惩罚，相对 barrier 权重 0.06）与 **LK**（把轨迹吸引到车道中心线）。
- 三类能量的稀疏性不同：DA、DD 是稀疏事件型，LK 是稠密连续型，这一差异直接导致"全叠加反而更差"（见 Key Results）。

**5. 评测设置**

| 项 | 设置 |
|:--|:--|
| Benchmark | NAVSIM v2 hard-split：450 个真实场景 + 5,462 个由扰动初始位姿渲染的合成场景 |
| 指标 | EPDMS，分 S1（真实）/ S2（合成）两阶段报告，附九项子指标 |
| 被引导模型 | DriveLaW、DriveWAM、SimWAM（均为公开 checkpoint，全部冻结） |
| 采样器 | 确定性 Euler 积分器；三个 seed 取均值与标准差 |
| 主实验能量 | 单一 camera-based DA（不使用特权信息） |
| 额外开销 | 每步引导耗时 DriveLaW 0.1 ms / SimWAM 1.4 ms / DriveWAM 1.9 ms，均小于前向耗时的 10% |

## Pipeline

![GuidedWAM 的引导效果与投影对比](https://arxiv.org/html/2611.07912v1/fig_guidance_vs_projection_6333f19e.png)

*同一个 NAVSIM v2 场景上，原计划（灰）驶出可行驶区域；事后投影（蓝）在违反约束前不动、触发后突然横向折回；能量引导（橙）从第一个去噪步起就沿整条轨迹平滑修正，最终两者都通过可行驶区域门限，但引导版本的相邻帧一致性更好。*

1. **冻结模型前向**：在噪声水平 $t$，采样器把当前动作流 $\tau_t$、观测上下文 $X$（历史帧、ego 状态、导航命令）送进 WAM，得到速度场 $v_\theta$。
2. **构造干净估计**：用 $\hat\tau_0=\tau_t-t\,v_\theta$ 得到"模型认为的最终轨迹"，并 detach，切断对网络的求导路径。
3. **解码并评估能量**：把 $\hat\tau_0$ 解码成 waypoint 的 $(x,y,\psi)$，在其上解析计算各项能量（DA / DD / LK），得到梯度 $\nabla_{\hat\tau_0}\mathcal{E}_k$。
4. **归一化与加权**：每个能量的梯度按样本 L2 归一化，再按 $w_k$ 加权求和，得到总修正方向 $\Delta v$。
5. **调度与裁剪**：用梯形窗给出 $\lambda(t)$（中段最大、两端线性衰减），再用首步记录的速度范数 $\rho$ 把 $\lambda(t)\lVert\Delta v\rVert$ 裁剪到 $c\rho$ 以内。
6. **写入速度场并积分**：执行 $v'=v_\theta+\lambda(t)\Delta v$，按 Euler 积分更新 $\tau_t$，回到第 1 步，直到 $t=0$。
7. **输出**：末步的干净轨迹直接作为规划结果；整个过程不改任何权重，也不增加反向传播。

## Key Results

**1) 三个冻结 WAM 上的主结果（Table 1，NAVSIM v2 hard-split EPDMS）**

| Model | Stock | + EG | Δ |
|:--|--:|--:|--:|
| DriveLaW | 30.0 | **33.6 ± 0.4** | +3.6 |
| DriveWAM | 35.9 | **37.8 ± 0.8** | +1.9 |
| SimWAM | 37.8 | **39.6 ± 0.5** | +1.8 |

- 提升幅度与各模型原有的 DAC 短板强相关：DriveLaW 合成场景里驶出可行驶区域的失败最多，增益也最大。正文把 DriveWAM 的增益写作 +1.8，与表内 35.9→37.8 的 +1.9 属舍入口径差异。
- 引导后最强的是 SimWAM（39.6），高于表内所有 diffusion / flow 规划器；但作者明确承认，经 RL 后训练的 DriveVLA-M0（47.0）仍然更高。
- NC 与 DDC 基本不动（各模型都在 stock 的 1 分以内）：作者据此说明"常速度净空项作为梯度并不能阻止这个 split 上的碰撞"，没有 overclaim。

**2) 与事后投影的对照（Table 2，同一约束、同一特权 DA 信息）**

| Model | Stock | Guidance | Projection | EC（stock / guide / proj） |
|:--|--:|--:|--:|:--|
| DriveLaW | 30.0 | 34.8 | **37.7** | 57.9 / 58.0 / 51.2 |
| DriveWAM | 35.9 | 39.2 | **40.3** | 73.0 / 71.4 / 66.0 |
| SimWAM | 37.8 | 39.2 | **41.0** | 72.8 / 71.6 / 65.3 |

- 作为"修复 DAC 失败"的工具，投影更有效：它能清掉 5–9 个百分点的失败合成场景，引导只能清 1–3 个，EPDMS 也高 1–3 分。
- 但投影以一致性为代价：EC 掉 5–7 分，引导最多掉 2 分；在两者都能恢复 DAC 门限的场景上，引导版的两帧舒适度为 80.7，投影版 72.5（stock 82.6）。
- 机制解释：引导在每一步都动一点、修正沿时域摊开，冻结模型在每步后还会继续塑形；投影只在约 1/3 的场景上激活，且单次修正可达 2 m 以上。

**3) 能量组合（Table 3，特权信息）**

| Model | Stock | LK | DA | DD | LK+DA | DA+DD | LK+DA+DD |
|:--|--:|--:|--:|--:|--:|--:|--:|
| DriveLaW | 30.0 | 31.1 | 34.0 | 30.6 | 33.3 | **34.8** | 33.0 |
| DriveWAM | 35.9 | 37.5 | 39.2 | 36.4 | 39.5 | **40.0** | 39.6 |
| SimWAM | 37.8 | 37.9 | 39.2 | 37.8 | 38.2 | **39.6** | 38.9 |

- 单独看 DD 几乎无效（各模型与 stock 相差 1 分以内），但叠加在 DA 之上能再加 0.4–0.8。
- **三项全叠加反而最差**：DA、DD 是稀疏事件型能量，LK 是稠密连续型、在每个场景上都拉，代价是已通过场景的舒适度，因此 dense 项会稀释 sparse 项的收益。
- LK 的效果呈模型依赖：DriveWAM 上 LK+DA 到 39.5，但在 SimWAM / DriveLaW 上低于单 DA。

**4) 强度与泛化（Figure 2、Table 4）**

- 强度跨度两个数量级：DriveWAM 在 $\lambda\in[1,4]$ 有增益、$\lambda=6$ 低于 stock；SimWAM / DriveLaW 在 $[0.01,0.10]$ 有增益、$0.20$ 崩溃（DriveLaW 掉 4 分）。
- 用"相对修正占模型速度的比例"作为模型无关单位后，SimWAM 与 DriveLaW 都在约 **1.4%** 处达到峰值（$\lambda=0.05$）；DriveWAM 峰值在 **45%**（$\lambda=4$），高三十倍——说明它的策略对速度场扰动更不敏感。
- 由于 hard-split 没有验证集，论文把 450 个真实帧分成两半，一半搜 $\lambda$ 与能量超参、另一半评测，交换后平均（cross-fitting）：得到 1.7 / 2.2 / 3.4 分增益，配对 bootstrap（10,000 次）的 95% 区间分别为 $[+1.9,+6.3]$、$[+0.5,+4.2]$、$[+0.8,+3.8]$，三个模型都统一为正。

**5) 调度窗与 Jacobian 的代价（Figure 3、Table 5）**

| Model | 形式 | EPDMS↑ | 每步引导耗时↓ | 每步前向↓ | 峰值显存↓ |
|:--|:--|--:|--:|--:|--:|
| DriveLaW | Jacobian-free | **34.1** | 0.1 ms | 48 ms | 8.2 GB |
| DriveLaW | Jacobian | 29.9 | 50.4 ms | 42.4 ms | 9.8 GB |
| DriveWAM | Jacobian-free | **37.7** | 1.9 ms | 36 ms | 13.7 GB |
| DriveWAM | Jacobian | 36.3 | 72.5 ms | 35.9 ms | 14.3 GB |
| SimWAM | Jacobian-free | 40.0 | 1.4 ms | 16.3 ms | 12.7 GB |
| SimWAM | Jacobian | 40.0 | 14.3 ms | 16.2 ms | 12.7 GB |

- 精确 Jacobian 形式在两个大模型上**分数更低**（−4.2 / −1.4），因为 $J_t$ 会按去噪器的响应重标定推力，而为 Jacobian-free 选的强度无法迁移过去——删掉预条件子反而成了"更容易调"的形式。
- 代价差距体现在延迟：Jacobian-free 每步 0.1–1.9 ms，精确形式 14.3–72.5 ms（DriveWAM 上是 5B 参数 Transformer 的一次完整反向）。
- 梯形窗的作用在 40 个 DriveWAM DAC 失败场景上量化：均匀权重在 $\lambda=4$ 时已有 10% 场景抖动（判据是最终计划抖动超过 stock 的 3 倍），$\lambda=32$ 时到 40%；梯形窗在 $\lambda=4$ 时零抖动，直到 64 都低于均匀权重。DriveWAM 上用均匀窗会掉 6.5 EPDMS（37.7 → 31.2），S1 从 84 掉到 71。

## Strengths & Weaknesses

**Strengths**

- **问题选得实**：把"部署期目标会变、但重训成本不可承受"讲成一个具体的控制通道问题，而不是再刷一个 benchmark；能量引导确实是少数只需前向访问就能改行为的机制。
- **推导链条完整且自洽**：从 score-velocity 恒等式 → 能量倾斜 → 辅助变量重写为条件化 → clean estimate 近似 → 删 Jacobian，每一步都写清了近似来源与代价（$J_t$ 作为预条件子被删、canonical 调度在 $t\to1$ 发散），比"我们加了个梯度项"清楚得多。
- **稳定化不是凑出来的**：梯形窗对应"高噪声端梯度无意义、低噪声端计划已定"，归一化对应"不同能量尺度不可比"，速度上界对应"正反馈放大"——每条都有明确失效模式，Figure 3 还给了抖动率与 ADE/FDE 的量化。
- **在三个公开 checkpoint 上验证**：DriveLaW / DriveWAM / SimWAM 全部冻结、只做前向，这比"自家模型 + 自家引导"更能说明通道的通用性；三个 seed 报告标准差。
- **诚实地报告了投影更强**：Table 2 明确写出投影在 DAC 上更有效、EPDMS 更高，只在舒适度上更差。这种"我们的方法不是最优修复器"的表述很少见，也让"一致性换分数"的取舍变得可信。
- **开销测量到组件级**：0.1 / 1.4 / 1.9 ms 的每步引导耗时对 48 / 16 / 36 ms 的前向，加上 Jacobian 形式的 50.4 / 72.5 ms 对照，把"训练无关"具体化成了可核对的数字。
- **cross-fitting 处理了没有验证集的问题**：一半搜参一半评测、交换平均并给 bootstrap 区间，比直接在全 split 上调参再报分要严谨。

**Weaknesses**

- **只验证了一种能量、一个 benchmark、一个 split**：主表所有增益都来自 camera-based DA，而 DA 恰好与 EPDMS 的 DAC 门限直接对应；换一个与指标无直接耦合的目标（如乘客舒适偏好、分段时间预算）能否同样work没有证据。跨 benchmark（NAVSIM v1 / HUGSIM / 闭环）完全缺失，作者在 Limitations 里也承认。
- **增益与失败修复是两回事**：正文说 DA 引导只清掉 1–3 个百分点的失败合成场景，而 EPDMS 涨 1.8–3.6。EPDMS 的乘性门限意味着少数场景从 0 变正会带来不成比例的分数变化，因此"分数提升"并不等于"驾驶变得更安全"，论文没有给出通过率 / 失败数的绝对统计。
- **碰撞与方向合规完全没动**：NC 与 DDC 在各模型上都在 stock 的 1 分以内，说明这个通道对"最危险的一类失败"无效。作者写得很诚实，但也意味着把它称为"驾驶目标的通用注入机制"目前只覆盖了几何边距这一类目标。
- **两个数量级的强度差异靠人工标定**：DriveWAM 需要 $\lambda=4$，另两个只需 $0.05$；虽然给出了"相对修正占速度比例"这个模型无关单位（1.4% vs 45%），但比例本身仍要靠扫参确定，论文没有给出"不看结果就能估出合适 $\lambda$"的方法。cross-fitting 只能证明"选出来的强度可迁移"，不能证明"无需调参"。
- **梯形的窗口形状基本是手工设计**：中段最大、两侧线性，起止位置与斜率都没有消融；Figure 3 显示 $\lambda=128$ 时梯形窗反而更差（末段权重仍大），说明窗口与强度的耦合没有被处理，实际使用需要联合调两个超参。
- **能量叠加只是加权和**：作者自己指出这无法调和互相矛盾的能量（例如"靠边停车"与"保持车道中心"），并把 STL robustness 列为下一步。这限制了它作为"通用目标注入"的适用范围，而 Table 3 的"全叠加更差"正是这一局限的直接体现。
- **DD 能量权重 0.06 的来源没有交代**：这个相对权重明显经过调参，而 DD 单独使用时几乎无效（≤1 分），因此"DA+DD 最优"这一结论对权重的敏感性未知。
- **确定性 Euler 采样器的前提**：所有实验都在确定性积分器上做，随机采样器（SDE / 随机 Euler）下的调度与强度范围未验证；作者在 Limitations 里承认，但这也意味着方法对"换采样器"的鲁棒性未知。
- **与事后投影的比较只在 DAC 上**：Table 2 用的是特权 DA 信息、0.5 m 边距，这一设置对投影天然有利（后验修正可以精确落在约束上），而引导用的是软能量。更公平的比较应是"同样软约束的事后优化"，论文没有做。

## Notes

+ **论文定位**：不是新的 WAM 架构，而是给冻结 WAM 加一条**推理期控制通道**。核心 insight 是：只要目标能写成干净轨迹上的可微函数，它的梯度就能在 flow matching 采样中当作 score 修正使用；而"删掉网络 Jacobian、只保留能量梯度"这个近似，把成本从一次反向传播降到可忽略，代价是必须用调度窗 + 归一化 + 速度上界把它按住在流形附近。
+ **可迁移点**：
  - **"detach clean estimate + 解析能量梯度"是通用的免训练约束注入配方**：任何 flow / diffusion 策略（不限于驾驶）都可以用它在推理期加新目标，前提是目标能写成几何量的可微函数。
  - **速度有界修正的参考量必须在首步冻结**：按当前速度设上界会形成正反馈，这是把"引导"变成"发散"的常见坑，作者的 $\rho$ 记录一次的做法可以直接复用。
  - **梯形调度窗**：高噪声端梯度是多个未来平均、低噪声端计划已定，因此修正应集中在中间噪声段；这一经验对任何在采样器上做外部修正的方法都成立。
  - **用"相对修正占模型速度的比例"做跨模型比较**：直接把 $\lambda$ 在不同模型间比较没有意义（这里差两个数量级），换成相对量后能看出真正的敏感度差异（1.4% vs 45%）。
  - **把 in-sampling 修正与 post-hoc 投影放在同一约束下对比**：这是评估"约束注入"类方法最该做的对照，因为投影通常分数更高但一致性更差，只看 EPDMS 会得出错误结论。
+ **待跟进**：
  - 用一个与 EPDMS 指标不直接耦合的目标（例如"乘客纵向 jerk 上限"或"与 VRU 的净空下限"）重做实验，检验这套通道是否只对"指标对应的约束"有效。
  - 在随机采样器（SDE / 随机 Euler）上重扫 $\lambda$ 与梯形窗，确认当前超参是否依赖确定性积分。
  - 把 DA 能量换成"多类目标的 STL robustness"实现组合，检验能否解决 Table 3 中"稠密项稀释稀疏项"的问题。
  - 补一个与"软约束事后优化"的对照（而非硬投影），判断引导的优势有多少来自 in-sampling、多少来自软约束本身。
