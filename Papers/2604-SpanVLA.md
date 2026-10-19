---
title: "SpanVLA: Learning from Negative-Recovery Samples with Fast Action Bridging for Vision-Language-Action Model"
authors: ["Zewei Zhou", "Ruining Yang", "Tony (Xuewei) Qi", "Yiluan Guo", "Sherry X. Chen", "Tao Feng", "Kateryna Pistunova", "Yishan Shen", "Lili Su", "Jiaqi Ma"]
institute: ["University of California, Los Angeles", "Motional", "Northeastern University"]
date_publish: 2026-04
venue: arXiv
tags: [VLA, flow-matching, agentic-RL]
url: https://spanvla.github.io/
arxiv_id: "2604.19710"
doi: "10.48550/arXiv.2604.19710"
cite_key: zhou2026spanvla
code: https://github.com/motional/SpanVLA
rating: 4
content_scope: full-text
date_added: 2026-10-14
---
## Summary

针对驾驶 VLA 的两个痛点——autoregressive action decoding 延迟高、且仅用正样本监督导致长尾鲁棒性不足——`SpanVLA` 提出 efficient action bridge（稀疏 KV-cache 条件 + 历史轨迹初始化的 flow-matching action expert）与 `NR-RFT`（基于 `GRPO` 的负样本惩罚 / 恢复奖励后训练），并发布首个同时含 negative 与 expert recovery 样本的真实世界驾驶数据集与 benchmark `nuReasoning-NR`。方法在 `NAVSIM` v1/v2、`nuReasoning` 与 `nuReasoning-NR` 上报告 SOTA，且 50-waypoint 下 action generation 相比 autoregressive 解码加速 21.5×。

## Problem & Motivation

- **现有驾驶 VLA 大多只从正样本（专家轨迹）做 imitation learning**：模型学到"该做什么"，却缺少"哪些行为必须避免、如何从失误中恢复"的监督，因此在长尾/交互密集场景鲁棒性有限。
- **真实世界的 negative + recovery 数据稀缺**：已有 takeover-based 方法主要依赖仿真，真实负样本与对应专家纠正样本仍是空白；而这类数据能提供非对称、定向的改进信号。
- **Action generation 延迟高**：把轨迹表示为离散 token 并 autoregressive decode，延迟随轨迹长度线性增长，高频规划代价大。已有的 continuous action expert 要么只用最后一层 VLM feature（丢失中间视觉/语义信息，如 `ReCogDrive`），要么用全部层的 dense KV cache + 大 action expert（显存与延迟开销大，如 `Alpamayo`）。
- **负/恢复样本的 RL 训练本身困难**：它们编码正/负/恢复的非对称监督；且困难场景的 rollout 组常被相似次优甚至不可行轨迹主导，advantage 信息量低、训练不稳定。

## Method

**1) VLM Backbone**（backbone 为 `Qwen2.5-VL-3B`）
- 输入：多视角多帧图像（默认 3 相机，每相机 4 历史帧 @2Hz）+ 语言 prompt（任务/驾驶指令）+ ego states（加速度、历史速度、历史轨迹）。
- Autoregressive decoding 生成结构化 reasoning；采用 `adaptive reasoning`（借鉴 `AutoVLA`）：slow thinking 先出 CoT 再出动作，fast thinking 直接出动作；推理时 VLM 发出 action-start token 后把连续轨迹生成交给 action bridge。
- 训练时在 reasoning 之后附加一个 **离散 action token 生成任务**（复用 `AutoVLA` 的 action codebook），把 reasoning 与 planning 统一进 `SFT`。

**2) Efficient Action Bridge**
- 轻量 Transformer，条件于**选定稀疏层**的 KV caches（多粒度上下文），保留 VLM 的 attention head 数但用更小的 hidden dim，降低 dense layer-wise conditioning 的开销。
- `Historical-trajectory initialization`：不像 `Alpamayo` 从纯高斯噪声起步，而是从历史轨迹 embedding 初始化，直接学习"过去动作→未来动作"的传输映射（optimal transport displacement map），训练时向历史 embedding 注入高斯噪声增强鲁棒性。
- 训练时在 VLM backbone 与 bridge 之间 stop gradient，先训 VLM 再单独 micro-tune bridge。

**3) SFT**：LM loss + action loss 联合监督推理与规划。

**4) NR-RFT（Negative–Recovery Reinforcement Fine-Tuning）**
- 基于 `GRPO`，对每个 query 采样 $G = 64$ 条响应解码成轨迹；奖励设计为非对称：
  $r = r_{\text{Driving}} - \lambda_N \cdot r_{\text{Negative}} + \lambda_R \cdot r_{\text{Recover}} - \lambda_C \cdot r_{\text{CoT}}$。
- negative 样本用 reference-matching 项惩罚"接近不良行为"（超阈值 $\delta$ 即归零），recovery 样本奖励"接近专家纠正"；$r_{\text{CoT}}$ 惩罚不必要地长的推理，reasoning 与预测 maneuver 不一致时给固定惩罚。
- `Advantage correction`：去掉 `GRPO` 的标准差归一化以保留原始 reward 差异；对 reward 全零的 rollout 组引入虚拟奖励 1，使每个样本得到 $A_i = -1/(G+1)$ 的惩罚信号，避免零 advantage、保留探索。

**数据集 `nuReasoning-NR`**：36K = 30K positive reasoning + 3K negative + 3K recovery；采集自 Las Vegas/Boston/Pittsburgh/Singapore 真实驾驶，negative/recovery 来自早期探索性测试的事件级配对（负片段 = 次优 ego 行为，recovery = 其后的专家纠正）。positive subset 的 reasoning 标注由基于 `Gemini-3-Pro` 的自动流水线生成，再经人工质量审核。

## Pipeline

![SpanVLA 整体架构](https://arxiv.org/html/2604.19710v2/Framework.png)

*`SpanVLA` 整体架构。`SFT` 对齐 reasoning 与 planning；`NR-RFT` 用真实 negative / recovery 样本惩罚不良行为、奖励专家纠正，并以修正后的 group-relative advantage 更新策略；高效推理时由轻量 action bridge 基于稀疏层 KV cache、以历史轨迹初始化，通过 flow matching 生成连续轨迹。*

## Key Results

- **`nuReasoning-NR` benchmark**（Table 1，`PDMS`）：negative 65.1→74.1、recovery 56.1→65.1、平均 60.6→69.6（+14.9%）加入 NR 监督后全面提升；同一 NR 训练把 `AutoVLA` 平均 `PDMS` 从 59.1 提到 64.2，说明收益不局限于 `SpanVLA` 架构。
- **`NAVSIM`**（Table 3）：`SpanVLA`(one-shot) v1 `PDMS` 82.1 / v2 `EPDMS` 79.4；post-RFT positive-only 90.8 / 86.9；best-of-N(N=6) 93.3 / 90.5；v2 `navhard` `EPDMS`：positive-only 40.1 → with NR 40.8 → best-of-N 47.9。论文声称在三个 benchmark 上 overall 最优。
- **`nuReasoning`**（Table 2）：`SpanVLA`(post-RFT) `NPS` 73.3、`ADE` 1.4，略优于 `nuVLA`（73.1 / 1.6）。
- **效率**（Table 4）：50-waypoint 下 bridge 相对 autoregressive decoding 的 action generation 加速 **21.5×**，总延迟约 **−74%**（`AutoVLA` 2.57s → `SpanVLA` FM 0.67s）；waypoint 从 10 增到 50，bridge 生成时间基本不变。
- **Action bridge ablation**（Table 5）：interval-2 + historical init = **90.8 `PDMS` / 0.08s**；interval-2 无 HI 86.4；interval-4 82.2；仅 last-layer 79.3；full caching 88.1 / 0.18s。

## Strengths & Weaknesses

**Strengths**
- **问题切得准**：把"仅正样本模仿学习"识为驾驶 VLA 鲁棒性瓶颈，并用真实 negative + recovery 数据 + 非对称奖励去补，而不是继续在正样本上堆规模。
- **工程与设计取舍清晰**：sparse KV-cache（多粒度但省算力）+ 历史轨迹初始化（而非纯噪声）两点针对性回应了 `ReCogDrive`（仅末层）与 `Alpamayo`（dense 全层）的缺陷；Table 5 的 ablation 支撑"过稀疏会掉点、和历史初始化几乎零延迟代价"。
- **NR 监督可迁移**：`AutoVLA` 加 NR 同样涨点，说明价值来自数据/训练范式而非仅架构。
- **数据集/benchmark 贡献明确**（首个含 negative+recovery 的真实驾驶数据），有望成为该子问题的评测基线。
- **`advantage correction` 针对性强**：直接处理困难场景 rollout 组 reward 同质导致的零 advantage 问题，属可复用的 `GRPO` 工程改进。

**Weaknesses / 边界**
- **SOTA 依赖 best-of-N(N=6) 选择**：主结果 `SpanVLA`* 用 6 候选选最高分，与单次推理的 positive-only 结果（90.8）不可直接比较；论文亦未与同样开 best-of-N 的 baseline 对齐。
- **数据未发布**：数据集标注 "Coming Soon"，benchmark/数据的可复现性与"first real-world" novelty 目前无法由外部核验。
- **对比基准偏窄**：efficiency（Table 4）对比对象基本是 `AutoVLA` 与本方法的 L1-head 变体；未系统对比其他 continuous action expert 的延迟-精度帕累托。
- **负/恢复样本规模小**（各 3K，benchmark 各 300）：在真实长尾分布上的泛化能力仍需更大规模验证；positive 标注依赖 `Gemini-3-Pro` 自动生成，存在标注偏差风险（虽有 250 样本人工抽查）。
- **领域偏驾驶**：作为 VLA + agentic-RL 的方法范式对通用具身/机器人有借鉴价值，但结论外推需谨慎。

## Notes

- **方法与既有工作的关系**：直接对标 `ReCogDrive`（末层 feature + diffusion）与 `Alpamayo`（dense 全层 KV + flow matching）；借鉴 `AutoVLA` 的 action codebook 与 adaptive reasoning；RFT 上以 `GRPO` + 非对称奖励区别于 takeover + `DPO` 路线。
- **可迁移点**：
  - "稀疏层 KV cache 条件"是省算力的多粒度特征聚合思路，可能可迁移到其他 VLM→action 的 bridge；
  - "历史初始化取代纯噪声"对任何 flow-matching / diffusion policy 都值得一试；
  - `advantage correction`（去 std 归一化 + 零 reward 组的虚拟参考）是通用 `GRPO` 技巧。
- **待跟进**：
  - 数据集/benchmark code 发布后应复核；
  - best-of-N 口径下的公平比较值得关注；
  - 该文与本 vault GUI/Embodied 方向的连接点主要在 VLA + agentic-RL 范式。
- **repo_candidate**：`code: https://github.com/motional/SpanVLA`（数据集 "Coming Soon"）。
