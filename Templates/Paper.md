---
title: ""
authors: []
institute: []
date_publish:
venue:
tags: []
url:
cite_key: # 留空、勿加引号（写成 "" 会让 assign_cite_keys.py 静默跳过）；由脚本分配，一旦写入永久冻结
arxiv_id: "%% arXiv id，如 2609.06055；非 arXiv 留空。由 assign_cite_keys.py 自动从 url 抽取 %%"
doi: "%% DOI，如 10.48550/arXiv.2609.06055；期刊/会议有则填，否则留空 %%"
code:
rating: "%% 1=不相关, 2=了解即可, 3=有参考价值, 4=重要, 5=必读 %%"
content_scope: "%% full-text / abstract-only %%"
verification_status: "%% source-checked / partial / unverified；仅表示原文一致性核查，不表示独立复现 %%"
date_added: "{{date}}"
---

%% 写作约定（全局，成稿时删除本块）
- 正文中文，模型名 / 方法名 / benchmark 名 / 指标名保持英文不翻译。
- bullet 用 `-`（子层缩进 2 空格）；公式一律独立 `$$ ... $$` 块，不用 `$` 包裹、不用 `\begin{aligned}`；多行公式在同一个 `$$` 内用 `\\` 换行。
- 解释公式的句子写进公式所在的 bullet，不要另起段落。
- 章节只保留下方 7 个二级标题，不要加 `## Mind Map`、`## Evidence Ledger`、`## Critical reading`；批评性阅读并入 `## Strengths & Weaknesses` 的 Weaknesses。
- 单次运行、无方差、开环评测、消融与主实验不同配置等边界必须写清，不 overclaim。 %%

## Summary

%% 一段、不超过 3 句：解决了什么问题 → 怎么解决的 → headline 数字（benchmark + 指标 + 数值）。关键结论与数字加粗。不要复述 abstract 的原句。 %%

## Problem & Motivation

%% 4-6 条 `-` 列表：先讲领域通用痛点与现有方法的局限（不要写成 related work），再讲作者的核心主张。每条落在一句话能说清的因果上。 %%

## Method

%% 用粗体编号小节分块，常见顺序：
**0. 总体框架** — 训练阶段 / 数据流 / teacher-student 关系，一句话给出输入与输出；
**1. <模块名>（§4.1）** — 关键设计与"为什么这样做"，写清输入输出与关键超参；
**5. 实现细节** — `| 项目 | 设置 |` 两列表格汇总参数量、token 宽度、数据规模、优化器、学习率、算力与训练时长。
块内用 `-` 列表；公式独立成块，符号在首次出现处定义。 %%

## Pipeline

%% 一张架构图 + 编号步骤，顺序一致：
![ShortTitle 整体架构](../images/ShortTitle-architecture.png)
图下可加一段斜体 caption（画了什么、左右/上下各是哪一段）。然后用 1. 2. 3. 按执行顺序列数据流，训练与推理分别写清哪一步在哪个阶段发生、哪些组件推理时被丢弃。 %%

## Key Results

%% 每组结果一个粗体小标题，格式：**1) <benchmark>（Table N，指标）**，紧跟 markdown 表格（数值列 `--:` 右对齐，最优值加粗），表格下用 `-` 列表给 2-4 条只属于该表的观察：增益来源哪一项、哪个指标反降、与 baseline 的口径是否可比、headline 数字取自哪一列。
正文不复述表格数字；推论、反常与边界写在这里，批评性判断留给 Strengths & Weaknesses。 %%

## Strengths & Weaknesses

**Strengths**

%% 4-8 条：方法层面真正的亮点，每条给出机制解释或消融证据（Table N / 具体数值），不写"效果好""设计巧妙"这类空评。 %%

**Weaknesses**

%% 4-10 条，笔记最有信息量的部分。至少覆盖：核心 claim 与证据链的缺口（哪一步没被实验隔离）、消融与主实验配置不一致、评测口径不可比（开环 / 伪闭环 / 单次无方差 / baseline 未统一 harness）、关键超参或数据缺失、延迟与显存不透明、训练成本未对齐、论文内部数字或表号矛盾、overclaim。措辞要落到具体行号、表格或数值上。 %%

## Notes

%% 三块，用 `+` 列表：
+ **论文定位**：这篇不是什么（防止误读）+ 真正的贡献 + 核心 insight。
+ **可迁移点**：把方法拆成可复用部件，写清适用条件与失效边界。
+ **待跟进**：每条落到"具体做什么实验 / 查哪个数字 / 补哪组对照"，不写"值得进一步研究"。
只写论文自身定位与可迁移性，不写与 notebook 内其他论文的对比。 %%
