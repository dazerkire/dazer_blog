---
title: "LLM 自回归推理的性能到底耗在哪里？"
description: "从 Prefill、Decode、KV Cache、TTFT 与 TPOT 出发，建立一套可复用的 LLM 推理性能分解框架。"
date: 2026-08-22 00:00:00 +0800
categories: [模型与系统, LLM 推理]
tags: [LLM, Prefill, Decode, KV Cache]
math: true
mermaid: true
---

分析 LLM 推理性能时，最常见的错误是把训练时的整段前向、离线吞吐测试和在线逐 Token 生成混在一起。它们调用的可能是同一个 Transformer，但执行形态、硬件利用率和最终关心的指标并不相同。

> **先给结论：**一次生成请求通常包含一个可高度并行的 **Prefill**，以及多个存在数据依赖的 **Decode** 步骤。短输入、长输出时，后者往往决定端到端延迟；长输入、短输出时，Prefill 与排队可能更重要。

## 先把“一次推理”拆开

假设请求包含 (N) 个输入 Token，模型最终生成 (M) 个输出 Token。从服务系统的视角看，总时间至少可以写成：

```text
T_total
  = T_queue + T_schedule + T_prefill
  + Σ(t=1…M) [T_decode,t + T_sample,t]
  + T_comm
```

这个表达式不是为了追求统一计时口径，而是提醒我们：**“模型前向耗时”只是完整请求的一部分。**在线服务还会受到排队、动态批处理、跨卡通信、Tokenizer、采样和结果回传影响。

| 阶段 | 发生了什么 | 常见影响因素 |
| --- | --- | --- |
| Queue / Schedule | 请求等待并被组成 Batch | 并发量、调度策略、优先级 |
| Prefill | 一次处理整段 Prompt | 输入长度、Attention 实现、Batch |
| Decode | 每一步生成一个或多个 Token | 输出长度、KV Cache、显存带宽 |
| Sample / Comm | 采样、同步与结果回传 | 词表大小、并行方式、网络 |

## Prefill：一次处理整段输入

Prefill 阶段把 Prompt 的 (N) 个 Token 一次送入模型。Causal Mask 限制第 (i) 个位置只能读取位置 (1…i)，但因为整段输入已经全部已知，各个位置的计算仍然可以在 GPU 上并行展开。

对 Decoder-only 模型而言，模型会计算所有输入位置的隐状态；真正用于生成下一个 Token 的，是最后一个输入位置对应的概率分布。与此同时，每一层此前位置的 Key 和 Value 会被保存，供后续生成使用。

Prefill 具有两个典型特点：

- **多 Token 并行。**大量计算呈现矩阵—矩阵乘法，更容易提高 GPU 计算单元利用率。
- **成本受输入长度影响。**线性层随 Token 数增长；标准全注意力部分还包含随序列长度平方增长的交互。

不过，“Prefill 一定是计算受限”不是硬规则。模型结构、序列长度、量化方式、算子实现和硬件都会改变瓶颈。更准确的说法是：**Prefill 通常比 Batch=1 的 Decode 具有更高的算术强度。**

## Decode：为什么只能一步接一步

第 (t) 个输出 Token 的概率依赖已经生成的前 (t-1) 个 Token。只有采样或选择出第 (t) 个 Token，模型才知道下一步的输入是什么。因此 (M) 个输出通常对应 (M) 次有先后依赖的模型调用：

```text
Prompt  →  y₁  →  y₂  →  y₃  →  …  →  yₘ
```

这解释了为什么训练时能够对整段目标序列并行计算，而在线生成不能直接“一次前向得到完整答案”：**训练时真实的前缀已经给定，生成时未来 Token 尚不存在。**

在小 Batch Decode 中，每一步只新增很少 Token，却仍要访问规模庞大的模型权重。计算常呈现类似矩阵—向量的形态，数据搬运相对于有效计算更多，因此显存带宽经常比峰值算力更早成为瓶颈。

投机解码的核心价值，也正是尝试让一次目标模型调用确认多个 Token，减少串行步骤；它不是在消除 Causal Mask，也不是把普通采样替换成另一种搜索策略。

## KV Cache 保存了什么，又没保存什么

如果每生成一个 Token 都重新计算整个历史序列，重复工作会越来越多。KV Cache 保存每一层历史 Token 的 Key 和 Value。下一步只需为新 Token 计算新的 Query、Key、Value，再用新 Query 与缓存中的所有 Key 做注意力。

单个请求的 KV Cache 可以用下面的式子近似估算：

```text
M_KV ≈ 2 × B × L × S × n_kv × d_head × bytes
```

- (B)：并发序列或 Batch 数；(L)：Transformer 层数；(S)：当前序列长度。
- (n_{kv})：KV Head 数；(d_{head})：每个 Head 的维度。
- 前面的 2 分别对应 Key 和 Value；GQA/MQA 通过减少 KV Head 数降低缓存规模。

KV Cache 消除了历史 K/V 的重复计算，但没有消除新 Token 经过所有 Transformer 层的计算，也没有消除读取模型权重和历史 KV 的成本。上下文越长，每一步需要访问的 KV 数据通常越多。

## TTFT、TPOT 与吞吐量为什么不能混用

**TTFT（Time to First Token）**是从请求进入系统到用户看到第一个输出 Token 的时间，通常包含排队、调度和 Prefill。

**TPOT（Time per Output Token）**是首 Token 之后，输出 Token 之间的平均间隔，更直接反映 Decode 速度。

**E2E Latency**是从请求开始到最后一个 Token 完成的时间。在一个简化模型中：

```text
T_E2E ≈ TTFT + (M − 1) × TPOT
```

**Throughput**描述整个系统单位时间处理的 Token 或请求数量。增大 Batch 可能显著提高吞吐量，但并不保证单个请求的延迟下降；排队和调度甚至可能让它变差。

面向交互聊天时，TTFT 影响“多久开始回答”，TPOT 影响“回答是否流畅”；面向必须等待完整结果的 Agent 或批量任务，端到端延迟通常更重要。

## 一个数值例子：应该优化哪一段？

假设一个请求的 TTFT 为 80 ms，TPOT 为 12 ms，最终输出 100 个 Token。简化后的端到端延迟约为：

```text
80 + 99 × 12 = 1,268 ms
```

如果把 TTFT 降低一半，只节省 40 ms；如果把 TPOT 降低 25%，则约节省 297 ms。因此在这个“输出较长”的例子中，优化 Decode 更有价值。

反过来，当输入很长、输出只有几个 Token 时，结论可能完全不同。这也是为什么脱离输入长度、输出长度和 Batch 谈“加速比”通常没有意义。

## 常见优化分别改动哪一部分

| 技术 | 主要目标 | 不应被简化成 |
| --- | --- | --- |
| FlashAttention | 减少注意力在 HBM 与片上存储之间的读写 | 消除自回归串行性 |
| 量化 | 降低权重与缓存的存储、搬运及部分计算成本 | 在所有硬件上等比例加速 |
| Continuous Batching | 让不同请求在迭代级别动态组成 Batch | 保证每个请求延迟下降 |
| PagedAttention | 分页管理 KV Cache，减少碎片并容纳更多并发 | 直接减少模型 FLOPs |
| Speculative Decoding | 用草稿与并行验证减少目标模型串行调用次数 | 一种 Top-k 或 Beam Search |

这些方法可能共同出现，因为它们针对的是不同层次：算子、数据格式、缓存管理、请求调度和解码算法。评估一项优化时，至少应说明工作负载、硬件、Batch、输入输出长度，以及比较的是 TTFT、TPOT 还是系统吞吐。

## 结论：先建立性能账本

LLM 推理不是“一次大矩阵计算”，也不是“每个 Token 都完整重算历史”。Prefill 利用已知输入并行计算；Decode 因为输出之间存在依赖而逐步进行；KV Cache 在两者之间保存历史注意力状态，换取计算复用，但带来持续增长的显存占用和读写成本。

在讨论任何推理优化前，最先应该记录的不是方法名称，而是四类信息：**工作负载、执行阶段、硬件瓶颈和目标指标。**没有这张性能账本，“加速 2 倍”通常无法被正确解释，更无法迁移到另一种场景。

## 原始资料

1. [Vaswani et al. — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [Dao et al. — FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
3. [Kwon et al. — Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
4. [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)

本文是机制性归纳，不绑定某个推理框架版本。最后更新：2026-08-22。
