---
title: "从指标到瓶颈：Prefill、Decode 与 LLM 推理的计算本质"
description: "从 TTFT、TPOT 出发，拆解 LLM 推理中的 attention、MLP、KV Cache、GEMM/GEMV 与 Roofline 下界。"
date: 2026-08-23 00:00:00 +0800
math: true
categories: [模型与系统, LLM 推理]
tags: [LLM, 在线推理, Prefill, Decode, KV Cache, Roofline]
---

上一篇从一次请求的生命周期出发，区分了 Prefill、Decode、TTFT 和 TPOT。这一篇把视角收窄到模型计算本身：同一个 Transformer，为什么在处理 Prompt 与逐 Token 生成时，会呈现出截然不同的性能特征？

本文不讨论具体的优化方案。量化、FlashAttention、连续批处理、PagedAttention 和投机解码会在后续文章中分别展开；这里先分析 TTFT 和 TPOT 背后分别发生了什么计算，以及它们为何会落到不同的硬件瓶颈上。

## 同一段回答里的两种等待

一次流式生成可以粗略分成两段：

```text
请求 ── TTFT ── 首 Token ── TPOT / ITL ── 后续 Token ── 完成
                 ↑
              Prefill 结束，开始 Decode
```

**Prefill** 接收完整、已知的 Prompt，一次处理其中所有 Token，并为后续生成建立 KV Cache。长系统提示词、长对话历史和 RAG 文档主要会抬高这一阶段的成本，因此也会影响 TTFT。

**Decode** 则在每一步只生成一个新 Token。新 Token 必须在上一个 Token 已经确定后才能开始计算，所以这条路径天然带有串行依赖；用户看到的 TPOT 或 ITL，主要反映它的速度。

这不是“一个阶段算得多、另一个阶段算得少”那么简单。两者使用的是同一组模型权重，却有不同的矩阵形状、数据复用方式和硬件瓶颈：Prefill 更像大矩阵乘，Decode 在小 batch 下更接近矩阵—向量乘。

## Prefill：长序列上的矩阵乘

让我们先回顾Transformer实际在算什么。
先只讨论 batch 为 1 的情形。设一个 Transformer 层的输入为 $X\in\mathbb{R}^{L\times d}$，其中 $L$ 是 Prompt 长度，$d$ 是隐藏维度。Prefill 一次处理全部 $L$ 行输入，因此线性层是矩阵乘矩阵，通常称为 **GEMM**（General Matrix Multiply）。

输入 $X$ 会被投影成 Query、Key 和 Value：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

多头 attention 将它们切分为多个 head。对第 $i$ 个 Query head，模型先用它和对应 KV head 的所有位置计算分数，再用归一化后的分数加权汇总 Value：

$$
A_i=\operatorname{softmax}\left(\frac{Q_iK_{g(i)}^\top}{\sqrt{d_h}}+M_{\mathrm{causal}}\right),
\qquad O_i=A_iV_{g(i)}
$$

$g(i)$ 表示第 $i$ 个 Query head 对应哪一个 KV head：MHA 中每个 Query head 各有一组 K、V；GQA 中多个 Query head 共享一组 K、V。最后，所有 $O_i$ 拼接并经过输出投影 $W_O$。这就是后面会分别计入的 $QK^\top$ 与 $AV$ 两次矩阵乘。[1]

attention 之后的 SwiGLU MLP 则是三次线性投影与一次逐元素门控：

$$
\operatorname{MLP}(X)=
\bigl(\operatorname{SiLU}(XW_{\mathrm{gate}})\odot XW_{\mathrm{up}}\bigr)W_{\mathrm{down}}
$$

$W_{\mathrm{up}}$、$W_{\mathrm{gate}}$ 将维度从 $d$ 扩展到中间维度 $m$，$W_{\mathrm{down}}$ 再将其投影回 $d$。有了这两个结构，后面的 FLOPs 统计就只是在数这些矩阵乘的形状。

若矩阵形状为 $(a\times b)(b\times c)$，一次前向矩阵乘约需：

$$
2abc\ \text{FLOPs}
$$

这里将一次乘加（FMA）按 2 次浮点运算计数。以此为口径，一层中 Q、K、V 和输出投影的计算量约为：

$$
4\times (2Ld^2)=8Ld^2
$$

SwiGLU 的 gate、up、down 三个矩阵乘则约为：

$$
3\times (2Ldm)=6Ldm
$$

attention 也有两次主要的矩阵乘：$QK^\top$ 用于计算分数，$AV$ 用于按权重汇总 Value。因此其主要 FLOPs 约为：

$$
2L^2d + 2L^2d = 4L^2d
$$

softmax、RoPE、RMSNorm、残差连接和 SiLU 也需要计算，但相对于大规模 GEMM 常是次要项；它们在实际 kernel 中依然会影响延迟，通常会尽量与相邻操作融合。

这里的 $8Ld^2$、$6Ldm$、$4L^2d$ 是**计算复杂度**，即模型需要完成的浮点工作量，不能直接等同于运行时间或显存占用。运行时间还取决于 GPU 能以多高的效率完成这些 FLOPs，以及需要搬运多少数据。

长 Prompt 下，$L^2$ 项会增长得很快；但对常见的中等上下文长度，MLP 和投影层中与 $d^2$、$dm$ 相关的 GEMM 往往仍是重要部分。FlashAttention 的价值正是改变 attention 的中间数据访问方式、降低 HBM I/O；它并不让精确 attention 的这两次矩阵乘从数学上消失。[2]

## Decode：单 Token、KV Cache 与带宽

当生成下一个 Token 时，新增输入只有一行：

$$
x_{t+1}\in\mathbb{R}^{1\times d}
$$

同一线性层从 $XW$ 变为 $x_{t+1}W$，更接近 **GEMV**（General Matrix–Vector Multiply）。每读取一次权重，只有一个新 Token 可以使用它；权重复用很低。

新 Token 仍要计算自己的 $Q,K,V$。但历史 Token 的 $K,V$ 已在每一层中缓存，无需重新从头计算：

```text
历史 Token：K₁,V₁ ... Kₜ,Vₜ  → KV Cache
新 Token：计算 Qₜ₊₁,Kₜ₊₁,Vₜ₊₁
          Qₜ₊₁ 与缓存的 K 匹配，再按权重读取缓存的 V
```

Query 只用于当前这一步 attention，未来 Token 不会再使用它，所以不需要缓存；历史 K、V 则会被每一个后续 Token 反复访问。

若模型有 $n_{\mathrm{kv}}$ 个 KV head、每个 head 维度为 $d_h$、数据类型占 $b$ 字节，那么一个请求的 KV Cache 大小约为：

$$
S_{\mathrm{KV}}
=2\times N_{\mathrm{layer}}\times L\times n_{\mathrm{kv}}
\times d_h\times b
$$

前面的 2 对应 K 与 V。MHA 中 $n_{\mathrm{kv}}=n_q$：每个 Query head 有自己的 K、V head。GQA 则让多组 Query head 共享一组 K、V，例如 32 个 Q head 对应 8 个 KV head。不同 Q head 仍有各自的 $W_Q$、各自的注意力分数和输出，只是共用较少组可检索的 K、V 表示。

因此，若 32 个 Q head 从 MHA 改为 8 个 KV head 的 GQA，KV Cache 会变为原来的 $8/32=1/4$。极端的 MQA 只保留一个 KV head，缓存最小，但所有 Query 都要从同一组 K、V 表示中检索，模型表达能力的约束也更强。

Decode 中的 attention 计算量随上下文长度 $L$ 线性增长，而不是 Prefill 那样以 $L^2$ 的方式处理整块矩阵；但它还要反复读取模型权重和越来越长的 KV Cache。小 batch 下，GPU 往往在等待 HBM 显存提供数据，Decode 因而通常是 memory-bound。

## 用 Llama 3.1 8B 估算工作量

下面统一使用 Llama 3.1 8B 的公开配置建立数量级直觉：

$$
N_{\mathrm{layer}}=32,\quad d=4096,\quad m=14336,
\quad n_q=32,\quad n_{\mathrm{kv}}=8,\quad d_h=128
$$

该模型使用 GQA：每 4 个 Query head 共享一个 KV head。这里的计算只用于说明数量级；不同模型的层数、隐藏维度、FFN 维度与 KV head 数都可能不同。[3]

假设 Prompt 长度 $L=2048$，采用 BF16（每元素 $b=2$ 字节）。先只统计 Transformer block 内的主要算子，暂不计 embedding、LM Head 和小算子：

$$
\begin{aligned}
\text{QKV + output} &\approx 5Ld^2 &&\approx 171.8\ \text{GFLOPs}\\
\text{SwiGLU MLP} &\approx 6Ldm &&\approx 721.6\ \text{GFLOPs}\\
\text{attention} &\approx 4L^2d &&\approx 68.7\ \text{GFLOPs}
\end{aligned}
$$

这里的 $5Ld^2$ 来自 GQA：Q 与 output projection 各为 $2Ld^2$；K、V 的总投影维度只有 $d/4$，合计再增加 $Ld^2$。合计约 $962.1$ GFLOPs/层，32 层约为 $30.8$ TFLOPs。这说明在这个长度下，MLP 仍是最大的单项；attention 已不可忽略，但尚未超过 MLP。

LM Head 的口径需要单独说明。设词表大小为 $V$：在线生成的 Prefill 只需要最后一个位置的 logits，因此 LM Head 的计算量为 $O(dV)$；训练或通用前向若要返回全部 $L$ 个位置的 logits，则会变为 $O(LdV)$。本文讨论前一种在线推理路径，输入 embedding 也只是在当前 Token 上做 lookup。对这个 Llama 3.1 8B 例子，LM Head 约为 $2\times4096\times128256\approx1.05$ GFLOPs，相对 $30.8$ TFLOPs 的 Transformer block 可以忽略。

同一配置下，GQA 的 KV Cache 约为：

$$
2\times 32\times 2048\times 8\times 128\times 2
=256\ \text{MiB}
$$

若使用同样 Q head 数的 MHA，KV Cache 会是这里的 4 倍，即约 $1$ GiB。KV Cache 是随上下文长度和并发请求数线性增长的**存储量**，与前面用于描述计算量的 FLOPs 是两件事。

进入 Decode 后，每个新 Token 的 QKV、输出投影和 MLP 计算量不再乘以 $L$；但是每层仍需对长度为 $L$ 的历史 KV 做一次读写相关的 attention。按上述配置，Transformer block 合计约为 $15$ GFLOPs/token，其中绝大多数仍来自读取权重后进行的线性层计算。此时不能再忽略 LM Head：它每一步都要为整个词表计算 logits，再增加约 $1.05$ GFLOPs。因此完整的主要计算量约为 $16.1$ GFLOPs/token。这个数看似很小，却不足以说明它会很快：关键还在于必须搬运多少字节。

## Roofline：FLOPs 与字节量共同决定下界

Roofline 模型给出一个很实用的判断框架。[4] 设一个计算需要 $F$ 次浮点运算、读写 $B$ 字节数据；GPU 的计算峰值为 $P_{\mathrm{peak}}$，显存带宽峰值为 $BW_{\mathrm{peak}}$。无论实现多好，理想时间都不能低于：

$$
T \geq
\max\left(
\frac{F}{P_{\mathrm{peak}}},
\frac{B}{BW_{\mathrm{peak}}}
\right)
$$

其中：

$$
I=\frac{F}{B}
$$

称为算术强度：每搬运 1 字节数据，完成多少 FLOPs。算术强度低时，性能受带宽限制；算术强度提高后，才可能碰到计算峰值这条“屋顶”。这就是 Roofline 这个名字的由来：带宽限制区是一条斜线，计算峰值区是一条水平线。

<figure class="text-center mt-3 mb-4">
  <img
    src="/assets/images/posts/llm-inference/roofline-model.svg"
    alt="Roofline 模型：性能在低算术强度时受显存带宽限制并沿斜线增长；达到计算峰值后形成水平平台。"
    style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 1：Roofline 性能模型。斜线的斜率由显存带宽决定，水平平台由计算峰值决定。</figcaption>
</figure>

为与后文的公开基准对齐，以下统一采用 H200 SXM。它与 H100 同属 Hopper 架构；这里按 dense BF16 的 $989$ TFLOPS 与 $4.8$ TB/s HBM3e 带宽计算，而不采用厂商表中“开启结构化稀疏”时的更高峰值。[5] 上述 2048-token Prefill 的纯计算下界约为：

$$
\frac{30.8\ \text{TFLOPs}}{989\ \text{TFLOPS}}
\approx 31\ \text{ms}
$$

计算峰值与 H100 的 dense BF16 峰值相同，因此 Prefill 的纯计算下界约为 $31$ ms；但 H200 的带宽更高。对 batch=1 的 Decode，字节量应按每步真正遍历的权重估计，而不是机械使用完整参数量：Transformer block 约为 $6.98$B 参数，LM Head 约为 $0.53$B 参数，合计约 $7.5$B 个 BF16 参数，即约 $15.0$ GB。输入 embedding 仅做当前 Token 的 lookup，不会在每一步读取整张 embedding 表；再加上约 256 MiB 的已有 GQA KV Cache，带宽下界约为：

$$
\frac{15.3\ \text{GB}}{4.8\ \text{TB/s}}
\approx 3.2\ \text{ms/token}
$$

这两个数字都不是实际服务延迟的承诺。峰值吞吐和峰值带宽很难同时、持续地达到；attention、归一化、采样、kernel launch、内存管理以及多 GPU 通信也未包含在内。更接近现实的表达是：

$$
T_{\mathrm{real}}
\gtrsim
\max\left(
\frac{F}{\eta_cP_{\mathrm{peak}}},
\frac{B}{\eta_bBW_{\mathrm{peak}}}
\right)+T_{\mathrm{other}}
$$

其中 $\eta_c$、$\eta_b<1$ 表示实际计算与带宽利用率。

不过，下界的数量级是有现实参照的。NVIDIA 公布的 H200、TensorRT-LLM、Llama 3.1 8B BF16、batch=1 基准为约 $173.8$ output tokens/s，即约 $5.75$ ms/token。[6] 这与这里的硬件、模型规模和精度已经对齐，但上下文长度、请求形态与统计口径仍可能不同，因此不能用来验证某个精确数值。它说明的是，$3.2$ ms/token 的理想带宽下界与约 $5.75$ ms/token 的公开结果处于同一量级，剩余差距来自有效带宽、算子效率与未计入的开销。比较前仍须对齐 batch、上下文长度、并行方式和统计口径。

## 小结

Prefill 和 Decode 的差异来自自回归生成本身：前者一次处理已知序列，能够形成较大的 GEMM；后者每步只处理一个新 Token，在小 batch 下权重复用不足，并不断访问 KV Cache，更容易成为显存带宽问题。

这给出了阅读后续优化方法的一条主线：有的技术减少 FLOPs，有的减少字节搬运，有的提高权重复用，有的压缩 KV Cache，有的则试图打破逐 Token 的串行过程。只有先区分瓶颈来自计算、带宽还是串行依赖，才知道一个优化为什么会有效，以及它会牺牲什么。

## 参考与延伸阅读

1. [Vaswani et al. — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [Dao et al. — FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
3. [Meta Llama 3.1 8B — config.json](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json)
4. [Williams et al. — Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785)
5. [NVIDIA H200 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h200/)
6. [NVIDIA Model Optimizer — Inference benchmark examples](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/benchmark.md)
