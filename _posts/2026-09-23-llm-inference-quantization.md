---
title: "LLM 推理系统（四）：量化究竟改变了什么——字节、算力与误差"
description: "从字节与算力两条线索分析量化的收益与适用负载，以及 scale、group size 与 outlier 如何决定量化误差。"
date: 2026-09-23 00:00:00 +0800
math: true
categories: [模型与系统, LLM 推理]
tags: [LLM, 在线推理, 量化, FP8, KV Cache]
---

上一篇结尾停在一个预告上：每步字节的两项，权重 15.0 GB 与每路 256 MiB 的 KV，都在吞吐天花板的分母上；量化减少的正是这些字节。这一篇把剩下的部分算完：压缩分别换来多少时间与显存，误差又从哪里进来。

讨论量化时最常见的说法是「INT4，压缩 4 倍」。这个 4 倍既不等于节省 4 倍显存，也不等于 4 倍加速：同一次 W4 压缩，B=1 时加速接近 3.8 倍，B=64 时降至 1.5 倍，对 Prefill 则几乎没有收益。差别不在压缩率，而在被压缩的字节位于每步字节公式的哪一项。

本文把量化的影响分成两条线索：**字节列**回答压缩的字节如何转化为时间收益；**算力列**回答哪些路线能同时抬高计算峰值。误差是第三个问题，它不出现在这两条线索里，却决定哪些方案可用。口径沿用第二、三篇：Llama 3.1 8B、H200 SXM、上下文 $L=2048$；FLOPs 与带宽用十进制前缀，显存容量用二进制前缀；除实测对照一节外均取理想口径（效率系数 $\eta=1$）。

## 收益：字节去了哪里

### 被压缩项的占比决定加速比

第三篇的每步字节公式是：

```text
每步字节 ≈ 权重 15.0 GB（固定项，与 B 无关） + B × 0.268 GB KV（不摊销项）
```

权重量化（下称 W4：权重以 4 bit 整数存储，读取后在片上反量化为 BF16 参与计算，即 weight-only 量化）只动其中的固定项。7.5B 参数从 BF16 换成 INT4，权重从 15.0 GB 变为 3.75 GB（实践中 LM Head 常保持较高精度，这里为与第二篇口径连续仍按 7.5B 全量估算）：

| | B=1 | B=64 |
| --- | --- | --- |
| 每步字节（BF16 → W4） | 15.3 → 4.0 GB | 32.2 → 20.9 GB |
| 每步时间 | 3.2 → 0.84 ms | 6.7 → 4.35 ms |
| 加速比 | **3.8x** | **1.5x** |

同样是 4 倍压缩，两种并发下的加速比相差 2.5 倍。规律是：**加速比的上限不由压缩率决定，而由被压缩项占总字节的比例决定**。B=1 时权重占每步字节的 98%；B=64 时只占 18%。即使把权重压到 0 字节，B=64 的加速比也只有 $32.2/17.2 \approx 1.9$ 倍，1.5x 已接近这一上限。

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/quant-speedup-vs-batch.svg" alt="Decode 每步时间相对 BF16 的加速比随并发 B（对数轴）变化：仅 W4 的加速比从 B=1 的约 3.8 倍衰减到 B=64 的约 1.5 倍并继续趋向 1；压缩 KV 的两条曲线趋向 2 倍渐近线。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 1：同一次 W4 压缩，B=1 时加速 3.8 倍，B=64 时降至 1.5 倍；加速比由被压缩项的字节占比决定，压缩 KV 的两条曲线趋向 2x 渐近线。</figcaption>
</figure>

这里有一个容易误读的细节：W4 之后算术强度从约 1 升到约 4 FLOPs/字节，仍在第二篇转折点 206 的左侧，Decode 依然是 memory-bound，但时间仍下降到约 1/4。Roofline 斜线区的下界是 $T = B/BW$：只要还在这条斜线上，时间就与字节量成正比，字节减少多少，时间就相应减少多少。

W4 同时使「KV 字节追平权重」的拐点左移：BF16 时 $0.268B=15$ 给出 $B\approx56$；W4 之后 $0.268B=3.75$ 给出 $B\approx14$。量化没有让系统离开带宽瓶颈，只是让它更早进入 KV 主导区，而典型服务的并发通常远大于 14。

### KV 量化：同时进入两个分母的字节

第三篇给出过两个上限：并发约 500 路（显存容量 135 GB ÷ 每路 256 MiB），吞吐天花板约 17.9k tok/s（带宽 4.8 TB/s ÷ 每 token 128 KiB）。把 KV 压成 FP8，两个上限同时翻倍：每 token 64 KiB，并发上限约 1007 路，天花板约 35.8k tok/s。

权重压缩为什么拿不到这份收益？天花板公式 $\frac{BW}{\text{每 token KV 字节}}$ 的分母中没有权重项。这不是近似，而是结构性的：天花板是 $B\to\infty$ 的渐近线；权重字节虽然每步都要读取，但它是常数，而 KV 字节随 B 线性增长，常数项在渐近线中被摊销为零。W4 腾出的 11.25 GB 显存即使全部交给 KV，也只把并发上限从 500 提高到 546（+9%）：权重只占总显存的 15 GB / 151 GB，能腾出的槽位有限。

第三篇用「权重摊销、KV 不摊销」解释并发为什么有效；同一条规律在这里决定量化收益的分布：

| 量化对象 | 作用在哪一项 | 兑现成什么 |
| --- | --- | --- |
| 权重（W4 / FP8） | 固定项 | 低并发下的 TPOT；少量腾出显存（+9%） |
| KV Cache（FP8） | 不摊销项 | 并发上限与吞吐天花板（各翻倍） |

### Prefill 与算力列：计算精度决定峰值

Prefill 是计算瓶颈。第二篇的分解是 $\max(31,\ 3.1)\ \text{ms}$：计算项 30.8 TFLOPs ÷ 989 TFLOPS，带宽项约 15 GB ÷ 4.8 TB/s。W4 把带宽项降到 0.8 ms，max 不变；压缩字节对处在计算瓶颈上的阶段无效。要改变 31 ms，只能改计算精度。

计算精度为什么能改变计算峰值？BF16 尾数为 7 位，FP8（E4M3）尾数只有 3 位；乘法器面积大致随尾数宽度的平方增长，节省的电路面积足以容纳双倍的乘加通路。H200 的规格表因此呈现每级恰好 ×2 的阶梯：

$$
\text{TF32}\ 495 \rightarrow \text{BF16}\ 989 \rightarrow \text{FP8}\ 1979\ \text{TFLOPS（dense）}
$$

两条主流路径的收益因此分属不同的列：

| 路径 | 字节列 | 算力列 |
| --- | --- | --- |
| W4A16（权重 INT4，激活 BF16） | 权重 −4x | 不动（仍按 BF16 计算），反量化开销另计 |
| FP8 W8A8（权重、激活均 FP8） | 权重与激活各 −2x | 989 → 1979 |

反量化开销在两个阶段的表现不同：Decode 的算力利用率只有 0.5%，大量算术单元空闲，转换开销可以忽略；Prefill 正处于计算峰值，转换开销是纯粹的损失，实测中 W4A16 的 Prefill 常与 BF16 持平甚至略慢，TTFT 没有收益。FP8 在两个阶段都有收益、幅度减半：B=1 的 Decode 每步字节 15.3 → 7.8 GB，约 2x；Prefill 计算项 31 → 15.6 ms，TTFT 接近减半。（乘法在 FP8、累加仍在 FP16/FP32，误差来自输入舍入而非累加漂移，误差一节会回到这里。）

### 端到端：两种负载的对比

端到端 = Prefill + 逐 Token Decode 之和。取两种负载：对话（2048 输入 / 128 输出）与 RAG（32K 输入 / 200 输出，Prefill 理想下界约 1030 ms，第三篇算过）。注意 RAG 的 Decode 每步要读取 32K 上下文的 KV，约 4.3 GB：KV 项随上下文长度增长，长上下文使 Decode 更深入 KV 主导区。

| 端到端（理想，ms） | 对话 2048/128 | RAG 32K/200 |
| --- | --- | --- |
| BF16 | 437 | 1830 |
| W4 | 138（**3.2x**） | 1364（1.3x） |
| FP8 | 221（2.0x） | 1005（**1.8x**） |

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/quant-end-to-end-workloads.svg" alt="对话（2048 输入 / 128 输出）与 RAG（32K 输入 / 200 输出）下 BF16、W4、FP8 的端到端理想耗时，每根柱由 Prefill（淡色）与 Decode（实色）叠成：对话中 W4 最快，RAG 中 FP8 最快。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 2：端到端耗时对比。对话负载下 W4 最快（3.2x），RAG 负载下 FP8 最快（1.8x）；两种精度的 1030 ms Prefill 相同，W4 不改变 Prefill 耗时。</figcaption>
</figure>

两种负载下的最优选择不同：对话负载选 W4，RAG 负载选 FP8。选型因此不是硬件问题，而取决于 workload 的三个参数：**Prefill 占比**（越高越有利于 FP8，因为它有算力列的收益）、**上下文长度**（越长 KV 项越重，越有利于 KV 量化、越稀释权重量化的占比）、**并发 B**（越高，效果与上一条相同）。

以上都是理想下界，实测还要打折。第二、三篇已经出现过 $\eta$ 与 $T_{\mathrm{other}}$；在量化场景下它们的影响更复杂，本文最后一节专门对照。

## 误差的机制

### 一次仿射量化与 a/2

把一组浮点数存进 4 bit 整数，至少需要两样东西：一组码，和一组编码参数。参数的通用形式是仿射变换：

$$
x = \operatorname{round}\!\left(\frac{w-b}{a}\right),\qquad \hat w = a\cdot x + b
$$

$b$ 是 zero point；$b=0$ 的特例叫对称量化。权重大体零对称，实践中几乎总用对称方案；非对称方案在矩阵乘中还需要额外维护 zero-point 修正项。误差的唯一来源是取整：码空间里 $|\Delta x|\le 0.5$，投影回浮点空间乘一次 $a$：

$$
|\Delta w| = a\cdot|\Delta x| \le \frac{a}{2}
$$

这个上界是理解全文的关键：**误差只由步长 $a$ 决定**。要提高精度只有两条路：缩小数值范围（$a$ 变小），或增加位数（码变密）。

用一个具体的例子验证。8 个权重，对称方案（$b=0$，码 −7…7），$a = 0.85/7 = 0.121$：

| $w$ | 0.12 | −0.85 | 0.31 | 0.02 | −0.44 | 0.67 | −0.03 | 0.58 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 码 | 1 | −7 | 3 | 0 | −4 | 6 | 0 | 5 |
| 误差 | +0.001 | 0 | +0.054 | −0.020 | −0.046 | **+0.059** | +0.030 | +0.027 |

实际最大误差 0.059，接近上界 $a/2 = 0.061$。误差最大的并不是极值：$a$ 按极值定义，极值的误差为零；误差最大的是落在两个码正中间的值。量化误差主要由这类中间值承担。

$a/2$ 的噪声为什么不会随维度放大？GEMM 中 $d$ 维点积的独立误差按 $\sqrt{d}$ 累积，而随机向量的点积本身也按 $\sqrt{d}$ 增长，因此相对误差与 $d$ 无关。这也是 4 bit 权重配合校准就能把精度损失控制在百分位量级的原因。

### outlier 与 group size

往上面那组数里加入第 9 个：$w_9 = +40$。$a$ 变成 $40/7 = 5.7$，码距扩大 47 倍。此时 $0.67/5.7 = 0.12$，取整后码为 0；逐个计算下去，原来 8 个权重的码全部为 0，还原值为 0，误差等于权重本身的大小。一个离群值使整组其余权重的信息完全丢失。

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/quant-error-mechanism.svg" alt="上图：对称 INT4 量化中，误差上界是步长 a 的一半，极值误差为零，落在两个码之间的中间值误差最大。下图：组内混入一个 +40 的 outlier 后，共享 scale 的码距扩大约 47 倍，其余 8 个权重全部落入码 0 的格、还原为 0。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 3：误差上界由步长 a 决定（上）；组内出现 +40 的 outlier 后，共享 scale 的码距扩大 47 倍，其余 8 个权重全部落入码 0（下）。</figcaption>
</figure>

解决办法是分组：各组使用自己的 $a$。每 128 个数配一个 FP16 scale，元数据开销为 16 bit ÷ 128 = **0.125 bit/数**，「INT4 模型」实际是 4.125 bit：$7.5\text{B}\times 4.125/8 \approx 3.87$ GB，与 3.75 GB 的粗估同一量级。HuggingFace 上 INT4 模型 config 里的 `group_size: 128` 就是这个 128（llama.cpp 常用 32）。

`g` 的权衡在于：`g` 越小量化越准，元数据开销也越大。极端情形 `g=1` 恰好暴露了量化的本质：每个数用自己的 $a=|w|/7$，码恒为 ±7，还原值精确等于原值，零误差；但每个数需要 16 bit scale 加 4 bit 码，约 20 bit，比直接存 BF16 的 16 bit 更多。量化的压缩因此全部来自共享：

> 量化的前提是一组相邻的数共享同一个动态范围。$g$ 划定这个前提的范围：$g=\infty$ 时共享范围最大，也最容易被 outlier 破坏；$g$ 越小前提越可靠，元数据开销越高。实践中常用的折中是 32–128。

### 动态 scale：激活为什么难量化

权重是静态的，$a$ 可以离线计算并保存。激活不同：每个 Token 的激活在现场计算，它的 $a$ 只能在运行时得到：先对激活行做一次 max-abs 归约求出 per-token scale，然后才能量化。这是一个真实存在的 kernel，每一步、每个 Token 都要执行，还引入一个同步点。

这项动态量化开销是两条路线的分水岭：**W4A16 不引入这项开销**，激活保持 BF16，只压缩静态权重，代价是计算峰值不变、Prefill 没有收益；**W8A8/FP8 引入这项开销**，换来低精度 tensor core 的 2 倍峰值与激活字节减半。

### 真实 LLM 的 outlier 与三条路线

真实 LLM 的激活比上面的例子更极端：每层有固定的一小部分隐藏维（不到 1%），幅度是其他通道的 20–100 倍，且跨 Token 稳定；LLM.int8() 首先系统记录了这一现象 [3]。这些通道会主导 per-tensor 的 $a$（即上面 +40 例子的一般化情形）；即使 per-token scale 也受影响，因为每行的 max 被这些通道占据，整行的量化步长随之变大。

| 路线 | 做法 | 代表 |
| --- | --- | --- |
| 混合分解 | outlier 通道走 FP16、其余走 INT8 | LLM.int8() |
| 尺度迁移 | 等价变换 $X' = X/s,\ W' = W\cdot s$，输出不变，把量化难度从激活挪给权重 | SmoothQuant |
| 不量化激活 | 只压权重；AWQ 按激活幅度给关键权重加保护 scale，GPTQ 用 Hessian 误差补偿逐列量化 | W4A16 派 |

两类路线的适用场景由此划分：**W4A16**（AWQ/GPTQ/llama.cpp）用于显存受限、单流与端侧部署；**W8A8/FP8**（vLLM/TensorRT-LLM on Hopper）用于在线服务。

### KV 的量化粒度

KV 量化的收益前面已经算过：并发上限与吞吐天花板各翻倍。误差的特性与权重不同：K 经过 RoPE 后通道间幅度差异大，V 接近高斯分布，因此常见的做法是 **K 用 per-channel、V 用 per-token** 的 scale（KIVI 一类方案把两者做到 2 bit 仍有可用精度 [6]）。生产 FP8 推理栈则普遍直接使用 FP8 KV Cache，配 per-tensor 或 per-head 的 scale。

## 实测：理想与真实之间

第二、三篇都用同一份公开基准做过现实检验；对量化而言，这份基准恰好构成对照组：同一模型、同一硬件、同一负载（输入 2048 / 输出 128），BF16 与三种量化并排（H200、TensorRT-LLM、Llama 3.1 8B，聚合吞吐 tok/s）[8]：

| B | BF16 | FP8 | INT4 AWQ（W4A16） | W4A8 AWQ |
| --- | --- | --- | --- | --- |
| 1 | 173.8 | 245.0（1.41x） | 231.8（1.33x） | 239.7（1.38x） |
| 8 | 803.1 | 1,051.2（1.31x） | 599.7（**0.75x**） | 801.7（1.00x） |
| 64 | 1,679.7 | 2,190.9（1.30x） | 1,392.8（**0.83x**） | 1,930.9（1.15x） |

理想模型给出的是结构与趋势，这张表是现实的修正。修正来自三层：

**一，$T_{\mathrm{other}}$ 不随精度缩水。** B=1 实测 BF16 每步 5.75 ms，理想字节项 3.18 ms，差值约 2.6 ms 是不随精度变化的杂项：attention kernel、采样、归一化、kernel 启动与同步，即第二篇 $T_{\mathrm{real}}$ 公式里的 $T_{\mathrm{other}}$。用这项固定开销预测量化后的步时：FP8 预测 $1.62+2.6\approx4.2$ ms，实测 4.08 ms，误差 3%；W4A16 预测 $0.84+2.6\approx3.4$ ms，实测 4.32 ms，多出的约 0.9 ms 是 W4 kernel 的反量化开销。两种量化在 B=1 的实测差距几乎全部由这项解释。（反量化开销可以通过精细的 kernel 设计压缩，Marlin 一类工作把 W4A16 做到接近理想带宽 [7]；但 $T_{\mathrm{other}}$ 的 2.6 ms 无法通过量化消除。）

**二，占比定律之外的 kernel 差异。** B≥8 后 W4A16 不再只是收益缩水，而是低于 BF16（0.75x、0.83x）。占比定律解释收益缩水，解释不了符号反转；这里起作用的是 kernel 形态差异：B=8 起线性层已是 GEMM，原生 BF16 GEMM 是硬件库中最成熟的路径；W4A16 的 GEMM 需要在读取侧解包、反量化再送入 BF16 tensor core，字节上省下的 11 GB 抵不过计算通路的损耗。

**三，原生低精度的收益。** W4A8 把激活换成 INT8、乘法交给原生 INT8 tensor core（峰值 ×2），B=64 时恢复到 1.15x；FP8 全程稳定在 1.3–1.4x。算力列的收益在实测中得到了验证。

「量化加速远小于压缩率」至此有了完整解释：占比定律、$T_{\mathrm{other}}$、kernel 开销，三层因素叠加。精度方面，同一份来源给出了量级：MMLU 相对 BF16 的损失，FP8 为 1.50%，INT4 AWQ 为 5.66%，W4A8 AWQ 为 6.00% [8]。

## 选型与取舍

量化方案按「误差由谁补偿、在什么时候补偿」分为两类。

**PTQ（Post-Training Quantization，训练后量化）**在训练完成后离线进行，不更新权重，只决定怎么量化：为每组数选合适的 scale，使量化后的模型尽量接近原模型。权重的范围可以直接从权重本身算出；激活的范围未知，需要跑几百条校准样本统计。GPTQ、AWQ 在此之上增加了轻量优化：前者用 Hessian 信息逐列补偿量化误差，后者按激活幅度调整关键通道的 scale。整个过程不需要原始训练集，不需要反向传播，一张 GPU 几分钟到几小时即可完成，任何已发布的模型都能直接处理。

**QAT（Quantization-Aware Training，量化感知训练）**把量化放进训练循环：在网络中插入伪量化（fake quantization）节点，前向时模拟「量化再还原」引入的误差，反向时更新权重，使模型主动适应量化噪声（取整操作不可导，梯度通过直通估计近似传递）。它需要训练数据、训练算力与相应的工程投入，产出的是一个为量化优化过的模型。

为什么两种方式并存？回到误差机制：误差只由步长 $a$ 决定，而可行的 $a$ 由被量化对象的分布决定。权重静态、分布集中，校准 scale 即可把误差控制在可用范围内，PTQ 足够；激活带 outlier，$a$ 的选择空间本身就小，仅靠调整 scale 无法补救，需要移动权重去适应误差，而修改权重就是训练。概括为一句话：误差在 scale 校准能力范围内的，用 PTQ；误差大到必须调整权重的，用 QAT。

两种路线的边界有一组直观的数字（Llama 2 7B，samsum 验证损失，BF16 为 1.036）[8]：只压权重时 PTQ 几乎无损失（1.059），QAT 只带来小幅改善（1.044）；激活也进入 8 bit 后 PTQ 完全失效（3.321），QAT 能恢复到可用水平（1.294）。数字与机制一致。因此在 8/4 bit 的常规场景里，事实标准是 PTQ；激活误差的另外两条出路是不量化激活（W4A16）与利用 FP8 更宽的动态范围（1.5% 的 MMLU 损失）；QAT 需要训练算力，用于更低位宽场景，或由模型厂商在出厂时提供。

**INT8 让位于 FP8 有三个原因**：FP8 带指数位，动态范围远宽于定点，scale 的管理更简单（per-tensor 通常即可，不需要 INT8 那套 per-tensor/per-channel 加 zero point 的管理）；Hopper 提供原生 2 倍计算峰值；训练侧 Transformer Engine 已把 FP8 标准化，推理沿用同一套格式与 scale 语义，训推一致性无需额外成本 [5]。INT8 退守 CPU 与旧架构 GPU，即 SmoothQuant 一代 W8A8 的主要部署场景。

**INT4 的适用位置由显存决定**：单卡容纳更大的模型、同一张卡承载更多并发、或端侧离线部署，权重字节是第一约束，W4 weight-only（GPTQ/AWQ，group 128）是标准方案，代价是约 5% 的精度损失与只在低并发兑现的加速。Blackwell 一代把 4-bit 浮点（FP4）做成原生计算格式，4 bit 开始同时进入算力列 [9]。

| 场景 | 默认选择 | 理由 |
| --- | --- | --- |
| 在线服务，Hopper 级硬件 | FP8（权重 + 激活 + KV） | 字节列与算力列都有收益，精度代价最小 |
| 单流、端侧、显存受限 | W4（GPTQ/AWQ）+ BF16 激活 | 显存是第一约束，Decode 收益最大 |
| 长上下文、高并发 | 上两者叠加 KV FP8 | KV 同时进入容量与带宽两个分母 |
| 更低位宽（<4 bit） | QAT | PTQ 的误差补偿已达上限 |

## 小结

量化改变的是每步字节公式里不同位置的项，收益由位置决定：压缩固定项（权重）只在低并发兑现，加速比等于被压缩项的字节占比；压缩不摊销项（KV）同时抬高并发上限与吞吐天花板，因为它的字节进入容量与带宽两个分母；换计算精度（FP8）才能抬高 Prefill 的计算峰值：尾数每减半，峰值翻一倍。误差由步长 $a$ 决定，靠共享（group）摊薄，并受 outlier 影响；量化的前提是一组数共享同一个动态范围。

不变的方面同样重要：FLOPs 几乎没有变化（反量化甚至略有增加），每步对上一步的串行依赖也没有改变。量化降低了每一步的成本，但没有改变「一步一步生成」本身：下一步开始的前提仍是上一步已经完成。要让一步产出多个 Token，需要其他方法，下一篇讨论投机解码。

## 参考与延伸阅读

1. [Frantar et al. — GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
2. [Lin et al. — AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
3. [Dettmers et al. — LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
4. [Xiao et al. — SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)
5. [Micikevicius et al. — FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
6. [Liu et al. — KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)
7. [Frantar et al. — Marlin: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://arxiv.org/abs/2408.11743)
8. [NVIDIA Model Optimizer — Inference benchmark examples](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/benchmark.md)
9. [NVIDIA — Blackwell Architecture](https://www.nvidia.com/en-us/data-center/blackwell/)
