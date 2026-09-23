---
title: "LLM 推理系统（四）：量化究竟改变了什么——字节、算力与误差"
description: "把量化放进两本账：压缩的字节在哪种负载下兑现成时间，低精度计算何时抬屋顶；以及 scale、group size 与 outlier 如何决定误差。"
date: 2026-09-23 00:00:00 +0800
math: true
categories: [模型与系统, LLM 推理]
tags: [LLM, 在线推理, 量化, FP8, KV Cache]
---

上一篇的账本停在一句预告上：每步字节的两项——权重 15.0 GB 与每路 256 MiB 的 KV——都坐在天花板公式的分母上；量化减少的正是这些字节。这一篇把这笔账算完：压缩到底换来多少时间和显存，误差又从哪里进来。

讨论量化绕不开一个宣传数字：「INT4，压缩 4 倍」。但它既不等于省 4 倍显存，更不等于快 4 倍：同一次 W4 压缩，B=1 时接近 3.8 倍加速，B=64 时只剩 1.5 倍，到了 Prefill 一分不赚。差别不在压缩率，而在被压缩的字节坐在账本的哪个位置。

本文把量化放进两本账：**字节列**回答压缩的字节如何兑现成时间；**算力列**回答哪条路线能连计算屋顶一起抬起来。误差是第三样东西，它不进这两本账，却决定哪些方案能用。口径沿用第二、三篇：Llama 3.1 8B、H200 SXM、上下文 $L=2048$；FLOPs 与带宽用十进制前缀，显存容量用二进制前缀；除实测对照一节外均取理想口径（效率系数 $\eta=1$）。

## 收益的账：字节去了哪里

### 被压缩项的占比决定加速比

第三篇的每步账本是：

```text
每步字节 ≈ 权重 15.0 GB（固定项，与 B 无关） + B × 0.268 GB KV（不摊销项）
```

权重量化（下称 W4：权重以 4 bit 整数存储，读取后在片上反量化为 BF16 参与计算，即 weight-only 量化）只动其中的固定项。7.5B 参数从 BF16 换成 INT4，权重从 15.0 GB 变为 3.75 GB（实践中 LM Head 常保持较高精度，这里为与第二篇口径连续仍按 7.5B 全量估算）：

| | B=1 | B=64 |
| --- | --- | --- |
| 每步字节（BF16 → W4） | 15.3 → 4.0 GB | 32.2 → 20.9 GB |
| 每步时间 | 3.2 → 0.84 ms | 6.7 → 4.35 ms |
| 加速比 | **3.8x** | **1.5x** |

同一个 4 倍压缩，两种并发差出 2.5 倍的加速比。规律一句话：**加速比封顶不由压缩率决定，由被压缩项占总字节的比例决定**。B=1 时权重占每步字节的 98%；B=64 时只占 18%——哪怕把权重压到 0 字节，B=64 的加速比也只有 $32.2/17.2 \approx 1.9$ 倍，1.5x 已经贴着上限。

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/quant-speedup-vs-batch.svg" alt="Decode 每步时间相对 BF16 的加速比随并发 B（对数轴）变化：仅 W4 的加速比从 B=1 的约 3.8 倍衰减到 B=64 的约 1.5 倍并继续趋向 1；压缩 KV 的两条曲线趋向 2 倍渐近线。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 1：同一个 W4 压缩，B=1 加速 3.8 倍、B=64 只剩 1.5 倍——加速比由被压缩项的字节占比决定；KV 字节减半的两条曲线则趋向 2x 渐近线。</figcaption>
</figure>

这里有一个容易误读的细节：W4 之后算术强度从约 1 升到约 4 FLOPs/字节，仍在第二篇屋脊点 206 的左侧——**还是 memory-bound，但时间照样降到约 1/4**。Roofline 斜线区的下界是 $T = B/BW$：还在线上不等于没变快，字节砍多少，时间就砍多少。

W4 还让「KV 追平权重」的拐点左移：BF16 时 $0.268B=15$ 给出 $B\approx56$，W4 之后 $0.268B=3.75$ 给出 $B\approx14$。量化没有让系统离开带宽瓶颈，而是让它**更早进入 KV 主导区**——典型服务的并发本来就在 14 右侧很深的地方。

### KV 量化：坐在两个分母上的字节

第三篇给出过两个上限：并发约 500 路（显存容量 135 GB ÷ 每路 256 MiB），吞吐天花板约 17.9k tok/s（带宽 4.8 TB/s ÷ 每 token 128 KiB）。把 KV 压成 FP8，两个数一起翻倍：每 token 64 KiB，并发上限约 1007 路，天花板约 35.8k tok/s。

权重压缩为什么拿不到这份收益？天花板公式 $\frac{BW}{\text{每 token KV 字节}}$ 的分母里**没有权重项**。这不是近似，是结构：天花板是 $B\to\infty$ 的渐近线，权重字节虽然每步照读，却是常数；KV 字节随 B 线性增长，常数在渐近线里被摊成零。W4 腾出的 11.25 GB 显存交给 KV，也只把并发上限从 500 抬到 546（+9%）——权重只占总显存 15 GB / 151 GB，省不出几个槽位。

第三篇用「权重摊销、KV 不摊销」解释并发为什么有效；这里同一条定理决定量化收益的分布：

| 量化对象 | 作用在哪一项 | 兑现成什么 |
| --- | --- | --- |
| 权重（W4 / FP8） | 固定项 | 低并发下的 TPOT；少量腾出显存（+9%） |
| KV Cache（FP8） | 不摊销项 | 并发上限与吞吐天花板（各翻倍） |

### Prefill 与算力列：换精度，换屋顶

Prefill 是计算瓶颈。第二篇的分解是 $\max(31,\ 3.1)\ \text{ms}$：计算项 30.8 TFLOPs ÷ 989 TFLOPS，带宽项约 15 GB ÷ 4.8 TB/s。W4 把带宽项砍到 0.8 ms，max 纹丝不动——**压缩字节对踩在计算屋顶上的阶段无效**。想动 31 ms，只能改计算精度。

改精度为什么能改屋顶？看尾数：BF16 尾数 7 位，FP8（E4M3）尾数只有 3 位，乘法器面积大致随尾数宽度平方增长，省出的硅片足以塞下双倍乘加通路。H200 的规格表上是一条每级恰好 ×2 的阶梯：

$$
\text{TF32}\ 495 \rightarrow \text{BF16}\ 989 \rightarrow \text{FP8}\ 1979\ \text{TFLOPS（dense）}
$$

于是两条主流路径的账目各是一列：

| 路径 | 字节列 | 算力列 |
| --- | --- | --- |
| W4A16（权重 INT4，激活 BF16） | 权重 −4x | 不动（仍按 BF16 计算），反量化开销另计 |
| FP8 W8A8（权重、激活均 FP8） | 权重与激活各 −2x | 989 → 1979 |

反量化税有两副面孔：Decode 的算力利用率只有 0.5%，大量算术单元闲着，转换开销近乎免费；Prefill 正踩着计算峰值，转换开销是纯亏损——实测中 W4A16 的 Prefill 常持平甚至略慢，TTFT 不赚反可能小亏。FP8 则两头都赚、幅度减半：B=1 的 Decode 每步字节 15.3 → 7.8 GB，约 2x；Prefill 计算项 31 → 15.6 ms，TTFT 近乎砍半。（乘法在 FP8、累加仍在 FP16/FP32，误差来自输入舍入而非累加漂移——误差一节会回到这里。）

### 端到端：两种负载，两个赢家

端到端 = Prefill + 逐 Token Decode 之和。取两种负载：对话（2048 输入 / 128 输出）与 RAG（32K 输入 / 200 输出，Prefill 理想下界约 1030 ms，第三篇算过）。注意 RAG 的 Decode 每步要读 32K 上下文的 KV，约 4.3 GB——KV 项随上下文长度增长，长上下文把 Decode 推得更深进 KV 区：

| 端到端（理想，ms） | 对话 2048/128 | RAG 32K/200 |
| --- | --- | --- |
| BF16 | 437 | 1830 |
| W4 | 138（**3.2x**） | 1364（1.3x） |
| FP8 | 221（2.0x） | 1005（**1.8x**） |

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/quant-end-to-end-workloads.svg" alt="对话（2048 输入 / 128 输出）与 RAG（32K 输入 / 200 输出）下 BF16、W4、FP8 的端到端理想耗时，每根柱由 Prefill（淡色）与 Decode（实色）叠成：对话中 W4 最快，RAG 中 FP8 最快。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 2：端到端两个赢家——对话里 W4 胜（3.2x），RAG 里 FP8 胜（1.8x）；RAG 的两根 1030 ms Prefill 一样高，W4 压不动它。</figcaption>
</figure>

互有胜负：对话里 W4 赢，RAG 里 FP8 赢。选型因此不是硬件问题，而是 workload 的三个参数：**Prefill 占比**（越高越利好 FP8，它有算力列收入）、**上下文长度**（越长 KV 项越重，越利好 KV 量化、越稀释权重量化）、**并发 B**（越高越同前）。

以上都是理想下界。实测还要再打折扣——第二、三篇已经出现过 $\eta$ 与 $T_{\mathrm{other}}$，量化篇里它们会以更有趣的方式现身，本文最后一节专门对账。

## 误差的机制

### 一次仿射量化与 a/2

把一组浮点数存进 4 bit 整数，最少需要两样东西：一组码，和一组编码参数。参数的通用形式是仿射变换：

$$
x = \operatorname{round}\!\left(\frac{w-b}{a}\right),\qquad \hat w = a\cdot x + b
$$

$b$ 是 zero point；$b=0$ 的特例叫对称量化。权重大体零对称，几乎总用对称方案——非对称在矩阵乘里还要额外维护 zero-point 修正项。误差的唯一来源是取整：码空间里 $|\Delta x|\le 0.5$，投影回浮点空间乘一次 $a$：

$$
|\Delta w| = a\cdot|\Delta x| \le \frac{a}{2}
$$

这个上界的读法是全文最重要的一句话：**误差只由步长 $a$ 决定**。想更准只有两条路——把数值范围缩小（$a$ 变小），或加位数（码变密）。

用一个玩具组看它落地。8 个权重，对称方案（$b=0$，码 −7…7），$a = 0.85/7 = 0.121$：

| $w$ | 0.12 | −0.85 | 0.31 | 0.02 | −0.44 | 0.67 | −0.03 | 0.58 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 码 | 1 | −7 | 3 | 0 | −4 | 6 | 0 | 5 |
| 误差 | +0.001 | 0 | +0.054 | −0.020 | −0.046 | **+0.059** | +0.030 | +0.027 |

实际最大误差 0.059，贴着上界 $a/2 = 0.061$。注意误差最大的不是极值——$a$ 按极值定义，极值零误差；最受苦的是落在两个码之间的**中间值**。量化误差是一场「中间值税」。

这个 $a/2$ 的噪声为什么不会随维度放大？GEMM 把 $d$ 维点积里的独立误差按 $\sqrt{d}$ 累积，而随机向量的点积本身也按 $\sqrt{d}$ 增长，相对误差与 $d$ 无关。这就是 4 bit 权重加校准就能把精度损失压到百分位的原因。

### outlier 劫持与 group size

往上面那组数里加第 9 个：$w_9 = +40$。$a$ 变成 $40/7 = 5.7$，码距撑大 47 倍。 $0.67/5.7 = 0.12$，取整后码为 0——逐个算下去，**原来 8 个数全部归零**，误差等于它们自身的全部大小。一个离群值把整组同伴的信息清空了。

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/quant-error-mechanism.svg" alt="上图：对称 INT4 量化中，误差上界是步长 a 的一半，极值零误差，落在两个码之间的中间值误差最大。下图：组内混入一个 +40 的 outlier 后，共享 scale 的码距撑大约 47 倍，其余 8 个权重全部落入码 0 的格、还原为 0。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 3：误差只由步长 a 决定（上）；一个 outlier 劫持整组共享的 scale，把同伴全部挤进码 0（下）。</figcaption>
</figure>

救法只有一种：分组，各组用自己的 $a$。每 128 个数配一个 FP16 scale，元数据开销 16 bit ÷ 128 = **0.125 bit/数**，「INT4 模型」实际是 4.125 bit——$7.5\text{B}\times 4.125/8 \approx 3.87$ GB，与 3.75 GB 的粗估同一量级。HuggingFace 上 INT4 模型 config 里的 `group_size: 128` 就是这个 128（llama.cpp 常用 32）。

`g` 的两难在于：越小越准、元数据越贵。极端的 `g=1` 点破量化的本质——每个数自己的 $a=|w|/7$，码恒为 ±7，还原值精确等于原值，零误差；但每个数要背 16 bit scale 加 4 bit 码，约 20 bit，比直接存 BF16 的 16 bit 还胖。所以量化的压缩**全部来自共享**：

> 量化是一次赌博：赌一组相邻的数共享同一个动态范围。$g$ 是赌注大小——$g=\infty$ 赌最大、最容易被 outlier 劫持；$g$ 越小赌得越稳、抽水越贵。市场落点在 32–128。

### 动态 scale：激活为什么难量化

权重是静态的，$a$ 可以离线算好刻进文件。激活不行：每个 Token 的激活现场算出，它的 $a$ 只能在运行时产生——对激活行做一次 max-abs 归约得到 per-token scale，然后才能量化。这是一个真实存在的 kernel，每步、每 Token 都要付，还多一个同步点。

这笔「动态量化税」是两条路线的分水岭：**W4A16 拒绝交税**——激活保持 BF16，只压缩静态权重，代价是算力列不动、Prefill 不赚；**W8A8/FP8 交税**——换来低精度 tensor core 的 2 倍屋顶和激活字节减半。

### 真实 LLM 的病与三条出路

真实 LLM 的激活比玩具更恶劣：每层有固定的一小撮隐藏维（不到 1%），幅度是其他通道的 20–100 倍，且跨 Token 稳定——LLM.int8() 首先系统记录了这一现象 [3]。它们劫持 per-tensor 的 $a$（玩具里 +40 的现实版）；连 per-token scale 都遭殃，因为每行的 max 被它霸占，全行替它交中间值税。

| 路线 | 做法 | 代表 |
| --- | --- | --- |
| 混合分解 | outlier 通道走 FP16、其余走 INT8 | LLM.int8() |
| 尺度迁移 | 等价变换 $X' = X/s,\ W' = W\cdot s$，输出不变，把量化难度从激活挪给权重 | SmoothQuant |
| 不量化激活 | 只压权重；AWQ 按激活幅度给关键权重加保护 scale，GPTQ 用 Hessian 误差补偿逐列量化 | W4A16 派 |

市场格局由此成形：**W4A16**（AWQ/GPTQ/llama.cpp——显存受限、单流、端侧）对 **W8A8/FP8**（vLLM/TensorRT-LLM on Hopper——在线服务）。

### KV 的量化粒度

KV 量化的收益已经算过：并发上限与吞吐天花板各翻倍。误差侧的形状与权重不同：K 经过 RoPE 后通道间幅度差异大，V 近高斯分布，所以常见做法是 **K 用 per-channel、V 用 per-token** 的 scale（KIVI 一类方案把两者做到 2 bit 仍有可用精度 [6]）。生产 FP8 推理栈则普遍直接开 FP8 KV Cache，per-tensor 或 per-head 的 scale。

## 实测：理想账与真实账之间

第二、三篇都用同一份公开基准做过现实检验，这一篇它恰好成了量化的对照组：同一模型、同一硬件、同一负载（输入 2048 / 输出 128），BF16 与三种量化并排（H200、TensorRT-LLM、Llama 3.1 8B，聚合吞吐 tok/s）[8]：

| B | BF16 | FP8 | INT4 AWQ（W4A16） | W4A8 AWQ |
| --- | --- | --- | --- | --- |
| 1 | 173.8 | 245.0（1.41x） | 231.8（1.33x） | 239.7（1.38x） |
| 8 | 803.1 | 1,051.2（1.31x） | 599.7（**0.75x**） | 801.7（1.00x） |
| 64 | 1,679.7 | 2,190.9（1.30x） | 1,392.8（**0.83x**） | 1,930.9（1.15x） |

理想模型给的是结构与趋势，这张表是它的现实修正项。三层折扣：

**一，$T_{\mathrm{other}}$ 不缩水。** B=1 实测 BF16 每步 5.75 ms，理想字节项 3.18 ms，差值约 2.6 ms 是不随精度缩水的杂项——attention kernel、采样、归一化、kernel 启动与同步，即第二篇 $T_{\mathrm{real}}$ 公式里的 $T_{\mathrm{other}}$。拿这个固定开销去预测量化后的步时：FP8 预测 $1.62+2.6\approx4.2$ ms，实测 4.08 ms，吻合到 3%；W4A16 预测 $0.84+2.6\approx3.4$ ms，实测却是 4.32 ms——多出的约 0.9 ms 是 W4 kernel 的反量化税。两个量化在 B=1 的实测差距，几乎全由这一项解释。（反量化税可以靠精心排布的 kernel 压缩，Marlin 一类工作把 W4A16 做到接近理想带宽 [7]；但 $T_{\mathrm{other}}$ 那 2.6 ms 谁也省不掉。）

**二，占比定律的极端版。** B≥8 后 W4A16 不再只是收益缩水，而是直接**倒亏**（0.75x、0.83x）。占比定律解释缩水，解释不了符号翻转——这里起作用的是 kernel 形态差：B=8 起线性层已是 GEMM，原生 BF16 GEMM 是硬件库里最成熟的路径；W4A16 的 GEMM 要在读取侧解包、反量化再喂给 BF16 tensor core，字节上省下的 11 GB 抵不过计算通路的损耗。

**三，原生低精度救场。** W4A8 把激活换成 INT8、乘法交给原生 INT8 tensor core（屋顶 ×2），B=64 拉回 1.15x；FP8 全程稳定在 1.3–1.4x。算力列的收入在实测里兑现了。

「量化加速远小于压缩率」至此有了完整答案：占比定律、$T_{\mathrm{other}}$、kernel 税，三层折扣叠加。精度侧同一份来源给出了量级：MMLU 相对 BF16 的损失，FP8 1.50%，INT4 AWQ 5.66%，W4A8 AWQ 6.00% [8]。

## 选型与取舍

**PTQ 与 QAT 的分工有一组直观数字**（Llama 2 7B，samsum 验证损失，BF16 为 1.036）[8]：只压权重时 PTQ 几乎免费（1.059），QAT 仅微调（1.044）；激活也进 8 bit 后 PTQ 崩溃（3.321），QAT 能拉回（1.294）。与机制部分的分水岭完全同构：权重的误差离线校准即可压住；激活的误差要么不量化它（W4A16），要么靠训练补偿（QAT），要么换 FP8 的宽容动态范围（1.5% 的 MMLU 损失）。8/4 bit 的事实标准是 PTQ；QAT 花训练算力，用于更低位宽或由模型厂出厂提供。

**INT8 让位 FP8 的三个原因**：FP8 带指数，动态范围远宽于定点，scale 的管理宽容得多（per-tensor 常常就够，不需要 INT8 那套 per-tensor/per-channel 加 zero point 的记账）；Hopper 原生 2 倍屋顶；训练侧 Transformer Engine 已把 FP8 标准化，推理沿用同一套格式与 scale 语义，训推一致性免费获得 [5]。INT8 退守 CPU 与旧架构 GPU——SmoothQuant 一代 W8A8 的主战场。

**INT4 的位置由显存决定**：单卡塞下更大的模型、同一张卡放下更多并发、或端侧离线部署，权重字节是第一约束，W4 weight-only（GPTQ/AWQ，group 128）是标准答案，代价是约 5% 的精度损失与只在低并发兑现的加速。Blackwell 一代把 4-bit 浮点（FP4）做成原生计算格式，4 bit 开始同时进入算力列 [9]。

| 场景 | 默认选择 | 理由 |
| --- | --- | --- |
| 在线服务，Hopper 级硬件 | FP8（权重 + 激活 + KV） | 两列都赚，精度代价最小 |
| 单流、端侧、显存受限 | W4（GPTQ/AWQ）+ BF16 激活 | 显存是第一约束，Decode 收益最大 |
| 长上下文、高并发 | 上两者叠加 KV FP8 | KV 坐在容量与带宽两个分母上 |
| 更低位宽（<4 bit） | QAT | PTQ 的误差补偿已到头 |

## 小结

量化改变的是账本上不同位置的数，而收益由位置决定：压缩固定项（权重）只在低并发兑现——加速比等于被压缩项的字节占比；压缩不摊销项（KV）同时抬高并发上限与吞吐天花板，因为它的字节坐在两个分母上；换计算精度（FP8）才动得了 Prefill 的屋顶——尾数每减半，峰值翻一倍。误差由步长 $a$ 决定，靠共享（group）摊薄，被 outlier 胁迫；量化的本质是赌一组数共享同一个动态范围。

不变的东西同样重要：FLOPs 几乎没变（反量化甚至略增），每步对上一步的串行依赖也没变。量化把每一步变便宜，没有打破「一步一步生成」本身——下一步快的前提仍是上一步已经算完。要让一步产出多个 Token，需要另一种办法，下一篇讨论投机解码。

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
