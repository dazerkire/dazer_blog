---
title: "LLM 推理系统（三）：KV Cache、并发与请求调度——上限、浪费与取舍"
description: "从 KV Cache 的显存占用出发，拆解静态 batch、continuous batching、chunked prefill 与 PagedAttention 的原理，以及并发、吞吐与延迟之间的取舍。"
date: 2026-08-25 00:00:00 +0800
math: true
categories: [模型与系统, LLM 推理]
tags: [LLM, 在线推理, KV Cache, Continuous Batching, PagedAttention, 调度]
---

上一篇的结论可以浓缩成一个数字：在简化 Roofline 模型中，batch=1 的 Decode 算术强度很低，算力利用率上限约为 0.5%，带宽下界约为 3.2 ms/token。这一篇讨论如何把这些闲置的算力用起来，以及为此需要付出什么。

办法是并发：让多个请求的 Decode 拼进同一个 batch，提高权重复用。随之需要处理动态组批、KV 分配和准入控制，并在吞吐与单请求延迟之间取舍。

本文先从每步字节出发，算清楚并发为什么有效，再沿着调度和显存两条线，分别看朴素的实现浪费在哪里、现代推理系统如何减少，最后把所有零件合进每步时间模型，回答并发到底能开多大。

口径沿用第二篇：Llama 3.1 8B（GQA）、H200 SXM、BF16、上下文 $L=2048$；FLOPs 与带宽用十进制前缀，显存容量用二进制前缀；推导均取理想口径（效率系数 $\eta=1$），用来建立结构，不承诺数值。这些设计在 vLLM、TensorRT-LLM、SGLang 等框架里的具体实现与差异，留到系列末篇的框架对比；量化与投机解码各自成篇。

## 并发为什么有效

并发之所以有效，原因藏在权重读取的一个性质里。batch=1 的 Decode 每步只做 16.1 GFLOPs，却要搬运约 15.3 GB 数据，其中约 15.0 GB 是模型权重——而这 15 GB 每步读一遍，与这一步在算谁的 token 无关。单请求时线性层是 $(1\times d)\cdot(d\times m)$ 的 GEMV，读一遍权重只服务一个 token；若这一步同时算 $B$ 路请求各一个 token，形状变成 $(B\times d)\cdot(d\times m)$ 的 GEMM，在理想复用下权重只需读一遍。忽略激活、KV 写入和重复加载，主要读流量与计算量为：

```text
每步字节 ≈ 15.0 GB 权重（与 B 无关） + B × 256 MiB KV（各读各的）
每步 FLOPs ≈ B × 16.1 GFLOPs
```

权重项随并发摊销，KV 项不摊销——每路请求每一步都要完整读取自己的历史 KV。这个不对称会贯穿全文。以 $B=64$ 为例：

| 项 | B=1 | B=64 |
| --- | --- | --- |
| 每步 FLOPs | 16.1 G | 1.03 T |
| 每步字节 | 15.3 GB | 32.2 GB |
| 理想每步时间下界 | 3.2 ms | 6.7 ms |
| 理想吞吐上界 | ≈310 tok/s | ≈9,500 tok/s |
| 固定 batch 下理想 Token 间隔 | 3.2 ms | 6.7 ms |
| 此下界对应的算力利用率上限 | ≈0.5% | ≈16% |

在这个模型中，吞吐上界涨约 31 倍，单路间隔下界增加约 2.1 倍；从 3.2 到 6.7 ms 的增量来自 KV 项，也就是 $64\times256$ MiB 的读取。算术强度从约 1 升到约 32，相当于在第二篇的 Roofline 上沿斜线右移，买回了一部分算力，但仍在转折点 206 的左侧。

那并发能开到多大？按第二篇的公式，每 token 的 KV 是：

$$
S_{\mathrm{KV}}=2\times N_{\mathrm{layer}}\times n_{\mathrm{kv}}\times d_h\times b
=2\times32\times8\times128\times2\ \text{B}=128\ \text{KiB}
$$

2048 上下文的一路请求共 256 MiB，即约 0.268 GB。容量算例单独假设：扣除 target 的全部常驻权重、workspace 和其他预留后，可用于 KV 的预算为 120 GB（约 111.8 GiB）。这不是由 H200 的标称容量直接推得的保证值，实际预算应从运行时读取；常驻权重还包括 embedding 等每步不完整遍历的张量，不能直接用 15 GB 读取量代替。不同资源给出的理想约束如下：

| 硬件资源 | 每 token / 每路的成本 | 上限 |
| --- | --- | --- |
| 算力（989 TFLOPS） | 16.1 GFLOPs / token | 吞吐 ≤ 61k tok/s |
| 显存带宽（4.8 TB/s） | 0.268 GB / token | 吞吐 ≤ 17.9k tok/s |
| 显存容量（120 GB） | 0.268 GB / 路 | 并发最多 447 路 |

三行成本的来历：每个生成的 token 至少要做 16.1 GFLOPs 的计算、完整读一遍自己的 KV；每一路请求至少要在显存里放下 256 MiB。算力和带宽由全体请求共享，于是各有各的吞吐天花板，带宽那条不到算力天花板的三成；容量则直接封住并发路数。换句话说，无论扩吞吐还是扩并发，先碰到的限制都在 KV 一侧，61k 的算力天花板在两个方向上都够不着。

KV 的占用还是动态的：每路每步新增 128 KiB，请求完成后才归还，并发打满时总占用每个毫秒都在变，调度器必须实时跟踪。447 路的成立依赖两个前提——槽位不空转、显存不浪费，而朴素的实现恰恰两样都做不到；前者是「调度」一节的问题，后者是「显存」一节的问题。

## 调度：batch 怎么组

### 静态 batch：槽位绑定整个生命周期

最朴素的组 batch 方式沿袭自训练代码的习惯：引擎挑 $B$ 个请求绑成一个 batch，每步按固定形状 $(B,\ldots)$ 做一次前向，直到最长的请求生成完毕，整个 batch 才解散重组。batch 维上的每个位置是一个槽位（slot），请求一旦占住，这个位置就属于它到全组结束——哪怕它早就生成完毕，槽位每步照样参与计算（实现上通常填 padding token），KV Cache 也照常占着。

假设 4 路请求，输出长度分别是 32、64、128、256 token。

```text
batch 总步数 = 256（最长者决定）
总槽位步 = 4 × 256 = 1024，其中有效 = 480
槽位利用率 ≈ 47%，平均在跑的只有 1.875 路
```

这里的浪费体现在三个地方：

- **算力**：53% 的槽位步在为已完成的请求计算 padding，$B=4$ 时 15 GB 的权重搬运本来只服务 4 个 token，再打对折，实际只服务了不到 2 个。
- **显存**：已完成请求的 256 MiB KV 要等到整个 batch 解散才归还。
- **准入**：第 33 步到达的第 5 路请求必须等 223 步，按每步约 3.4 ms 计，纯排队就接近 0.75 秒；如果最长的请求要输出 1000 token，等待会以秒计。这些时间全部花在第一个 token 之前，最终都计入 TTFT——静态 batch 下 TTFT 的长尾往往不是模型慢，而是在等前面的请求做完。

<figure class="text-center mt-3 mb-4">
  <img
    src="/assets/images/posts/llm-inference/batching-timeline.svg"
    alt="静态 batch 与 continuous batching 的时间线对比：静态 batch 中短请求完成后槽位空占直到最长请求结束，新请求排队等待；continuous batching 中完成的请求立即释放槽位，新请求下一步加入。"
    style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 1：静态 batch 把槽位绑定整个生命周期（上），continuous batching 按步重组 batch（下）。</figcaption>
</figure>

### continuous batching：按步重组 batch

OSDI 2022 的推理服务系统 Orca 把组 batch 的粒度从请求的整段生命周期降到单个 decode step，称为 iteration-level scheduling，业界更常用的名字是 continuous batching：调度器每一步重新决定 batch 里有哪些请求，生成完毕的请求立即退出并归还 KV Cache，等待中的请求随即加入。[1] 这样能显著减少已完成请求的空占，并缩短等待新槽位的时间；padding、调度开销和负载不均衡仍取决于实现。

但成员每一步都可能变化，batch 的形状就不再固定，这就引出一个新的实现问题：长度各异的请求，该怎么拼进同一个前向？Orca 给出的方案叫 selective batching，核心是按算子分别处理。[1] 区分的判据，是看请求各自的长度 $L_i$（或请求个数 $B$）出现在矩阵运算的哪个维度上。

线性层（QKV 投影、MLP）以及 Norm、激活这类逐 token 的算子，对每一行输入的处理方式完全相同，因此 $B$ 路请求各自新增的那个 token 可以直接堆叠成 $(B\times d)$，做一次 GEMM、读一遍权重；请求个数只出现在矩阵的行数上，4 行与 512 行都能用 GEMM 表达，但 tile 利用率、并行度和执行效率可以相差很大。

attention 则完全是另一回事：每路的 query 要与这路请求独有的历史 KV 配对，即 $(1\times128)\cdot(128\times L_i)$，四路上下文分别为 2100、350、1800、900 的请求，会产出四种宽度不同的分数行——$L_i$ 出现在乘法的内维和产出的宽度上，天然拼不成一个矩形张量。

一种直接做法是把所有请求 pad 到最长再 mask，以上面四路为例利用率只有约 61%，而且补进去的零同样占用带宽；训练实现中也可用 padding，但序列打包与变长 kernel 同样可以减少这类浪费。推理引擎则普遍使用支持变长序列（ragged / varlen）的 kernel，在一次调用里按各请求的实际长度分段处理。

| 算子 | 形状 | $L_i$ 出现的位置 | 拼 batch 方式 |
| --- | --- | --- | --- |
| 线性层、逐元素算子 | $(B\times d)\cdot(d\times m)$ | 只影响行数（外维） | 直接堆叠，共享权重读取 |
| attention | $(1\times128)\cdot(128\times L_i)$ | 内维与产出宽度 | padding 或变长 kernel |

所以 selective batching 的含义是：形状统一的算子拼进 batch，让所有请求共享一次权重读取；形状依赖各路长度的算子按请求分别计算。这要求服务端支持动态 batch 和变长 attention，而不只是修改一个固定 batch 的前向接口。

### 新请求的加入：Prefill 的干扰与 chunked prefill

continuous batching 解决了请求的退出，还有加入这一半：新请求进入 batch 的第一件事不是 decode，而是一次完整的 Prefill。

按第二篇的因果有效 FLOPs，2048 token 的 Prefill 约为 29.7 TFLOPs，纯计算下界约 30 ms；64 路 Decode 的简化带宽下界约 6.7 ms。若调度器让完整 Prefill 独占执行，这段工作就会延迟已有请求的下一次 Decode。32K Prompt 的情况更明显：主要线性层约 457.4 TFLOPs，因果 attention 约 281.5 TFLOPs，合计约 738.9 TFLOPs，对应纯计算下界约 747 ms。实际时间还取决于 kernel 和资源竞争，不能直接由这两个下界推算用户会停顿多少毫秒。

调度器需要平衡两类等待：优先 Prefill 有利于新请求的 TTFT，却可能拉长已有请求的 ITL；优先 Decode 则可能让新请求积压。把 Prefill 和 Decode 放到不同节点的 PD 分离是另一种处理方式，但还需考虑 KV 传输与资源配比，留到框架篇展开。

**chunked prefill** 按 token 预算把长 Prompt 切成若干片，限制单轮进入引擎的 Prefill 工作量。Sarathi-Serve 进一步把 Decode token 与 Prefill 片合在同一个 batch 中：线性层共享权重读取，attention 按请求及已有前缀处理。[3] 因此，混合迭代的时间不能由“单独 Prefill 片的下界 + 单独 Decode 的下界”精确相加得到。

切片的 attention 成本也不是常数。若本片有 $C$ 个 token、此前已有 $P$ 个 token，每层的因果 attention 有效 FLOPs 约为：

$$
4d\left(CP+\frac{C(C+1)}{2}\right)
$$

同样是 512-token 的片，越靠近长 Prompt 的尾部，读取前缀和 attention 计算就越多。切片大小应与现有 Decode 数、前缀长度及延迟目标一起决定。

| 切片策略 | 对新请求的影响 | 对已有 Decode 的影响 | 需要实测的量 |
| --- | --- | --- | --- |
| 整段 Prefill | 完成所需迭代数少 | 单轮占用可能较长 | TTFT、最长迭代时长 |
| 中等切片并混合 Decode | 分多轮建立 KV | 每轮仍可推进 Decode | 混合 batch 效率、ITL 分位数 |
| 很小的切片 | 迭代次数与调度开销增加 | 更容易限制单轮工作量 | 总 TTFT、GPU 利用率 |

“让一片 Prefill 的估计工作量与一次 Decode 同量级”可以作为搜索起点，不能推出通用的最优 chunk。应对几档预算做同负载测量，用 TTFT 与 ITL 的联合结果选型。Sarathi-Serve 的 stall-free 指每个混合迭代仍推进 Decode，不意味着原有 ITL 完全不变。

## 显存：KV 怎么放

### 连续预留与按需分配的两种失败

447 路的上限要成立，显存必须接近零浪费。而几百路长度各异、随时生长、随时进出的 KV，用朴素的方式管理只有两种做法，各有各的失败方式。

- **按最大长度预留**：每路请求开工时划走 $8192\times128\ \text{KiB}=1\ \text{GiB}$ 的连续空间；实际平均用量不足一半，120 GB 只能容纳 $\lfloor120\times10^9/2^{30}\rfloor=111$ 路，预留的大半空间从不写入一个字节。
- **按需分配**：用到多少申请多少；但长短不一的请求先后进出之后，显存里只剩下碎片——空闲总量往往够，最大的连续空洞却装不下新请求。这就是外部碎片。

两种做法的败因相同：都要求一个请求的 KV 是一整块连续显存。

<figure class="text-center mt-3 mb-4">
  <img
    src="/assets/images/posts/llm-inference/kv-allocation-failures.svg"
    alt="KV Cache 两种朴素管理方案的失败：按最大长度连续预留，斜纹部分永不写入却始终占着显存；按需分配产生外部碎片，空闲总量足够但最大的连续空洞装不下新请求。"
    style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 2：按最大长度预留的浪费与按需分配的外部碎片——两种做法都要求 KV 是一整块连续显存。</figcaption>
</figure>

### 分页：block 与 block table

操作系统在几十年前就遇到过同样的问题，答案是分页。vLLM 的 PagedAttention 把这套机制搬进了 KV 管理：[2] KV 显存被划成固定大小的 block（论文的典型配置为 16 token 一块，实际可选值依赖版本与后端），每路请求持有一张 block table，记录自己的每个逻辑块对应哪个物理块，作用相当于进程的页表。逻辑上连续的 KV，物理上可以散布在显存各处。

固定块池减少了两类浪费，但仍有块内碎片与元数据成本：

- **不再预留**：按实际已有 token 申请块，随序列生长追加；尾块未填满形成块内碎片。若长度余数近似均匀，平均不足半块，16-token 块对应约 1 MiB/路。
- **不再碎片**：所有块等大，任何请求释放的块都能被任何请求复用，「空闲但不连续」这个状态根本不会出现。

vLLM 论文给出的对照是：此前系统的 KV 显存有效利用率只有 20.4%–38.2%，PagedAttention 做到 near-zero waste，同等延迟下吞吐比 FasterTransformer、Orca 等高 2–4 倍。[2]

块大小本身是个需要权衡的参数。块太小，block table 冗长，attention kernel 按块遍历的开销和访存的碎片化都会上升；块太大，等于退回到按块预留，1024 token 的块平均每路浪费 64 MiB。论文配置的 16 是一种折中，并非所有后端的固定默认值。[2]

<figure class="text-center mt-3 mb-4">
  <img
    src="/assets/images/posts/llm-inference/kv-paged-memory.svg"
    alt="分页管理：两个请求的逻辑块经 block table 映射到物理显存中散布的固定大小块，相同前缀的三个逻辑块指向同一批物理块，只存一份。"
    style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 3：block table——逻辑连续、物理散布；紫色为两个请求共享的前缀块。</figcaption>
</figure>

### 共享与 prefix cache

分页结构使共享更容易以统一的块引用与引用计数管理：多张 block table 指向同一批物理块，以只读方式使用。这带来两个直接用途。

一是 copy-on-write。beam search 的多个候选共享完全相同的 Prompt 块，KV 只存一份；当某个候选要在一个共享块上追加写入时，才把这一块复制出来私有化，其余块照旧共享。共享和 copy-on-write 并不以分页为必要条件；连续缓冲区、分段结构等也能实现。分页的价值是让共享、分配和回收采用相同的块粒度。

二是 prefix cache。第二篇讨论过为什么只缓存 K、V 而不缓存 Q：历史 K、V 是不可变的，写好即冻结。不可变意味着不但可以在同一时刻被多个请求共享，还可以跨时间留存复用。多轮对话进行到第 3 轮、携带 8K 历史时，引擎按前缀匹配直接挂上上一轮的块，命中的部分既不计算也不新占显存；没有这层缓存，这 8K 要重付约 132 TFLOPs（因果有效计算下界约 133 ms）的重复 Prefill。第一篇讲过的 Agent 负载——每轮把全部历史重发一遍——从中受益最大。收益还不止于算力：命中前缀可减少重复 Prefill，缓解它对正在 Decode 的请求的干扰；未命中尾部、缓存元数据处理和首个输出的计算仍需执行，TTFT 是否改善还取决于排队。

### 显存满时：准入与抢占

块池将尽时（运行中的请求还在生长，新请求还在到达），调度器手里有两个动作：

- **准入控制**：不再放新请求进入，等有请求自然结束后再放行。
- **抢占**：把某一路暂停，收回它的全部块。

KV 是恢复计算所需的主要张量状态；引擎还要保留 token 序列、采样与调度状态。KV 的两种处置方式如下：

| 处置方式 | 做法 | 主要成本 |
| --- | --- | --- |
| recompute | 丢弃可回收的 KV，恢复时重跑所需前向 | 线性层随长度增长，完整 Prefill 的因果 attention 随长度平方增长 |
| swap | 将 KV 移到 CPU，恢复时搬回 | KV 大小、实测链路带宽、传输重叠和 CPU 内存占用 |

例如，在无缓存命中、采用第二篇因果 FLOPs 的模型中，32K 序列重算的计算下界约 747 ms，2200-token 序列约 32.3 ms。若单向有效 PCIe 带宽假设为 50 GB/s，两者 KV 往返传输分别约 $2\times32768\times128\ \text{KiB}/50\ \text{GB/s}=172$ ms 和 11.5 ms。这只是两种成本的数量级算例：前者是计算下界，后者是假定带宽下的传输时间，不能据此断言 swap 总是更快。

prefix cache 只有在对应块仍然驻留或可恢复时，才能减少重算；已被回收、覆盖的块不能继续命中。共享前缀、部分保留、异步传输、总线争用以及引擎支持的抢占方式，都会改变实际选择。

## 并发开多大：拐点、天花板与 SLO

### 每步时间模型：拐点与天花板

把前面的零件装回每步时间模型（$L=2048$，理想口径）：

$$
T_{\mathrm{ideal}}(B)=
\max\left(\frac{16.1B\ \text{GFLOPs}}{989\ \text{TFLOPS}},
\ \frac{15.0+0.268B\ \text{GB}}{4.8\ \text{TB/s}}\right)
\approx 3.13+0.056B\ \text{ms}
$$

第一个结论关于拐点。令 KV 总流量与权重流量相等，$0.268B=15.0$，解得 $B\approx56$：远低于此值时，主要读流量由权重项主导，权重摊销的收益较明显，吞吐随并发近似线性上升；越过这个点之后，KV 项开始主导，在这个模型中，每增加一路都给单轮带宽项增加约 56 µs（拐点前后均如此）。

第二个结论关于天花板。$B$ 趋于无穷时，吞吐趋于 $\frac{4.8\ \text{TB/s}}{268\ \text{MB}}\approx17.9\text{k}$ tok/s，一个只取决于每步 KV 字节数的数字，与前文资源上限表里带宽一行的上限一致。算力那边的天花板约有 61k tok/s，但在 $L=2048$ 下够不着，因为每增加一路请求的边际带宽成本（56 µs）始终大于边际算力成本（16.3 µs），若暂时固定每 token 的计算量，粗估两者相等在 $L\approx600$。缩短上下文会降低 KV 读取成本，使大 batch 更可能受计算限制；精确位置还要重新计入随 $L$ 变化的 attention FLOPs，实际瓶颈仍需逐算子确认。

### SLO 定运营点

先只在固定上下文、固定 batch、无排队的理想模型里，将每步时长预算换算为并发与吞吐：

| 理想每步时长预算 | 并发 B | 理想吞吐上界 | 起约束作用的因素 |
| --- | --- | --- | --- |
| 10 ms | 123 | 12.3k | 理想时长预算 |
| 20 ms | 301 | 15.0k | 理想时长预算 |
| 约 28.1 ms | 447 | 15.9k | 假设的 KV 预算 |
| 无限制 | → | 17.9k（渐近） | KV 带宽 |

这条曲线增益递减：SLO 从 10 放宽到 20 ms，只多换约 22% 的吞吐，最后一段吞吐最贵。起约束作用的因素也在切换：SLO 紧时是承诺本身，SLO 松时是显存容量，而 KV 带宽始终悬在头顶。

SLO 由此成为运营决策而非物理常数——这张表只是资源模型的候选点，不能把配置并发直接当成可兑现的 SLO。真实服务还需对请求到达率、长度分布、Prefill 混入和尾延迟进行压测。生产系统据此引入 goodput 的视角：只统计满足 SLO 的请求的吞吐，把 SLO 与并发一起作为容量规划的输入。

### 与公开基准对照

沿用第二篇的 NVIDIA 数据：H200、Model Optimizer v0.21.1、TensorRT-LLM v0.15、Llama 3.1 8B BF16、输入 2048 / 输出 128。[4]

| 基准中的 batch size | 实测输出吞吐 |
| --- | --- |
| 1 | 173.80 tok/s |
| 8 | 803.11 tok/s |
| 64 | 1679.74 tok/s |

这张表支持“该基准中 batch 增加时，吞吐增长但未线性增长”的观察。它没有给出逐请求 TPOT，不能用 $B/\text{吞吐}$ 填出“实测 TPOT”，也不能将其与纯 Decode 下界的比值解释为带宽效率下降。Prefill、请求调度、形状变化和不同算子的效率都可能影响结果，需要对应的延迟测量和 profile 才能归因。

## 小结

全文可以收成几句话。并发的上限是显存给的，KV 容量决定能同时跑多少路；吞吐的天花板是带宽给的，约等于带宽除以每步 KV 字节；调度的职责是不浪费，continuous batching 减少槽位空占，PagedAttention 减少分配浪费，chunked prefill 限制并分散 Prefill 的干扰；运营点由 SLO 在吞吐与延迟的前沿上选定。

每步字节的公式也预告了下一篇。每步字节的两项——权重 15.0 GB 与每路 256 MiB 的 KV——都是字节数，而天花板公式 $\frac{BW}{L\times S_{\mathrm{KV}}}$ 里，字节数就坐在分母上。量化减少的正是字节数：压缩权重削减每步的固定项，压缩 KV 同时抬高并发上限和吞吐天花板。第四篇见。

## 参考与延伸阅读

1. [Yu et al. — Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu)
2. [Kwon et al. — Efficient Memory Management for Large Language Model Serving with PagedAttention (SOSP 2023)](https://arxiv.org/abs/2309.06180)
3. [Agrawal et al. — Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve (OSDI 2024)](https://arxiv.org/abs/2403.02310)
4. [NVIDIA Model Optimizer — Inference benchmark examples](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/benchmark.md)
