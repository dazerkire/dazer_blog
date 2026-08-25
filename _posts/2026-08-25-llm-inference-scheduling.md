---
title: "LLM 推理系统（三）：KV Cache、并发与请求调度——上限、浪费与取舍"
description: "从 KV Cache 的显存账本出发，拆解静态 batch、continuous batching、chunked prefill 与 PagedAttention 的原理，以及并发、吞吐与延迟之间的取舍。"
date: 2026-08-25 00:00:00 +0800
math: true
categories: [模型与系统, LLM 推理]
tags: [LLM, 在线推理, KV Cache, Continuous Batching, PagedAttention, 调度]
---

上一篇的结论可以浓缩成一个数字：batch=1 的 Decode 只用掉约 0.5% 的计算峰值，在 3.2 ms/token 的带宽下界里，GPU 大部分时间在等显存。这一篇讨论如何把这些闲置的算力用起来，以及为此需要付出什么。

办法是并发：让几十、上百路请求的 Decode 拼进同一个 batch。但并发并不是一个免费的旋钮，它带来一串新的问题——上限由什么决定？请求长短不一、随时到达也随时结束，batch 该怎么组？几百路请求的 KV Cache 共享同一块显存，该怎么放置？并发开到多大，单个用户的体验会差到什么程度？

本文先把并发的账本算清楚，再沿着调度和显存两条线，分别看朴素的实现浪费在哪里、现代推理系统如何消除，最后把所有零件合进每步时间模型，得到吞吐与延迟的总账。

口径沿用第二篇：Llama 3.1 8B（GQA）、H200 SXM、BF16、上下文 $L=2048$；FLOPs 与带宽用十进制前缀，显存容量用二进制前缀；推导均取理想口径（效率系数 $\eta=1$），用来建立结构，不承诺数值。这些设计在 vLLM、TensorRT-LLM、SGLang 等框架里的具体实现与差异，留到系列末篇的框架对比；量化与投机解码各自成篇。

## 并发的账本

并发之所以有效，原因藏在权重读取的一个性质里。batch=1 的 Decode 每步只做 16.1 GFLOPs，却要搬运约 15.3 GB 数据，其中约 15.0 GB 是模型权重——而这 15 GB 每步读一遍，与这一步在算谁的 token 无关。单请求时线性层是 $(1\times d)\cdot(d\times m)$ 的 GEMV，读一遍权重只服务一个 token；若这一步同时算 $B$ 路请求各一个 token，形状变成 $(B\times d)\cdot(d\times m)$ 的 GEMM，权重仍然只读一遍。于是每步的账本是：

```text
每步字节 ≈ 15.0 GB 权重（与 B 无关） + B × 256 MiB KV（各读各的）
每步 FLOPs ≈ B × 16.1 GFLOPs
```

权重项随并发摊销，KV 项不摊销——每路请求每一步都要完整读取自己的历史 KV。这个不对称会贯穿全文。以 $B=64$ 为例：

| 项 | B=1 | B=64 |
| --- | --- | --- |
| 每步 FLOPs | 16.1 G | 1.03 T |
| 每步字节 | 15.3 GB | 32.2 GB |
| 每步时间 | 3.2 ms | 6.7 ms |
| 聚合吞吐 | ≈310 tok/s | ≈9,500 tok/s |
| 每路 TPOT | 3.2 ms | 6.7 ms |
| 算力利用率 | ≈0.5% | ≈16% |

吞吐涨了 31 倍，代价是每个用户的 token 间隔慢了 2.1 倍；而且从 3.2 到 6.7 ms 的新增耗时全部来自 KV 项，也就是 $64\times256$ MiB 的读取。算术强度从约 1 升到约 32，相当于在第二篇的 Roofline 上沿斜线右移，买回了一部分算力，但仍在屋脊点 206 的左侧。

那并发能开到多大？算力维度很宽裕，显存维度很快见底。按第二篇的公式，每 token 的 KV 是：

$$
S_{\mathrm{KV}}=2\times N_{\mathrm{layer}}\times n_{\mathrm{kv}}\times d_h\times b
=2\times32\times8\times128\times2\ \text{B}=128\ \text{KiB}
$$

2048 上下文的一路请求共 256 MiB。H200 的 141 GiB（约 151 GB）扣除 15 GB 权重和若干 workspace，粗放按 135 GB 给 KV：

$$
135\ \text{GB}\ \div\ 0.268\ \text{GB/路}\ \approx\ 500\ \text{路}
$$

这是理想并发上限，而且由显存决定：就算并发开到无限，吞吐天花板也只有约 1.8 万 tok/s（「总账」一节推导），离 989 TFLOPS 能支撑的约 6.1 万 tok/s 还很远，显存却在 500 路就先满了。换句话说，沿着扩并发这个方向，先碰到的限制是 KV Cache 的容量，不是算力。

这本账还是动态的：每路每步新增 128 KiB，请求完成后才归还，并发打满时这个账户每个毫秒都在变，调度器必须实时记账。500 路的成立依赖两个前提——槽位不空转、显存不浪费，而朴素的实现恰恰两样都做不到；前者是「调度」一节的问题，后者是「显存」一节的问题。

## 调度：batch 怎么组

### 静态 batch：槽位绑定整个生命周期

最朴素的组批方式沿袭自训练代码的习惯：引擎挑 $B$ 个请求绑成一个 batch，每步按固定形状 $(B,\ldots)$ 做一次前向，直到最长的请求生成完毕，整个 batch 才解散重组。batch 维上的每个位置是一个槽位（slot），请求一旦占住，这个位置就属于它到全组结束——哪怕它早就生成完毕，槽位每步照样参与计算（实现上通常填 padding token），KV Cache 也照常占着。

用一组典型数字看清代价：4 路请求，输出长度分别是 32、64、128、256 token。

```text
batch 总步数 = 256（最长者决定）
总槽位步 = 4 × 256 = 1024，其中有效 = 480
槽位利用率 ≈ 47%，平均在跑的只有 1.875 路
```

浪费体现在三个地方。最直接的是算力：53% 的槽位步在为已完成的请求计算 padding，$B=4$ 时 15 GB 的权重搬运本来只服务 4 个 token，再打对折，实际只服务了不到 2 个。其次是显存：已完成请求的 256 MiB KV 要等到整个 batch 解散才归还。最后是准入：第 33 步到达的第 5 路请求必须等 223 步，按每步约 3.4 ms 计，纯排队就接近 0.75 秒；如果最长的请求要输出 1000 token，等待会以秒计。这些时间全部花在第一个 token 之前，最终都计入 TTFT——静态 batch 下 TTFT 的长尾往往不是模型慢，而是在等前面的请求做完。

<figure class="text-center mt-3 mb-4">
  <img
    src="/assets/images/posts/llm-inference/batching-timeline.svg"
    alt="静态 batch 与 continuous batching 的时间线对比：静态 batch 中短请求完成后槽位空占直到最长请求结束，新请求排队等待；continuous batching 中完成的请求立即释放槽位，新请求下一步加入。"
    style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 1：静态 batch 把槽位绑定整个生命周期（上），continuous batching 按步重组 batch（下）。</figcaption>
</figure>

### continuous batching：按步重组 batch

Orca 把组批的粒度从请求的整段生命周期降到单个 decode step，称为 iteration-level scheduling，业界更常用的名字是 continuous batching：调度器每一步重新决定 batch 里有哪些请求，生成完毕的请求立即退出并归还 KV Cache，等待中的请求随即加入。[1] 上一小节的三层浪费因此同时消失。

但成员每一步都可能变化，batch 的形状就不再固定，这就引出一个新的实现问题：长度各异的请求，该怎么拼进同一个前向？Orca 给出的方案叫 selective batching，核心是按算子分别处理。[1] 区分的判据，是看请求各自的长度 $L_i$（或请求个数 $B$）出现在矩阵运算的哪个维度上。

线性层（QKV 投影、MLP）以及 Norm、激活这类逐 token 的算子，对每一行输入的处理方式完全相同，因此 $B$ 路请求各自新增的那个 token 可以直接堆叠成 $(B\times d)$，做一次 GEMM、读一遍权重；请求个数只出现在矩阵的行数上，而行数对 GEMM 来说是无关紧要的，4 行与 512 行并无本质区别。attention 则完全是另一回事：每路的 query 要与这路请求独有的历史 KV 配对，即 $(1\times128)\cdot(128\times L_i)$，四路上下文分别为 2100、350、1800、900 的请求，会产出四种宽度不同的分数行——$L_i$ 出现在乘法的内维和产出的宽度上，天然拼不成一个矩形张量。硬要拼，只能把所有请求 pad 到最长再 mask，以上面四路为例利用率只有约 61%，而且补进去的零同样占用带宽；训练代码采取的正是这种做法，因为训练时一个 batch 是固定的一组样本，同进同出，padding 加 mask 的矩形形状并无不妥。推理引擎则普遍使用支持变长序列（ragged / varlen）的 kernel，在一次调用里按各请求的实际长度分段处理。

| 算子 | 形状 | $L_i$ 出现的位置 | 拼批方式 |
| --- | --- | --- | --- |
| 线性层、逐元素算子 | $(B\times d)\cdot(d\times m)$ | 只影响行数（外维） | 直接堆叠，共享权重读取 |
| attention | $(1\times128)\cdot(128\times L_i)$ | 内维与产出宽度 | padding 或变长 kernel |

所以 selective batching 的含义是：形状统一的算子拼进 batch，让所有请求共享一次权重读取；形状依赖各路长度的算子按请求分别计算。推理服务不能直接复用训练代码，原因大半在这里。

### 新请求的加入：Prefill 的干扰与 chunked prefill

continuous batching 解决了请求的退出，还有加入这一半：新请求进入 batch 的第一件事不是 decode，而是一次完整的 Prefill。

2048 token 的 Prefill 是 30.8 TFLOPs 的计算量，理想下界 31 ms，计入效率因素后现实约 40–60 ms；而运行中 64 路请求的正常 decode 步只要 6.7 ms。Prefill 执行的这段时间里，本来可以完成 6 到 9 个 decode 步、每位用户多收 6 到 9 个 token，现在全部停滞——流式界面上，用户看到的是输出突然停住。长上下文的情形更严重：一个 32K token 的 RAG 请求，Prefill 约 1020 TFLOPs（attention 的 $L^2$ 项占大半），理想下界就超过 1 秒。一个新用户的 TTFT，就这样变成了所有已有用户的停顿。

调度器因此面对一个两难：Prefill 优先，新请求的 TTFT 好，但运行中用户的 TPOT 出现尖峰；Decode 优先，运行中用户的输出流畅，但新请求在队列里积压，高负载下 TTFT 失控。这并不是实现水平的问题——Prefill 落在第二篇 Roofline 的屋脊点右侧，Decode 在左侧，两种瓶颈属性相反的负载共享同一块 GPU，必然互相挤压。

chunked prefill 是目前普遍采用的缓解方案：按 token 预算把长 Prompt 切成若干片，每个 iteration 只处理一片、与 decode 步交错执行，使单个 iteration 的时长有界。[3] 以 512 token 一片为例，约 7.2 TFLOPs，理想耗时 7.3 ms，与一步 decode 同量级；停顿的性质随之改变——从一次 31 ms 的完全中断，变成每步从 6.7 到 14 ms 的均匀变慢，输出流不再中断，只是整体节奏放慢。三档切片的对比（理想口径）：

| 切片 | 新请求 TTFT | 运行中用户感受 | GEMM 效率 |
| --- | --- | --- | --- |
| 2048（不切） | ≈31 ms | 一次 31 ms 的完全中断 | 最高（M=2048） |
| 512 | ≈56 ms | 每步 6.7→14 ms | 仍接近峰值 |
| 128 | ≈138 ms | 每步 6.7→8.6 ms | M=128 有折损，iteration 翻四倍，开销摊薄不了 |

合适的切片大小可以直接估出来：让一片 prefill 的耗时约等于一步 decode，每个 iteration 就相当于把一次 decode 和一片 prefill 合并执行。

$$
\text{chunk}\times\frac{15\ \text{GFLOPs}}{989\ \text{TFLOPS}}\approx 6.7\ \text{ms}
\ \Rightarrow\ \text{chunk}\approx 440
$$

各引擎的默认预算多在几百到一两千 token，与这个量级一致。Sarathi-Serve 走得更远，把运行中请求的 decode token 直接拼进 prefill 片的 batch，做到 decode 不因新请求的加入而空转，即所谓 stall-free。[3] 各框架的具体策略留到框架篇比较。

## 显存：KV 怎么放

### 连续预留与按需分配的两种失败

500 路的上限要成立，显存必须接近零浪费。而几百路长度各异、随时生长、随时进出的 KV，用朴素的方式管理只有两种做法，各有各的失败方式。

按最大长度预留，是每路请求开工时划走 $8192\times128\ \text{KiB}=1\ \text{GiB}$ 的连续空间；实际平均用量不足一半，135 GB 只够约 135 路，预留的大半空间从不写入一个字节。按需分配，是用到多少申请多少；但长短不一的请求先后进出之后，显存里只剩下碎片——空闲总量往往够，最大的连续空洞却装不下新请求，显存名义上还有空闲，却已经放不进新的请求。这就是外部碎片。

两种做法的败因相同：都要求一个请求的 KV 是一整块连续显存。

### 分页：block 与 block table

操作系统在几十年前就遇到过同样的问题，答案是分页。vLLM 的 PagedAttention 把这套机制搬进了 KV 管理：[2] KV 显存被划成固定大小的 block（vLLM 默认 16 token 一块），每路请求持有一张 block table，记录自己的每个逻辑块对应哪个物理块，作用相当于进程的页表。逻辑上连续的 KV，物理上可以散布在显存各处。

```text
逻辑视图（请求以为自己是连续的）：
  token:  [ 0..15 ][16..31 ][32..47 ][48..63 ] ...
物理视图（实际散在各处）：
  block#:    77       5       142      61     ...
```

预留和碎片的问题一起消失。不再预留：请求只分第一个块，写满才申请下一块，浪费只剩最后一个未满的块，平均半块约 8 token，摊到 32 层约 1 MiB 一路。也不再碎片：所有块等大，任何请求释放的块都能被任何请求复用，「空闲但不连续」这个状态根本不会出现。vLLM 论文给出的对照是：此前系统的 KV 显存有效利用率只有 20.4%–38.2%，PagedAttention 做到 near-zero waste，同等延迟下吞吐比 FasterTransformer、Orca 等高 2–4 倍。[2]

块大小本身是个需要权衡的参数。块太小，block table 冗长，attention kernel 按块遍历的开销和访存的碎片化都会上升；块太大，等于退回到按块预留，1024 token 的块平均每路浪费 64 MiB。默认的 16 是两者的折中。[2]

<figure class="text-center mt-3 mb-4">
  <img
    src="/assets/images/posts/llm-inference/kv-paged-memory.svg"
    alt="KV Cache 显存管理三方案对比：连续预留造成预留浪费；按需分配造成外部碎片；分页管理用固定块与 block table 映射，两个请求的相同前缀共享物理块。"
    style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 2：朴素显存管理的两种失败与分页方案；块号即 block table 记录的物理块号，相同前缀共享物理块。</figcaption>
</figure>

### 共享与 prefix cache

分页结构还让「共享」第一次可以被表达：多张 block table 指向同一批物理块，以只读方式使用。这带来两个直接用途。

一是 copy-on-write。beam search 的多个候选共享完全相同的 Prompt 块，KV 只存一份；当某个候选要在一个共享块上追加写入时，才把这一块复制出来私有化，其余块照旧共享。没有分页结构，这种共享无从表达，只能每个候选各存一份完整 KV。

二是 prefix cache。第二篇讨论过为什么只缓存 K、V 而不缓存 Q：历史 K、V 是不可变的，写好即冻结。不可变意味着不但可以在同一时刻被多个请求共享，还可以跨时间留存复用。多轮对话进行到第 3 轮、携带 8K 历史时，引擎按前缀匹配直接挂上上一轮的块，命中的部分既不计算也不新占显存；没有这层缓存，这 8K 要重付约 150 TFLOPs（理想约 150 ms）的重复 Prefill。第一篇讲过的 Agent 负载——每轮把全部历史重发一遍——从中受益最大。

### 显存满时：准入与抢占

块池将尽时（运行中的请求还在生长，新请求还在到达），调度器可以选择不再放新请求进入，等有请求自然结束后再放行，这是准入控制；也可以把某一路暂停、收回它的全部块，这是抢占。被抢占请求的全部状态就是它的 KV，处置方式有两种。

recompute 是扔掉 KV，恢复时把整段序列（Prompt 加已生成的全部 token）当作新 Prompt 重新 Prefill，理想代价约 15 µs/token。swap 是把 KV 搬到 CPU 内存、恢复时再搬回，按 PCIe 5.0 x16 有效带宽 50 GB/s 计，128 KiB/token 约 2.6 µs，往返翻倍。

两笔账都正比于总序列长度，因为恢复需要全部上下文的 KV；差别只在系数，纯算术上 swap 几乎总占优。32K prompt 的 RAG 请求，recompute 约 1.0 秒，swap 往返约 172 ms；200 token prompt、已生成 2000 的对话请求，34 ms 对 11 ms。recompute 仍然存在的理由有两个。其一，prefix cache 能给重算打折：命中缓存的前缀部分免费，只需付真正要重算的尾部，而 swap 的搬运量没有折扣。其二，swap 要占用 CPU 内存，还要与其他传输竞争 PCIe 这块共享带宽。所以决策大致可以概括为：swap 的代价是总长乘固定系数，recompute 的代价取决于缓存命中后真正要重算的部分。

## 总账：并发开多大

把前面的零件装回每步时间模型（$L=2048$，理想口径）：

$$
T_{\mathrm{step}}(B)\approx
\max\left(\frac{16.1B\ \text{GFLOPs}}{989\ \text{TFLOPS}},
\ \frac{15.0+0.268B\ \text{GB}}{4.8\ \text{TB/s}}\right)
\approx 3.13+0.056B\ \text{ms}
$$

第一个结论关于拐点。令 KV 总流量与权重流量相等，$0.268B=15.0$，解得 $B\approx56$：在此之前，每步时间几乎完全由权重项决定，增加并发近乎免费，吞吐随并发近似线性上升；越过这个点之后，KV 项开始主导，每多一路请求，所有请求的每步都要多付约 56 µs。

第二个结论关于天花板。$B$ 趋于无穷时，吞吐趋于 $\frac{4.8\ \text{TB/s}}{268\ \text{MB}}\approx17.9\text{k}$ tok/s，一个只取决于每步 KV 字节数的数字。算力那边的天花板约有 61k tok/s，但在 $L=2048$ 下够不着，因为每增加一路请求的边际带宽成本（56 µs）始终大于边际算力成本（16.3 µs），两者相等要求 $L\approx600$。也就是说，上下文短于约 600 token 时，增加并发还有机会触及算力屋顶；长于它，瓶颈就只在 KV 带宽上，而 GQA 模型的常规上下文都落在这个范围里。

运营层面，服务方给定 TPOT 承诺之后，并发和吞吐就一起定了：

| TPOT SLO | 并发 B | 聚合吞吐 | 起约束作用的因素 |
| --- | --- | --- | --- |
| 10 ms | 123 | 12.3k | SLO 本身 |
| 20 ms | 301 | 15.0k | SLO 本身 |
| 31 ms | 500 | 16.1k | 显存容量 |
| 无限制 | → | 17.9k（渐近） | KV 带宽 |

这条曲线增益递减：SLO 从 10 放宽到 20 ms，只多换约 22% 的吞吐，最后一段吞吐最贵。起约束作用的因素也在切换：SLO 紧时是承诺本身，SLO 松时是显存容量，而 KV 带宽始终悬在头顶。SLO 由此成为运营决策而非物理常数——同一台机器，改一行配置，就可以从 123 路、每路 100 tok/s 的流畅体验，切换到 500 路、每路 32 tok/s 的大容量。生产系统据此引入 goodput 的视角：只统计满足 SLO 的请求的吞吐，把 SLO 与并发一起作为容量规划的输入。

最后用公开基准做一次现实检验，沿用第二篇引用的 NVIDIA 数据（H200、TensorRT-LLM、Llama 3.1 8B BF16、输入 2048/输出 128，基准方注明并非峰值口径）：[4]

| B | 理想 TPOT | 实测聚合吞吐 | 实测 TPOT |
| --- | --- | --- | --- |
| 1 | 3.2 ms | 173.8 tok/s | 5.75 ms |
| 8 | 3.6 ms | 803 tok/s | 10.0 ms |
| 64 | 6.7 ms | 1,680 tok/s | 38 ms |

形状与模型一致：吞吐随 B 次线性增长，TPOT 随 B 上升。但数值差距随 B 扩大——$B=1$ 时理想与实测差 1.8 倍（第二篇解释过的 $\eta$ 与开销），到 $B=64$ 已经 5.7 倍。理想模型假设 $\eta$ 是常数，现实里并发越大、有效带宽越差：KV 访问更零散，调度与采样开销更重，这个场景下 Prefill 还在持续混入。模型给出的是结构与趋势——权重摊销、KV 不摊销、约束切换——不是绝对数值。

## 小结

全文可以收成几句话。并发的上限是显存给的，KV 容量决定能同时跑多少路；吞吐的天花板是带宽给的，约等于带宽除以每步 KV 字节；调度的职责是不浪费，continuous batching 消除槽位的浪费，PagedAttention 消除显存的浪费，chunked prefill 把 Prefill 的干扰摊平；运营点由 SLO 在吞吐与延迟的前沿上选定。

这张账也预告了下一篇。每步字节的两项——权重 15.0 GB 与每路 256 MiB 的 KV——都是字节数，而天花板公式 $\frac{BW}{L\times S_{\mathrm{KV}}}$ 里，字节数就坐在分母上。量化减少的正是字节数：压缩权重削减每步的固定项，压缩 KV 同时抬高并发上限和吞吐天花板。第四篇见。

## 参考与延伸阅读

1. [Yu et al. — Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu)
2. [Kwon et al. — Efficient Memory Management for Large Language Model Serving with PagedAttention (SOSP 2023)](https://arxiv.org/abs/2309.06180)
3. [Agrawal et al. — Sarathi-Serve: Efficient LLM Inference by Taming Throughput-Latency Tension (OSDI 2024)](https://arxiv.org/abs/2403.02310)
4. [NVIDIA Model Optimizer — Inference benchmark examples](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/benchmark.md)
