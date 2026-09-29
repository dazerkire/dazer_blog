---
title: "LLM 推理系统（四）：量化究竟改变了什么——字节、算力与误差"
description: "从字节与算力两条线索分析量化的收益与适用负载，以及 scale、group size 与 outlier 如何决定量化误差。"
date: 2026-09-23 00:00:00 +0800
math: true
categories: [模型与系统, LLM 推理]
tags: [LLM, 在线推理, 量化, FP8, KV Cache]
---

上一篇结尾停在一个预告上：每步字节的两项，权重 15.0 GB 与每路 256 MiB 的 KV，都在吞吐天花板的分母上；量化减少的正是这些字节。这一篇把剩下的部分算完：压缩分别换来多少时间与显存，误差又从哪里进来。

讨论量化时最常见的说法是「INT4，压缩 4 倍」。这个 4 倍既不等于节省 4 倍显存，也不等于 4 倍加速：在仅比较主要读流量的理想模型中，同一次 W4 压缩，B=1 时加速比约 3.8，B=64 时约 1.5；对计算受限的 Prefill，单纯减少权重字节并不降低计算下界。差别不在压缩率，而在被压缩的字节位于每步字节公式的哪一项。

本文把量化的影响分成两条线索：**字节量**回答压缩的字节如何转化为时间收益；**计算峰值**回答哪些路线能同时抬高计算峰值。误差是第三个问题，它不出现在这两条线索里，却决定哪些方案可用。口径沿用第二、三篇：Llama 3.1 8B、H200 SXM、上下文 $L=2048$；FLOPs 与带宽用十进制前缀，显存容量用二进制前缀；除实测对照一节外均取理想口径（效率系数 $\eta=1$）。

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
| 主要读流量对应的时间下界 | 3.2 → 0.84 ms | 6.7 → 4.35 ms |
| 加速比 | **3.8x** | **1.5x** |

同样是 4 倍压缩，两种并发下的加速比相差 2.5 倍。规律是：**加速比的上限不由压缩率决定，而由被压缩项占总字节的比例决定**。B=1 时，BF16 权重占原始字节量约 98%；B=64 时约为 $15/(15+64\times0.268)=46.6\%$。18% 是 W4 压缩后的权重占比，不能用来解释压缩前的成本。若原始总字节中可压缩部分的占比为 $f$、压缩倍数为 $q$，且两种路径都受相同有效带宽限制，则：

$$
S_{\mathrm{bytes}}=\frac{1}{(1-f)+f/q}
$$

加速比由占比和压缩倍数共同决定，并不等于占比本身。即使把权重压到 0 字节，B=64 的加速比也只有 $32.2/17.2 \approx 1.9$ 倍，1.5x 已接近这一上限。

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/quant-speedup-vs-batch.svg" alt="主要读流量模型中，相对 BF16 的流量比随并发变化：仅 W4 从约 3.8 降至接近 1；压缩 KV 的曲线趋向 2。流量比不是实测加速比，容量另行约束。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 1：主要读流量的比值由压缩前的占比和压缩倍数共同决定。真实加速还受转换、算子效率与其他成本影响。</figcaption>
</figure>

这里有一个容易误读的细节：W4 之后算术强度从约 1 升到约 4 FLOPs/字节，仍在第二篇转折点 206 的左侧，Decode 依然是 memory-bound，但时间仍下降到约 1/4。Roofline 斜线区的下界是 $T = B/BW$：只有在主导访存项、有效带宽和其他开销都可比时，减少字节才会近似按比例缩短时间；真实反量化 kernel 还会改变执行效率。

W4 同时使「KV 字节追平权重」的拐点左移：BF16 时 $0.268B=15$ 给出 $B\approx56$；W4 之后 $0.268B=3.75$ 给出 $B\approx14$。量化没有让系统离开带宽瓶颈，只是让它更早进入 KV 主导区，而典型服务的并发通常远大于 14。

### KV 量化：同时进入两个分母的字节

第三篇假设可供 KV 使用的预算为 120 GB（约 111.8 GiB），在 $L=2048$ 下最多容纳 447 路完整的 BF16 KV。必须区分两个单位：每个**历史 token** 存储 128 KiB KV；生成一个新 token 时，需读取整段历史，主要读量为 $L\times128\ \text{KiB}=256\ \text{MiB}$。因此带宽吞吐上界是：

$$
\frac{4.8\times10^{12}}{2048\times131072}\approx17881\ \text{tok/s}
$$

把 KV 压成 FP8，每个历史 token 的数据从 128 KiB 降至 64 KiB；忽略 scale、保留高精度区和其他开销，容量上限约为 894 路，KV 带宽吞吐上界约为 35.8k tok/s。实际吞吐能否接近翻倍，还取决于是否先遇到计算或 kernel 效率限制。

权重项在 $B\to\infty$ 的每输出 token 成本中趋于零，因此不进入这个固定 $L$ 下的 KV 带宽渐近式。按前述全量 W4 假设释放 11.25 GB，并全部交给 KV，预算变成 131.25 GB，可容纳 488 路，约增加 9%。这只是显存预算算例，实际常驻权重大小、未量化层与 workspace 必须单独统计。

第三篇用「权重摊销、KV 不摊销」解释并发为什么有效；同一条规律在这里决定量化收益的分布：

| 量化对象 | 作用在哪一项 | 兑现成什么 |
| --- | --- | --- |
| 权重（W4 / FP8） | 固定项 | 低并发下的 TPOT；少量腾出显存（+9%） |
| KV Cache（FP8） | 不摊销项 | 理想 KV 容量与带宽上界（数据减半时各翻倍） |

### Prefill 与计算峰值：计算精度决定峰值

第二篇的因果有效 FLOPs 给出 2048-token Prefill 约 29.7 TFLOPs，H200 上纯计算下界约 30 ms。主要权重读流量对应约 3.1 ms。W4A16 将权重读量压缩，却仍使用 BF16 算术，因此不会降低这条计算下界；实际时间还受反量化、激活流量和算子效率影响。

低精度可以让硬件用不同的执行通路完成更多运算，但具体峰值是硬件设计的结果，不能仅由尾数位数推导。H200 的 dense 规格为 TF32 约 495、BF16 约 989、FP8 约 1979 TFLOPS。[10]

| 路径 | 字节量 | 计算通路 |
| --- | --- | --- |
| W4A16 | 理想权重字节为 BF16 的 1/4 | 解包、反量化后走 BF16，转换与资源占用另计 |
| FP8 W8A8 | 权重与激活的数据位宽减半 | 支持的矩阵乘可走 FP8 Tensor Core，峰值更高 |

低 batch Decode 的 Tensor Core 利用率低，也不意味着解包和反量化免费：它们会使用指令、寄存器、共享内存和带宽，并影响融合与占用率。高效实现可把这些成本部分隐藏，但需要测量确认。FP8 的计算峰值优势同样只适用于支持该通路的算子；attention、归一化和采样不会自动全部加速两倍。累加精度与中间归约方式也要以 kernel 配置为准，不能只由“FP8”三个字符判断。

### 端到端：两种负载的对比

下面比较单请求模型执行时间的简化估算，不含排队、网络和采样。采用因果 Prefill FLOPs；首 Token 由 Prefill 产生，随后只需 $M-1$ 次 Decode。第 $j$ 次 Decode 的上下文按 $L+j$ 计，保留增长的 KV 读取量：

$$
T_{\mathrm{model,ideal}}=\frac{F_{\mathrm{prefill}}(L)}{P}
+\sum_{j=1}^{M-1}\frac{W+(L+j)s_{\mathrm{KV}}}{BW}
$$

这里三种方案的 KV 都保留 BF16，$s_{\mathrm{KV}}=128$ KiB；W4 假设全部参与遍历的权重降至四分之一，FP8 权重减半、计算峰值翻倍。省略的算子与低精度转换开销使结果只能用于说明负载差异。

| 简化模型时间（ms） | 对话：2048 输入 / 128 输出 | RAG：32768 输入 / 200 输出 |
| --- | --- | --- |
| BF16 | 434 | 1548 |
| W4A16 | 137（约 3.18x） | 1081（约 1.43x） |
| FP8 W8A8 | 221（约 1.97x） | 863（约 1.79x） |

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/quant-end-to-end-workloads.svg" alt="两种负载的简化模型时间：因果 Prefill 加 M-1 次 Decode，三种方案均保留 BF16 KV；对话算例中 W4A16 较短，长输入 RAG 算例中 FP8 较短。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 2：同一组硬件与量化假设下，输入和输出长度改变收益分布。柱形是公式估算，不是端到端实测；W4A16 与 BF16 的计算峰值相同。</figcaption>
</figure>

这两个算例提示了三个需要测量的维度：Prefill 占比越高，低精度计算峰值越有价值；上下文越长，KV 读取越重；并发越高，权重摊销越充分。它们可以帮助缩小候选方案范围，不能单独决定线上选型。

## 误差的机制

### 仿射量化：舍入与裁剪

把浮点数映射到有限整数码，需要 scale、zero point 和码的取值范围。常见定义是：

$$
q=\operatorname{clip}\left(\operatorname{round}(w/a)+z,\ q_{\min},q_{\max}\right),\qquad
\hat w=a(q-z)
$$

$a>0$ 是浮点 scale，$z$ 是整数 zero point，使实数 0 映射到整数码 $z$。若用浮点偏移写成 $\hat w=aq+b$，则 $b=-az$，不能把这个浮点偏移直接当成整数 zero point。对称量化常取 $z=0$，非对称量化也广泛用于权重量化，需看具体方法和 kernel 支持。

在没有裁剪、scale 可精确表示的前提下，最近邻舍入误差满足：

$$
\lvert\hat w-w\rvert\le a/2
$$

超出码范围会出现裁剪误差；有限精度 scale 又会引入表示误差。这两者不受上述单纯舍入界约束。

用一个具体的例子验证。8 个权重，对称方案（$z=0$，码 −7…7），$a = 0.85/7 = 0.121$：

| $w$ | 0.12 | −0.85 | 0.31 | 0.02 | −0.44 | 0.67 | −0.03 | 0.58 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 码 | 1 | −7 | 3 | 0 | −4 | 6 | 0 | 5 |
| 误差 | +0.001 | 0 | +0.054 | −0.020 | −0.046 | **+0.059** | +0.030 | +0.027 |

实际最大误差 0.059，接近上界 $a/2 = 0.061$。误差最大的并不是极值：$a$ 按极值定义，在这个精确 scale 的例子中，定义 scale 的极值可被精确还原；落在两个码正中间的值舍入误差最大。

误差经过点积和多层网络后会怎样，不能只靠 $a/2$ 判断。若各项误差独立、零均值且方差受控，误差和的标准差才可能按 $\sqrt d$ 增长；要进一步说相对误差不随维度变化，还需要原始点积具有相应的随机性与尺度。真实激活中的相关性、outlier、信号抵消和非线性都会破坏这些假设。具体任务精度必须通过评测确认，不能从这个舍入界推出“固定几个百分点”的损失。

### outlier 与 group size

往上面那组数里加入第 9 个：$w_9 = +40$。$a$ 变成 $40/7 = 5.7$，码距扩大 47 倍。此时 $0.67/5.7 = 0.12$，取整后码为 0；逐个计算下去，原来 8 个权重的码全部为 0，还原值为 0，误差等于权重本身的大小。一个离群值使整组其余权重的信息完全丢失。

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/quant-error-mechanism.svg" alt="上图：在无裁剪、scale 精确的对称 INT4 示例中，最近邻舍入误差上界为 a/2，定义 scale 的极值可精确还原。下图：组内混入一个 +40 的 outlier 后，共享 scale 的码距扩大约 47 倍，其余 8 个权重全部落入码 0 的格、还原为 0。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 3：无裁剪、精确 scale 示例的舍入误差上界为 a/2（上）；组内出现 +40 的 outlier 后，共享 scale 的码距扩大 47 倍，其余 8 个权重全部落入码 0（下）。</figcaption>
</figure>

解决办法是分组：各组使用自己的 $a$。每 128 个数配一个 FP16 scale，元数据开销为 16 bit ÷ 128 = **0.125 bit/数**，「INT4 模型」实际是 4.125 bit：$7.5\text{B}\times 4.125/8 \approx 3.87$ GB，与 3.75 GB 的粗估同一量级。HuggingFace 上 INT4 模型 config 里的 `group_size: 128` 就是这个 128（llama.cpp 常用 32）。

`g` 的权衡在于：`g` 越小量化越准，元数据开销也越大。极端情形 `g=1` 下，若 scale 无限精确，可对非零值取 $a=\lvert w\rvert/7$、码取 ±7，使它精确还原；实际 FP16 scale 通常不能精确表示这个比值，仍有误差。每个数还需要 16 bit scale 加 4 bit 码，约 20 bit，比直接存 BF16 更多。这个例子说明共享 scale 能摊薄元数据成本：

> 分组量化让一组数共享 scale，不要求它们在原张量中一定相邻。$g$ 划定这个前提的范围：$g=\infty$ 时共享范围最大，也最容易被 outlier 破坏；$g$ 越小前提越可靠，元数据开销越高。实践中常用的折中是 32–128。

### 激活的 scale：静态校准与动态计算

权重固定，scale 可以离线计算；激活随输入变化，但它的 scale 不一定只能在线求得。静态方案用校准数据估计范围，推理时复用；动态方案在运行时按 tensor、token 或其他粒度计算。前者成本较低，但对校准分布外的输入更敏感；后者能适应当前范围，但要付归约和转换成本。

SmoothQuant 的不同配置就包含动态 per-token、动态 per-tensor 与静态激活 scale。[4] 动态量化也不一定产生独立 kernel 或额外完整显存读写：融合实现可把范围统计和量化接入相邻算子。W4A16 不量化激活，W8A8/FP8 则需要选择适合模型和后端的激活量化策略，这比“是否有一次独立同步”更准确。

### 真实 LLM 的 outlier 与三条路线

真实 LLM 的激活比上面的例子更极端：每层有固定的一小部分隐藏维（不到 1%），幅度是其他通道的 20–100 倍，且跨 Token 稳定；LLM.int8() 首先系统记录了这一现象 [3]。这些通道会主导 per-tensor 的 $a$（即上面 +40 例子的一般化情形）；即使 per-token scale 也受影响，因为每行的 max 被这些通道占据，整行的量化步长随之变大。

| 路线 | 做法 | 代表 |
| --- | --- | --- |
| 混合分解 | outlier 通道走 FP16、其余走 INT8 | LLM.int8() |
| 尺度迁移 | 等价变换 $X' = X/s,\ W' = W\cdot s$，输出不变，把量化难度从激活挪给权重 | SmoothQuant |
| 不量化激活 | 只压权重；AWQ 按激活幅度给关键权重加保护 scale，GPTQ 用 Hessian 误差补偿逐列量化 | W4A16 派 |

两类路线常用于不同的部署目标，但并非互斥的适用范围：**W4A16**（AWQ/GPTQ/llama.cpp）用于显存受限、单流与端侧部署；**W8A8/FP8**（vLLM/TensorRT-LLM on Hopper）用于在线服务。

### KV 的量化粒度

KV 位宽减半时，理想的数据容量与带宽上界都可改善，但误差需要单独验证。KIVI 根据其评测模型中 K、V 的分布差异，采用 K per-channel、V per-token 的非对称粒度，并保留一段高精度残余区。[6] 这不能概括所有模型及 RoPE 前后布局。FP8 KV 的 scale 粒度、校准方式与支持的 attention kernel 也依赖推理后端。

## 实测：理想与真实之间

第二、三篇都用同一份公开基准做过现实检验；对量化而言，这份基准恰好构成对照组：同一模型、同一硬件、同一负载（输入 2048 / 输出 128），BF16 与三种量化并排（H200、Model Optimizer v0.21.1、TensorRT-LLM v0.15、Llama 3.1 8B，输出吞吐 tok/s）[8]：

| B | BF16 | FP8 | INT4 AWQ（W4A16） | W4A8 AWQ |
| --- | --- | --- | --- | --- |
| 1 | 173.8 | 245.0（1.41x） | 231.8（1.33x） | 239.7（1.38x） |
| 8 | 803.1 | 1,051.2（1.31x） | 599.7（**0.75x**） | 801.7（1.00x） |
| 64 | 1,679.7 | 2,190.9（1.30x） | 1,392.8（**0.83x**） | 1,930.9（1.15x） |

这张表能说明在该版本与负载下，FP8 的整体吞吐较高，而 W4A16 在 B=8、64 时低于 BF16。它不能单独解释每一个差额来自哪里：输出吞吐不是 Decode TPOT，不能取倒数后减去带宽下界，进而断言存在固定 2.6 ms 杂项或 0.9 ms 反量化开销。

候选原因包括 Prefill 占比、低精度 kernel 的形状支持、解包与转换、有效带宽和调度。要分离这些因素，需要逐阶段计时、kernel profile 或受控消融。Marlin 等工作说明优化后的 W4A16 kernel 可以显著改善执行效率，但不能据此预测任意后端的结果。[7]

此表中的 **W4A8 AWQ 是 INT4 权重配 FP8 激活**，走 TensorRT-LLM 对应的 W4A8 路径；不能将其描述为 INT8 Tensor Core，也不能与后面的 INT4+INT8 QAT 实验混为一谈。[11]

同一来源另报 Llama 3.1 8B Instruct 的 MMLU loss：FP8 为 1.50%，INT4 AWQ 为 5.66%，W4A8 AWQ 为 6.00%。这里保留来源的指标名称与百分比写法；它们只适用于该模型和评测设置，不是四位量化必然损失 5% 的规律。

## 选型与取舍

量化方案按「误差由谁补偿、在什么时候补偿」分为两类。

**PTQ（Post-Training Quantization，训练后量化）**在预训练完成后转换模型，通常使用较少校准数据，不进行完整的量化感知再训练。它不只是选 scale，也不意味着权重数值保持不动：GPTQ 会用二阶信息补偿后续列的权重量化误差，AWQ 会调整通道尺度。[1][2] 所需数据、优化过程和耗时依算法与模型规模而定。

**QAT（Quantization-Aware Training，量化感知训练）**把量化放进训练循环：在网络中插入伪量化（fake quantization）节点，前向时模拟「量化再还原」引入的误差，反向时更新权重，使模型主动适应量化噪声（取整操作不可导，梯度通过直通估计近似传递）。它需要训练数据、训练算力与相应的工程投入，产出的是一个为量化优化过的模型。

两种路线的区别在于是否把量化误差纳入训练循环，而不是“是否修改过权重”。可以先评测成本较低的 PTQ；若目标位宽下任务质量不足，再比较更好的校准、混合精度、其他 PTQ 方法与 QAT。不能预先断言激活进入 8 bit 后 PTQ 就失效，或低于 4 bit 必须使用 QAT。

NVIDIA 给出的一个具体实验是 Llama 2 7B、samsum、Model Optimizer v0.11.0，基线已在目标数据上微调。[8]

| 量化设置 | BF16 验证损失 | PTQ | QAT |
| --- | --- | --- | --- |
| INT4 权重 + FP16 激活 | 1.036 | 1.059 | 1.044 |
| INT4 权重 + INT8 激活 | 1.036 | 3.321 | 1.294 |

它显示 QAT 改善了这两种配置，尤其是该 INT4+INT8 配置。结论不能扩展为所有 W8A8 或 FP8 PTQ 的效果，验证损失也不能直接换算成 MMLU 损失。

Hopper 上 FP8 有原生硬件支持和较宽的动态范围，是值得优先评测的方案；INT8、W4A16 仍有各自的硬件和 kernel 生态。FP8 仍需要 scale 管理与校准，并不自动解决所有 outlier。Blackwell 增加了原生低精度浮点路径，但支持什么格式、达到什么峰值，应查对应硬件和引擎规格。[9]

| 场景 | 优先评测的候选 | 验收重点 |
| --- | --- | --- |
| Hopper 在线服务 | FP8 W8A8，分别测试 KV FP8 | TTFT、TPOT、任务质量及不同 batch 的吞吐 |
| 单流、端侧、显存受限 | W4A16（GPTQ/AWQ 等） | 常驻显存、目标形状的 kernel 效率 |
| 长上下文、高并发 | 叠加低精度 KV | 长上下文质量、attention 支持与实际容量 |
| 更低位宽或严格质量目标 | 更强 PTQ、混合精度与 QAT | 校准/训练成本和真实任务回归 |

## 小结

量化作用于不同的成本项：权重压缩的收益取决于原始权重占比与压缩倍数；KV 压缩同时改变容量和每步历史读取量；原生低精度计算可能降低计算受限部分的时间。实际收益还取决于 kernel 效率、转换成本和请求形态。

误差同样需要分层分析：无裁剪、精确 scale 下有 $a/2$ 的舍入界，实际模型还包含裁剪、scale 表示误差及误差传播。分组和校准用于控制这些问题，最终仍要用目标任务评测决定方案是否可用。

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

10. [NVIDIA — H200 Tensor Core GPU（硬件规格）](https://www.nvidia.com/en-us/data-center/h200/)
11. [NVIDIA TensorRT-LLM — Quantization in TensorRT-LLM（W4A8 类型）](https://nvidia.github.io/TensorRT-LLM/blogs/quantization-in-TRT-LLM.html)
