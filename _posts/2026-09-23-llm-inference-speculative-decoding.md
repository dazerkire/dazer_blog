---
title: "LLM 推理系统（五）：投机解码如何突破逐 Token 串行——draft、验证与接受率"
description: "draft 模型先猜、target 模型一次验证：验证如何复用权重与历史 KV，接受率如何影响收益，以及严格修正采样如何保持目标分布。"
date: 2026-09-23 12:00:00 +0800
math: true
categories: [模型与系统, LLM 推理]
tags: [LLM, 在线推理, 投机解码, Speculative Decoding]
---

上一篇结尾留下的问题是：量化降低了每一步的成本，但没有改变「一步只产出一个 Token」。Decode 的串行依赖在模型内部：第 $t+1$ 个 Token 的输入包含第 $t$ 个 Token 的取值。批处理能并行不同请求，却不能并行同一条序列的相邻两步；KV Cache 免去了重算历史，也只是把每步做薄，一步还是只有一个 Token。

直接把多步并行也做不到。每个位置的输出分布必须由完整模型自己算出才算数，用别的东西近似，得到的就不再是这个模型的输出。**投机解码（speculative decoding）**绕开了这个矛盾：让一个小得多的**草稿模型（draft model）**先把 $k=4$ 个候选 Token 逐个猜出来，再让完整模型（下称 **target**，口径仍取 Llama 3.1 8B）用一次前向把候选全部验证。串行的部分转移给了便宜的 draft，昂贵的 target 一次处理多个位置。

本文沿用前几篇的口径：Llama 3.1 8B 与 Llama 3.2 1B，H200 SXM，上下文 $L=2048$，理想效率 $\eta=1$；实测对照一节再引入公开数据修正。

## 一轮的完整流程

一轮投机解码分两段。第一段，draft 以当前上下文为起点，自回归地生成 4 个候选 $t_1, t_2, t_3, t_4$：4 次 draft 前向，串行。第二段，target 做一次**验证前向（verification forward）**，输入是 5 个位置：

```text
[ 上下文末 Token, t1, t2, t3, t4 ]    ← 5 个位置，一次前向
```

为什么是 5 个位置而不是 4 个：判定 $t_i$ 需要的是「看见了 $t_1 \ldots t_{i-1}$」时 target 的分布，即位置 $i-1$ 的 logits；因果注意力下，要得到位置 4（$t_4$）的 logits，就必须把前面所有位置都过一遍。于是验证前向从上下文末 Token 开始，共 $4+1$ 个位置，一次读完。

前向得到 5 个位置的分布，用途各不相同：

- 位置 0–3 的分布分别用于判定 $t_1$–$t_4$：按接受规则保留或拒绝（规则在「输出分布不变性」一节给出）；
- 首个被拒的位置：从归一化的 $\max(0,p-q)$ 修正分布采样一个 Token，其后候选全部作废；这里 $p$、$q$ 分别是该位置的 target 与 draft 分布。直接从 $p$ 重采一般不能保持输出分布，证明见后文；
- 位置 4 的分布预测的是「$t_4$ 之后的下一个 Token」。若 4 个候选全部被接受，直接从它采样，第 5 个 Token 不需要再跑任何 target 前向。

位置 4 是这个设计的额外收益来源，通常称为 **bonus**：验证本身只需要位置 0–3 的分布，位置 4 的分布是因果前向的自然产物，跑到 $t_4$ 这一行的 logits 顺手就得到了。它不经受验证，直接从 target 分布采样，天然合规。全接受的一轮产出 5 个 Token；第一个候选就被拒的一轮也有修正采样兜底，产出 1 个。**一轮的产出区间是 $[1, 5]$，下限为 1 是设计保证，不是运气**：验证前向无论接受与否，至少兑现位置 0 的分布。

一个实现细节：验证前向对 5 个位置都计算了各自的 K、V 并写入缓存，因此被接受的候选不需要再前向补 KV；被拒绝的候选已写入的部分直接丢弃。

## 验证如何复用权重与历史 KV

验证前向与普通 B=1 的 Decode 一步相比，多算了 4 个位置。字节与 FLOPs 两侧分别是：

| 项 | 普通 Decode（1 位置） | 验证前向（5 位置） |
| --- | --- | --- |
| 权重字节 | 15.0 GB | 15.0 GB（读一遍，服务 5 行） |
| 历史 KV 的最低读量 | 0.268 GB | 理想复用下约 0.268 GB |
| FLOPs | 16.1 GFLOPs | $5 \times 16.1 \approx 80.5$ GFLOPs |
| 时间下界 | 3.2 ms | $\max(15.3/4.8,\ 80.5/989) \approx 3.2$ ms |

表中只保留权重和历史 KV 的主要读流量。多个位置可以复用这些数据，FLOPs 则约增至五倍；在这个简化模型中，计算项约 0.08 ms，仍低于约 3.2 ms 的带宽项。这说明低 batch 下有用额外计算换取数据复用的空间。

**相同的带宽下界不等于相同的实测时间。** 验证还会增加新 KV 写入、激活、多个位置的 logits、采样与回滚工作；历史 KV 是否只从 HBM 读取一次，也取决于 tiling、缓存和 kernel。整段前向取一次 max 只是宽松估计，不能保证增加位置没有成本。

要把一个容易混淆的点说清楚：GEMM 合并的是内存访问，不是算术。5 个位置各自都要与同一组权重做完整的乘加，一次乘法也不少；省下的是「为每个 Token 单独再读一遍 15 GB 权重」的字节项。第三篇「权重摊销」的规律在这里重现：跨请求复用与跨位置复用，经济结构相同。投机解码通常增加候选与验证工作；它的收益来自减少串行 target 调用，并提高每次读取数据所服务的有效输出数量。

draft 一侧的成本。Llama 3.2 1B：1.24B 参数约 2.5 GB（BF16），16 层、8 个 KV head、head dim 64，每 Token KV 为 32 KiB，2048 上下文共 64 MiB。每步字节与 8B 的口径同构：

$$
\frac{(2.5 + 0.067)\ \text{GB}}{4.8\ \text{TB/s}} \approx 0.53\ \text{ms/步}
$$

4 步共 2.14 ms。下面为便于比较，取 draft 步时 0.535 ms、验证 3.2 ms，建立固定长度、忽略其他开销的算例。它假定 draft 总跑满 4 步，验证总处理 5 个位置；真实步时和回滚成本仍会变化。

$$
T_{\text{round}} = 4 \times 0.535 + 3.2 \approx 5.34\ \text{ms}
$$

在这个算例中，全接受时一轮出 5 个 Token，模型内最大加速比为：

$$
\frac{5 \times 3.2}{5.34} \approx 3.0\text{x}
$$

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/spec-round-timeline.svg" alt="普通路径 5 步各 3.2 ms 共 16 ms；投机路径 draft 4 步各 0.535 ms 加一次 3.2 ms 的验证前向共 5.34 ms，同样产出 5 个 token。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 1：同一时间轴下，普通 Decode 出 5 个 Token 需 16 ms；投机一轮（全接受）draft 4 步加一次验证共 5.34 ms。固定步时算例的全接受加速比约 3.0x，不代表真实执行的上限。</figcaption>
</figure>

## 收益公式：接受率决定落点

全接受是上界，实际每轮的产出是随机的。设每个位置的接受概率独立、同为 $\alpha$（理想化，其确切含义在下一节落定），一轮产出的 Token 数 $N$ 取值 1–5。逐项写尾概率：

```text
P(N ≥ 1) = 1        拒绝也有修正采样兜底
P(N ≥ 2) = α        第 1 个候选被接受
P(N ≥ 3) = α^2
P(N ≥ 4) = α^3
P(N ≥ 5) = α^4      4 个全接受，加 bonus
```

由尾概率求和 $E[N] = \sum_i P(N\ge i)$：

$$
E[N] = 1 + \alpha + \alpha^2 + \alpha^3 + \alpha^4 = \frac{1-\alpha^{5}}{1-\alpha}
$$

两个极端可以校验：$\alpha=1$ 时 $E[N]=5$，$\alpha=0$ 时 $E[N]=1$，都与机制一致。在固定步时算例中，一轮耗时取 5.34 ms，接受率影响的是单位 Token 需要的轮数，因此：

$$
\text{每 Token 耗时} = \frac{5.34}{E[N]},\qquad \alpha = 0.8 \text{ 时 } E[N]=3.36 \Rightarrow 1.59\ \text{ms/token}
$$

对比普通 Decode 的 3.2 ms/token，加速比 $2.0$x。$\alpha=0.8$ 落在两个极端之间（上界 3.0x、反向 0.6x），这里只是演示参数，不代表 Llama 3.2 1B 与 Llama 3.1 8B 的实测接受率。

反解临界值：加速比为 1 要求 $E[N] = 5.34/3.2 \approx 1.67$，即

$$
1+\alpha+\alpha^2+\alpha^3+\alpha^4 = 1.67 \ \Longrightarrow\ \alpha^{\ast} \approx 0.41
$$

直观读法：一轮的成本是普通 Decode 一步的 1.67 倍，平均每轮至少要兑现 1.67 个 Token 才能覆盖成本。$\alpha$ 低于 0.41 时，在这个固定步时模型中，额外产出不足以覆盖 draft 成本，投机比普通路径更慢。

$k$ 也不是定死的 4。同样的口径下，$k=2$ 与 $k=8$ 的一轮成本分别为 $3.2+2\times0.535=4.27$ ms 与 $3.2+8\times0.535=7.48$ ms，期望产出相应变为 $(1-\alpha^3)/(1-\alpha)$ 与 $(1-\alpha^9)/(1-\alpha)$：

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/spec-speedup-vs-alpha.svg" alt="固定步时算例中，k=2、4、8 的收益随相同条件接受率变化；真实产出应使用前缀存活概率，真实时间需按配置测量。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 2：固定步时算例中，加速比由 $\alpha$ 与 $k$ 共同决定。$k$ 越大上界越高（2.3x / 3.0x / 3.9x），盈亏平衡的 $\alpha$ 也越晚（0.26 / 0.41 / 0.57）；$\alpha=0.8$ 附近 $k=4$ 已优于 $k=8$。</figcaption>
</figure>

三条曲线只在固定步时和同一 $\alpha$ 假设下比较 $k$。例如 $\alpha=0.8$ 时，$k=4$ 优于 $k=8$：后四步 draft 的成本超过额外期望产出的收益。实际应测每个位置在前缀已接受条件下的接受概率，以及不同长度的 draft、验证时间。

不要求独立同分布时，更一般的表达是：

$$
E[N]=1+\sum_{i=1}^{k}\Pr(A_1\cap\cdots\cap A_i)
$$

其中 $A_i$ 是第 $i$ 个候选被接受的事件。若把 $\alpha_i$ 定义为前 $i-1$ 个候选都接受时的条件接受率，则对应项为 $\prod_{j=1}^{i}\alpha_j$。直接代入全局平均接受率可能高估，也可能低估；不存在“独立同分布一定乐观”的普遍结论。

启动、采样、同步和回滚会影响每轮总时间，但不能给每次前向机械加上同一个固定开销。要把这张估算曲线用于部署，需分别测量 target 普通步、draft 段和验证段，并保留对应的 batch 与上下文条件。

## 输出分布不变性

随机采样下，接受规则不能简单替换为 top-1 比对，而应采用一套保证输出分布严格等于 target 分布的采样程序。设某位置上 draft 的分布为 $q(x)$，target 的分布为 $p(x)$，$x$ 遍历词表。

draft 抽中 $x$ 时，接受概率取

$$
A(x) = \min\!\left(1,\ \frac{p(x)}{q(x)}\right)
$$

这个形式是逐点最优的：接受路径要求 $q(x)A(x) \le p(x)$，即 $A(x) \le p(x)/q(x)$，同时 $A \le 1$；取 min 是在每个 $x$ 上都取到合法上限。经此路径，$x$ 的输出质量为 $\min(p(x), q(x))$，对全词表求和定义总接受率：

$$
\alpha = \sum_x \min\!\left(p(x),\ q(x)\right)
$$

拒绝不是丢弃，而是重采：从 $\max(0,\ p(x)-q(x))$ 归一化后的分布抽取。归一化常数为

$$
Z = \sum_x \max\!\left(0,\ p(x)-q(x)\right) = \sum_x\Big[p(x)-\min(p(x),q(x))\Big] = 1-\alpha
$$

当 $\alpha=1$ 时从不进入拒绝分支，无需构造修正分布；当 $\alpha<1$ 时，合并两条路径，输出 $x$ 的概率为

$$
\underbrace{\min(p(x),q(x))}_{\text{接受}} + \underbrace{(1-\alpha)\cdot\frac{\max(0,\ p(x)-q(x))}{1-\alpha}}_{\text{拒绝后重采}} = \min(p,q)+\max(0,\ p-q) = p(x)
$$

最后一步用了恒等式 $\min(a,b)+\max(0,\ a-b)=a$。接受路径走到哪里，重采路径恰好补齐缺口，词表上每个 $x$ 严格等于 $p(x)$。这就是「无损加速」的准确含义：逐 Token 的输出分布与 target 自己一步步解码完全一致，这是算法在精确概率运算和正确实现下的保证；浮点误差、随机数使用方式和采样器实现仍需检验。相同输出分布也不要求相同随机种子下逐字相同。

$\alpha$ 的含义也由此落定。由定义与 $\sum_x(p-q)=0$：

$$
\alpha = \sum_x \min(p(x),q(x)) = 1 - \mathrm{TV}(p,q),\qquad
\mathrm{TV}(p,q)=\frac{1}{2}\sum_x \lvert p(x)-q(x)\rvert
$$

$\mathrm{TV}$ 是总变差距离，逐词表求和度量两个分布的差异。这里的接受率由当前位置的两个条件分布决定：$q=p$ 时全接受，分布不相交时全拒；draft 越接近 target，TV 越小，$\alpha$ 越高。

$\alpha$ 还逐位置、逐负载地变化：两模型的条件分布越接近，$\alpha$ 越高；分布本身的不确定性并不能单独决定接受率。全局平均值也不能替代前缀存活概率。

贪心解码是这个框架的极限情形，不变性不但成立，形式还更强。采样温度为 $T$ 时，验证作用于重标定的分布 $p_T(x)\propto p(x)^{1/T}$，上述证明对每个 $T$ 都成立；$T\to 0$ 时 $p_T$ 收缩为 $\arg\max$ 上的点质量，接受概率退化为「$t_i$ 是否等于 target 的 $\arg\max$」，拒绝后的重采退化为直接取该位置的 $\arg\max$，在相同的 argmax 并列处理规则下，整套概率机制变成逐位置的确定性比对。归纳可得每步输出都与普通贪心解码相同：采样基线下证明保证的是输出分布相等，贪心基线下保证的是输出序列逐位相同，后者更强。贪心口径的接受率也因此是另一个量，即 draft 与 target 的 top-1 一致率，不能与随机采样下的分布重叠量直接比较；实测一节温度为 0 的数据测的正是这个口径。对 top-p、top-k 采样，规则作用于截断重归一后的分布，结论同样成立。

## 实测对照：接受率与 draft 选型

两篇原始论文给出了系统的实测。Leviathan 等在 T5-XXL（11B）上以 batch=1、单个 TPU-v4、英译德（EnDe）任务上测量了不同 draft 的接受率与加速比（$\gamma$ 即本文的 $k$）[1]：

| draft | 温度 | $\gamma$ | $\alpha$ | 实测加速比 |
| --- | --- | --- | --- | --- |
| T5-small（77M） | 0 | 7 | 0.75 | **3.4x** |
| T5-small（77M） | 1 | 7 | 0.62 | 2.6x |
| T5-base（250M） | 0 | 7 | 0.80 | 2.8x |
| T5-base（250M） | 1 | 5 | 0.68 | 2.4x |
| T5-large（800M） | 0 | 7 | 0.82 | 1.7x |
| T5-large（800M） | 1 | 3 | 0.71 | 1.4x |

这张表说明 draft 大小、接受率与执行成本需要一起看。T5-large 的接受率更高，但该任务的加速比低于 T5-small；更大的 draft 不一定划算。该实验中的时间比不能按参数量直接外推到另一组模型和硬件。

温度和任务也会改变结果。表中从温度 0 改成 1 后接受率下降，但这只是这些配对与任务的观测，不是“温度越高，TV 必然越大”的定律。极高温度下两个完整词表分布都可能趋向均匀，距离反而缩小。模型是否同家族也不能保证某个接受率。

论文的解析估计与测量会有差距，有时高估、有时低估，不能用一个固定折扣覆盖全部设置。Chen 等在 Chinchilla 70B 上报告的 2–2.5x 来自另一组模型、硬件与任务，只能作为独立实例。[2]

Leviathan 还讨论了 bigram draft：在 $\alpha\approx0.2$、draft 时间近似为零的分析中，收益可接近 $1/(1-0.2)=1.25$。有限 $k=3$ 时公式给出 1.248；这不是一个独立测得的“$k=3$ 实测 1.25x”实验，不能用它反过来验证公式。候选可以不来自神经网络，这一点会在下一篇展开。

## 选型与取舍

| 场景 | 优先检查 | 判断依据 |
| --- | --- | --- |
| 单流、低并发 | draft 段与验证段是否足够便宜 | 实测每轮时间与有效产出 |
| 不同 tokenizer 的模型 | 引擎是否支持对齐与重分词 | 基础算法要求同一 token 空间；跨 tokenizer 需扩展实现 |
| 高并发服务 | 验证行数、资源竞争和容量 | 单流收益不能直接外推，见第七篇 |
| 换任务或采样参数 | 重新测前缀接受概率 | 旧负载的接受率可能不再适用 |

同 tokenizer 是本文基础实现的假设，不是所有投机方法的硬限制。Universal Assisted Generation 等实现可通过重分词和对齐支持不同 tokenizer，但成本和支持的采样规则必须另查。[6] 同家族小模型可作为候选起点，尺寸和 $k$ 则应在目标负载上搜索，不能规定通用的 1/10 比例或默认长度。

投机主要优化首 Token 之后的生成。若 draft 的 Prefill 与 target 争用资源、首次起草阻塞输出，或调度策略改变排队，TTFT 也可能变化；应将 TTFT、请求 TPOT 和 ITL 分位数一起测量。

## 小结

投机解码把候选生成交给较便宜的路径，再用 target 批量验证。固定步时算例取 $k=4$、$T_{\mathrm{round}}=5.34$ ms；在各位置条件接受率均为 0.8 时，$E[N]=3.36$，得到约 2.0x 的估算收益。这些数字是模型内的比较，不是该 Llama 配对的实测结论。

严格无损采样要求同时使用 $\min(1,p/q)$ 的接受概率与归一化的 $\max(0,p-q)$ 修正分布。实际收益则取决于有效产出是否足以覆盖 draft、验证与其他执行成本；带宽复用创造了机会，但不保证验证行数没有代价。

本篇假设候选来自一个独立的小模型，这只是设计空间的一角。候选可以来自跳层自推（self-speculative）、多头预测（Medusa）、树形草稿与并行验证（EAGLE），甚至来自 n-gram 统计；它们改变的是「候选从哪来」，保持目标分布的方案仍需正确的接受与修正规则，近似接受方案则应明确标注。下一篇展开这个设计空间。

## 参考与延伸阅读

1. [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
2. [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
3. [Zhang et al. — Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://arxiv.org/abs/2309.08168)
4. [Cai et al. — Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
5. [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)

6. [Hugging Face — Universal Assisted Generation](https://huggingface.co/blog/universal_assisted_generation)
