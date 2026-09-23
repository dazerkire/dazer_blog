---
title: "LLM 推理系统（五）：投机解码如何突破逐 Token 串行——draft、验证与接受率"
description: "draft 模型先猜、target 模型一次验证：验证前向为什么近乎免费，接受率如何决定加速比，以及输出分布为何严格不变。"
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
- 首个被拒的位置：从 target 在该位置的分布重采一个 Token 作为修正，其后候选全部作废；
- 位置 4 的分布预测的是「$t_4$ 之后的下一个 Token」。若 4 个候选全部被接受，直接从它采样，第 5 个 Token 不需要再跑任何 target 前向。

位置 4 是这个设计的额外收益来源，通常称为 **bonus**：验证本身只需要位置 0–3 的分布，位置 4 的分布是因果前向的自然产物，跑到 $t_4$ 这一行的 logits 顺手就得到了。它不经受验证，直接从 target 分布采样，天然合规。全接受的一轮产出 5 个 Token；第一个候选就被拒的一轮也有修正采样兜底，产出 1 个。**一轮的产出区间是 $[1, 5]$，下限为 1 是设计保证，不是运气**：验证前向无论接受与否，至少兑现位置 0 的分布。

一个实现细节：验证前向对 5 个位置都计算了各自的 K、V 并写入缓存，因此被接受的候选不需要再前向补 KV；被拒绝的候选已写入的部分直接丢弃。

## 验证为什么近乎免费

验证前向与普通 B=1 的 Decode 一步相比，多算了 4 个位置。字节与 FLOPs 两侧分别是：

| 项 | 普通 Decode（1 位置） | 验证前向（5 位置） |
| --- | --- | --- |
| 权重字节 | 15.0 GB | 15.0 GB（读一遍，服务 5 行） |
| 上下文 KV | 0.268 GB | 0.268 GB（读一遍） |
| FLOPs | 16.1 GFLOPs | $5 \times 16.1 \approx 80.5$ GFLOPs |
| 时间下界 | 3.2 ms | $\max(15.3/4.8,\ 80.5/989) \approx 3.2$ ms |

字节项完全不变：GEMV 变 GEMM，行数从 1 到 5，权重仍然只读一遍；5 个位置同属一条序列，共享同一段上下文 KV，也只读一遍。FLOPs 项实打实涨了 5 倍，但只有 0.08 ms。这一步的算术强度约为 $80.5/15.3 \approx 5.3$ FLOPs/字节，深在第二篇转折点 206 的左侧，时间由字节决定。**验证 5 个位置的时间与普通 Decode 一步几乎相同**，这句话就是整个方法的经济基础。

要把一个容易混淆的点说清楚：GEMM 合并的是内存访问，不是算术。5 个位置各自都要与同一组权重做完整的乘加，一次乘法也不少；省下的是「为每个 Token 单独再读一遍 15 GB 权重」的字节项。第三篇「权重摊销」的规律在这里重现：跨请求复用与跨位置复用，经济结构相同。投机解码没有省任何计算量，它省的是访存。

draft 一侧的成本。Llama 3.2 1B：1.24B 参数约 2.5 GB（BF16），16 层、8 个 KV head、head dim 64，每 Token KV 为 32 KiB，2048 上下文共 64 MiB。每步字节与 8B 的口径同构：

$$
\frac{2.5 + 0.067}{4.8\ \text{TB/s}} \approx 0.53\ \text{ms/步}
$$

4 步共 2.14 ms。一轮的总成本由此确定，且与接受结果无关：draft 总是跑满 4 步，验证前向总是 5 个位置。

$$
T_{\text{round}} = 4 \times 0.53 + 3.2 \approx 5.34\ \text{ms}
$$

全接受时一轮出 5 个 Token，得到加速比的上界：

$$
\frac{5 \times 3.2}{5.34} \approx 3.0\text{x}
$$

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/spec-round-timeline.svg" alt="普通路径 5 步各 3.2 ms 共 16 ms；投机路径 draft 4 步各 0.53 ms 加一次 3.2 ms 的验证前向共 5.34 ms，同样产出 5 个 token。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 1：同一时间轴下，普通 Decode 出 5 个 Token 需 16 ms；投机一轮（全接受）draft 4 步加一次验证共 5.34 ms。上界约 3.0x，来自 draft 的 2.14 ms 是必须预付的成本。</figcaption>
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

两个极端可以校验：$\alpha=1$ 时 $E[N]=5$，$\alpha=0$ 时 $E[N]=1$，都与机制一致。一轮耗时是常数 5.34 ms，接受率影响的是单位 Token 需要的轮数，因此：

$$
\text{每 Token 耗时} = \frac{5.34}{E[N]},\qquad \alpha = 0.8 \text{ 时 } E[N]=3.36 \Rightarrow 1.59\ \text{ms/token}
$$

对比普通 Decode 的 3.2 ms/token，加速比 $2.0$x。$\alpha=0.8$ 落在两个极端之间（上界 3.0x、反向 0.6x），是同家族小模型配对时常见的区间。

反解临界值：加速比为 1 要求 $E[N] = 5.34/3.2 \approx 1.67$，即

$$
1+\alpha+\alpha^2+\alpha^3+\alpha^4 = 1.67 \ \Longrightarrow\ \alpha^{\ast} \approx 0.41
$$

直观读法：一轮的成本是普通 Decode 一步的 1.67 倍，平均每轮至少要兑现 1.67 个 Token 才不亏。$\alpha$ 低于 0.41 时，draft 的 2.14 ms 是纯支出，投机比不投机更慢。

$k$ 也不是定死的 4。同样的口径下，$k=2$ 与 $k=8$ 的一轮成本分别为 $3.2+2\times0.53=4.26$ ms 与 $3.2+8\times0.53=7.44$ ms，期望产出相应变为 $(1-\alpha^3)/(1-\alpha)$ 与 $(1-\alpha^9)/(1-\alpha)$：

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/spec-speedup-vs-alpha.svg" alt="不同候选数 k（k=2、k=4、k=8）下加速比随接受率 α 的变化：盈亏平衡 α 分别约 0.26、0.41、0.57；α=0.8 时三者分别约 1.8x、2.0x、1.9x；上界分别为 2.3x、3.0x、3.9x。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 2：加速比由 $\alpha$ 与 $k$ 共同决定。$k$ 越大上界越高（2.3x / 3.0x / 3.9x），盈亏平衡的 $\alpha$ 也越晚（0.26 / 0.41 / 0.57）；$\alpha=0.8$ 附近 $k=4$ 已优于 $k=8$。</figcaption>
</figure>

三条曲线给出选 $k$ 的依据。$\alpha$ 低于约 0.7 时 $k=2$ 最优，高于约 0.85 才轮到 $k=8$；$\alpha=0.8$ 时 $k=4$（2.0x）已经压过 $k=8$（1.9x），因为多出的 4 步 draft 成本超过了多接受的候选。图还系统性地高估了大 $k$：独立同分布的 $\alpha$ 假设对后半段候选偏乐观，draft 越猜越远，第 $i$ 个位置的接受率随 $i$ 递减，$k=8$ 的尾部候选实际更少被接受。实践中 $k$ 取 3–7、多数实现默认 4，正是这些效应折中的结果。

以上都是理想下界。第二篇建立的 $T_{\mathrm{other}}$ 在这里被放大：一轮要执行 5 次 kernel 序列（4 次 draft 加 1 次验证），每次都付一遍启动、采样与同步，draft 层数少，杂项在步时中的占比反而更高。理想模型的 2.0x 因此偏乐观，幅度见实测一节。

## 输出分布不变性

接受规则不是「top-1 一致就接受」这类启发式，而是一套保证输出分布严格等于 target 分布的采样程序。设某位置上 draft 的分布为 $q(x)$，target 的分布为 $p(x)$，$x$ 遍历词表。

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

合并两条路径，输出 $x$ 的概率为

$$
\underbrace{\min(p(x),q(x))}_{\text{接受}} + \underbrace{(1-\alpha)\cdot\frac{\max(0,\ p(x)-q(x))}{1-\alpha}}_{\text{拒绝后重采}} = \min(p,q)+\max(0,\ p-q) = p(x)
$$

最后一步用了恒等式 $\min(a,b)+\max(0,\ a-b)=a$。接受路径走到哪里，重采路径恰好补齐缺口，词表上每个 $x$ 严格等于 $p(x)$。这就是「无损加速」的准确含义：逐 Token 的输出分布与 target 自己一步步解码完全一致，任何统计检验都无法区分。

$\alpha$ 的含义也由此落定。由定义与 $\sum_x(p-q)=0$：

$$
\alpha = \sum_x \min(p(x),q(x)) = 1 - \mathrm{TV}(p,q)
$$

其中 $\mathrm{TV}(p,q)=\frac{1}{2}\sum_x|p(x)-q(x)|$ 是总变差距离。**接受率不是工程调参的结果，它就是两个分布在当前位置的距离**：$q=p$ 时全接受，分布不相交时全拒；draft 越接近 target，TV 越小，$\alpha$ 越高。它还逐位置、逐负载地变化：分布集中且两模型一致的位置 $\alpha$ 高，真正不确定的位置 $\alpha$ 低。前面「独立同分布」的理想化因此是偏乐观的，实际收益落在理论曲线之下。

贪心解码是这个框架的极限情形，不变性不但成立，形式还更强。采样温度为 $T$ 时，验证作用于重标定的分布 $p_T(x)\propto p(x)^{1/T}$，上述证明对每个 $T$ 都成立；$T\to 0$ 时 $p_T$ 收缩为 $\arg\max$ 上的点质量，接受概率退化为「$t_i$ 是否等于 target 的 $\arg\max$」，拒绝后的重采退化为直接取该位置的 $\arg\max$，整套概率机制变成逐位置的确定性比对。归纳可得每步输出都与普通贪心解码相同：采样基线下证明保证的是输出分布相等，贪心基线下保证的是输出序列逐位相同，后者更强。贪心口径的接受率也因此是另一个量，即 draft 与 target 的 top-1 一致率，通常高于采样口径的 $\alpha$；实测一节温度为 0 的数据测的正是这个口径。对 top-p、top-k 采样，规则作用于截断重归一后的分布，结论同样成立。

## 实测对照：接受率与 draft 选型

两篇原始论文给出了系统的实测。Leviathan 等在 T5-XXL（11B）上以 batch=1、单卡 TPU-v4 测量了不同 draft 的接受率与端到端加速比（$\gamma$ 即本文的 $k$）[1]：

| draft | 温度 | $\gamma$ | $\alpha$ | 实测加速比 |
| --- | --- | --- | --- | --- |
| T5-small（77M） | 0 | 7 | 0.75 | **3.4x** |
| T5-small（77M） | 1 | 5 | 0.62 | 2.6x |
| T5-base（250M） | 0 | 7 | 0.80 | 2.8x |
| T5-base（250M） | 1 | 5 | 0.68 | 2.4x |
| T5-large（800M） | 0 | 7 | 0.82 | 1.7x |
| T5-large（800M） | 1 | 3 | 0.71 | 1.4x |

这张表对前几节的三处理想化各给了一条修正。

**draft 不是越大越好。** T5-large 的 $\alpha$ 最高（0.82），加速比却最低：draft 步时随参数量线性增长（论文中 draft 与 target 的步时比 $c$ 从 T5-small 的 0.02 升到 T5-large 的 0.11），接受率的一点提升抵不过步时的抬升。同一论文在 LaMDA 137B 上的配对更直接：draft 从 2B 换到 8B，$\alpha$ 只从 0.71 到 0.75。对照本文口径：1B draft 的步时比 $c = 0.53/3.2 \approx 0.17$，已与 T5-large 的量级相当；若换 3B draft（步时约 1.35 ms），一轮成本升至约 8.6 ms，临界 $\alpha$ 从 0.41 抬到约 0.69，同家族 1B 换 3B 的接受率提升很难覆盖这个抬升。**draft 应当选得小**：接受率靠分布相近，不靠容量。

**$\alpha$ 是负载属性，不是模型属性。** 采样温度从 0 升到 1，六组配置的 $\alpha$ 全部下降约 0.1：温度越高分布越平，draft 与 target 的 TV 越大。换任务（摘要 vs 翻译）同样改变 $\alpha$。为一种负载测出的接受率不能直接外推到另一种。

**理想公式要打折。** 论文自己的预测公式在 T5-small 上给出 3.2x、实测 3.4x，几乎吻合；在 T5-large（$c=0.11$）上预测 2.5x、实测 1.7x，缺口显著。draft 步数越多、每步越贵，kernel 启动等杂项吃掉理想值的比例越大，与第四篇 $T_{\mathrm{other}}$ 不随精度缩水的教训同源。DeepMind 在 Chinchilla 70B 上的分布式实验报告 2–2.5x，也落在理想模型打完折的区间 [2]。

同一组实验还留了一个值得记住的数字：不用神经网络、只用 bigram 统计当 draft，$\alpha$ 仅约 0.2，$\gamma=3$ 时仍测得 1.25x。候选不必来自一个更小的模型，这把问题推向了更广的设计空间。

## 选型与取舍

| 场景 | 判断 | 依据 |
| --- | --- | --- |
| 单流、低并发（本篇口径） | 收益明确 | 验证近乎免费；同家族配对 $\alpha$ 约 0.6–0.8，典型 1.5–2x |
| draft 与 target 不同家族 | 一般无收益 | tokenizer 不一致无从验证；$\alpha$ 常低于临界值 |
| 高并发、满 batch 服务 | 常为负收益 | 算力瓶颈下验证的行数开始计价（第七篇展开） |
| 高采样温度、开放生成 | 收益缩水 | $\alpha$ 随温度下降约 0.1 |

三条经验法则。其一，draft 与 target 同 tokenizer 是硬约束，同家族（同训练数据分布）是软约束，后者决定 $\alpha$ 的量级。其二，draft 尺寸取 target 的 1/10 上下：再小则 $\alpha$ 掉，再大则步时吃掉收益；「8B 配 1B、70B 配 8B」的社区惯例与实测数据一致。其三，$k$ 取 3–7 并按 $\alpha$ 微调：$\alpha$ 高加一两个候选，$\alpha$ 低坚决减。

TTFT 基本不受影响：首 Token 仍由常规路径产出，投机的作用对象是 TPOT 与 ITL。输出长的负载收益更稳定，轮数越多，draft 的固定成本均摊得越充分。

## 小结

投机解码没有删除串行依赖，而是把它搬到了便宜的地方：draft 的 4 步仍然串行，但每步只付 0.53 ms；target 的时间从「每个 Token 一次前向」变为「每轮一次前向的 5 行」，每轮固定 5.34 ms，产出 0–4 个被接受的候选，外加 1 个修正或 bonus 的 Token。收益公式因此只有三个变量：接受率 $\alpha$ 决定 $E[N]=(1-\alpha^5)/(1-\alpha)$，draft 步时决定一轮成本，$k$ 在两者之间权衡。同家族配对实测 $\alpha$ 约 0.6–0.8，对应 1.5–2x；临界 $\alpha$ 约 0.41，低于它投机就是纯支出。输出分布不变性由修正采样保证：接受概率取 $\min(1, p/q)$、拒绝后从 $\max(0, p-q)$ 重采，边缘分布逐 Token 等于 target 自身，而 $\alpha = 1-\mathrm{TV}(p,q)$ 把接受率还原为两个分布的距离。

它也没有违反「每个位置的分布必须由 target 计算」：所有位置都是 target 算的，只是组织方式从 5 次串行前向变成 1 次前向的 5 行。方法成立的前提是 Decode 停在带宽瓶颈上：行数不计价，验证近乎免费；这个前提在高并发服务里不再成立，第七篇回到这个问题。

本篇假设候选来自一个独立的小模型，这只是设计空间的一角。候选可以来自跳层自推（self-speculative）、多头预测（Medusa）、树形草稿与并行验证（EAGLE），甚至来自 n-gram 统计；它们改变的是「候选从哪来」，验证与接受的骨架不变。下一篇展开这个设计空间。

## 参考与延伸阅读

1. [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
2. [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
3. [Zhang et al. — Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://arxiv.org/abs/2309.08168)
4. [Cai et al. — Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
5. [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
