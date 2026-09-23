---
title: "LLM 推理系统（六）：投机解码的设计空间——从外部 draft 到块级起草"
description: "候选从哪里来：外部小模型、跳层自推、Medusa 附加头、EAGLE 与 MTP 特征层头、树形验证、n-gram，以及 2026 年的块级起草与验证调度。"
date: 2026-09-23 18:00:00 +0800
math: true
categories: [模型与系统, LLM 推理]
tags: [LLM, 在线推理, 投机解码, Medusa, EAGLE, MTP]
---

第五篇把投机解码压成了一个公式：一轮耗时 $T_{\text{round}}$，一轮产出 $E[N]$ 个 Token，加速比 $=3.2\,E[N]/T_{\text{round}}$。那篇只讨论了一种候选来源：独立的小模型。本篇沿着这个公式把整个设计空间走一遍：候选还能从哪里来，组织成什么形状，成本落在哪一项。结论可以先摆出来：**所有方案都是同一笔账的不同分摊方式，差别只在 draft 步时、每步固定项、额外显存与接受率这四个数**。

口径沿用前篇：Llama 3.1 8B、H200 SXM、上下文 $L=2048$、理想效率 $\eta=1$、验证前向 3.2 ms。文末再看 2026 年这个空间正在往哪些方向扩展。

## 成本框架与设计空间

回顾第五篇的账：一轮 = $k$ 次 draft 步 + 一次验证前向，$T_{\text{round}} = k\,T_{\text{draft}} + 3.2$ ms；产出 $E[N]$ 由接受率决定；加速比为 1 的条件是 $E[N] = T_{\text{round}}/3.2$，这个值记作盈亏平衡 $E^{\ast}$，它越低，方案对接受率越宽容。Medusa 一类还会引入第三项：随每步固定收取的附加头开销。

「候选从哪来」的结构性答案有四类：外部小模型是第五篇的基线，跳层、附加头、特征层头三类在各自小节配机制图；另有一类零结构的来源，没有任何机制可画。下一节从它开始：成本为零的一端，是整个空间的原点。

与「来源」正交的一维是**形状**：候选排成一条链，还是一棵树。这一维同样由成本公式决定，第六节单独讲。

## 零成本候选：n-gram 与 prompt lookup

成本最低的一端不需要任何模型：从 Prompt、已生成文本或历史请求里找前缀匹配的片段，直接当作候选（prompt lookup / n-gram）。没有 draft 前向、没有额外显存，唯一成本是查找本身与被拒候选的少量 KV 写入，远小于一次前向。它也是成本公式的最简形态：draft 项清零后，整条公式只剩验证一项。

$$
T_{\text{round}} = 3.2\ \text{ms},\qquad E^{\ast} = 1.0
$$

门槛是理论下限：接受率只要不为零就是净赚。用第五篇引过的 Leviathan bigram 数据复算：$\alpha=0.2$、$k=3$ 时 $E[N]=1+0.2+0.04+0.008=1.248$，加速比 $=3.2\times1.248/3.2\approx1.25$x，与论文实测的 1.25x 闭合到小数点后两位。这个交叉验证同时校验了 $E[N]$ 公式的形状与「无模型 draft 步时可忽略」的假设，后面的方案都在这个已验证的框架上累加结构。

它的 $\alpha$ 不由模型决定，由负载的复制结构决定，方差远大于模型类：代码补全（标识符与样板反复出现）、RAG 问答（答案摘录 Prompt 中的文档片段）、多轮对话与 Agent 的格式化输出（每轮重 emit 相同的模板与键名）里命中率很高；开放生成与推理链里接近零。选型上它与模型类正交且可叠加：作为第一层常开、未命中再落到模型类 draft（分层投机），是「复制密集负载 + 无兄弟模型 + 显存为零」象限的默认答案。

## 模型自身：跳层起草

从零成本往上一档，回到模型本身。这一类回答的问题是：没有兄弟小模型时怎么办。答案是根本不引入新模型，用 target 自己的前几层起草。draft 路径 = 主干前 $M$ 层 + LM Head，后面的层不读；验证路径 = 完整 32 层（Draft & Verify 是这个思路的代表 [1]）。

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/spec-layer-skip.svg" alt="跳层起草：draft 路径只读主干前 M 层加 LM Head，早期退出产出候选链；验证路径走完整 32 层。LM Head 是与 M 无关的固定项，M=8 时占每步字节的 23%。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 1：跳层起草。draft 路径只读前 $M$ 层与 LM Head；LM Head 是与 $M$ 无关的固定项，决定了这种 draft 的成本下限。</figcaption>
</figure>

每步字节与 $M$ 的关系：

$$
T_{\text{draft}}\ \text{对应字节} = M\times(0.436 + 0.008)\ \text{GB} + 1.06\ \text{GB（LM Head）}
$$

$M=8$：4.61 GB → 0.96 ms；$M=16$：8.17 GB → 1.70 ms。注意 LM Head 这一项与 $M$ 无关：draft 层数砍到四分之一，每步时间只降到 3.2 ms 的 30%，LM Head 占了 $M=8$ 时每步字节的 23%。**廉价 draft 的成本下限由输出层决定**，因为它每次起草都要为整个词表算 logits。这条规律后面还会出现。

$k=4$ 下 $M=8$ 的 $T_{\text{round}} = 7.04$ ms，盈亏平衡 $E^{\ast}=2.20$，临界 $\alpha\approx0.57$；$M=16$ 则是 10.0 ms、$E^{\ast}=3.13$、$\alpha\approx0.76$。对比外部 1B draft 的 $\alpha^{\ast}\approx0.41$：接受率门槛高出 0.16 到 0.35。这里的结构性困难在于截断网络与 LM Head 的错配：LM Head 是在「第 32 层输出」上训练的，跳层草案喂给它的是中间层特征，即使补一个轻量适配层，通常也达不到同规模独立模型的预测水平。而 $\alpha(M)$ 的增长是递减的（前十几层承担了绝大部分预测内容），$\alpha^{\ast}(M)$ 却随步时近似线性上升，两条曲线的间隙很薄。实测落点与此一致：LLaMA-2 系列上最高 1.99x [1]，温和于外部 draft 的 2–3x。

它的收益在成本公式之外：额外显存为零、无需维护第二个模型、不依赖兄弟模型的存在（自研模型往往没有 1B 版本）、天然同 tokenizer 同训练谱系。**定位是「没有更好选择时的可靠兜底」，而不是「更省的 draft」**。

## 附加头：Medusa

第二类变体把 draft 的显式前向彻底去掉：在主干最后一层的隐状态 $h$ 上挂若干小头，一次前向同时产出当前 Token 与未来若干位置的候选。头是两层 MLP（$d\to d\to V$），首个未来位置由原 LM Head 自己负责，通常再挂 2 个头覆盖 +2、+3（图 2，结构改绘自 [2]）。

代价在两头。其一，每步固定变贵：128k 词表下每个头约 0.54B 参数、1.08 GB，两头使每步字节从 15.3 涨到 17.5 GB，3.2 → 3.64 ms（+14%）；若词表是 32k，则只涨 3.4%。附加头开销正比于词表大小，且乘以所有步数而不只投机轮。其二，也是结构性的：**所有候选都从同一个 $h$ 出发，预测 +2、+3 位置时看不到 +1 的实际取值，是并行盲猜**。链式 draft 猜第 $i$ 个候选时前 $i-1$ 个已经确定；Medusa 的头先天缺这个条件，接受率因此明显偏低。

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/spec-medusa-heads.svg" alt="Medusa：主干最后一层的隐状态 h 同时送入 LM Head 与多个附加头，一次前向产出 +1 真输出与 +2、+3 候选；后两个头在预测时看不到 +1 的实际取值，是并行盲猜。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 2：Medusa 的并行头（结构改绘自 [2]）。预测 +2、+3 的头看不到 +1 的实际取值；每头的末层与 LM Head 同量级，开销随词表大小走。</figcaption>
</figure>

$$
T_{\text{round}} = 3.64\ \text{ms/步},\qquad E^{\ast} = 1.14
$$

盈亏平衡是全部方案里最低的（外部 draft 1.67、跳层 2.2–3.1），因为开销摊进了每步固定项而不是按轮收取；但上限同理由此封住：加速比 $=0.88\times E[N]$，而 $E$ 上限是 1 + 头数。训练分两档：Medusa-1 冻结主干只训头，贪心解码下无损；Medusa-2 联合微调，接受率更高但 target 本身变了。采样模式用的是「典型接受」这类近似规则，不是第五篇的严格拒绝采样。实测：Medusa-1 超过 2.2x，Medusa-2 达 2.3–3.6x [2]——高于成本公式对「两个头的裸链」的预期，因为这些配置实际用了更多头加树形验证，形状的作用下一节展开。

## 特征层头：EAGLE 与 MTP

第三类变体来自一个实证观察：**相邻位置的隐藏状态高度相似（余弦相似度普遍在 0.9 上下），而相邻 Token 的分布可以截然不同**。特征是连续量，偏差一点不致命；Token 是离散选择，错就是错。在特征空间里猜，先天比在 Token 空间里猜容易。

EAGLE 的 draft 头顺着这个观察构造：一个 decoder 层，输入上一位置的主干特征与 Token embedding 的拼接，输出预测的下一层特征；预测特征过 target 自己的 LM Head 得到候选；候选的 embedding 再喂回头部，串行推进（图 3，结构改绘自 [3]）。它与 Medusa 的关键差别是候选串行产生，第 $i$ 个候选建立在第 $i-1$ 个之上，不盲；而串行的代价被头部极小这件事吃掉了。每步字节 $=0.436$（一层）$+1.06$（LM Head）$+0.008$（该层 KV）$\approx1.50$ GB → **0.31 ms，其中 LM Head 占 71%**。$k=4$：$T_{\text{round}}=4.44$ ms，$E^{\ast}=1.39$，临界 $\alpha^{\ast}\approx0.28$，是所有模型类 draft 中门槛最低的。同时它的实际接受率又是最高一档：特征可预测、候选不盲、头部对齐冻结主干训练。**门槛最低而命中最高**，这是 EAGLE 系实测领先（LLaMA2-Chat 70B 上 2.7–3.5x [3]，吞吐翻倍）的结构原因。

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/spec-eagle-loop.svg" alt="EAGLE 与 MTP 的 draft 头在特征层串行推进：主干特征 h 与上一候选的 embedding 拼接进一层 draft 头，预测特征过复用的 LM Head 得到候选，候选再回馈进下一步，第 i 个候选以第 i-1 个为条件。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 3：EAGLE 与 MTP 的特征层串行链（结构改绘自 [3]）。候选的 embedding 回馈进下一步输入，串行不盲；LM Head 与主干复用。</figcaption>
</figure>

MTP（Multi-Token Prediction）是同一种头在训练侧的出身。多 Token 预测作为训练目标由 Gloeckle 等系统研究 [6]；DeepSeek-V3 把它做成了出厂配置：主干旁一个 MTP 模块，结构与 EAGLE 头同构（一层、拼接 $(h, e)$、复用 embedding 与 LM Head），区别在于它与主干**联合从头训练**、随 checkpoint 发布 [7]。投机解码由此从推理期的附加组件变成训练期的设计决策；生产环境的采用也证明了这条路（第九节 DSpark 的基线就是单 MTP 模块）。

谱系按时间线看更清楚：

```text
2023-02  Chen / Leviathan     严格无损的 draft-verify 框架（token 级）
2023-09  Draft & Verify       跳层自推（token 级、模型自身）
2024-01  Medusa               并行多头（token 级、附加头）
2024-01  EAGLE                特征级串行头：concat(h, e) → 一层 → 复用 LM Head
2024-04  Gloeckle et al.      多 Token 预测作为训练目标
2024-12  DeepSeek-V3 MTP      同构模块搬进预训练，联合从头训练
2025-03  EAGLE-3              转直接 token 预测 + 多层特征融合
2026-02  DFlash               块扩散起草
2026-07  DSpark               半自回归起草 + 验证调度
```

两条线在互相走近：V3 把投机能力预训练进模型，EAGLE-2 引入按置信度动态分配的草稿树 [4]，EAGLE-3 反过来放弃特征预测目标、改直接 Token 预测并融合多层特征（实测至 6.5x，较 EAGLE-2 再提约 1.4x）[5]。「特征级」是这条线的起点而非终点，头与主干的耦合深度已经成为连续谱。

## 链与树：候选的组织方式

形状这一维由一个几乎反直觉的事实驱动：**验证前向的成本与候选行数近似无关**。树形 draft 在每个节点留 top-k 个孩子，一次验证整棵树；$m$ 个节点作为 $m$ 行一次前向，行与行之间只有注意力掩码的差别，每个节点的行只注意自己的祖先路径（树掩码），不看旁支。字节项照旧是权重与上下文 KV 各读一遍；FLOPs 项 $m\times16.1$ GFLOPs，$m=15$ 时 0.24 ms，仍深在转折点左侧。字节项与 FLOPs 项打平处在

$$
m^{\ast} \approx \frac{206\times15.3}{16.1} \approx 196\ \text{行}
$$

**在约 200 行以内，加节点几乎免费**：这一侧的算力当前利用率不到 1%。树往闲置的 FLOPs 侧扩张，这与 Medusa 头（往字节侧扩张、正比于词表）形成对照。

免费行数买到的是命中率的提升。链形下每个位置只有一次机会，$P(N\ge i)=\alpha^{i-1}$，$\alpha$ 是 top-1 命中率；树形把每层的事件变成「真实 Token 落在该层候选集合里」，概率是 top-k 覆盖率 $\beta$，天然高于 $\alpha$（同一个还行的 draft，top-1 命中 0.7 时 top-2 覆盖往往到 0.9）。逐层连乘近似为

$$
E_{\text{tree}} \approx 1+\beta+\beta^2+\beta^3+\beta^4
$$

$\alpha=0.7$ 的链 $E=2.77$，$\beta=0.9$ 的树 $E\approx4.1$，验证耗时同为 3.2 ms 量级。树用有效深度换每层机会：同样 15 行，链能探到 15 层深但每层只有一个候选，树只到 4 层但每层多个。实际系统再叠一层优化：动态树按置信度分配宽度，把行数花在不确定的位置（EAGLE-2 起 [4]，Medusa 的树同理 [2]）。

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/spec-chain-vs-tree.svg" alt="左：链形候选每个位置只有一次机会，接受概率为 top-1 命中率 α 的连乘。右：树形候选每个位置有 k 次机会，逐层命中率为 top-k 覆盖率 β；验证时每个节点只注意自己的祖先路径（树掩码），同为一次前向。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 4：链与树。同为一次 3.2 ms 的验证前向，树把每层的 top-1 命中换成 top-k 覆盖；加粗链是一个节点的祖先路径，即树掩码下它唯一能看见的部分。</figcaption>
</figure>

## 训练内置与块级起草：2026 年的两个方向

设计空间最新的两个成员各自打开一条新轴。

**DFlash：把 draft 成本对 $k$ 的斜率压平** [8]。串行链的 draft 成本 $\propto k$，这正是 EAGLE 被迫只做一层的原因；DFlash 用一个小型块扩散模型**一次并行前向产出整块候选**（评测配置 16 个 Token、单步去噪），条件信号取自 target 多层特征、注入 draft 每一层的 KV（EAGLE 只喂第一层，深度会稀释信号），embedding 与 LM Head 复用。成本结构由此改变：16 行对一个小模型就是一次微型 Prefill，行数在带宽瓶颈区照旧近乎免费，示意地按 3 层 draft 算，一次前向约 2.4 GB → 0.5 ms 出 16 个候选，串行方式则要 $16\times0.31\approx5$ ms。报告的实测：Qwen3-8B、温度 0 下 2.3–6.2x，同任务的 EAGLE-3 为 1.9–2.5x；已进入 vLLM 与商用部署（Baseten 报告 2.9x）。以上数字来自其项目页与摘要，成稿口径以论文为准。

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/spec-dflash-block.svg" alt="DFlash 用小型块扩散模型一次并行前向产出整块候选（16 个），draft 成本对候选数近似平坦；串行起草方式出同样数量候选的成本随数量线性增长。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 5：DFlash 的块扩散起草。一次并行前向出整块，draft 成本对候选数近似平坦；条形为同一时间轴的示意，按论文核对。</figcaption>
</figure>

**DSpark：把验证变成可调度量** [9]。它的出发点在服务侧：并行起草的块越长尾部衰减越重，而验证整块会把 batch 容量浪费在注定被拒的后缀上，高并发下直接伤吞吐。draft 侧用「并行主干 + 轻量串行模块」建模块内依赖，把并行起草的盲修掉一半；验证侧按估计的前缀存活概率与引擎吞吐画像，逐请求裁剪验证长度（图 6）。实测相对单 MTP 模块的生产基线（MTP-1），同吞吐下每用户速度提升 60–85%，已部署于 DeepSeek-V4 线上。注意它的目标函数已经从单流延迟扩展到服务吞吐：**验证长度第一次成为可以按系统状态调节的量**，这正是下一篇的主题。

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/spec-dspark-schedule.svg" alt="DSpark 按估计的前缀存活概率逐请求裁剪验证长度：置信边界之前的候选正常验证，之后的后缀按存活概率舍弃，不浪费 batch 容量；验证长度成为可调度量。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 6：DSpark 的置信调度验证 [9]。置信边界之后的候选按存活概率舍弃，验证长度成为可调度量。</figcaption>
</figure>

至此设计空间多了三个维度：**出身**（外挂训练还是预训练内置，MTP）、**draft 成本对 $k$ 的形状**（线性还是平坦，DFlash）、**验证的可调度性**（固定成本还是按置信与系统状态调节，DSpark）。验证与接受概率的骨架始终未变。

## 实测对照

各方案公开报告的加速比，条件各不相同，放在一起只看量级与相对位置：

| 方案 | 报告加速比 | 条件 |
| --- | --- | --- |
| 外部 draft（T5-XXL 配对） | 最高 **3.4x** | TPU-v4，B=1（第五篇表） |
| 跳层（Draft & Verify） | 最高 1.99x | LLaMA-2 系 [1] |
| Medusa-1 / Medusa-2 | >2.2x / 2.3–3.6x | 含树形验证 [2] |
| EAGLE | 2.7–3.5x | LLaMA2-Chat 70B，B=1 [3] |
| EAGLE-3 | 至 6.5x；B=64 吞吐 1.38x | 含动态树 [5] |
| DFlash | 2.3–6.2x | Qwen3-8B，温度 0 [8] |
| DSpark | +60–85% | 对 MTP-1，线上 [9] |

读这张表要注意三点。其一，口径不同：多数是单流延迟，EAGLE-3 的第二行与 DSpark 是吞吐或线上指标，高并发下投机未必是正收益，下一篇专门算。其二，实测的排序与本文成本表一致：门槛最低、命中最高的特征层头与块扩散落在最前，跳层垫后。其三，所有数字都是各自论文的最优配置（头数、$k$、树形状各不相同），横向比较只到量级为止。

## 各家在用什么

闭源服务不公开推理内部实现，能确认的信息集中在开源与开源权重一侧：

| 厂商 | 公开的方案 | 证据 |
| --- | --- | --- |
| DeepSeek | V3 出厂 MTP 模块；V4 线上 DSpark（半自回归 + 验证调度） | 技术报告与论文 [7][9] |
| 智谱（GLM） | GLM-4.5 / 4.5-Air 内置 MTP 层（额外一层 MoE），SGLang、vLLM 官方支持 | 技术报告 [10] |
| OpenAI | gpt-oss（开源权重）带 MTP 组件、权重随 checkpoint 发布、明确为投机解码设计；ChatGPT / API 内部未公开 | 官方发布 [11] |
| Google | 投机采样源于 DeepMind（第五篇 [2]）；Gemma 4 以 MTP 为官方投机解码路径；Gemini 服务端未明示 | 官方博客与文档 |
| Anthropic | 无官方披露；行业分析普遍认为生产中采用了投机解码 | 无一手来源 |

三条观察。其一，**开源阵营已经收敛到「训练内置 MTP」这一条线**：DeepSeek、智谱、OpenAI 的开源权重、Google 的 Gemma 全部如此。第五篇的外部小模型与本篇的外挂头路线服务于存量模型，新模型的出厂配置里 draft 已是预训练的一部分，且引擎（SGLang、vLLM）原生支持，部署链条打通。其二，闭源 API 的内部做法基本不可考，第三方转述的加速数字无法核实，选型参考应以开源证据为准。其三，DSpark 在 MTP 底座上靠调度再拿 60–85% 的提升，说明训练内置只是起点，服务侧还有独立的一层收益，这正是下一篇的主题。

## 选型与取舍

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/spec-round-cost-breakdown.svg" alt="六种候选来源在 k=4 下的一轮耗时构成：跳层 M=16 共 10.0 ms、跳层 M=8 共 7.04 ms、外部 1B 共 5.34 ms、EAGLE 与 MTP 共 4.44 ms、Medusa 两头每步 3.64 ms、n-gram 仅验证 3.2 ms；右侧标注盈亏平衡所需的每轮期望产出。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 7：一轮成本的构成与盈亏平衡。draft 步时（橙）与验证（紫）是所有方案共有的两项，Medusa 的附加头（蓝）按每步收取；门槛最低的三个方案恰好也是实测最快的三个。</figcaption>
</figure>

| 场景 | 默认选择 | 依据 |
| --- | --- | --- |
| 自研模型、无兄弟小模型 | 跳层，或 prompt lookup | 不依赖外部模型的存在；两者可叠加 |
| 有同家族小模型、单流在线 | 外部 draft 或 EAGLE 头 | 前者基线可靠，后者门槛更低、实测更高 |
| 追求当前最高单流加速 | EAGLE-3 / 块扩散 | 门槛低且命中高；需要训头 |
| 复制密集负载（代码、RAG、模板输出） | prompt lookup 常开 | 零成本，可与模型类 draft 分层叠加 |
| 词表较小、不接受显存增长 | Medusa 头 | 开销正比词表；32k 词表时近乎免费 |
| 高并发、满 batch 服务 | 谨慎，先算验证成本 | 行数开始计价；下一篇展开 |

三条横切的法则。其一，同 tokenizer 是硬约束，同家族（同训练分布）决定 $\alpha$ 的量级。其二，**LM Head 是廉价 draft 的成本下限**：跳层 $M=8$ 的 23%、EAGLE 的 71%、DFlash 与 MTP 靠复用而非重造来摊薄它。其三，方案没有绝对优劣，只有「开销出现在哪」的分摊差异：按轮收 draft 步时的（外部、跳层、EAGLE）看临界 $\alpha$，摊进每步固定项的（Medusa）看词表，零成本的（n-gram）看负载，验证可调度的（DSpark）看系统状态。

## 小结

第五篇问「一次前向能不能出多个 Token」，本篇问「候选从哪来、什么形状、账分摊到哪」。四类来源把 draft 步时从 1.70 ms（跳层 $M=16$）一路压到 0（Medusa、n-gram），把开销搬进每步固定项（Medusa，正比词表）或搬进主干训练（MTP）；形状这一维用近乎免费的验证行数（200 行以内）把每层 top-1 命中换成 top-k 覆盖，同一份 3.2 ms 的验证前向，$E$ 从 2.77 提到 4.1 的量级。2026 年的两个新方向各开一条轴：块扩散把 draft 成本对候选数的斜率压平，置信调度把验证长度变成系统可调节的量。骨架自始至终没变：所有位置都由 target 计算与修正，输出分布不变性对贪心与采样分别成立。

本篇全部数字都是单流口径，隐含前提是 Decode 停在带宽瓶颈上、验证行数不计价。进入多请求服务，这个前提开始动摇：draft 与 target 如何进 batch、验证行长怎样吃掉吞吐、接受率波动如何扰动调度。下一篇把投机解码放进服务系统，算清「单请求加速」与「服务吞吐」之间的这笔转换。

## 参考与延伸阅读

1. [Zhang et al. — Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://arxiv.org/abs/2309.08168)
2. [Cai et al. — Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
3. [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
4. [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858)
5. [Li et al. — EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840)
6. [Gloeckle et al. — Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
7. [DeepSeek-AI — DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
8. [Chen et al. — DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
9. [Cheng et al. — DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147)
10. [Zhipu AI — GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models](https://arxiv.org/abs/2508.06471)
11. [OpenAI — Introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/)
