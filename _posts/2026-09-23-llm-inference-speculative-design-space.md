---
title: "LLM 推理系统（六）：投机解码的设计空间——从外部 draft 到块级起草"
description: "候选从哪里来：外部小模型、跳层自推、Medusa 附加头、EAGLE 与 MTP 特征层头、树形验证、n-gram，以及 2026 年的块级起草与验证调度。"
date: 2026-09-23 18:00:00 +0800
math: true
categories: [模型与系统, LLM 推理]
tags: [LLM, 在线推理, 投机解码, Medusa, EAGLE, MTP]
---

第五篇用一轮执行时间与有效产出描述投机收益：固定请求形态下，加速比近似为 $T_{\mathrm{base}}E[N]/T_{\mathrm{round}}$。外部小模型只是候选来源之一。本篇比较跳层、附加头、特征条件 draft、文本查找和块级起草，重点是它们改变了哪些成本，以及需要什么训练与运行条件。

数值算例继续使用 Llama 3.1 8B、H200 SXM、上下文 $L=2048$。其中 3.2 ms 是主要读流量对应的简化 target 步时，并非任何投机方案的实测验证时间。下面的结构算例用于拆解成本；论文数据在后面单独列出，不能用异构实验给这些算例排序。

## 统一成本框架

将一轮分成 draft、验证和未包含在两者中的开销：

$$
T_{\mathrm{round}}=T_{\mathrm{draft,total}}+T_{\mathrm{verify}}+T_{\mathrm{residual}},\qquad
E^{\ast}=\frac{T_{\mathrm{round}}}{T_{\mathrm{base}}}
$$

$E^{\ast}$ 是这一固定配置下获得加速所需的平均产出门槛。串行 draft 近似有 $T_{\mathrm{draft,total}}=kT_d$；并行头和块级 draft 则应测整次起草的时间，不能仍机械乘 $k$。附加头的成本计入执行它的轮次，再由 $E[N]$ 摊到输出 token。

候选来源与候选形状是两个维度：同一种 draft 可以生成链，也可以生成树。除此之外，还要单独检查常驻参数、draft KV、临时验证状态，以及接受算法能否保持指定的 target 分布。

## 文本查找：n-gram 与 prompt lookup

这类方法从 Prompt、已生成文本或维护的统计表中找匹配前缀，把后续片段作为候选。它免去了神经网络 draft 前向，适合包含复制结构的代码、文档摘录与模板输出，但仍有查找、索引、候选传输和验证成本，不能称为零成本。

若在一个理想算例中忽略查找开销，并假设验证与普通一步同价，则 $E^{\ast}=1$。取各位置条件接受率均为 0.2、$k=3$，有 $E[N]=1+0.2+0.04+0.008=1.248$。这只是第五篇公式的代入；Leviathan 的 bigram 1.25x 也是近似零 draft 时间下的分析，不能当作独立测量对该公式的验证。

适用性取决于文本中是否有可复用片段。应测匹配率、有效接受长度及查询耗时，必要时与模型 draft 组合。开放生成可能很少命中，但不能仅凭任务名称断言接受率为零；也不应未经测量就把查找设为常开。

## 模型自身：选择性跳层与提前退出

**Draft & Verify** 在起草阶段选择性跳过部分 attention 或 MLP 层，验证阶段仍执行完整模型；论文使用离线搜索选择跳过的层。[1] 它不是固定执行“前 $M$ 层 + LM Head”的提前退出方案。后者属于另一类设计，例如 LayerSkip 通过训练让中间层支持提前退出，再利用剩余层验证。[12]

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/spec-layer-skip.svg" alt="Draft & Verify 在起草路径中选择性跳过部分中间层，验证仍经过完整模型；这种选择性跳层不同于只运行前 M 层的提前退出。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 1：选择性跳层与完整验证。图中层号仅作结构示意，具体跳层集合由搜索决定。</figcaption>
</figure>

可以用“保留相当于 $M$ 个完整 block 的工作量”建立粗略字节算例，但这不是论文的实际跳层模式。按每层权重约 0.436 GB、该上下文每层 KV 约 0.008 GB、LM Head 约 1.06 GB：

$$
T_{d,\mathrm{ideal}}\approx\frac{0.444M+1.06}{4.8}\ \mathrm{ms}
$$

$M=8$ 时约 0.96 ms，$M=16$ 时约 1.70 ms。保留同一个大词表输出层会限制压缩幅度；但缓存复用、跳过 attention 还是 MLP、特征分布和 kernel 形状也会改变结果。不能从这个算式推导接受率随 $M$ 的固定曲线，更不能据此断言它总不如同规模外部模型。

Draft & Verify 在其 LLaMA-2 实验中报告最高约 1.99x。[1] 它不需要单独存储另一套完整 draft 权重，是显存受限时值得评测的方案；运行中仍可能需要额外缓存和临时状态。“不增加第二套模型权重”不等于执行过程完全没有额外显存。

## 附加预测头：Medusa

Medusa 在 target 最后层隐状态上增加预测未来不同位置的头。[2] 多个头并行工作，所以预测更后面位置时，不能直接以之前刚选出的候选 token 为条件。它通过训练和树形候选增加覆盖机会，实际接受长度需测量，不能仅凭这种结构宣称一定低于所有串行 draft。

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/spec-medusa-heads.svg" alt="Medusa 将同一个主干隐状态送入原 LM Head 和多个未来位置预测头；各头并行产生候选，后续结合树形验证与指定接受规则决定输出。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 2：Medusa 并行预测头。图中 +1、+2、+3 表示预测位置，不代表未经接受规则就能直接输出。</figcaption>
</figure>

若用 $d\to d\to V$ 的头估算，$d=4096$、$V\approx128$k 时，一个头约有 0.54B 参数，BF16 约占 1.08 GB。两个头合计新增约 2.16 GB 常驻权重。小词表可以减轻这项成本，但它仍是显存增量。

假定一轮只额外读取一次这两个头的权重，忽略临时张量与额外算术，主要读流量对应的时间可由 3.2 ms 增至约 3.64 ms。于是 $E^{\ast}\approx1.14$。这是两头结构的成本示例，不能替代论文的多头、树形和具体 kernel 配置，也不能把这项开销按最终输出 token 重复计数。

输出质量需要区分训练和接受规则。Medusa-1 冻结 target、只训练附加头；Medusa-2 联合微调，target 本身已经改变。严格拒绝采样可以相对当前 target 保持分布，而论文常用的 typical acceptance 是近似接受规则，不具有同一分布保证。贪心模式若逐位按 target argmax 验证，可保留该 target 的贪心序列；这也不等于保留微调前模型的输出。论文 v3 的摘要报告 Medusa-1 超过 2.2x、Medusa-2 约 2.3–2.8x，引用时需保留版本与配置。[2]

## 特征条件 draft：EAGLE 与 MTP

EAGLE 利用 target 的隐藏特征起草，并引入已经选定的 token 信息处理下一步的不确定性；候选 token 再反馈给后续 draft 步。[3] 与并行位置头相比，这种链式结构显式建模候选之间的依赖，同时增加串行起草步骤。隐藏特征是否容易预测是经验问题，不能仅靠“连续量比离散量容易”得出接受率排序。

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/spec-eagle-loop.svg" alt="EAGLE 类起草利用 target 特征和候选 token 的 embedding，逐步生成后续候选；图示为特征条件串行起草的简化结构，具体版本的模块和训练目标不同。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 3：特征条件串行起草。复用 target 的输出层可以减少独立参数，但输出层计算与读取仍有成本。</figcaption>
</figure>

仍用 Llama 8B 的单层算例，假设 draft 层、LM Head 和该层 KV 分别读 0.436、1.06、0.008 GB，一步对应约 0.31 ms；$k=4$ 加上 3.2 ms 验证，约为 4.44 ms。这说明 LM Head 可以成为轻量 draft 的主要读量，但并非 EAGLE 或 MTP 的实测延迟：输入投影、词表策略、层数和 KV 布局都可能不同。

**MTP（Multi-Token Prediction）**首先是一类训练目标，不是唯一固定的 draft 结构。[6] DeepSeek-V3 的 MTP 模块组合隐藏状态与后续 token embedding、共享 embedding 和输出层，并与主干联合训练；它可用于投机解码。[7] 不能把所有名为 MTP 的实现都等同于一层 EAGLE。

EAGLE-2 使用动态 draft 树；EAGLE-3 改用直接 token 预测，并融合 target 多层特征。[4][5] 这些变化说明，特征来源、训练目标、起草结构与候选形状都可以独立调整。选择时应确认具体版本和 checkpoint 支持，而非只比较方法名称。

## 链与树：覆盖率与验证成本

树形候选在一个位置保留多个分支。验证时，每个节点只关注共同历史和自己的祖先路径，不读取旁支；这个树形 attention mask 保持每条路径的因果条件。一次前向可同时验证多个路径，但并不意味着成本与节点数无关。

在“权重与历史 KV 完全复用”的粗模型中，$m$ 行的计算项为 $m\times16.1/989$ ms，主要读流量项约 3.2 ms，两者约在 $m=196$ 处相交。这个交点只是两个汇总下界的交点；新增 logits、KV 写入、激活、tile 效率和 attention 访问早已可能增加实际时间，不能称为“200 行以内免费”。

覆盖率也必须注明采样口径。以贪心解码的示意例子为例，若链每层包含 target argmax 的条件概率为 0.7，树每层候选集合的相应覆盖概率为 0.9，四层的产出期望分别为：

$$
E_{\mathrm{chain}}=\sum_{i=0}^{4}0.7^i\approx2.77,\qquad
E_{\mathrm{tree}}=\sum_{i=0}^{4}0.9^i\approx4.10
$$

0.7 与 0.9 是假设值，不是通用实测关系。树的分支预算还必须足以实现这些覆盖率；任意 15 节点树并不自动拥有四层 top-2 覆盖。随机采样下，top-k 覆盖率不能代替第五篇的 $1-\mathrm{TV}(p,q)$，树形算法仍需正确处理提议概率与拒绝修正。

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/spec-chain-vs-tree.svg" alt="链在每个位置保留一个候选，树保留多个候选分支；验证节点只能注意共同历史与自身祖先。图示的 top-1 与集合覆盖率采用贪心口径，不等同于随机采样接受率。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 4：候选形状与树掩码。树可提高覆盖机会，同时增加验证行数；收益应由接受长度和验证耗时共同衡量。</figcaption>
</figure>

## 块级起草与验证调度

**DFlash** 用小型块扩散 draft，在一次并行前向中生成一块候选，并把 target 多层特征注入 draft 各层的 KV。[8] 本文采用论文 v2：block size 为 16 时，包含 1 个已经确定的 anchor 和 15 个待预测位置，不能写成 16 个新候选。

这种结构减少了串行起草步数，但 block 变大仍增加计算、激活和验证工作，draft 时间不会在任意长度下恒定。具体层数、训练块长、后端和温度都属于配置，不能用假设的“三层 0.5 ms”当作论文测量。

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/spec-dflash-block.svg" alt="DFlash 的 block size 16 包含一个已知 anchor 与十五个待预测位置；小型块扩散 draft 一次并行前向生成十五个新候选，再由 target 验证。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 5：DFlash 的块结构，按论文 v2 绘制。结构图不表示相对延迟，也不把 anchor 计为新候选。</figcaption>
</figure>

**DSpark** 同时调整起草结构、训练和服务调度：并行主干与轻量串行模块建模块内依赖，调度器结合前缀存活概率与硬件执行画像，为请求选择验证长度。[9] 裁掉的是预计收益较低的后缀，不是已经确定会被拒绝的 token；估计也可能失准。

论文 v1 在 DeepSeek-V4-Flash 线上流量下，相对生产 MTP-1 基线报告：匹配聚合吞吐时，每用户生成速度提高 60%–85%。这是整套方法的结果，不能全部归因于调度，也不是首次提出动态长度。验证长度缩至零时，若 draft 已执行，其成本仍然存在，不能视为完整退回普通 Decode。

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/spec-dspark-schedule.svg" alt="DSpark 根据估计的前缀存活概率和硬件执行成本，为每个请求选择验证长度；未选择的候选后缀不进入本轮验证，已执行的起草成本仍然存在。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 6：置信度与执行成本共同参与验证长度决策。边界是调度选择，不是候选必然正确或错误的分界。</figcaption>
</figure>

## 公开实验：先对齐条件，再比较结果

| 来源与版本 | 报告结果 | 关键条件与限制 |
| --- | --- | --- |
| Draft & Verify v2 | 最高约 1.99x | LLaMA-2 系列，选择性跳层；不能外推到任意 target |
| Medusa v3 | Medusa-1 >2.2x；Medusa-2 2.3–2.8x | 多头与树形配置；需区分微调 target 及接受规则 |
| EAGLE-3 v3 §4.3 | B=64 时约 1.38x | H100、Llama 3.1 8B Instruct、SGLang v0.4.4、MT-Bench；该实验关闭树形、使用长度 3 的链 |
| DFlash v2，Transformers | Qwen3-8B 贪心任务最高约 6.1x | B=1，论文表 1 的任务集合；不是所有任务的平均值 |
| DFlash v2，SGLang | Math500：B=1 约 5.1x，B=32 约 2.8x | 单 B200、Qwen3-8B、FlashAttention-4、Spec v2 调度 |
| DSpark v1 | 匹配吞吐时每用户速度 +60%–85% | DeepSeek-V4-Flash 线上，相对 MTP-1，而非无投机基线 |

这些实验的模型、硬件、后端、任务、采样和基线不同，表格用于指向可复查的实例，不构成性能排名，也不能验证本文 H200 简化算例。真正选型应在同一 target、同一质量约束和同一服务负载下复测。

## 公开模型采用了什么

| 模型或服务 | 可以从一手来源确认的内容 | 不能据此推断的内容 |
| --- | --- | --- |
| DeepSeek-V3 / V4 | V3 报告 MTP；DSpark 报告 V4 线上应用 [7][9] | 其他服务的默认配置或全部请求的行为 |
| GLM-4.5 系列 | 技术报告包含 MTP 设计 [10] | 任意引擎版本均以相同方式启用 |
| OpenAI gpt-oss | 官方发布与公开参考实现可供检查 [11] | 所引来源没有证实随 checkpoint 发布 MTP；也不能推断 ChatGPT/API 内部做法 |
| Google Gemma 4 | 官方提供配套 MTP draft，文档描述轻量四层助手 [13] | “MTP”名称不证明它与 target 联合预训练；不能推断 Gemini 内部实现 |
| Anthropic 服务 | 本文未取得可支持具体结构的一手披露 | 不根据行业猜测填写已采用的方案 |

公开证据表明训练内置和配套 draft 都在发展，尚不足以得出所有开源模型已经收敛到同一种方案。源码、技术报告与服务部署是不同层次的证据，引用时应保持边界。

## 选型与取舍

<figure class="text-center mt-3 mb-4">
  <img src="/assets/images/posts/llm-inference/spec-round-cost-breakdown.svg" alt="固定单流假设下的候选来源成本示例：起草、附加头与验证分别计入一轮时间，再计算所需平均产出门槛；这些数值不是各论文的实测排名。" style="width: 100%; max-width: 1120px; height: auto;">
  <figcaption class="text-muted mt-2">图 7：统一假设下的成本拆分。验证固定取 3.2 ms，跳层用等效完整 block 数近似；实际选型还需接受长度、显存和实测步时。</figcaption>
</figure>

| 约束 | 可优先评测的路线 | 需要核实 |
| --- | --- | --- |
| 无外部小模型，显存紧 | prompt lookup、选择性跳层 | 查找命中、跳层接受率、临时状态 |
| 有兼容小模型 | 外部 draft | tokenizer 支持、draft KV、起草速度 |
| 可以训练配套模块 | EAGLE 类、Medusa、MTP、块级 draft | 训练数据、额外参数、接受规则和引擎支持 |
| 复制内容较多 | 文本查找，可与模型 draft 组合 | 查找成本是否被有效输出覆盖 |
| 高并发或严格流式延迟 | 小候选预算、动态长度 | 混合负载吞吐、ITL 分位数与容量 |

基础算法在同一 token 空间验证；不同 tokenizer 需要对齐或重分词扩展，并非一概不可用。复用 LM Head 可以省独立参数，却不会自动省掉输出层的读取与计算。各种方法应统一比较每轮总时间、有效产出和资源占用，避免把“每轮成本低”直接等同于“每输出 token 快”。

## 小结

候选可以来自另一个模型、原模型的部分计算、附加头或文本匹配；链与树决定如何组织这些候选。训练方式、候选覆盖、串行起草深度、输出层与 KV 成本共同影响收益，没有单一结构参数能给出通用排名。

保持 target 分布还依赖具体接受算法：严格拒绝修正、贪心比对和 typical acceptance 应分开描述。下一篇将这些成本放进多请求服务，进一步分析验证行数、持久 KV 与流式延迟怎样改变调度决策。

## 参考与延伸阅读

1. [Zhang et al. — Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://arxiv.org/abs/2309.08168v2)
2. [Cai et al. — Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774v3)
3. [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
4. [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858)
5. [Li et al. — EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840v3)
6. [Gloeckle et al. — Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
7. [DeepSeek-AI — DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
8. [Chen et al. — DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036v2)
9. [Cheng et al. — DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147v1)
10. [Zhipu AI — GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models](https://arxiv.org/abs/2508.06471)
11. [OpenAI — Introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/)

12. [Hugging Face — LayerSkip: Faster LLM Inference through Self-Speculative Decoding](https://huggingface.co/blog/layerskip)
13. [Google — Gemma 4 Multi-Token Prediction](https://ai.google.dev/gemma/docs/mtp/mtp)
