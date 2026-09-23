# LLM 推理系统系列写作提纲

本文档记录系列的暂定路线，用于后续写作讨论；它不是博客文章，不参与站点发布。

## 已发布 / 已完成草稿

1. **LLM 推理系统（一）：从一次请求理解在线推理——应用、流程与指标**
   - 应用形态、Prefill / Decode、TTFT / TPOT 与在线请求生命周期。

2. **LLM 推理系统（二）：从指标到瓶颈——Prefill、Decode 与计算本质**
   - attention、MLP、GEMM / GEMV、KV Cache、Roofline 与公开基准的数量级比较。

3. **LLM 推理系统（三）：KV Cache、并发与请求调度——上限、浪费与取舍**
   - 并发的收益来源（权重摊销、KV 不摊销、显存上限）、静态 batch、continuous batching 与 selective batching、
     chunked prefill、PagedAttention（分页、共享与 prefix cache、准入与抢占）、SLO 视角的吞吐–延迟总账。
   - 原理在本篇讲透，第八篇只做框架落地对比。

4. **LLM 推理系统（四）：量化究竟改变了什么——字节、算力与误差**
   - 字节列（带宽）与算力列（计算峰值）两条线索；误差是独立的第三个问题。
   - 收益：加速比 = 被压缩项的字节占比（同一次 W4：B=1 约 3.8x，B=64 约 1.5x）；拐点左移（56→14）；
     KV FP8 使并发上限与吞吐天花板各翻倍，W4 腾显存仅 +9%；prefill 的 max 不动，FP8 尾数减半换来
     计算峰值翻倍（989→1979）；端到端两场景互有胜负（对话 W4 胜、RAG FP8 胜）。
   - 误差：仿射量化与 a/2 上界、中间值承担最大误差、outlier 主导 scale、group size 与 0.125 bit 元数据、
     g=1 悖论（量化的压缩全部来自共享动态范围）、动态 scale 分水岭（W4A16 vs W8A8/FP8）、
     outlier channels 三条出路（LLM.int8 / SmoothQuant / weight-only）、KV 粒度（K per-channel、V per-token）。
   - 实测与选型：NVIDIA H200 基准的三层折扣（T_other 不缩水、占比定律、W4A16 kernel 开销致 B≥8 低于 BF16）；
     PTQ/QAT 分工；INT8 让位 FP8 的三个原因；INT4 的端侧位置与 Blackwell 原生 FP4。

5. **LLM 推理系统（五）：投机解码如何突破逐 Token 串行——draft、验证与接受率**
   - 一轮流程（5 位置验证、bonus、产出区间 [1,5]）；验证近乎免费（字节项不变，GEMM 合并访存不合并算术）；
     draft 1B 每步 0.53 ms、一轮固定 5.34 ms、上界 3.0x（图 1）。
   - 收益公式：E[N] = (1−α^(k+1))/(1−α)；α=0.8 → 2.0x；临界 α≈0.41；k=2/4/8 权衡（图 2，k 取 3–7）。
   - 分布不变性：A = min(1, p/q) 加 (p−q) 正部重采，边缘恰为 p；α = 1−TV(p,q)；
     greedy 为 T→0 极限，退化为逐位置确定性比对，保证序列逐位相同。
   - 实测：Leviathan T5-XXL 表（draft 越大加速比越低、温度升 α 降约 0.1、理想公式打折、bigram α≈0.2 仍 1.25x）；
     Chen Chinchilla 70B 2–2.5x。
   - 选型：同 tokenizer 硬约束、draft 取 target 约 1/10、k=3–7；高并发负收益留待第七篇。

## 后续文章

### 六、投机解码的设计空间

核心问题：除了“小模型起草、大模型验证”，还有哪些投机解码路线？

- 外部 draft model；
- self-speculative / early exit；
- 多头预测与 Medusa 类方法；
- tree decoding 与并行候选验证；
- 用候选结构、验证方式、额外模型、接受率作为统一比较维度。

### 七、投机解码进入服务系统后

核心问题：单请求加速为什么不必然转化为服务吞吐提升？

- draft 与 target 如何进入 batch；
- 一次验证多个 Token 如何影响 KV Cache；
- 接受率波动怎样改变 batch 长度与调度；
- TTFT、TPOT、总吞吐各自可能如何变化；
- 长上下文、高并发、短输出等工作负载下的取舍；
- 判断标准：额外工作是否换来了更少的 target Decode step。

### 八、推理框架与引擎如何落实这些设计

核心问题：第三篇及前文的原理，在 vLLM、TensorRT-LLM、SGLang 等框架中分别如何落地？实际选型时如何比较？

- 同一原理的不同实现：block manager 与调度器的实现差异、prefix cache（含 RadixAttention 一类变体）、chunked prefill 的具体策略、PD 分离架构；
- 图优化与 kernel：编译、fusion、量化路径；
- 分布式推理：tensor parallel、pipeline parallel、通信；
- 框架选择由模型、硬件、部署目标与可维护性共同决定；
- 回看整个系列：每个框架都在处理前文的一类瓶颈。

## 可选插篇

**长上下文与 Prefill**

若后续讨论发现内容足够独立，可放在量化之前，覆盖 FlashAttention 与 context parallelism（chunked prefill 的原理已在第三篇随调度讲掉）。当前不占用固定编号。
