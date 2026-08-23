# LLM 推理系统系列写作提纲

本文档记录系列的暂定路线，用于后续写作讨论；它不是博客文章，不参与站点发布。

## 已发布 / 已完成草稿

1. **LLM 推理系统（一）：从一次请求理解在线推理——应用、流程与指标**
   - 应用形态、Prefill / Decode、TTFT / TPOT 与在线请求生命周期。

2. **LLM 推理系统（二）：从指标到瓶颈——Prefill、Decode 与计算本质**
   - attention、MLP、GEMM / GEMV、KV Cache、Roofline 与公开基准的数量级比较。

## 后续文章

### 三、KV Cache、并发与请求调度

核心问题：单请求 Decode 已受带宽限制；当请求变成几十、几百条时，KV Cache 如何决定并发上限？

- 从每 Token KV Cache 增量和多请求并发的例子开始；
- 静态 batch 为什么会被不同请求长度拖累；
- continuous batching 如何按 Decode step 重组 batch；
- PagedAttention 如何用 block 管理 KV，避免连续预分配与碎片；
- 并发、吞吐、TTFT、TPOT 之间的取舍。

### 四、量化究竟改变了什么

核心问题：量化减少了哪些字节搬运，又在哪些地方引入误差和额外计算？

- 回到 memory-bound Decode，解释压缩权重的收益；
- 权重量化、激活量化、KV Cache 量化分别作用于什么；
- 对称 / 非对称量化，scale、zero point、group size；
- PTQ 与 QAT；
- INT8、INT4、FP8 的取舍；
- 用同一个 8B 模型估算权重与 KV Cache 的显存变化；
- 理论压缩率与端到端加速比的差异。

### 五、投机解码如何突破逐 Token 串行

核心问题：在不改变 target model 输出分布的前提下，如何一次推进多个 Token？

- 普通 Decode 的串行链条；
- draft model 提出 $k$ 个候选 Token；
- target model 的并行验证；
- 接受、拒绝与修正采样；
- 接受率、draft 成本与验证成本如何决定收益；
- 哪些模型组合可能没有收益。

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

核心问题：缓存、调度、量化和 kernel 如何在 vLLM、TensorRT-LLM 等框架中变成系统能力？

- KV 管理：PagedAttention、block manager、prefix cache；
- 调度：continuous batching、Prefill / Decode 取舍；
- 图优化与 kernel：编译、fusion、量化路径；
- 分布式推理：tensor parallel、pipeline parallel、通信；
- 框架选择由模型、硬件、部署目标与可维护性共同决定；
- 回看整个系列：每个框架都在处理前文的一类瓶颈。

## 可选插篇

**长上下文与 Prefill**

若后续讨论发现内容足够独立，可放在量化之前，覆盖 FlashAttention、chunked prefill 与 context parallelism。当前不占用固定编号。
