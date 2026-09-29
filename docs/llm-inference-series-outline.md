# LLM 推理系统系列写作提纲

本文档记录系列路线与维护状态，不参与站点发布。七篇正文与配图于 2026-09-29 完成内容校订；统一数值口径见 `CLAUDE.md`。

## 已发布文章

1. **从一次请求理解在线推理：应用、流程与指标**
   - 模型 API TTFT 与应用首字等待分开；定义 ITL、请求 TPOT 与端到端恒等式。
   - Token 间隔分位数与请求平均 TPOT 分位数区分统计。
2. **从指标到瓶颈：Prefill、Decode 与计算本质**
   - GEMM/GEMV、GQA、KV 与 Roofline；因果 attention 有效 FLOPs 统一为 `2L(L+1)d`。
   - 2048-token Prefill 约 29.7 TFLOPs；Decode 下界约 3.2 ms。公开吞吐不能转换成实测 Decode TPOT。
3. **KV Cache、并发与请求调度：上限、浪费与取舍**
   - 权重摊销、历史 KV 不摊销；连续批处理、分块 Prefill、分页、共享与抢占。
   - 混合 batch 时间不由两个独立下界相加；重算含二次 attention 成本。
   - 120 GB KV 预算对应 447 路；按最大 8192-token 预留仅 111 路。实际 SLO 需压测。
4. **量化究竟改变了什么：字节、算力与误差**
   - 字节模型加速比 `1/(1-f+f/q)`；吞吐分母使用完整历史 KV 读量 `L*s_KV`。
   - 因果 Prefill 加 `M-1` 次 Decode，随输出更新上下文长度；图中三种路径均保留 BF16 KV。
   - 舍入界注明无裁剪、精确 scale；解释静态/动态激活量化、PTQ 权重补偿和 QAT。
   - NVIDIA 表的 W4A8 为 INT4+FP8；独立的 QAT 表为 INT4+INT8，不混用。
5. **投机解码如何突破逐 Token 串行：draft、验证与接受率**
   - 正确拒绝修正分布；说明理想读量复用不等于相同实测延迟。
   - 一般产出期望使用前缀存活概率；固定接受率只作算例。
   - 修正 EnDe 的 T5-small 温度 1 候选长度为 7，bigram 1.25x 标为分析估计。
   - 跨 tokenizer 需扩展实现；TTFT 也可能受 draft 与调度影响。
6. **投机解码的设计空间：从外部 draft 到块级起草**
   - 选择性跳层与提前退出分开；Medusa 的参数、训练与接受规则分别讨论。
   - EAGLE/MTP 的结构差异、链与树的条件覆盖率、DFlash 的 anchor 与 15 个新候选。
   - DSpark 的结构、训练与调度共同贡献；裁剪为零仍可能保留起草成本。
   - 公开数据注明配置，不做异构排名；不再声称 gpt-oss 已公开 MTP 或所有模型收敛到 MTP。
7. **投机解码进入服务系统：当行数开始计价**
   - 约 122 的汇总下界交点不等于盈亏平衡；B=64 理想模型约 1.82x。
   - 外部 draft 容量计入持久 KV，示例上限 350 路，吞吐图限于共同比较区间。
   - ITL 长间隔出现在每轮边界，示例比例为 `1/E[N]`，不等于首拒概率。
   - 统一以实测轮次时间与产出比较，再单独检查显存、TTFT、ITL 和 goodput。

## 配图维护

- 各篇现有插图随正文同步校订；保留原文件名与文章 URL。
- 数值图由 `tools/generate_inference_figures.py` 重算生成；静态结构图直接维护 SVG。
- 第六篇保留七张图，第七篇保留两张图；SVG 描述、正文 alt 与图注需一致。

## 后续文章

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
