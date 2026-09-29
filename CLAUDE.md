# CLAUDE.md

Jekyll 博客（GitHub Pages，push main 自动部署）。`_drafts/` 不参与发布；通用单篇骨架见 `_drafts/post-template.md`，本文件是写作风格与流程的权威约定。

## 写作工作流

- 成稿走对话优先：先多轮讨论、让用户自己推导数字，用户明确确认后才写入 `_drafts/`，再按确认移入 `_posts/`。
- 发布 = 移入 `_posts/YYYY-MM-DD-<slug>.md`（front matter 补 `date: ... 00:00:00 +0800`）→ 提交（结尾带 `Co-Authored-By: Claude Code <noreply@anthropic.com>`）→ push → 等 pages-deploy workflow 成功 → curl 线上页面抽查。

## 文风：严谨的技术博客口吻（2026-09 用户反馈确立，勿回退）

- 克制的书面语；机制与结论用直白的技术陈述。
- 不造比喻、不用修辞框架：禁「赌博/赌注/抽水、劫持、XX 税、主场、赢家、救场」一类说法；系列早期自造术语「账本/两本账/屋脊点/屋顶」也已弃用（2026-09，旧文已全部替换）。替换口径：账本 → 每步字节公式/字节构成；屋脊点 → 转折点（ridge point，首次出现标英文）；屋顶 → 计算峰值/峰值。通用词「吞吐天花板」不受限。
- 破折号（——）尽量少：一段至多一处，多数改逗号、冒号或拆句。
- 加粗只用于：术语首次定义、单句关键结论、表格中标记最优解的数值；不做节奏性强调。
- 不用反问句或「一句话：」「看尾数：」式话头承接段落。
- 避免口语词（倒亏、压不动、谁也省不掉、塞得下等）。
- 引入新概念先给定义与存在理由（为什么有它），再上数据；不直接甩数字。
- 数字给出可复算的算式或来源。

## 「LLM 推理系统」系列结构与口径

- 结构：开局接上一篇钩子（1–2 段）→ 机制分析若干节 → 实测对照（公开基准做现实修正）→ 选型与取舍（场景表）→ 小结（结尾预告下一篇）→ 参考与延伸阅读（编号引用 [n] + 链接）。
- 统一口径：Llama 3.1 8B、H200 SXM（dense BF16 989 TFLOPS、HBM 4.8 TB/s）、上下文 L=2048、理想口径 η=1；FLOPs/带宽用十进制前缀，显存容量用二进制前缀。
- 系列数值口径（2026-09-29 校订）：Prefill 使用因果有效 attention FLOPs `2L(L+1)d`，2048 token 约 29.7 TFLOPs → 纯计算下界约 30 ms；32768 token 约 738.9 TFLOPs → 747 ms。满矩阵计数须单独标注，不能作为所有因果实现的延迟下界。
- Decode B=1 主要读流量约 15.3 GB → 带宽下界约 3.2 ms，主要计算量约 16.1 GFLOPs/token；GQA KV 每历史 token 128 KiB、2048 token 每路 256 MiB。常驻权重与每步读取量分开统计。
- 容量算例统一假设：扣除 target 全部常驻权重与其他预留后，KV 预算 120 GB（约 111.8 GiB）；BF16 target KV 最多 447 路。外部 1B draft 再扣约 2.5 GB 权重及每路 64 MiB 持久 KV，乐观上限 350 路，额外 workspace/临时状态另计。
- 公开吞吐不倒推为排除 TTFT 的 TPOT，不据此拟合固定 2.6 ms 开销。汇总 Roofline 的 max 是宽松下界；计算/带宽交点不是“免费行数”或投机开关。数值算例、实测与经验建议分别标注。
- 引用性能结果注明模型、硬件、后端版本、负载、采样和基线。异构实验不做排名或精确模型验证；论文有版本差异时固定版本。ITL 分位数与请求级 TPOT 分位数分别统计。
- 修改算例后运行 `python tools/generate_inference_figures.py --font <本机中文字体路径>` 同步相关 SVG，并核对正文与图注。此脚本仅是可选绘图维护工具（依赖 matplotlib），不参与 Jekyll 构建。
- Front matter：`categories: [模型与系统, LLM 推理]`；tags 含 LLM 与本篇关键词；`math: true`。
- 提纲与状态记录在 `docs/llm-inference-series-outline.md`，成稿/配图/发布状态同步更新。

## 配图

- SVG，宽 1120，浅底圆角卡片（bg #fbfcfe）；色板：blue #2563eb / orange #f97316 / purple #7c3aed（浅 tint #dbeafe/#ffedd5/#ddd6fe）、ink #344054、次级 #64748b、网格 #dce3ea 虚线；字体栈 `-apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif`；`role="img"` + `<title>/<desc>`。
- 嵌入：`<figure class="text-center mt-3 mb-4">` + `<img src="/assets/images/posts/llm-inference/<name>.svg" alt="<复用 SVG desc>" style="width: 100%; max-width: 1120px; height: auto;">` + `<figcaption class="text-muted mt-2">图 N：<读图结论></figcaption>`。
- 图内文字与图注同受文风规则约束。
- 本地渲染校验：cairosvg 不支持多字体列表，需先把字体栈替换成 "Noto Sans CJK SC"；视觉检查报告须用 grep/坐标计算交叉核对后再采信。

## 已知技术坑

- 行内公式 `$...$` 内不能有裸 `*`：kramdown 会把它配对成 `<em>`，破坏公式（块级 `$$...$$` 不受影响）；改用 `\ast`。
- 行内公式内同样不能有裸 `|`：kramdown 会把竖线解析成表格单元、把公式拆散并生成多余表格（块级不受影响）；改用 `\lvert ... \rvert`。
- 网页字体缺字形的字符会显示为方块：Unicode 上下标（₁ ₜ ₊ ₙ 等）改 ASCII 写法（x1、K1,V1、Q(t+1)）；└ ─ ↑ ↓ µ … 已验证安全。
