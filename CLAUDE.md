# CLAUDE.md

Jekyll 博客（GitHub Pages，push main 自动部署）。`_drafts/` 不参与发布；通用单篇骨架见 `_drafts/post-template.md`，本文件是写作风格与流程的权威约定。

## 写作工作流

- 成稿走对话优先：先多轮讨论、让用户自己推导数字，用户明确确认后才写入 `_drafts/`，再按确认移入 `_posts/`。
- 发布 = 移入 `_posts/YYYY-MM-DD-<slug>.md`（front matter 补 `date: ... 00:00:00 +0800`）→ 提交（结尾带 `Co-Authored-By: Claude Code <noreply@anthropic.com>`）→ push → 等 pages-deploy workflow 成功 → curl 线上页面抽查。

## 文风：严谨的技术博客口吻（2026-09 用户反馈确立，勿回退）

- 克制的书面语；机制与结论用直白的技术陈述。
- 不造比喻、不用修辞框架：禁「赌博/赌注/抽水、劫持、XX 税、主场、赢家、救场」一类说法；系列既有装置（账本、两本账、屋脊点、天花板、屋顶）可继续用。
- 破折号（——）尽量少：一段至多一处，多数改逗号、冒号或拆句。
- 加粗只用于：术语首次定义、单句关键结论、表格中标记最优解的数值；不做节奏性强调。
- 不用反问句或「一句话：」「看尾数：」式话头承接段落。
- 避免口语词（倒亏、压不动、谁也省不掉、塞得下等）。
- 引入新概念先给定义与存在理由（为什么有它），再上数据；不直接甩数字。
- 数字给出可复算的算式或来源。

## 「LLM 推理系统」系列结构与口径

- 结构：开局接上一篇钩子（1–2 段）→ 机制/账若干节 → 实测对照（公开基准做现实修正）→ 选型与取舍（场景表）→ 小结（结尾预告下一篇）→ 参考与延伸阅读（编号引用 [n] + 链接）。
- 统一口径：Llama 3.1 8B、H200 SXM（dense BF16 989 TFLOPS、HBM 4.8 TB/s）、上下文 L=2048、理想口径 η=1；FLOPs/带宽用十进制前缀，显存容量用二进制前缀。
- 系列账底（已建立，沿用勿重推）：Prefill 2048 ≈ 30.8 TFLOPs → 31 ms；Decode B=1 每步 15.3 GB → 3.2 ms、16.1 GFLOPs/token；KV 每路 256 MiB（GQA）；实测 BF16 B=1 每步约 5.75 ms（T_other ≈ 2.6 ms）。
- Front matter：`categories: [模型与系统, LLM 推理]`；tags 含 LLM 与本篇关键词；`math: true`。
- 提纲与状态记录在 `docs/llm-inference-series-outline.md`，成稿/配图/发布状态同步更新。

## 配图

- SVG，宽 1120，浅底圆角卡片（bg #fbfcfe）；色板：blue #2563eb / orange #f97316 / purple #7c3aed（浅 tint #dbeafe/#ffedd5/#ddd6fe）、ink #344054、次级 #64748b、网格 #dce3ea 虚线；字体栈 `-apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif`；`role="img"` + `<title>/<desc>`。
- 嵌入：`<figure class="text-center mt-3 mb-4">` + `<img src="/assets/images/posts/llm-inference/<name>.svg" alt="<复用 SVG desc>" style="width: 100%; max-width: 1120px; height: auto;">` + `<figcaption class="text-muted mt-2">图 N：<读图结论></figcaption>`。
- 图内文字与图注同受文风规则约束。
- 本地渲染校验：cairosvg 不支持多字体列表，需先把字体栈替换成 "Noto Sans CJK SC"；视觉检查报告须用 grep/坐标计算交叉核对后再采信。

## 已知技术坑

- 行内公式 `$...$` 内不能有裸 `*`：kramdown 会把它配对成 `<em>`，破坏公式（块级 `$$...$$` 不受影响）；改用 `\ast`。
- 网页字体缺字形的字符会显示为方块：Unicode 上下标（₁ ₜ ₊ ₙ 等）改 ASCII 写法（x1、K1,V1、Q(t+1)）；└ ─ ↑ ↓ µ … 已验证安全。
