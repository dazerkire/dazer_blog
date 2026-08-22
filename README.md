# Dazer笔记

记录对模型、算法与系统问题的理解，以及公开论文与技术资料的阅读笔记。内容以长文系列为主，尝试把模型原理、工程实现和实际性能问题放在同一条脉络中讨论。

## 当前内容

- **LLM 推理系列**：从一次在线请求出发，逐步讨论推理流程、性能指标、KV Cache、量化、投机解码和推理框架。
- **模型与系统笔记**：围绕模型、算法、系统与工程实践的阶段性记录。

## 写作方式

文章位于 `_posts/`，文件名使用 `YYYY-MM-DD-英文短标题.md`。新文章可从 `_drafts/post-template.md` 复制，正文使用 Markdown；分类、标签、归档和文章目录由站点自动生成。

常用的 Front Matter 如下：

```yaml
---
title: "文章标题"
description: "一句话摘要"
categories: [模型与系统]
tags: [LLM, 推理]
---
```

## 项目结构

```text
_posts/                 # 已发布文章
_drafts/                # 草稿与文章模板
_tabs/                  # 关于、分类、标签、归档页面
assets/images/posts/    # 文章配图
_config.yml             # 站点信息与主题配置
```

## 本地预览

本地预览需要 Ruby 3.4 和 Bundler：

```bash
bundle install
bundle exec jekyll serve
```

## 主题与许可证

站点基于 [Chirpy Starter](https://github.com/cotes2020/chirpy-starter) 和 [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) 构建；模板代码采用 MIT License，详见 [LICENSE](LICENSE)。
