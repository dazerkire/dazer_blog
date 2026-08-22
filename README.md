# 模型与系统笔记

这是基于官方 [Chirpy Starter](https://github.com/cotes2020/chirpy-starter) 配置的 GitHub Pages 技术博客。

站点保留 Chirpy 的原生结构和视觉，不需要手工维护首页、文章列表、分类、标签、归档或文章目录。日常使用只需编辑 Markdown。

## 发布新文章

复制草稿模板，并把文件名改为 `年-月-日-英文短标题.md`：

```bash
cp _drafts/post-template.md _posts/2026-08-23-my-new-note.md
```

然后修改文件顶部的 Front Matter：

```yaml
---
title: "文章标题"
description: "一句话摘要"
categories: [技术笔记]
tags: [LLM, 推理系统]
math: true
mermaid: true
---
```

正文直接使用 Markdown。推送到 GitHub 后：

- 首页会自动出现新文章；
- 分类、标签和归档页面会自动更新；
- 二级标题会自动进入文章目录；
- 公式、Mermaid 图和代码高亮由主题处理。

## 修改站点信息

主要配置集中在 `_config.yml`：

- `title`：站点名称；
- `tagline`：侧栏副标题；
- `description`：站点摘要；
- `github.username`：GitHub 用户名；
- `social`：作者名称和公开链接；
- `avatar`：侧栏头像；
- `theme_mode`：浅色、深色或跟随系统。

“关于”页面直接编辑 `_tabs/about.md`。

## 部署到 GitHub Pages

1. 创建名为 `<用户名>.github.io` 的公开仓库；
2. 将本目录全部内容放入仓库根目录；
3. 推送到 `main` 分支；
4. 在 `Settings → Pages → Build and deployment` 中选择 `GitHub Actions`；
5. 等待自带的 `Build and Deploy` 工作流完成。

普通项目仓库同样支持，工作流会自动处理仓库路径前缀。

## 本地预览

需要 Ruby 3.4 和 Bundler：

```bash
bundle install
bundle exec jekyll serve
```

然后打开终端中显示的本地地址。

## 目录

```text
_posts/                 # 已发布文章
_drafts/                # 草稿与文章模板
_tabs/                  # 关于、分类、标签、归档页面
_config.yml             # 站点设置
.github/workflows/      # GitHub Pages 自动部署
```

## 上游与许可证

- Starter：[cotes2020/chirpy-starter](https://github.com/cotes2020/chirpy-starter)
- Theme：[cotes2020/jekyll-theme-chirpy](https://github.com/cotes2020/jekyll-theme-chirpy)
- 当前基础版本：Chirpy 7.6

模板代码采用 MIT License，详见 [LICENSE](LICENSE)。
