# QuamtumDusk Notes Engine

> **Live Website**: [https://blog.yjluo.ccwu.cc/](https://blog.yjluo.ccwu.cc/)  
> **Original Template Base**: [Fuwari Template (Vercel)](https://fuwari.vercel.app/) | [Fuwari GitHub](https://github.com/saicaca/fuwari)

一个从开源项目 [Fuwari](https://github.com/saicaca/fuwari) 深度二次开发而来的**检索优先型学术知识库**。区别于传统个人博客“流水账/看完即走”的呈现方式，本系统围绕“从零手搓 LLM 训练”与“点集拓扑”等长篇硬核知识体系建模，重构了排版、检索、导航与关联机制，旨在打造极致沉浸、无噪音、可精确落点的“数字学术书房”。

![Node.js >= 20](https://img.shields.io/badge/Node.js-%3E%3D20-brightgreen)
![Astro v5](https://img.shields.io/badge/Astro-v5-orange)
![pnpm >= 9](https://img.shields.io/badge/pnpm-%3E%3D9-blue)
![License MIT](https://img.shields.io/badge/License-MIT-purple)

---

## 目录

1. [项目背景与二开动机](#项目背景与二开动机)
2. [系统核心能力与架构设计](#系统核心能力与架构设计)
3. [学术阅读美学与极简克制规范](#学术阅读美学与极简克制规范)
4. [使用场景与知识检索工作流](#使用场景与知识检索工作流)
5. [内容结构与分类规范](#内容结构与分类规范)
6. [Markdown 写作契约与格式规范](#markdown-写作契约与格式规范)
7. [自定义 Remark / Rehype 插件管线](#自定义-remark--rehype-插件管线)
8. [开发与构建命令](#开发与构建命令)
9. [CI / CD 部署门禁](#ci--cd-部署门禁)
10. [运维踩坑与注意事项](#运维踩坑与注意事项)
11. [License & Acknowledgments](#license--acknowledgments)

---

## 项目背景与二开动机

原始 Fuwari 模板是一个视觉出色的**展示型个人博客**，但在承载大体量、高密度、多定理关联的学术与技术笔记时，存在若干无法妥协的“使用痛点”。本工程基于真实的高频检索与阅读需求，进行了全面的重构：

| 原始模板痛点 | 本项目重构与解决方案 |
| :--- | :--- |
| **侧栏仅有营销卡片**：缺乏层级树，无法在阅读中查看同目录下其他章节 | **全新文件树导航**：支持无限层级目录展开，在阅读文章时可随时在左侧切换相关笔记 |
| **TOC 扁平且占位**：只支持 h1/h2，且独占右侧固定宽度 | **3 级 TOC + 向左交互展开**：支持 h1/h2/h3 完整的结构树，向左悬浮展开，激活项随滚动高亮 |
| **正文过度挤压**：两侧被大宽度侧栏与浅 TOC 挤占 | **48rem (768px) 黄金阅读列**：保持每行 70~80 中文字符的最佳阅读视野 |
| **孤立无交叉引用**：文章间无法像 Obsidian 般互链 | **Wiki 链接体系 `[[...]]`**：支持文件名/别名/小节锚点直达，构建期诊断校验断链 |
| **视觉高亮喧宾夺主**：黄色色块、红虚线干扰阅读 | **极简学术视觉（Zen Minimalist）**：平时保持中性文字层次，仅在 Hover/交互时显示主题色 |

---

## 系统核心能力与架构设计

### 1. Obsidian 式 Wiki 链接体系 (`[[...]]`)
提供天然的知识网路图谱链接，支持 4 种书写形态：

```markdown
[[损失函数与-100掩码]]              # 1. 文件名直连（主键，最稳定）
[[07-紧致性#单调子序列引理]]        # 2. 锚点直达小节
[[自动求导与梯度|自动求导]]         # 3. 别名展示
[[pytorch/dataset]]                # 4. 相对路径
```

- **构建期严格诊断**：内嵌 AST 诊断器，在 `WIKI_LINK_STRICT=true` 下存在未解析链接或文件名冲突直接中断构建，阻止断链上线。
- **稳定性优先**：在编辑器热更新（文件删除+重建）场景下具备 Try/Catch 崩溃防护。

### 2. 精确检索锚点与 Anchor 标记
支持 `## <<定理名称>>` 语法，构建期自动去除标记并注册为全局平滑滚动锚点。

### 3. amsthm 规范学术证明块 (`:::proof`)
```markdown
:::proof
设 $p \ne q$，令 $\varepsilon = \tfrac{1}{2}d(p,q) > 0$。

$$
d(p,q) \le d(p,p_n) + d(p_n,q) < 2\varepsilon.
$$

矛盾，故 $p = q$。
:::
```
渲染为规范的学术证明排版：主题色“证明”标头 + 散文段落/居中 LaTeX 公式 + 右下角 `■` QED 结尾符。

### 4. 动态多层目录树 (Sidebar Directory Tree)
侧栏默认收起为干净的文件夹结构，展开即可预览全站目录，配有分类统计徽章，并与 SPA 无刷新加载（Swup）无缝集成。

---

## 学术阅读美学与极简克制规范

为避免高饱和度色彩破坏阅读专注力，全站遵循以下**极简克制（Minimalist Restraint）**视觉原则：

- **平时素雅，悬停点亮**：
  - **标签胶囊 (`.tag-capsule`)**：默认显示为浅调中性灰包围框，仅在鼠标 Hover 悬停时触发主题 Accent 亮色。
  - **正文链接与 Wiki 链接**：默认保持与正文完全一致的颜色（`color: inherit`），消除满屏彩字；悬停时触发下划线与主题色渐变。
- **背景透亮与自然过渡**：
  - **证明块与引用块**：完全移除任何异色背景填充，与纸面背景保持 100% 融为一体。
  - **顶栏 Banner 渐变**：保留顶部 85% 图像的完整清晰度，仅在底边缘 15% 紧凑自然消隐。
- **纯粹无位移动画**：从首页点击文章导航时，取消 SideBar 的位移/滑动特效，改为原地的纯渐入渐出（Fade-in / Fade-out）。

---

## 使用场景与知识检索工作流

- **场景 A：定理推导与查表**
  进入 `点拓/00-总览` 索引地图 → 表格查找目标定理 → 点击 `[[07-紧致性#单调子序列引理]]` 锚点直接跳入对应推导段落。
- **场景 B：代码实现与数学符号对照**
  展开侧栏 `pytorch/` → 打开 `[[数学符号到代码]]` → 查阅权重矩阵转置与广播契约判定表。

---

## 内容结构与分类规范

```
src/content/posts/
├── huggingface/   # Transformers / Datasets 核心契约与生成机制
├── pytorch/       # 深度学习引擎（张量切片、Module、梯度与优化器）
├── python/        # 语言特性、上下文管理器与常用工具
├── 点拓/          # 点集拓扑（12 篇完整讲义 + 00-总览索引地图）
└── guide/         # 本站 Markdown 延伸扩展语法说明
```

---

## Markdown 写作契约与格式规范

### Frontmatter 标准模版
```yaml
---
title: 文章标题
published: 2026-08-07
description: 一句话摘要（作为卡片预览与 Pagefind 索引关键字）
tags: [主题, 关键字]
category: 所属分类
---
```

### 标准文章小节结构
```markdown
# 主题

> 归属：**PyTorch · `torch.nn`**
> 索引：[[00-总览]]。上一篇：[[05-前篇]]。下一篇：[[07-后篇]]。

## <<定理名称>>

> 位置：原笔记 p22（中）

**定理**：陈述内容。

**思路**：一句话证明直觉。

:::proof
证明推导过程。
:::

**链接**：[[相关章节#锚点]]
```

---

## 自定义 Remark / Rehype 插件管线

| 插件名称 | 文件路径 | 核心职责 |
| :--- | :--- | :--- |
| `remark-wiki-link.js` | `src/plugins/` | 解析 `[[...]]` Wiki 链接，构建文件名/路径/标题三级映射，收集未解析断链诊断 |
| `remark-highlight-heading.js` | `src/plugins/` | 处理 `<< >>` 锚点标题，注入 `hl-heading` 类名并注册 slug |
| `rehype-component-admonition.mjs` | `src/plugins/` | 解析 `:::proof` 并转换为包含 `proof-title` 与 `proof-qed` 的 DOM 结构 |

**管线执行顺序**（`astro.config.mjs`）：  
`remarkMath` ➔ `remarkDirective` ➔ `remarkSectionize` ➔ `parseDirectiveNode` ➔ `remarkHighlightHeading` ➔ `remarkWikiLink` （Wiki 插件严格保持最后执行）。

---

## 开发与构建命令

| 命令 | 说明 |
| :--- | :--- |
| `pnpm dev` | 启动本地热更新开发服务器 (`http://localhost:4321`) |
| `pnpm build` | 静态编译生成产物并运行 Pagefind 全文搜索引擎索引构建 |
| `WIKI_LINK_STRICT=true pnpm build` | 严格构建模式：存在未解析 Wiki 链接或冲突时中断构建（CI 校验用） |
| `pnpm check` | 运行 Astro 类型检查 |

---

## CI / CD 部署门禁

推荐在 Cloudflare Pages 或 GitHub Actions 构建管线中使用严格构建模式：

```yaml
# Cloudflare Pages / CI Pipeline
build_command: "WIKI_LINK_STRICT=true pnpm build"
```

这可保证任何未解析的 Wiki 链接或潜在文件命名冲突均无法静默部署至生产环境。

---

## 运维踩坑与注意事项

1. **`pnpm dev` 与 `pnpm build` 并发限制**：两者共享 `.astro` 缓存目录，并发运行可能导致 Tailwind `@apply` 样式顺序竞争异常。建议构建前先关闭 dev 开发服务。
2. **WSL / 9P 文件监听机制**：WSL 环境下多次重命名文件后，热更新可能残留旧缓存。如遇样式异常，清理 `.astro` 与 `node_modules/.vite` 后重启开发服务器即可。

---

## License & Acknowledgments

- **Live Site**: [https://blog.yjluo.ccwu.cc/](https://blog.yjluo.ccwu.cc/)
- **Upstream Base**: [Fuwari Template](https://github.com/saicaca/fuwari) (MIT License)
- **License**: MIT License.
