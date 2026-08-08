# QuamtumDusk Notes

一个基于 [Astro](https://astro.build) + [Fuwari](https://github.com/saicaca/fuwari) 的**知识库**——围绕"从零手搓 LLM 训练"与"点集拓扑"两个主题，面向**检索**而不是"刷完即走"的阅读。

> 它看起来像个博客，但设计目标是知识库：文章是**可被锚点直达的条目网络**，不是线性流水。这篇 README 说明它为什么这样设计、怎么用、怎么维护。

![Node.js >= 20](https://img.shields.io/badge/node.js-%3E%3D20-brightgreen)
![pnpm >= 9](https://img.shields.io/badge/pnpm-%3E%3D9-blue)

---

## 目录

1. [为什么从 fuwari 改造](#为什么从-fuwari-改造)
2. [核心能力](#核心能力)
3. [使用工作流](#使用工作流)
4. [知识库组织哲学](#知识库组织哲学)
5. [内容结构](#内容结构)
6. [写作规范](#写作规范)
7. [自定义插件架构](#自定义插件架构)
8. [命令](#命令)
9. [CI / 部署建议](#ci--部署建议)
10. [运维注意](#运维注意)
11. [License](#license)

---

## 为什么从 fuwari 改造

原始 fuwari 是为"个人博客"设计的，但本站的用途是**知识库**——需要编译发布、无后台、文章量大会长期累积。实测原始模板（fuwari.vercel.app）在阅读和管理上有几个灾难级缺陷：

| 原始模板的缺陷 | 为什么是灾难 |
| --- | --- |
| 阅读页侧栏只有"头像名片 + 分类 + 标签" | **没有任何文章导航**——读一篇时无法跳到相关知识，只能回首页。营销组件占了整条侧栏 |
| TOC 浅（h1/h2 扁平）且独占右侧大半宽度 | 长文章没有结构导航，TOC 占位不干活 |
| 文章列只有 ~800px（视口 31%） | 两侧被无用的侧栏和浅 TOC 挤占，正文细条 |
| 文章之间零交叉引用 | 知识之间没有链接，读完一篇就断了 |
| 顶部 775px banner | 进正文前先滚过一个横幅 |
| 无检索锚点 | 想"找到某个定理"只能全文搜索，没有精确落点 |

**设计哲学**：外观好 ≠ 好用。这个模板的外观不错，但显然没有"大量读写知识"的使用习惯支撑——本站的全部改造，都是把"营销展示型模板"改造成"检索优先的知识库"。

---

## 核心能力

### 1. Wiki 链接体系 `[[...]]`

Obsidian 式文章互链，支持四种形态：

```markdown
[[损失函数与-100掩码]]              # 文件名（主键，推荐）
[[07-紧致性#单调子序列引理]]        # 锚点直达小节
[[自动求导与梯度|自动求导]]         # 显示别名
[[pytorch/dataset]]                # 相对路径（逃生通道）
```

**设计要点**：

- **文件名是主键**——内容是重组最频繁的（改标题、挪文件夹都不断链），文件名是最稳定的标识。解析顺序：**文件名 > 相对路径 > 标题**。
- **锚点 = 荧光笔标题去空格**：`## <<序列极限的唯一定理>>` 的锚点是 `#序列极限的唯一定理`，写链接零规则零歧义。
- **构建门禁**：未解析链接、标题命中、文件名冲突在构建时全部报警（`file:line`）；`WIKI_LINK_STRICT=true` 时存在未解析/冲突**直接构建失败**——断链不可能悄悄出现。
- **崩溃防护**：索引构建对"文件在 readdir 后消失"（编辑器保存 = 删除+改名）做了 try/catch，不会因为保存动作崩掉整个管线。

### 2. 检索锚点 + 荧光笔标题

```markdown
## <<序列极限的唯一定理>>
```

`<< >>` 包裹的标题渲染成**荧光笔高亮**（半透明底色、圆角、只包文字宽度），同时锚点 id = 标题去掉空格。高亮让"这是检索锚点"一眼可见，不是普通小节标题。

### 3. amsthm 式证明块 `:::proof`

````markdown
:::proof
设 $p \ne q$，令 $\varepsilon = \tfrac{1}{2}d(p,q) > 0$。

$$
d(p,q) \le d(p,p_n) + d(p_n,q) < 2\varepsilon.
$$

矛盾，故 $p = q$。
:::
````

渲染为论文级证明排版：加粗"证明"标题 + 散文/居中数学 + 右对齐 ■ QED。**display 数学必须跨行**（`$$` 独占一行）——单行 `$$x$$` 会被 micromark 当成行内渲染。

### 4. 目录树侧栏

- 默认**收起为文件夹形态**，点开展开（比原始模板的"营销组件侧栏"强：这是真正的导航）
- 文件夹带**分类徽章**，点击跳转归档页
- 高亮跟随 SPA 导航（swup `visit:start` 同步）

### 5. 3 级 TOC + 向左展开

完整结构导航（h1/h2/h3），悬浮按钮收起/展开，激活项跟随滚动高亮。替代原始模板"又浅又占位"的 TOC。

### 6. 数学渲染优化

KaTeX `output: html`——砍掉冗余 MathML（读屏器可访问性换 DOM 减半，页面解析成本大降）。跨行 `$$` 居中 display。

---

## 使用工作流

**场景一：写论文推导需要某个定理**
1. 打开对应知识库的 `00-总览`（索引地图）
2. 表格里找到定理名 → 点锚点直达
3. 从 `[[07-紧致性#单调子序列引理]]` 跳到证明，复制思路

**场景二：写 LoRA 代码，矩阵运算晕**
1. 侧栏目录树展开 `pytorch/` → 进 `[[数学符号到代码]]`
2. "两个世界，一个转置"判定表直接查：组合权重普通 `@`，作用到数据才 `x @ W.T`

**场景三：增量写拓扑笔记**
1. 侧栏默认收起，点 `点拓/` 展开
2. 写完一篇后，回 `00-总览` 补索引行（新定理的锚点）
3. `WIKI_LINK_STRICT=true pnpm build` 确认没断链再提交

---

## 知识库组织哲学

| 原则 | 为什么 |
| --- | --- |
| **单一主题** | 每篇只讲一件事，检索时"名字即位置"，不散 |
| **检索锚点** | 每个定理/引理/定义独立标题 + 锚点，可被 `[[篇#锚点]]` 直达 |
| **索引地图** | 每个知识库 `00-总览` 把全部条目列成可点击表——论文推导的查表入口 |
| **位置→定理→思路→证明→链接** | 统一模板：先给结论，再讲思路，证明用标准竖写，最后互链 |
| **归属行** | 每篇标注库归属（transformers / PyTorch / Python），跨库不混淆 |
| **构建门禁** | 断链在 CI 失败，不允许"悄悄坏掉" |

---

## 内容结构

```
src/content/posts/
├── huggingface/   # transformers / datasets 使用契约（类型、特殊 token、生成机制）
├── pytorch/       # 训练引擎（张量、模块、数据管线、数学符号→代码）
├── python/        # 语言基础 + 常用工具（f-string、tqdm、迭代）
├── 点拓/          # 点集拓扑（12 篇完整课程 + 00-总览索引，99+ 检索锚点）
└── guide/         # 本站使用说明（wiki 链接语法等）
```

---

## 写作规范

### Frontmatter

```yaml
---
title: 文章标题
published: 2026-08-07
description: 一句话摘要（也是检索关键词来源）
tags: [主题, 关键词]
category: 所属分类
---
```

### 一篇标准文章的结构

```markdown
# 主题

> 归属：**PyTorch · `torch.nn`** —— 一句话定位。
> 索引：[[00-总览]]。上一篇：[[xx]]。下一篇：[[yy]]。

## 概念本体：先讲清楚是什么

## <<定理/引理/定义名>>        ← 荧光笔锚点

> 位置：原笔记 pXX（页码溯源）

**定理**：陈述。

**思路**：一句话证明直觉。

:::proof
标准证明（散文 + 跨行居中数学）。
:::

**链接**：[[相关篇#锚点]] · [[另一篇]]

## 小结
| 知识 | 要点 |
```

### 其他约定

- Admonitions：`::::note / tip / important / warning / caution`（**4 冒号**，项目约定）
- 代码块用 Expressive Code：` ```python title="xxx.py" `
- Wiki 链接规范见 [guide](src/content/posts/guide/index.md)

---

## 自定义插件架构

本站相对 fuwari 新增/重写的插件（维护时先看这里）：

| 插件 | 位置 | 职责 |
| --- | --- | --- |
| `remark-wiki-link.js` | `src/plugins/` | Wiki 链接解析：三张映射表（文件名/路径/标题）、锚点支持、诊断收集（冲突/标题命中/未解析）、签名缓存、崩溃防护 |
| `remark-highlight-heading.js` | `src/plugins/` | `<< >>` 荧光笔标题：去标记 + 加 `hl-heading` 类 + **自设锚点 id**（rehype-slug 见已有 id 不覆盖） |
| `rehype-component-admonition.mjs` | `src/plugins/` | `ProofComponent`：`:::proof` 渲染为 amsthm 式证明块 |
| `remark-reading-time.mjs` 等 | `src/plugins/` | 原有插件保留 |

管线顺序（`astro.config.mjs`）：`remarkMath → … → remarkDirective → remarkSectionize → parseDirectiveNode → remarkHighlightHeading → remarkWikiLink`（wiki 插件必须最后）。rehype 端：`rehypeKatex({output:'html'}) → rehypeSlug → rehypeComponents`。

---

## 命令

| 命令 | 作用 |
| --- | --- |
| `pnpm dev` | 本地开发 `localhost:4321` |
| `pnpm build` | 生产构建（含 pagefind 索引） |
| `WIKI_LINK_STRICT=true pnpm build` | 严格构建：存在断链/冲突直接失败（CI 用） |
| `pnpm check` | Astro 类型检查 |
| `pnpm new-post <name>` | 新建文章 |

---

## CI / 部署建议

Cloudflare Pages 随 push 自动部署。CI 里建议加严格构建：

```yaml
# Cloudflare Pages build config 或 CI workflow
build_command: "WIKI_LINK_STRICT=true pnpm build"
```

这样任何未解析 wiki 链接、文件名冲突都会让部署失败，而不是上线一个带灰链的站点。

---

## 运维注意（踩坑记录）

- **`astro build` 与 dev server 不能同时跑**：共享 `.astro` 缓存，并发会损坏导致 tailwind `@apply` 报错（`btn-regular-dark`）。构建前先停 dev。
- **WSL/9P 文件监听不可靠**：改文件后 dev 可能不热更新、甚至喂旧缓存——大改后重启 dev；仍异常时清 `.astro` + `node_modules/.astro` + `node_modules/.vite` 再重启。
- **改标题/文件名前**先 `grep` 确认没有按旧标题写的 wiki 链接（链接以文件名为主键，标题改动一般不波及）。
- **`The btn-regular-dark class does not exist`**：`@apply` 跨文件自定义类有顺序竞争，已在 markdown.css 内联展开；再出现先清 vite 缓存。

---

## License

MIT（基于 [Fuwari](https://github.com/saicaca/fuwari)，原模板 MIT 协议）。
