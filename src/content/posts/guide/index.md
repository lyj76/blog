---
title: Simple Guides for Fuwari
published: 2024-04-01
description: "How to use this blog template."
image: "./cover.jpeg"
tags: ["Fuwari", "Blogging", "Customization"]
category: Guides
draft: false
---

> Cover image source: [Source](https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/208fc754-890d-4adb-9753-2c963332675d/width=2048/01651-1456859105-(colour_1.5),girl,_Blue,yellow,green,cyan,purple,red,pink,_best,8k,UHD,masterpiece,male%20focus,%201boy,gloves,%20ponytail,%20long%20hair,.jpeg)

This blog template is built with [Astro](https://astro.build/). For the things that are not mentioned in this guide, you may find the answers in the [Astro Docs](https://docs.astro.build/).

## Front-matter of Posts

```yaml
---
title: My First Blog Post
published: 2023-09-09
description: This is the first post of my new Astro blog.
image: ./cover.jpg
tags: [Foo, Bar]
category: Front-end
draft: false
---
```

| Attribute     | Description                                                                                                                                                                                                 |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `title`       | The title of the post.                                                                                                                                                                                      |
| `published`   | The date the post was published.                                                                                                                                                                            |
| `description` | A short description of the post. Displayed on index page.                                                                                                                                                   |
| `image`       | The cover image path of the post.<br/>1. Start with `http://` or `https://`: Use web image<br/>2. Start with `/`: For image in `public` dir<br/>3. With none of the prefixes: Relative to the markdown file |
| `tags`        | The tags of the post.                                                                                                                                                                                       |
| `category`    | The category of the post.                                                                                                                                                                                   |
| `draft`        | If this post is still a draft, which won't be displayed.                                                                                                                                                    |

## Where to Place the Post Files



Your post files should be placed in `src/content/posts/` directory. You can also create sub-directories to better organize your posts and assets.

```
src/content/posts/
├── post-1.md
└── post-2/
    ├── cover.png
    └── index.md
```

## Wiki Links：文章间链接（Obsidian 式）

文章之间可以用 `[[...]]` 互相引用，渲染后变成可点击的链接。

**推荐写法**：

```markdown
[[tensor-and-parameter]]                   # ① 用文件名（推荐，最稳定）
[[自动求导与梯度|自动求导]]                # ② 文件名 + 显示别名（| 后面是显示文本）
```

**其他可用写法**（能用，但不推荐——见下方警告规则）：

```markdown
[[pytorch/dataset]]                        # ③ 相对路径（文件夹重组会断链）
[[PyTorch 核心：Tensor 与 nn.Parameter]]   # ④ 用文章标题（标题改名会断链，且命中会报警）
```

**解析与校验规则**：

- **解析顺序**：文件名 > 相对路径 > 标题
- **文件名是主键**：内容重组时改标题、改路径都不影响链接——文件名是文章最稳定的标识
- **找不到目标**：渲染成灰色虚线文字，构建时打印 `file:line` 警告
- **标题命中**：构建时打印警告，建议改用文件名（标题是可变的展示层）
- **文件名冲突**：两个文件同名时构建报警，需用相对路径形式区分（如 `[[a/xxx]]` 和 `[[b/xxx]]`）
- **CI 强制**：`WIKI_LINK_STRICT=true pnpm build` 时，存在未解析或冲突链接会让构建失败
- **代码块里不会解析**：``` 包裹的代码中的 `[[...]]` 保持原样
- **性能**：链接映射在构建时扫描一次并缓存（按文件 mtime 自动失效），文章多了也不会拖慢构建

**配合知识库的用法**：写 PyTorch/HuggingFace 笔记时，用 `[[标题]]` 把相关概念串起来——例如在 [[损失函数与-100掩码]] 里写"掩码机制见 [[causallm-前向契约]]"，两篇文章就建立了互相引用的关系（点击上面的链接可以直接跳转）。
