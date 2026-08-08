---
title: Qwen 阅读地图
published: 2026-08-08
description: Qwen 系列索引：主线的完整链路、抽象数学结构、实现层、分词向量化、交叉熵损失、GQA、生成采样。
tags: [论文阅读, Qwen, 阅读地图]
category: 论文阅读
---

# Qwen 阅读地图

本系列按三层拆解 Qwen2-0.5B：**抽象数学结构**（前向就是一堆矩阵乘法）、**具体实现层**（config.json 与代码里的投影矩阵）、**原理**（分词、损失、GQA、采样）。主文先给完整链路，专题篇负责把每一环讲透。

- [[论文阅读/qwen/01-qwen主线]]：文本 → 下一个词的完整链路。
- [[论文阅读/qwen/02-qwen-decoder-block]]：一个 Decoder block 的完整数学过程。
- [[论文阅读/qwen/03-qwen输入与训练]]：token、embedding、右移标签与交叉熵。
- [[论文阅读/qwen/04-qwen推理与采样]]：GQA、KV Cache、logits 与采样。
- [[论文阅读/qwen/05-qwen模型文件与代码]]：模型目录、config 与 Python 模块映射。

**依赖关系**：01 是总览，02 是数学核心，03 解释训练，04 解释推理，05 负责从模型目录回到代码。Transformer 系列提供架构语言（[[论文阅读/transformer/00-transformer阅读地图]]）；tokenizer 契约、generate 机制等复用 huggingface 库（见各篇链接）。

## 参考材料

- [Qwen2.5 技术报告](https://qwenlm.github.io/blog/qwen2.5/)
- [Qwen2 技术报告](https://arxiv.org/abs/2407.10671)
- 本地笔记：`E:\test\qwen抽象数学结构.md`、`qwen模型的使用.md`、`qwen模型与理解2.md`
