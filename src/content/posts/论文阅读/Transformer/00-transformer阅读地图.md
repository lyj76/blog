---
title: Transformer 阅读地图
published: 2026-08-08
description: Transformer 系列索引：论文主线、位置编码演化、多头注意力、归一化、FFN 演化。
tags: [论文阅读, Transformer, 阅读地图]
category: 论文阅读
---

# Transformer 阅读地图

本系列给一切现代模型提供**架构语言**。从 2017 年原论文的主线开始，把每个组件单独拆出来讲它的演化——位置编码、多头、归一化、FFN——每条线都指向现代模型（Qwen/LLaMA）为什么长成今天这样。

- [[论文阅读/transformer/01-transformer主线]]：从序列建模困境到"只要注意力"，Encoder-Decoder 全流程。
- [[论文阅读/transformer/02-位置编码的演化]]：正弦波 → learned PE → RoPE → ALiBi。
- [[论文阅读/transformer/03-多头注意力]]：切分不增计算，14 双眼睛怎么分工。
- [[论文阅读/transformer/04-残差与归一化]]：LayerNorm → RMSNorm，Post-LN → Pre-LN。
- [[论文阅读/transformer/05-ffn的演化]]：ReLU → GELU → SwiGLU 三矩阵结构。

**依赖关系**：01 是主文，其余四篇是它的组件深潜。02/03/04/05 都直接服务 [[论文阅读/qwen/01-qwen主线]]。

## 原始论文

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：Vaswani et al., 2017，Transformer 开山。
- [RoFormer](https://arxiv.org/abs/2104.09864)：Su et al., 2021，RoPE 旋转位置编码。
- [Attention is not all you need](https://arxiv.org/abs/2305.13245)：RoPE 外推性分析。
- [Train Short, Test Long](https://arxiv.org/abs/2108.12409)：ALiBi。
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)：SwiGLU。
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)：RMSNorm。
