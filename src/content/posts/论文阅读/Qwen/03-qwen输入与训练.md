---
title: Qwen 输入与训练：从 token 到交叉熵
published: 2026-08-08
description: 把分词、chat template、embedding、右移标签、causal mask、交叉熵和 -100 label mask 串成一条训练闭环。
tags: [论文阅读, Qwen, tokenizer, 交叉熵, 训练]
category: 论文阅读
---

# Qwen 输入与训练：从 token 到交叉熵

> Qwen 训练时并不直接比较两段字符串，而是在每个位置比较“词表上的预测分布”和“下一个真实 token”。这篇把输入契约和损失契约接成一条线。

## <<文本如何变成训练样本>>

字符串先经过 tokenizer：

$$
s\xrightarrow{\operatorname{encode}}t=(t_1,\ldots,t_T),
\qquad t_i\in\{0,\ldots,V-1\}.
$$

聊天数据还要先经过 chat template，把角色和消息边界变成特殊 token。最终送进模型的是 `input_ids`，不是原始字符串；padding 时还会有 `attention_mask`。

同一段文字换 tokenizer，得到的 $t$ 可能完全不同。因此 tokenizer、模型权重和 special token 配置必须配套，详细文件契约见 [[tokenizer-类型契约]]。

## <<Embedding：ID 只负责索引>>

embedding 矩阵为 $E\in\mathbb{R}^{V\times896}$：

$$
H^0=E[t]\in\mathbb{R}^{T\times896},
\qquad H^0_i=E[t_i].
$$

`input_ids` 中的整数不是向量坐标，也不参与大小比较；它们只用来索引 $E$ 的行。后面的 Decoder 才把这些向量变成上下文相关的 hidden states。

## <<右移标签：第 t 个位置预测谁>>

训练目标是下一个 token。给定序列

$$
(t_1,t_2,t_3,\ldots,t_T),
$$

输入与标签错开一位：

$$
\text{input}=(t_1,t_2,\ldots,t_{T-1}),
$$

$$
\text{label}=(t_2,t_3,\ldots,t_T).
$$

模型在输入位置 $i$ 产生对 $t_{i+1}$ 的预测。causal mask 保证位置 $i$ 看不到右侧真实 token；因此一次前向可以并行计算所有位置，同时保持自回归约束。

## <<交叉熵：只惩罚真实 token 的概率>>

位置 $i$ 的 logits 为 $z_i\in\mathbb{R}^{V}$，概率为 $p_i=\operatorname{softmax}(z_i)$，真实下一个 token 为 $y_i$：

$$
L_i=-\log p_i[y_i].
$$

写成分布交叉熵：

$$
H(q_i,p_i)=-\sum_{j=1}^{V}q_{i,j}\log p_{i,j},
$$

one-hot 目标 $q_i$ 只在 $y_i$ 处为 1，所以正好化成 $-\log p_i[y_i]$。真实 token 概率越小，惩罚越大；$p_i[y_i]\to0$ 时损失趋于无穷。

## <<为什么不用概率空间的 MSE>>

如果把 one-hot 向量和概率向量做 MSE：

$$
L_{MSE}=\frac{1}{V}\sum_{j=1}^{V}(p_{i,j}-q_{i,j})^2,
$$

它在概率单纯形上有界。真实 token 的概率从 $10^{-2}$ 降到 $10^{-20}$，MSE 都只看到它们接近 0；交叉熵会继续增大，区分“低概率”和“极低概率”。

此外，MSE 对 logits 的梯度还要经过 softmax Jacobian：

$$
\frac{\partial L_{MSE}}{\partial z}
=\frac{\partial p}{\partial z}\frac{\partial L_{MSE}}{\partial p}.
$$

交叉熵与 softmax 合并后则有简洁梯度：

$$
\frac{\partial L_i}{\partial z_{i,j}}
=p_{i,j}-q_{i,j}.
$$

语言建模关心的是类别概率和似然，这正是交叉熵的对象；MSE 更像连续向量回归的距离。

## <<-100：哪些位置不计入损失>>

实际训练不一定让每个位置都参与平均。例如 padding、只用于提供上下文的 prompt 部分，可能不应成为监督目标。用 `ignore_index=-100` 标记这些位置：

$$
\mathcal{I}=\{i:\operatorname{label}_i\ne-100\},
$$

$$
L=\frac{1}{|\mathcal{I}|}\sum_{i\in\mathcal{I}}-\log p_i[y_i].
$$

`-100` 不是词表 token，它是损失函数识别的忽略标记；不要把它送进 tokenizer，也不要把它当成真实类别。

## 训练闭环

$$
\text{text}
\to\text{token IDs}
\to\text{embedding}
\to\text{24 blocks}
\to\text{logits}
\to\text{shifted cross-entropy}
\to\text{backward}.
$$

模型学习的是“给定前文，下一个 token 的条件分布”。训练目标、输入序列和 mask 三者必须位置对齐；错一位，损失仍然能算，但学习目标已经变了。
