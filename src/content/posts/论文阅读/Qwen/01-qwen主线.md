---
title: Qwen 主线：从文本到下一个词的完整链路
published: 2026-08-08
description: 一条流水线讲完 Qwen：文本 → token ID → embedding → 24 层 Decoder → LM Head → Softmax → 下一个词。
tags: [论文阅读, Qwen, 主线]
category: 论文阅读
---

# Qwen 主线：从文本到下一个词的完整链路

> Qwen 是 Decoder-only 语言模型。本文只抓一条主线：文本如何变成 token，token 如何经过 24 层 block，最后如何变成词表上的概率分布。一个 block 的完整数学见 [[论文阅读/qwen/02-qwen-decoder-block]]，架构背景见 [[论文阅读/transformer/01-transformer主线]]。

## 一条流水线：文本变成下一个词

设输入文本为 $s$，tokenizer 输出长度为 $T$ 的整数序列：

$$
\operatorname{Tokenizer}(s)=t=(t_1,\ldots,t_T),\qquad t_i\in\{0,\ldots,V-1\}.
$$

以用户当前的 Qwen2-0.5B 配置为例：$d_{model}=896$、$L=24$、$h_q=14$、$h_{kv}=2$、$d_h=64$、$d_{ff}=4864$、$V=151936$。

整个前向可以压成：

$$
t
\xrightarrow{\text{Embedding}}
H^0\in\mathbb{R}^{T\times896}
\xrightarrow{\text{24 Decoder blocks}}
H^{24}\in\mathbb{R}^{T\times896}
\xrightarrow{\text{final RMSNorm}}
\xrightarrow{\text{LM Head}}
Z\in\mathbb{R}^{T\times151936}
\xrightarrow{\text{取最后一行 + softmax}}
p_T\in\mathbb{R}^{151936}.
$$

$p_T$ 是当前位置预测下一个 token 的概率分布。生成一个 token 后，把它追加到输入序列，重新进行下一轮预测；推理时用 KV Cache 保存历史层的 $K,V$，避免每轮重复计算历史 token。

## <<Token：文本到数字的第一跳>>

模型的输入不是字符串，而是 token ID：

$$
s\longrightarrow(t_1,\ldots,t_T).
$$

$t_i$ 只是词表索引，不代表一个完整的"词"。它可能是一个汉字、一个英文子词、一个标点或一个控制 token。Qwen 的词表大小在该配置中为 $151936$，分词和训练输入见 [[论文阅读/qwen/03-qwen输入与训练]]。

## <<Embedding：查表不是计算>>

令 embedding 矩阵为 $E\in\mathbb{R}^{V\times896}$。第 $i$ 个 token 直接取矩阵第 $t_i$ 行：

$$
H^0_i=E[t_i]\in\mathbb{R}^{896},
$$

整句写成

$$
H^0=E[t]\in\mathbb{R}^{T\times896}.
$$

这一步是索引，不是把 token ID 乘某个数字。Qwen 不在这里加经典 Transformer 的绝对正弦位置向量；位置通过每层 attention 内部的 RoPE 作用到 $Q,K$ 上。

## <<24 层 Decoder：骨架在循环什么>>

Qwen 每层是 Pre-RMSNorm 的 Decoder block。设第 $l-1$ 层输出为 $H^{l-1}$：

$$
U^l=H^{l-1}+\operatorname{Attention}\!\left(\operatorname{RMSNorm}(H^{l-1})\right),
$$

$$
H^l=U^l+\operatorname{SwiGLU}\!\left(\operatorname{RMSNorm}(U^l)\right),
\qquad l=1,\ldots,24.
$$

每个张量的外部形状保持 $T\times896$。attention 内部把 $896$ 投影为 14 个 $64$ 维 Q 头、2 个 $64$ 维 K/V 头；SwiGLU 内部把 $896$ 升到 $4864$，逐元素门控后再降回 $896$。完整 block 见 [[论文阅读/qwen/02-qwen-decoder-block]]，文件与代码映射见 [[论文阅读/qwen/05-qwen模型文件与代码]]。

Qwen 是 causal decoder。当前位置的 attention 只能看自己和左侧 token，不能读取右侧答案；这由 causal mask 把未来位置的分数设为 $-\infty$ 实现。

## <<LM Head：896 维回到词表>>

最终 hidden state 经过最后 RMSNorm，得到 $H\in\mathbb{R}^{T\times896}$。LM Head 是一个把 hidden size 映射到词表的线性层：

$$
Z=H W_U,\qquad W_U\in\mathbb{R}^{896\times151936},\qquad
Z\in\mathbb{R}^{T\times151936}.
$$

$Z_{i,j}$ 是第 $i$ 个位置对词表第 $j$ 个 token 的 logit。它还不是概率，也不要求落在 $[0,1]$；每一行要单独经过 softmax。

当 `tie_word_embeddings=true` 时，$W_U$ 与 embedding 矩阵共享参数：如果 $E\in\mathbb{R}^{V\times896}$，则通常有 $W_U=E^T$。输入查表和输出分类使用同一组词向量参数。

## <<Softmax：logits 变成概率>>

生成时只取最后一个位置的 logits $z_T\in\mathbb{R}^{151936}$：

$$
p_T=\operatorname{softmax}(z_T),\qquad
p_{T,j}=\frac{e^{z_{T,j}}}{\sum_{k=1}^{V}e^{z_{T,k}}}.
$$

$p_T$ 的每个分量非负且总和为 1。接下来有两类选择：

$$
t_{T+1}=\arg\max_j p_{T,j}
$$

是贪心解码；按 $p_T$ 随机抽样，再配合 temperature、top-k、top-p，是采样解码。具体见 [[论文阅读/qwen/04-qwen推理与采样]]。

训练时，模型对每个位置都产生一行 logits，目标 token 整体右移一位对齐；损失通常是所有有效位置的 causal language-model cross-entropy，见 [[论文阅读/qwen/03-qwen输入与训练]]。

## 各环节地图

| 链路环节 | 数学对象 | 深入阅读 |
| --- | --- | --- |
| 文本 → token ID | $s\to(t_1,\ldots,t_T)$ | [[论文阅读/qwen/03-qwen输入与训练]]、[[tokenizer-类型契约]] |
| ID → embedding | $H^0=E[t]$ | [[论文阅读/qwen/03-qwen输入与训练]] |
| Decoder block | RMSNorm + GQA + SwiGLU + residual | [[论文阅读/qwen/02-qwen-decoder-block]] |
| hidden → logits | $Z=HW_U$ | [[论文阅读/qwen/02-qwen-decoder-block]] |
| logits → token | softmax + decode | [[论文阅读/qwen/04-qwen推理与采样]]、[[generate-生成机制]] |
