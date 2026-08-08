---
title: Qwen Decoder Block：从 RMSNorm 到 SwiGLU
published: 2026-08-08
description: 把 Qwen 一个 Decoder block 的完整数学过程列出来：RMSNorm、Q/K/V、RoPE、GQA、因果 mask、残差与 SwiGLU。
tags: [论文阅读, Qwen, Decoder, GQA, RoPE, SwiGLU]
category: 论文阅读
---

# Qwen Decoder Block：从 RMSNorm 到 SwiGLU

> 这是 Qwen 知识库的核心篇。输入一个 $(T,896)$ 的 hidden state，经过一层 block 后仍然得到 $(T,896)$；中间只在 attention head 和 SwiGLU 隐层里改变形状。完整主线见 [[论文阅读/qwen/01-qwen主线]]。

## 一个 block 先看完整公式

以本地 Qwen2-0.5B 配置为例：

$$
d_{model}=896,\qquad h_q=14,\qquad h_{kv}=2,\qquad d_h=64,\qquad d_{ff}=4864.
$$

设上一层输出为 $H\in\mathbb{R}^{T\times896}$。Qwen 使用 Pre-RMSNorm：

$$
R=\operatorname{RMSNorm}(H),
$$

$$
H'=H+\operatorname{Attention}(R),
$$

$$
S=\operatorname{RMSNorm}(H'),
$$

$$
H_{out}=H'+\operatorname{SwiGLU}(S).
$$

这就是一层 block 的骨架：**归一化 → attention → 残差 → 归一化 → MLP → 残差**。下面只展开中间两个函数。

## <<Attention：Q/K/V 如何形成相关性>>

从 $R\in\mathbb{R}^{T\times896}$ 投影：

$$
Q_{flat}=RW_Q\in\mathbb{R}^{T\times896},
$$

$$
K_{flat}=RW_K\in\mathbb{R}^{T\times128},\qquad
V_{flat}=RW_V\in\mathbb{R}^{T\times128}.
$$

拆头：

$$
Q\in\mathbb{R}^{14\times T\times64},
\qquad K,V\in\mathbb{R}^{2\times T\times64}.
$$

RoPE 作用在每个 Q/K 头内部的二维通道对上，形状不变。然后按每 7 个 Q 共享一组 K/V 对齐：

$$
K,V:(2,T,64)\longrightarrow(14,T,64).
$$

对齐后的 K/V 与 Q 做 causal attention：

$$
A=\operatorname{softmax}\left(
\frac{Q\widetilde K^T}{\sqrt{64}}+M
\right)\in\mathbb{R}^{14\times T\times T},
$$

$$
C=A\widetilde V\in\mathbb{R}^{14\times T\times64}.
$$

$M$ 是因果 mask：未来位置填 $-\infty$，softmax 后对应权重为 0。拼接 14 个 head，再过 `o_proj`：

$$
\operatorname{Concat}(C)\in\mathbb{R}^{T\times896},
\qquad
O=\operatorname{Concat}(C)W_O\in\mathbb{R}^{T\times896}.
$$

于是 attention 分支与 $H$ 形状相同，可以直接残差相加：$H'=H+O$。

## <<SwiGLU：两路升维，一路降维>>

从 $S\in\mathbb{R}^{T\times896}$ 走三组投影：

$$
G=\operatorname{SiLU}(SW_{gate})\in\mathbb{R}^{T\times4864},
$$

$$
P=SW_{up}\in\mathbb{R}^{T\times4864},
$$

$$
M=(G\odot P)W_{down}\in\mathbb{R}^{T\times896}.
$$

其中 $\odot$ 是逐元素乘法，$W_{gate}$ 产生门控路，$W_{up}$ 产生内容路，$W_{down}$ 把中间表示压回 hidden size。最后

$$
H_{out}=H'+M\in\mathbb{R}^{T\times896}.
$$

## <<RMSNorm：只沿 hidden 维控制尺度>>

对一个 token 的向量 $x\in\mathbb{R}^{896}$：

$$
\operatorname{RMS}(x)=\sqrt{\frac{1}{896}\sum_{j=1}^{896}x_j^2+\epsilon},
$$

$$
\operatorname{RMSNorm}(x)=g\odot\frac{x}{\operatorname{RMS}(x)}.
$$

序列矩阵逐行归一化：$\operatorname{RMSNorm}(H)_{i,:}=\operatorname{RMSNorm}(H_{i,:})$。它不改变 $T\times896$ 形状，也不在 batch 维统计量。

## 一层 block 的形状总表

$$
(T,896)
\xrightarrow{\operatorname{RMSNorm}}(T,896)
\xrightarrow{Q/K/V,\ \text{RoPE},\ \text{GQA}}(14,T,64)
\xrightarrow{\text{attention + concat + }o\_proj}(T,896)
$$

$$
\xrightarrow{\operatorname{RMSNorm}}
(T,896)
\xrightarrow{gate/up}(T,4864)
\xrightarrow{\odot,\ down\_proj}(T,896).
$$

24 层只是把这条 block 计算重复 24 次；层数改变深度，单层的形状契约不变。
