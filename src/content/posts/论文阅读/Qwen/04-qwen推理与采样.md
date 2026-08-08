---
title: Qwen 推理：从 GQA 与 KV Cache 到采样
published: 2026-08-08
description: 把 Qwen 的自回归推理串起来：单 token 解码、GQA、KV Cache、LM Head、temperature、top-k 与 top-p。
tags: [论文阅读, Qwen, GQA, KV Cache, 生成]
category: 论文阅读
---

# Qwen 推理：从 GQA 与 KV Cache 到采样

> 训练时可以并行整段序列，生成时却是一 token 一 token 地向右延伸。推理篇只回答一个问题：每一轮到底重算什么、缓存什么、最后如何选 token。

## <<生成循环：每轮只新增一个位置>>

已有序列为 $t_{1:T}$，模型只需要得到最后位置的 logits：

$$
t_{1:T}\xrightarrow{\text{Decoder}}z_T\in\mathbb{R}^{151936}
\xrightarrow{\text{decode rule}}t_{T+1}.
$$

把新 token 追加后进入下一轮：

$$
t_{1:T}\to t_{1:T+1}\to t_{1:T+2}\to\cdots.
$$

如果每轮都从头计算所有历史 token，成本会随生成长度迅速增加。KV Cache 保存每层历史位置的 K/V，使下一轮只为新 token 计算 Q/K/V。

## <<KV Cache：保存历史 K/V>>

当前 query $q_t$ 需要访问历史：

$$
A_t=\operatorname{softmax}\left(
\frac{q_tK_{\le t}^T}{\sqrt{d_h}}+M_t
\right),
\qquad
o_t=A_tV_{\le t}.
$$

$K_{\le t},V_{\le t}$ 来自 cache。对每层、单 batch，若 K/V head 数是 $h_{kv}$、head dim 是 $d_h$、元素字节数为 $b$，缓存规模近似为

$$
M_{KV}=2\,T\,h_{kv}\,d_h\,b.
$$

前面的 2 是 K 和 V 两份。层数、batch、上下文长度都会再乘上去。

## <<GQA：14 个 Q 头共享 2 个 KV 头>>

Qwen 的配置是

$$
h_q=14,\qquad h_{kv}=2,\qquad d_h=64.
$$

投影后的形状：

$$
Q\in\mathbb{R}^{14\times T\times64},
\qquad K,V\in\mathbb{R}^{2\times T\times64}.
$$

每 7 个 Q 头共享一组 K/V：

$$
(Q_1,\ldots,Q_7)\leftrightarrow(K_1,V_1),
\qquad
(Q_8,\ldots,Q_{14})\leftrightarrow(K_2,V_2).
$$

实现 attention 前通常把 K/V 按组 repeat 到 14 个头参与计算，但 cache 中只保存原始 2 头：

$$
(2,T,64)\xrightarrow{\operatorname{repeat\_kv}}(14,T,64).
$$

因此 GQA 把 KV Cache 的头数从 14 降到 2，理想缓存比例是 $2/14=1/7$；这描述存储量，不等于端到端延迟一定降低 7 倍。

## <<LM Head：hidden state 变成 logits>>

最后一个 hidden vector 经最终 RMSNorm 后，通过词表投影：

$$
h_T\in\mathbb{R}^{896}
\xrightarrow{W_U\in\mathbb{R}^{896\times151936}}
z_T\in\mathbb{R}^{151936}.
$$

$z_T$ 是未归一化分数。它的排序和差距交给解码策略处理，模型前向本身只负责产生 logits。

## <<Temperature：改变分布锐度>>

温度作用在 logits 上：

$$
p_j^{(\tau)}=\operatorname{softmax}\left(\frac{z}{\tau}\right)_j.
$$

$\tau>1$ 使分布变平，$0<\tau<1$ 使分布变尖；$\tau\to0^+$ 逼近 argmax。温度不会改变 token 排名。

## <<Top-k 与 Top-p：限制采样候选>>

Top-k 只保留概率最高的 $k$ 个 token：

$$
S_k=\operatorname{TopK}(z,k),
\qquad
z'_j=\begin{cases}z_j,&j\in S_k,\\-\infty,&j\notin S_k.\end{cases}
$$

Top-p 取按概率排序后累计概率达到阈值 $p$ 的最小集合：

$$
k^*=\min\left\{k:\sum_{i=1}^{k}p_{(i)}\ge p\right\}.
$$

Top-k 的候选数固定，top-p 的候选数随分布形状变化。过滤后重新归一化，再按概率抽样。

## 推理的一轮完整路径

$$
\text{新 token}
\to\text{当前 hidden}
\to\text{Q/K/V}
\to\text{读取 KV Cache}
\to\text{GQA attention}
\to\text{SwiGLU}
\to\text{LM Head logits}
\to\text{temperature/top-k/top-p}
\to\text{下一个 token}.
$$

生成配置属于解码层，不改变模型参数；KV Cache 属于计算复用层，不改变模型的数学目标。两者都服务于同一个自回归循环。
