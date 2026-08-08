---
title: FFN 的演化：从 ReLU 到 SwiGLU
published: 2026-08-08
description: FFN 的知识存储猜想、ReLU 死神经元、GELU、GLU 家族、SwiGLU 三矩阵结构为何成为现代标配。
tags: [论文阅读, Transformer, FFN, SwiGLU]
category: 论文阅读
---

# FFN 的演化：从 ReLU 到 SwiGLU

> 承接 [[论文阅读/transformer/01-transformer主线]]。注意力负责 token 之间的信息交换，FFN 负责对每个 token 的表示独立变换。它从两个矩阵的 ReLU 结构，演化到 Qwen 使用的 gate/up/down 三矩阵结构。

## FFN：逐位置的升维与降维

Transformer 的 FFN 对序列矩阵的每一行独立计算：

$$
\operatorname{FFN}(X)=\operatorname{ReLU}(XW_1+b_1)W_2+b_2.
$$

以原论文配置为例，$X\in\mathbb{R}^{T\times512}$：

$$
W_1\in\mathbb{R}^{512\times2048},\qquad
W_2\in\mathbb{R}^{2048\times512},
$$

$$
(T,512)\to(T,2048)\to(T,512).
$$

注意力改变不同 token 之间的信息，FFN 的每一行互不通信。它把每个 token 暂时送进更宽的特征空间，再压回 hidden size；大模型中大量参数也集中在这组投影矩阵里，因此常被视为知识变换的主要位置。"FFN 就是记忆库"是解释性假说，不是架构定义。

## <<ReLU 的死神经元问题>>

经典 FFN 的激活是

$$
\operatorname{ReLU}(x)=\max(0,x),
$$

其导数为

$$
\operatorname{ReLU}'(x)=
\begin{cases}
0,&x<0,\\
1,&x>0.
\end{cases}
$$

某个神经元长期落在负半轴，它对当前样本的梯度为 0，参数更新会停滞，这就是 dead ReLU。它不一定永远无法恢复，但负半轴没有局部梯度，恢复困难。

ReLU 还有一个结构限制：每个中间维度是否激活，只由该维自己的 pre-activation 决定。后来的门控结构让"信息内容"和"通过比例"由两路投影共同决定。

## <<GELU：平滑版 ReLU>>

GELU 用标准正态分布的累积分布函数 $\Phi$ 对输入进行平滑门控：

$$
\operatorname{GELU}(x)=x\Phi(x).
$$

常用近似为

$$
\operatorname{GELU}(x)\approx\frac{x}{2}\left(1+\tanh\left[\sqrt{\frac{2}{\pi}}\left(x+0.044715x^3\right)\right]\right).
$$

它在负半轴保留连续的非零梯度，过渡比 ReLU 平滑。GPT 等模型大量使用 GELU，但 GELU 仍然是一条输入到输出的单路激活；它没有把门控和内容拆成两组投影。

## <<GLU 家族：门控机制入场>>

GLU 的基本形式是两路线性投影的逐元素乘积：

$$
\operatorname{GLU}(x)=\big(xW_a\big)\odot g\big(xW_b\big),
$$

其中 $xW_a$ 提供内容，$g(xW_b)$ 提供门控系数，$\odot$ 是 Hadamard 逐元素乘法。门控路为 sigmoid 时得到原始 GLU；换成 ReLU、GELU、SiLU 就得到不同变体。

门控的关键变化是：第 $j$ 个中间特征的输出同时依赖两路结果，

$$
u_j=\underbrace{(xW_a)_j}_{\text{内容}}
\cdot
\underbrace{g(xW_b)_j}_{\text{门控}}.
$$

因此模型可以根据当前 token 的上下文调节某个特征通过多少。

## <<SwiGLU：现代三矩阵结构>>

SwiGLU 取 SiLU 作为门控函数：

$$
\operatorname{SiLU}(x)=x\,\sigma(x),
$$

$$
\operatorname{SwiGLU}(x)
=\big(\operatorname{SiLU}(xW_{gate})\odot(xW_{up})\big)W_{down}.
$$

三组矩阵的职责和形状：

| 矩阵 | 形状 | 作用 |
| --- | --- | --- |
| $W_{gate}$ | $d_{model}\times d_{ff}$ | 产生门控路，经 SiLU |
| $W_{up}$ | $d_{model}\times d_{ff}$ | 产生内容路 |
| $W_{down}$ | $d_{ff}\times d_{model}$ | 降回 hidden size |

Qwen2-0.5B 中 $d_{model}=896$、$d_{ff}=4864$：

$$
(T,896)\xrightarrow{W_{gate},W_{up}}(T,4864)
\xrightarrow{\odot,\operatorname{SiLU}}(T,4864)
\xrightarrow{W_{down}}(T,896).
$$

对应代码里的 `gate_proj`、`up_proj`、`down_proj`，实现定位见 [[论文阅读/qwen/05-qwen模型文件与代码]]。

SwiGLU 的优势来自乘法门控带来的输入依赖特征选择；它在多种语言模型实验中表现良好，但"门控一定更强"不是由公式直接推出的定理。实际配置还要同时考虑 $d_{ff}$、参数预算和训练规模。

## 双矩阵 vs 三矩阵对照表

| 项目 | 经典 Transformer FFN | SwiGLU FFN |
| --- | --- | --- |
| 结构 | $\operatorname{ReLU}(XW_1)W_2$ | $(\operatorname{SiLU}(XW_{gate})\odot XW_{up})W_{down}$ |
| 升维路径 | 一路 | 内容路 + 门控路 |
| 投影矩阵 | 2 个 | 3 个 |
| 中间操作 | 激活 | 逐元素乘法门控 |
| 输出形状 | $(T,d_{model})$ | $(T,d_{model})$ |
| Qwen 对应模块 | — | `gate_proj` / `up_proj` / `down_proj` |

FFN 演化的主线可以压成一句：**从"每个维度独立过激活"，走到"内容先生成，再由另一条路决定通过多少"。**
