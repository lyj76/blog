---
title: LoRA 主线：从有效低秩到 BA 适配器
published: 2026-08-07
description: 从全量微调更新的有效低秩观察，到 SVD 截断的数学逼近，再到 LoRA 直接训练 BA、初始化、合并与推理。
tags: [论文阅读, LoRA, 参数高效微调, 有效秩]
category: 论文阅读
---

# LoRA 主线：从有效低秩到 BA 适配器

> 论文：Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*（[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)）。数学与实现细节拆在 [[论文阅读/lora/00-阅读地图]]。

## 微调到底更新了什么

预训练模型的线性层权重是 $W_0$。微调结束后变成

$$
W_{fine}=W_0+\Delta W.
$$

任务带来的信息全部在 $\Delta W$ 里。真正要紧的是：这个更新量，需要占满整个 $d_{out}\times d_{in}$ 的空间吗？全量微调要对 $d_{out}d_{in}$ 个参数求梯度、维护优化器状态，成本摆在那里。

## 满秩，不等于方向多

对 $\Delta W$ 做奇异值分解：

$$
\Delta W=U\Sigma V^T,\qquad \Sigma=\operatorname{diag}(\sigma_1,\sigma_2,\ldots).
$$

"代数 rank"数的是非零奇异值有多少个；"有效 rank"数的是能量真正集中在几个方向。一个矩阵可以代数满秩，但大部分奇异值小得可以忽略。

看这个例子：$\sigma=(10,1,0.1,0.01)$。代数 rank 是 4，但奇异值平方和 $101.01$ 里，最大方向占 $99\%$，前两个方向占 $99.99\%$。从严格维度看它是四维，从作用看它几乎是一维的。

LoRA 的出发点就是这个观察：任务更新真正起作用的方向常常很少。这是实验现象，不是什么定理——而且"有效 rank"本身没有统一定义，按能量阈值、相对阈值、stable rank 度量的结果可以不同。

## SVD 截断：自然的最佳 rank-r 近似

既然完整的 $\Delta W$ 已经拿到，最直接的办法是只保留前 $r$ 个奇异方向：

$$
\Delta W_r=U_r\Sigma_rV_r^T.
$$

Eckart–Young–Mirsky 定理保证这是所有 rank 不超过 $r$ 的矩阵里的最佳逼近，误差分别是尾部奇异值平方和（Frobenius）与第 $r+1$ 个奇异值（谱范数）：

$$
\lVert\Delta W-\Delta W_r\rVert_F^2=\sum_{i>r}\sigma_i^2,\qquad
\lVert\Delta W-\Delta W_r\rVert_2=\sigma_{r+1}.
$$

这条链是通的：能量集中 → 截断小奇异值 → 最佳 rank-r 近似。而这个近似本身就能写成 BA：

$$
\Delta W_r=\underbrace{U_r\Sigma_r}_{B}\underbrace{V_r^T}_{A}.
$$

## 为什么不先 SVD 再 LoRA

SVD 截断有个前提：你得先有完整的 $\Delta W$。也就是先全量微调一遍，再做压缩。全量微调的成本已经付过了，压缩只是事后补救，没解决"怎么低成本训练"。

LoRA 把思路反过来：既然相信有用更新集中在低维方向，那就从第一步起把搜索空间限制成低秩因子：

$$
\Delta W=\frac{\alpha}{r}BA.
$$

它借的是 SVD 的数学动机，做的却是训练时的参数化约束。两件事的差别：

|  | 何时限制低秩 | 优化什么 | 需要完整 $\Delta W$ |
| --- | --- | --- | --- |
| SVD 截断 | 全量微调之后 | 矩阵重构误差 | 需要 |
| LoRA | 训练一开始 | 任务损失 | 不需要 |

## BA 参数化

$W_0\in\mathbb{R}^{d_{out}\times d_{in}}$，LoRA 引入

$$
A\in\mathbb{R}^{r\times d_{in}},\qquad B\in\mathbb{R}^{d_{out}\times r},\qquad r\ll\min(d_{in},d_{out}).
$$

训练时使用

$$
W'=W_0+\frac{\alpha}{r}BA.
$$

由 $\operatorname{rank}(BA)\leq r$，$BA$ 的秩至多 $r$。这是参数化空间的秩上界，跟"真实最优更新秩多小"是两回事。

## 只训练 A 和 B

训练时冻结 $W_0$，优化器只碰 $A,B$：

$$
h=W_0x+\frac{\alpha}{r}BAx.
$$

```python title="lora-trainable.py"
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable, lr=lr)
```

LoRA 参数要注册进 `nn.Module` 并设 `requires_grad=True`。实现细节见 [[nn-module]]。

## 初始化：B 为零

$A$ 随机初始化，$B_0=0$：

$$
W'_0=W_0+\frac{\alpha}{r}B_0A_0=W_0.
$$

插入适配器的瞬间不改变原模型函数，再从原模型行为开始学。这只能保证初始函数一致，不证明后面的非凸优化一定收敛（$B=0$ 引起的初期梯度不对称性与 $\alpha/r$ 缩放动力学见 [[论文阅读/lora/04-动力学与目标模块演进]]）。

## 合并权重

训练完，低秩更新合回原权重：

$$
W_{merge}=W_0+\frac{\alpha}{r}BA.
$$

合并后的普通线性层和 LoRA 分支数学等价：

$$
W_0x+\frac{\alpha}{r}BAx
=\left(W_0+\frac{\alpha}{r}BA\right)x.
$$

推理时删掉适配器分支，不多一层计算。

## forward 的转置

权重组合在映射世界：`B @ A` 不转置；把 `(out,in)` 权重作用到 PyTorch 的 `[batch,in]` 行数据时，才写 `x @ W.T`：

```python title="merge-vs-forward.py"
W_merge = W0 + (alpha / r) * (B @ A)  # 合并：无转置
h = x @ W_merge.T                     # forward：作用到行数据，转置
```

详见 [[数学符号到代码#权重组合不动转置]]。

## 省了什么

完整更新要 $d_{out}d_{in}$ 个参数，LoRA 只要

$$
N_{LoRA}=r(d_{in}+d_{out}).
$$

省的是可训练参数、梯度和优化器状态；基础模型权重和激活还在。具体数字见 [[论文阅读/lora/03-参数量与显存]]。

## 边界

原论文提出的是低秩适配参数化，并在任务实验里展示参数效率和竞争性效果。不要把它读成"任意任务的 $\Delta W$ 都严格低秩"，也不要把 SVD 误差定理算到 LoRA 论文头上。论文脉络见 [[论文阅读/lora/05-相关论文与方法演进]]。
