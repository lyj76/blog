---
title: LoRA 数学基础：SVD 与最佳低秩近似
published: 2026-08-07
description: 秩、SVD、奇异值、Eckart–Young 最佳低秩近似，以及 SVD 截断与 LoRA 学习 BA 的边界。
tags: [论文阅读, LoRA, SVD, 线性代数]
category: 论文阅读
---

# LoRA 数学基础：SVD 与最佳低秩近似

## <<矩阵的秩>>

矩阵 $M$ 的秩是列空间维数。若 $M=BA$，$B\in\mathbb{R}^{m\times r}$、$A\in\mathbb{R}^{r\times n}$，则

$$
\operatorname{rank}(BA)\leq\min(\operatorname{rank}(B),\operatorname{rank}(A))\leq r.
$$

这解释了 LoRA 因子化为何把更新秩限制在 $r$ 以内，但不说明 $BA$ 是某个给定矩阵的最佳逼近。

## <<奇异值分解>>

任意实矩阵可分解为：

$$
M=U\Sigma V^T,
$$

其中 $U,V$ 正交，$\Sigma$ 的非负对角元素为奇异值 $\sigma_1\geq\sigma_2\geq\cdots$。非零奇异值个数等于 $\operatorname{rank}(M)$。

几何上，$V^T$ 旋转输入，$\Sigma$ 沿正交方向缩放，$U$ 旋转到输出坐标。奇异值是拉伸倍数，不是旋转角度。

## <<Eckart-Young最佳低秩近似>>

保留最大的前 $r$ 个奇异值：

$$
M_r=\sum_{i=1}^{r}\sigma_i u_i v_i^T.
$$

Eckart–Young–Mirsky 定理：$M_r$ 是所有秩不超过 $r$ 的矩阵中对 $M$ 的最佳近似。

## <<截断奇异值的误差>>

$$
\lVert M-M_r\rVert_F^2=\sum_{i>r}\sigma_i^2,
\qquad
\lVert M-M_r\rVert_2=\sigma_{r+1}.
$$

前者是 Frobenius 误差，后者是谱范数误差。奇异值尾部小，说明存在误差小的 rank-$r$ 近似。

## <<SVD截断不是LoRA训练>>

SVD 截断的输入是已经知道的完整矩阵 $\Delta W$：

$$
\Delta W=W_{fine}-W_0 \longrightarrow (\Delta W)_r.
$$

LoRA 则从训练开始就直接优化：

$$
\Delta W=\frac{\alpha}{r}BA.
$$

因此“存在最佳低秩近似”不能推出“LoRA 训练出的 $BA$ 等于 SVD 截断结果”。前者是后处理压缩，后者是受限参数化下的任务优化。
