---
title: Qwen 推理：Prefill/Decode、KV Cache 与采样
published: 2026-08-08
description: 自回归推理的物理瓶颈：Prefill 为什么算力受限、Decode 为什么带宽受限、KV Cache 的精确显存模型、采样为什么用 Top-p。
tags: [论文阅读, Qwen, 推理, KV Cache, GQA, 采样]
category: 论文阅读
---

# Qwen 推理：Prefill/Decode、KV Cache 与采样

> 承接 [[论文阅读/qwen/01-qwen主线]] 与 [[论文阅读/qwen/04-qwen输入与训练]]：01 给出因果掩码 → KV Cache 的理论基础，04 拆解了训练目标。本篇从推理侧切入——**生成一个词为什么这么慢、显存花在哪、采样在干什么**。索引见 [[论文阅读/qwen/00-qwen阅读地图]]。

## Prefill 与 Decode：为什么生成是两步

自回归推理被物理地分成两个阶段，瓶颈完全不同：

```text title="two_phases.txt"
【阶段 1：Prefill】
输入完整 Prompt ──> 矩阵×矩阵 (GEMM) ──> 算力密集 ──> 【计算受限】
                                      │
                                      ▼ 产出第 1 个 Token 与全部历史 KV Cache
【阶段 2：Decode】
单步输入 1 个 Token ──> 矩阵×向量 (GEMV) ──> 算力空转 ──> 【带宽受限】
```

**Prefill（计算受限）**：一次性把整条 Prompt 并行送进模型，算的是 GEMM，GPU 的 Tensor Core 满载——瓶颈是算力（TFLOPS）。

**Decode（带宽受限）**：每步只输入上一个生成的 Token，算的是 GEMV。但为了算这一个 Token，GPU **必须把全部权重从显存读一遍**——0.5B 的模型约 1GB，70B 约 140GB。算力密度极低，GPU 大部分时间在等数据搬运——瓶颈是显存带宽（GB/s）。

这两个阶段的剪刀差决定了生成为什么慢：Prompt 越长，Prefill 越久；生成 Token 越多，Decode 步数越多，每步都被带宽卡死。

## KV Cache：为什么值得用显存换时间

Decode 每步都要算新 Token 的注意力，而注意力需要和**所有历史 Token** 的 K、V 做点积。如果没有缓存，第 $T$ 步要把前 $T-1$ 个 Token 的 K、V 全部重算一遍——成本随序列长度二次方增长。

因果掩码保证了关键性质：**位置 $i$ 的 K、V 只依赖前 $i$ 个 Token**。生成第 $T+1$ 个词时，前 $T$ 个词的 K、V 一个都没变。缓存下来，每步只需算新 Token 的 $k_t, v_t$，和缓存拼接后使用。

代价是显存。KV Cache 的精确公式：

$$
M_{\text{KV}} = 2 \times B \times L \times h_{kv} \times d_h \times T \times b_{\text{bytes}}.
$$

- 系数 $2$：K 和 V 两份张量；
- $b_{\text{bytes}}$：精度字节数，BF16/FP16 为 2。

逐项代入 Qwen2-0.5B（$L=24, h_{kv}=2, d_h=64$），$B=4$、$T=32{,}768$：

$$
M_{\text{KV}} = 2 \times 4 \times 24 \times 2 \times 64 \times 32768 \times 2 \approx 0.804\text{ GB}.
$$

如果把 $h_{kv}$ 换成 MHA 的 14：$2 \times 4 \times 24 \times 14 \times 64 \times 32768 \times 2 \approx 5.63\text{ GB}$——**GQA 让 KV Cache 缩到 1/7**。这就是 02 里"KV 头数出现在乘数位置"的后果：长文本并发时，这一项往往决定单卡能承载多少请求。

## 采样：Logits 怎么变成 Token

最后一层输出 $z \in \mathbb{R}^{151936}$ 是 Logits，不是 Token。把它变成离散 Token 有三道闸门。

**温度 $\tau$**：Softmax 前除一个系数，控制分布的锐利程度：

$$
p_j^{(\tau)} = \frac{e^{z_j / \tau}}{\sum_k e^{z_k / \tau}}.
$$

- $\tau \to 0$：分布趋向狄拉克脉冲，等价于贪心解码（取 $\arg\max$）；
- $\tau = 1$：原始分布；
- $\tau > 1$：分布被拉平，长尾词更容易被采到。

**Top-p 核采样**：动态截断候选集。按概率降序，取累积概率刚过阈值 $p$ 的最小集合：

$$
k^* = \min \left\{ k : \sum_{i=1}^k p_{(i)} \ge p \right\}.
$$

Top-p 的自适应性是关键：分布尖锐时候选集自动缩小，平坦时自动扩大。对比 Top-k 固定截断 $k$ 个——分布尖锐时 Top-k 会混入不必要的长尾，平坦时又会漏掉应该考虑的候选。Top-p 用概率总量替代固定数量，规避了这两个问题。

**重复惩罚**：对已出现的 Token 衰减其 Logit：

$$
z_i' = \begin{cases} z_i / \alpha_{\text{penalty}}, & z_i > 0, \\ z_i \cdot \alpha_{\text{penalty}}, & z_i \le 0, \end{cases} \quad (\alpha_{\text{penalty}} \ge 1).
$$

正 Logit 被压、负 Logit 被推，已生成过的词更难再被采到。

## 单步自回归：完整执行流

把四块拼起来，一步生成的全过程：

$$
t_n \xrightarrow{\text{Embedding}} h_n^0 \xrightarrow{\text{24 层 + KV Cache}} h_n^{24} \xrightarrow{\text{LM Head}} z_n \in \mathbb{R}^{151936} \xrightarrow{\tau / \text{Top-p}} \text{Sample}(p_n) \to t_{n+1}.
$$

## 边界

- **Prefill/Decode 的分野在"一个 Token"处**：Prefill 阶段第一个 Token 也是算力受限的 GEMM，从第二个 Token 起才转入带宽受限的 GEMV。
- **KV Cache 公式不含权重和激活**：它只是缓存那一项的显存。真实部署显存是权重 + 激活 + KV Cache 三项之和，KV Cache 只是其中可被 GQA 压缩的一项。
- **采样参数是工程调优，不是理论结论**：温度、Top-p、重复惩罚的取值依赖任务（代码生成偏好低温度、对话偏好中等），没有普适最优。
- **"GQA 是推理最大优化"的表述要限定在 KV Cache 范围内**：对训练显存，GQA 帮助有限（02 篇已标注）。

config 字段到 PyTorch 源码的映射见 [[论文阅读/qwen/06-qwen模型文件与代码]]。
