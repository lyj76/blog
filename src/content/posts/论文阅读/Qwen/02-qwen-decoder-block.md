---
title: Qwen Decoder Block：单层四个算子的深度解构
published: 2026-08-08
description: 单层 Decoder Block 四个核心算子：Pre-RMSNorm 为何扔掉均值、GQA 如何用 1/7 的 KV 头数、RoPE 的旋转几何、SwiGLU 的门控机理。
tags: [论文阅读, Qwen, Decoder Block, GQA, RoPE, SwiGLU, RMSNorm]
category: 论文阅读
---

# Qwen Decoder Block：单层四个算子的深度解构

> 承接 [[论文阅读/qwen/01-qwen主线]]：01 给出了 7 个权重矩阵的拓扑和端到端链路。本篇把单层 Block 内部四个算子逐个拆开——每个算子替代了什么旧方案、为什么这么换、代价是什么。索引见 [[论文阅读/qwen/00-qwen阅读地图]]。

## 单层 Block 长什么样

一个 Decoder Block 是两个子层的组合，每个子层都是 "归一化 → 变换 → 残差"：

```text title="block_forward_path.txt"
输入 H [T, 896]
  │
  ├──> RMSNorm ──> QKV 投影 ──> RoPE ──> GQA Attention ──> O 投影 ──> (+) ──┐
  │                                                                          │
  └──> RMSNorm ──> gate/up 升维 ──> SiLU ⊙ Up ──> down 降维 ──> (+) ──> 输出
```

上半场是注意力子层，做**序列内交互**——让每个 token 看见其他 token；下半场是 MLP 子层，做**知识存储**——把上下文压成更高维的非线性表达再压回来。两个子层共用同一套残差结构，这是它们能堆 24 层的前提。

下面四个小节分别拆解四个算子。所有数值以 Qwen2-0.5B 为准：$d_{\text{model}}=896$、$h_q=14$、$h_{kv}=2$、$d_h=64$、$d_{ff}=4864$、$L=24$。

## <<Pre-RMSNorm：为什么扔掉均值>>

LayerNorm 做两件事：减均值（去中心）、除标准差（缩放）。RMSNorm 把去中心扔掉，只保留缩放：

$$
\operatorname{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}}.
$$

分母里的 $\sqrt{\frac{1}{d}\sum x_i^2}$ 就是标准差公式去掉均值项——它只差这一步。

为什么这一步可以扔？两个理由。第一，深层网络里激活的均值携带的信息很少，减去它对表达能力几乎没有影响；第二，算均值要在 $d_{\text{model}}=896$ 维上做一次全局 reduce 同步，这是实打实的访存开销。砍掉这一步，单层带宽省 7–10%，24 层堆起来是显著的训练吞吐提升。

| | LayerNorm | RMSNorm |
|:---|:---|:---|
| 操作 | 减均值 + 除标准差 | 只除 RMS |
| 可学习参数 | $\gamma, \beta$ | $\gamma$ only |
| 计算开销 | 两次 reduce | 一次 reduce |
| 带宽节省 | 基准 | ~7–10% |

RMSNorm 放在每个子层**之前**（Pre-Norm），不是之后（Post-Norm）。差别在残差路径：Pre-Norm 让残差不经过归一化，梯度可以直通到底层；Post-Norm 会让残差被归一化缩放，深层梯度容易消失或爆炸。24 层能训得动，靠的就是这条干净的残差通道——这是架构层面的选择，不是实现细节。

## <<GQA：为什么 KV 头可以只有 2 个>>

标准 MHA 给每个查询头各配一份 K、V。Qwen 的配置是 $h_q=14$、$h_{kv}=2$——14 个查询头共享 2 份 KV。投影矩阵的形状随之改变：

$$
\begin{aligned}
W_Q &\in \mathbb{R}^{896 \times 896}, && \text{14 个头} \\
W_K &\in \mathbb{R}^{128 \times 896}, && \text{2 个头} \\
W_V &\in \mathbb{R}^{128 \times 896}, && \text{2 个头} \\
W_O &\in \mathbb{R}^{896 \times 896}, && \text{输出投影}.
\end{aligned}
$$

$W_K, W_V$ 的输出维度是 $128 = h_{kv} \times d_h = 2 \times 64$，不是 896。计算注意力前，K 和 V 从 2 头广播到 14 头：

```python title="gqa_broadcast.py"
K_rep = K.repeat_interleave(7, dim=0)   # [2, T, 64] -> [14, T, 64]
V_rep = V.repeat_interleave(7, dim=0)   # [2, T, 64] -> [14, T, 64]
```

广播是计算时的视图操作，**不复制显存**——真正驻留的 KV 只有 2 头。

为什么要共享？看 KV Cache 的显存公式：

$$
M_{\text{KV}} = 2 \times B \times L \times h_{kv} \times d_h \times T \times 2\text{ bytes}.
$$

$h_{kv}$ 出现在乘数位置。从 14 降到 2，KV Cache 缩到原来的 $2/14 = 1/7$，即削减 85.7%。长上下文并发场景下，这一项往往决定单卡能承载多少用户——GQA 是推理期最大的显存工程手段之一，但它牺牲的是 K、V 的表达多样性，这是它的代价。

## <<RoPE：为什么用旋转编码位置>>

绝对位置编码（正弦波或可学习参数）把位置信息**加**到输入 embedding 上。RoPE 换个思路：把位置信息**旋**进 Q 和 K 向量。

对维度对 $(2i, 2i+1)$，位置 $m$ 处的旋转角是：

$$
\theta_i = \text{base}^{-2i/d_h}, \qquad \text{base} = 1{,}000{,}000.
$$

旋转矩阵作用于 Q 和 K（V 不动）：

$$
R_{\Theta, m}^i = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}, \qquad
\tilde{q}_m = R_{\Theta, m}\, q_m, \quad \tilde{k}_n = R_{\Theta, n}\, k_n.
$$

关键性质在点积里出现：位置 $m$ 的 Q 和位置 $n$ 的 K 相乘时，两个旋转合成一个只依赖差值 $n-m$ 的旋转：

$$
\langle \tilde{q}_m, \tilde{k}_n \rangle = q_m^T R_{\Theta, n-m}\, k_n.
$$

**结果只依赖相对距离，不依赖绝对位置。** 这是相对位置编码想要的全部性质，而且它发生在 Q、K 点积内部，不污染输入 embedding 的几何结构。推理时序列比训练长也能工作，因为模型只需要理解相对距离，不需要见过绝对位置。

$\text{base}=10^6$ 控制频率谱：低维转得快（高频，管短程局部依赖），高维转得慢（低频，管长程位置区分），形成多尺度结构。实现上对相邻维度对做旋转，避免显式构造稀疏旋转矩阵：

```python title="rope_impl.py"
def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([-x2, x1], dim=-1)

q_rot = q * cos + rotate_half(q) * sin
k_rot = k * cos + rotate_half(k) * sin
```

代价：RoPE 只作用在 Q、K 上，V 仍是无位置信息的纯内容向量——位置信息只能通过注意力权重进入输出，这个设计取舍是明确的。

## <<SwiGLU：为什么用三矩阵门控>>

传统 FFN 是两层线性夹一个 ReLU。SwiGLU 换成三矩阵加一个门控：

$$
\operatorname{SwiGLU}(x) = \big(\operatorname{SiLU}(x W_{\text{gate}}) \odot x W_{\text{up}}\big) W_{\text{down}}.
$$

三个矩阵各司其职：

$$
\begin{aligned}
W_{\text{gate}} &\in \mathbb{R}^{4864 \times 896}, && \text{门控开关，决定激活哪些维度} \\
W_{\text{up}} &\in \mathbb{R}^{4864 \times 896}, && \text{被门控的内容} \\
W_{\text{down}} &\in \mathbb{R}^{896 \times 4864}, && \text{压回残差流}.
\end{aligned}
$$

$\operatorname{SiLU}(x) = x \cdot \sigma(x)$ 是平滑门控：输入接近零时输出几乎为零（静默该维），输入很大时近似线性（放行）。它和 ReLU 的硬截断（0 以下全死）相比，是可导的连续开关。gate 与 up 的 Hadamard 积是双线性交互——门控不是单纯地乘一个标量，而是逐维决定内容通道的开合。

| | ReLU FFN | SwiGLU |
|:---|:---|:---|
| 矩阵数 | 2 | 3 |
| 非线性 | $\max(0, xW_1)$ | $\operatorname{SiLU}(xW_g) \odot xW_u$ |
| 门控 | 硬截断 | 平滑连续 |
| 参数量 | $2 \cdot d \cdot d_{ff}$ | $3 \cdot d \cdot d_{ff}$ |

代价是显式的：多一个矩阵，参数量多了 50%（$2d \cdot d_{ff} \to 3d \cdot d_{ff}$）。回报是门控让知识存储可被选择性激活——MLP 承担约 2/3 的模型参数，是事实知识的主要载体，门控让这些知识按上下文按需唤醒。

## 完整前向：每一步的维度

把四个算子串起来，跟踪一个 $H \in \mathbb{R}^{T \times 896}$ 在单层 Block 内的完整演变：

```python title="decoder_block_trace.py"
# === Attention 半场 ===
R1 = RMSNorm(H)                      # [T, 896]
Q = R1 @ W_Q^T                       # [T, 896]  (14 头平铺)
K = R1 @ W_K^T                       # [T, 128]  (2 头平铺)
V = R1 @ W_V^T                       # [T, 128]

Q = Q.view(T, 14, 64).transpose(0, 1)  # [14, T, 64]
K = K.view(T, 2, 64).transpose(0, 1)   # [2, T, 64]
V = V.view(T, 2, 64).transpose(0, 1)   # [2, T, 64]

Q_rot, K_rot = RoPE(Q), RoPE(K)      # 形状不变

K_rep = K_rot.repeat_interleave(7, dim=0)   # [14, T, 64]
V_rep = V.repeat_interleave(7, dim=0)       # [14, T, 64]

Scores = Q_rot @ K_rep.transpose(-2, -1) / sqrt(64)   # [14, T, T]
A = softmax(Scores + causal_mask)                    # [14, T, T]
C = A @ V_rep                                        # [14, T, 64]

C_flat = C.transpose(0, 1).reshape(T, 896)   # 拼接多头 [T, 896]
O = C_flat @ W_O^T                           # [T, 896]
H1 = H + O                                   # [T, 896]  残差

# === MLP 半场 ===
R2 = RMSNorm(H1)             # [T, 896]
Gate = R2 @ W_gate^T         # [T, 4864]
Up = R2 @ W_up^T             # [T, 4864]
M = SiLU(Gate) * Up          # [T, 4864]  Hadamard
D = M @ W_down^T             # [T, 896]
H_out = H1 + D               # [T, 896]  残差
```

两个残差都发生在归一化与变换**之外**，梯度可以从 $H_{\text{out}}$ 直通到 $H$，不被任何归一化缩放——这就是 Pre-Norm 在反向传播里的价值，也是整条链能重复 24 次而不散的机制保障。

## 边界

- **GQA 的 85.7% 只算 KV Cache 一项**：不含权重、激活、优化器状态。它对推理显存是决定性优化，对训练显存帮助有限。
- **RoPE 的"长度外推"是性质，不是保证**：相对位置不变性给了外推的可能，实际外推长度还取决于训练时的位置分布覆盖，`base` 调大只是缓解手段。
- **"MLP 存知识、Attention 管路由"是功能分工的经验观察**（源自 LoRA 系列 04 篇的模块演进分析），不是 Qwen 报告里证明的定理，引用时注意归因。
- 四个算子的收益是相加的关系，不是替代关系——GQA 省显存、RMSNorm 省带宽、RoPE 给外推、SwiGLU 给知识密度，缺一个现代架构就不完整。

训练目标与损失函数见 [[论文阅读/qwen/03-qwen输入与训练]]。推理时的 KV Cache 与采样见 [[论文阅读/qwen/04-qwen推理与采样]]。config 到源码的映射见 [[论文阅读/qwen/05-qwen模型文件与代码]]。
