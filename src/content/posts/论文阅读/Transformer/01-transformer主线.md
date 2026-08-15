---
title: Transformer 主线：从序列建模困境到"只要注意力"
published: 2026-08-08
description: Attention Is All You Need 主线：RNN/CNN 的困境、注意力机制前身、Encoder-Decoder 全流程、缩放点积注意力、多头、位置编码与训练细节。
tags: [论文阅读, Transformer, 注意力机制, 位置编码]
category: 论文阅读
---

# Transformer 主线：从序列建模困境到"只要注意力"

> 论文：Vaswani et al., *Attention Is All You Need*（[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)）。数学与组件演化拆在 [[论文阅读/transformer/00-transformer阅读地图]]。

## 序列建模的旧世界：长距离与并行两座山

2017 年之前，序列建模的主力是 RNN（LSTM/GRU）和 CNN。它们各有一座跨不过去的山。

RNN 按时间步串行计算：第 $t$ 步的输出依赖第 $t-1$ 步，一个句子要跑 $T$ 步，每一步都在等上一步。GPU 的并行能力完全用不上——这是第一座山：**不能并行**。

第二座山是**长距离信息衰减**。信息要穿过 $T$ 步的循环才能从句子头传到尾，梯度沿时间反向传播时连乘衰减。句子一长，句首的词对句尾的预测几乎失去影响。LSTM 的门控缓解了梯度问题，但 $O(T)$ 的串行步数摆在那。

CNN 能并行，但靠堆层数换感受野：卷积核只能看局部窗口，要建立长距离依赖就得一层层加深网络。窗口是 $k$，感受野是 $k\times$层数，代价是参数和计算量。

两条路都在打补丁。Transformer 的主张是换个思路：**干脆别按顺序处理了**。

## 注意力的前身：从 Seq2Seq 的"回看"开始

注意力机制不是 Transformer 发明的。2015 年的 Seq2Seq 翻译模型里，编码器把整句压成一个向量，解码器每一步都只能看这一个固定向量——句子一长，压缩就丢信息。Bahdanau 加了个"回看"：解码时重新去看编码器的每一个隐状态，按相关性加权求和，而不是只盯压缩后的那个向量。

当时的注意力是辅助件：主架构还是 RNN，注意力只是给解码器多递了一张"该关注哪里"的纸条。但它证明了一件事——**按相关性加权聚合上下文，这个操作本身是有效的**。

## 核心洞察：把注意力从配角变成主角

Transformer 的关键决策：把注意力从辅助机制升级成**唯一**的机制，删掉循环和卷积。

删掉循环，是因为注意力本身就是"任意两位置直接交互"：$T$ 个位置两两计算相关性，一步到位，没有 $t-1\to t$ 的等待——并行。删掉卷积，是因为注意力的交互范围天然覆盖全句，不需要堆层数换感受野——长距离直连。

代价是注意力矩阵是 $T\times T$，复杂度 $O(T^2)$。2017 年这是"贵"，但并行带来的硬件红利远大于多出的计算。这笔账后来被证明划算。

## 全流程：Encoder-Decoder 的张量数学

配置：$d_{model}=512$、$h=8$、$d_k=64$、$d_{ff}=2048$、encoder 6 层 + decoder 6 层。整个架构只有两种形状：$T\times512$（向量序列）和 $T\times T$（注意力矩阵）。

**Encoder**——输入查表 + 逐元素加位置编码：

$$
X=\text{Embed}(t)+PE\in\mathbb{R}^{T\times512},\qquad (X_0)_{i,j}=X_{i,j}+PE_{i,j},
$$

$PE$ 公式见 [[01-transformer主线#位置编码：正弦波从哪来]]。之后 6 层同构堆叠，每层两个子层（Post-LN），形状全程不变：

$$
X^{(1)}=\text{LN}\big(X+\text{MHA}(X)\big),\qquad
X'=\text{LN}\big(X^{(1)}+\text{FFN}(X^{(1)})\big).
$$

**Decoder**——输入是右移一位的目标序列 $Y$（第 $t$ 步只见 $y_1,\ldots,y_{t-1}$），6 层，每层三个子层：

$$
Y^{(1)}=\text{LN}\big(Y+\text{MaskedMHA}(Y)\big),
$$

$$
Y^{(2)}=\text{LN}\big(Y^{(1)}+\text{CrossAttn}(Q=Y^{(1)},\;K=V=X_{out})\big),
$$

$$
Y'=\text{LN}\big(Y^{(2)}+\text{FFN}(Y^{(2)})\big).
$$

三种注意力的差别只在 Q/K/V 来源和 mask：

| 注意力 | Q | K/V | 差别 |
| --- | --- | --- | --- |
| encoder 自注意力 | 自己 | 自己 | 全可见 |
| decoder masked 自注意力 | 自己 | 自己 | 上三角 mask 成 $-\infty$，只看过去 |
| decoder cross-attention | 自己 | encoder 输出 $X_{out}$ | 从源句取信息 |

子层里的两个数学件（MHA 拆在下一节）：

$$
\text{FFN}(x)=\max(0,\,xW_1+b_1)W_2+b_2,\qquad W_1\in\mathbb{R}^{512\times2048},\;W_2\in\mathbb{R}^{2048\times512},
$$

$$
\text{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,\qquad
\mu=\frac{1}{512}\sum_{j}x_j,\quad\sigma^2=\frac{1}{512}\sum_{j}(x_j-\mu)^2,
$$

$\mu,\sigma^2$ 沿特征维逐 token 算，把每层输入拉回零均值单位方差（演化见 [[论文阅读/transformer/04-残差与归一化]]）。

收尾：$Z=Y_L W_{out}$，$W_{out}\in\mathbb{R}^{512\times V}$ 给出全体位置的 logits，取最后一行的 softmax 得下一个词的概率分布。

## <<缩放点积注意力：QKᵀ 在算什么>>

承接上一节：每个 token 要输出一个"融合了全句信息"的新向量。三步：**投影出 Q/K/V → 行向量内积求相关 → 加权求和取 V**。

**第一步，投影**。输入 $X\in\mathbb{R}^{T\times512}$（已加位置编码）乘三个矩阵：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$

$W_Q,W_K,W_V\in\mathbb{R}^{512\times64}$（单头情形），$Q,K,V\in\mathbb{R}^{T\times64}$。每一行是投影后的新向量，名字只是分工不同：

- $q_i$（第 $i$ 行）：token $i$ 的**查询**——"我要找谁"
- $k_j$（第 $j$ 行）：token $j$ 的**键**——"我是谁，用什么被找到"
- $v_j$（第 $j$ 行）：token $j$ 的**值**——"被选中后我提供什么内容"

**第二步，行向量内积求相关**。$QK^T$ 是相关矩阵：

$$
S=QK^T\in\mathbb{R}^{T\times T},\qquad S_{i,j}=q_i\cdot k_j=\sum_{m=1}^{64}q_{i,m}k_{j,m}.
$$

$S_{i,j}$ 是第 $i$ 个 token 的查询向量和第 $j$ 个 token 的键向量的**内积**——内积大说明方向相近、相关度高。于是 $S$ 第 $i$ 行 = "token $i$ 对全句每个 token 的相关性得分"，$T\times T$ 矩阵完整编码了句内两两关系。

**第三步，softmax 归一化成权重**。得分要变成"和为 1 的权重"才能加权：

$$
\text{softmax}(z)_i=\frac{e^{z_i}}{\sum_{j}e^{z_j}}.
$$

按行作用：指数把"稍高的得分"放大成"明显更高的权重"，同时保证非负、和为 1。实现上先减每行最大值再指数，防溢出。

**第四步，除以 $\sqrt{d_k}$，加权取 V**：

$$
\text{Attention}(Q,K,V)=\underbrace{\text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)}_{A\in\mathbb{R}^{T\times T}}V,\qquad
O_i=\sum_{j}A_{i,j}\,v_j.
$$

$O\in\mathbb{R}^{T\times64}$，第 $i$ 行 = 全体 $v_j$ 按权重 $A_{i,j}$ 加权求和——token $i$ 的输出是"全句内容按相关度加权后的平均"，相关度越高贡献越大。维度进出：$(T,64)\to(T,T)\to(T,64)$。

**除以 $\sqrt{d_k}$ 为何（承接 softmax）**。$S_{i,j}$ 是 64 个乘积之和，若每个乘积期望方差 $\sigma^2$，和式方差约 $64\sigma^2$——**维度越大，得分数值越大**。而 softmax 对大的输入敏感：得分一大，指数差拉爆，输出逼近 one-hot，进入梯度 $\approx 0$ 的饱和区，模型学不动。除以 $\sqrt{64}=8$ 把方差压回 $\sigma^2$ 量级，softmax 落回有梯度的区域。诚实标注：论文原句是"我们怀疑点积会随 $d_k$ 变大"——方差分析是后人补的严格解释。

**masked 版本**（decoder 用）：softmax 前给 $S$ 加三角掩码 $M$，上三角填 $-\infty$，$e^{-\infty}=0$，对应位置的权重归零——第 $t$ 个 token 看不到 $t$ 之后的内容。

## <<多头注意力：怎么分、怎么算、怎么合>>

单头一组 $W_Q,W_K,W_V$ 只能定义一种"相关"。多头把投影矩阵拆成 $h$ 份，每份独立做上面的缩放点积，最后拼回去。

**分**：$W_Q\in\mathbb{R}^{512\times512}$ 拆成 $h=8$ 块，每块 $\mathbb{R}^{512\times64}$（$d_k=512/8=64$）：

$$
Q_i=XW^i_Q,\; K_i=XW^i_K,\; V_i=XW^i_V,\qquad W^i_Q,W^i_K,W^i_V\in\mathbb{R}^{512\times64}.
$$

代码里等价做法：投影出 $Q\in\mathbb{R}^{T\times512}$，reshape 成 $(T,8,64)$ 再转置成 $(8,T,64)$——第 $i$ 个头拿第 $i$ 段。

**算**：每个头独立算缩放点积注意力：

$$
\text{head}_i=\text{softmax}\!\left(\frac{Q_iK_i^T}{\sqrt{64}}\right)V_i\in\mathbb{R}^{T\times64}.
$$

8 个头各自定义自己的相关性，互不干扰。

**合**：拼接 + 投影融合：

$$
\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\ldots,\text{head}_8)\,W_O,
$$

$W_O\in\mathbb{R}^{512\times512}$，输出回到 $\mathbb{R}^{T\times512}$。$W_O$ 把 8 个子空间的信息混合成最终输出。

**为什么"切分不增计算"**：单头 $T\times512$ 注意力开销 $\propto T^2\cdot512$；8 个头 $=8\times(T^2\cdot64)=T^2\cdot512$——**完全相等**。切分是分块并行，不是复制 8 份。多出的只有 $W_O$ 融合和调度开销。

**大致意思**：8 个头 = 8 个平行视角，一个头可能抓指代、另一个抓语法，谁管什么由训练自行涌现。维度进出：$(T,512)\to(8,T,64)\to(8,T,64)\to(T,512)$。

## <<位置编码：正弦波从哪来>>

不加位置时，自注意力对 token 置换是**等变**的：把输入 token 的顺序打乱，输出只会跟着打乱——计算本身对"位置"一无所知。RNN 的顺序信息是天然的，Transformer 删掉了顺序，就得显式补回来。

怎么补：**逐元素相加**，不是拼接。词向量和位置编码向量形状相同，同位置同维度直接相加：

$$
X_0=X+PE,\qquad (X_0)_{i,j}=X_{i,j}+PE_{i,j},\qquad X,PE,X_0\in\mathbb{R}^{T\times512}.
$$

$X_{i,j}$ 是第 $i$ 个 token 词向量的第 $j$ 维，$PE_{i,j}$ 是第 $i$ 个位置编码向量的第 $j$ 维——每个 token 的每个维度都加上一个位置数字。语义信息在 $X$，位置信息在 $PE$，加完两者叠进同一个向量。

$PE$ 的每一行（每个位置）用正弦波生成，第 $pos$ 个位置第 $2i$ 维和第 $2i+1$ 维分别是：

$$
PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right),\qquad
PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right).
$$

$pos$ 是位置，$2i$ 和 $2i+1$ 是维度的奇偶下标。频率随维度变化：低维用高频（变化快），高维用低频（变化慢），每个位置得到一个独特的指纹。

选正弦波有两个理由。第一，**无需训练**——任何位置（包括训练没见过的）都能直接算出编码，外推友好。第二，**相对位置可以用绝对位置表达**：三角恒等式保证了 $PE_{pos+k}$ 是 $PE_{pos}$ 的线性组合，模型理论上能学到"位置差"这种相对关系。

注意这两条是"设计动机"，论文没有实验证明正弦波严格优于可学习的 position embedding。演化到 RoPE 的完整故事见 [[论文阅读/transformer/02-位置编码的演化]]。

## 训练配方：Post-LN、warmup、label smoothing

原论文的训练细节后来大多被改了，值得记住原始版本长什么样：

- **Post-LN**：每个子层"先残差、后归一化"。这个摆法训练不稳，是后面一堆麻烦的根源（演化见 [[论文阅读/transformer/04-残差与归一化]]）。
- **Adam + warmup**：学习率先线性升 4000 步再衰减。warmup 用来缓解 Post-LN 结构在训练初期的敏感性，是原论文训练配方的一部分。
- **label smoothing**：$0.1$ 的标签平滑，防止模型过度自信。
- **dropout**：$0.1$，加在注意力权重和 FFN 上。

记住这条：**2017 年那一堆"调参细节"，大半是在给 Post-LN 的不稳定擦屁股**。后来 Pre-LN + RMSNorm 换掉 Post-LN 后，warmup 就没那么要命了。

## 边界：Transformer 解决了什么，没解决什么

解决的是序列建模的两座山：**并行**（$T$ 个位置同时算）和**长距离直连**（任意两位置一步交互，不需要 $T$ 步传递）。这两条让它在 2017 年的机器翻译实验中取得了很强的结果，也成为后来大量序列模型的基础。

没解决的，是后人接力的地方：

- **$O(T^2)$ 的注意力复杂度**：序列翻倍，注意力矩阵面积翻四倍。长上下文靠 GQA、KV Cache 这些工程手段缓解（见 [[论文阅读/qwen/05-qwen推理与采样]]），复杂度本身仍然存在。
- **位置先验缺失**：位置靠外部注入，注入方式本身就是前沿课题（[[论文阅读/transformer/02-位置编码的演化]]）。
- **训练稳定性**：Post-LN 的原版配方训练易崩，后来靠 Pre-LN 和 RMSNorm 解决（[[论文阅读/transformer/04-残差与归一化]]）。
- **encoder-decoder 不是必须的**：原论文的 cross-attention 后来被 GPT 砍掉，只剩 decoder 也能工作——这是架构上的大简化，见 [[论文阅读/qwen/01-qwen主线]]。
