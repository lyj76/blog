---
title: Qwen 阅读地图
published: 2026-08-08
description: Qwen 系列索引：符号系统、基准超参数表与从主线架构到源码映射的阅读依赖关系。
tags: [论文阅读, Qwen, 阅读地图, 索引]
category: 论文阅读
---

# Qwen 阅读地图

> 本系列用于拆解 Qwen2 / Qwen2.5 架构的数学原理、张量流转、训练目标与推理系统。主文负责建立端到端的因果自回归主线，专题篇负责承载各组件的数学证明、显存建模与代码实现细节。

---

## 篇目索引

- [[论文阅读/qwen/01-qwen主线]]：为什么是 Decoder-Only——因果掩码的训练并行与推理复用、单层精确公式、每层 7 个权重矩阵的参数拓扑。
- [[论文阅读/qwen/02-架构解剖]]：整机系统图——五模块逐个讲透、一条 prompt 的数据流、参数量分布与全系列装配图。
- [[论文阅读/qwen/03-qwen-decoder-block]]：单层四个算子深度解构——Pre-RMSNorm、GQA、RoPE、SwiGLU——含完整张量维度追踪。
- [[论文阅读/qwen/04-qwen输入与训练]]：152K 词表的信息密度、右移一位对齐、交叉熵解析梯度与 SFT -100 掩码契约。
- [[论文阅读/qwen/05-qwen推理与采样]]：Prefill 计算受限与 Decode 带宽受限的分水岭、KV Cache 精确显存模型、Top-p 采样。
- [[论文阅读/qwen/06-qwen模型文件与代码]]：config.json 字段决定的算子结构、safetensors 权重形状契约与 PyTorch 源码映射。

---

## 解剖标本基准参数表

全系列以 **Qwen2-0.5B / Qwen2.5-0.5B** 为基准标本进行具体计算与张量维度推导：

| 符号 | 超参数名称 | 对应配置字段 (`config.json`) | 标本基准数值 | 物理意义 |
| :--- | :--- | :--- | :---: | :--- |
| $d_{\text{model}}$ | 隐层维度 (Hidden Size) | `hidden_size` | $896$ | 残差流主干通道宽度 |
| $L$ | 网络层数 (Layers) | `num_hidden_layers` | $24$ | Decoder Block 堆叠深度 |
| $h_q$ | 查询头数 (Query Heads) | `num_attention_heads` | $14$ | 自注意力 Q 投影空间数 |
| $h_{kv}$ | 键值头数 (KV Heads) | `num_key_value_heads` | $2$ | GQA 分组共享的 K/V 投影空间数 |
| $d_h$ | 单头维度 (Head Dim) | `head_dim` | $64$ | 单个注意力头的特征维度（满足 $h_q \times d_h = 896$） |
| $d_{ff}$ | MLP 隐层维度 (FFN Intermediate) | `intermediate_size` | $4864$ | SwiGLU 门控升维宽度（约 $\approx 5.4 \times d_{\text{model}}$） |
| $V$ | 词表大小 (Vocab Size) | `vocab_size` | $151936$ | BPE 词表容量（152K 词表） |
| $\text{rope\_base}$ | RoPE 频率基数 | `rope_theta` | $1000000.0$ | 长文本外推频率参数（100万） |

---

## 依赖关系

- 经典 Transformer 架构语言与注意力推导见 [[论文阅读/transformer/00-transformer阅读地图]]；
- 参数高效微调机制与手写算子见 [[论文阅读/lora/00-阅读地图]]；
- 分词契约与模型生成机制见 [[huggingface/tokenizer-类型契约]] 与 [[huggingface/generate-生成机制]]。
