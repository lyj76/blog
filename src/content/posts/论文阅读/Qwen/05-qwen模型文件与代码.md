---
title: Qwen 模型文件与代码：从 Config 到 PyTorch 源码
published: 2026-08-08
description: 模型目录文件职责、config.json 字段决定的算子结构、权重张量存储契约与 HuggingFace PyTorch 源码映射。
tags: [论文阅读, Qwen, 源码, PyTorch, config, safetensors]
category: 论文阅读
---

# Qwen 模型文件与代码：从 Config 到 PyTorch 源码

> 承接 [[论文阅读/qwen/01-qwen主线]] 与 [[论文阅读/qwen/02-qwen-decoder-block]]：01 给了 7 个权重矩阵的拓扑，02 拆了单层算子。本篇把数学符号落到磁盘——**config 里的每个数字决定什么结构、权重在文件里长什么样、源码怎么对应公式**。索引见 [[论文阅读/qwen/00-qwen阅读地图]]。

## 一个模型目录里有什么

一个标准的 Qwen 权重目录，四类文件各管一件事：

```text title="model_dir.txt"
Qwen2.5-0.5B-Instruct/
├── config.json                 ──> 结构声明：层数、维度、头数
├── model.safetensors           ──> 浮点权重：训练收敛后的真实张量
├── tokenizer.json / vocab.json ──> 离散分词：自然语言 ↔ Token ID
└── generation_config.json      ──> 解码策略：默认采样超参数
```

`from_pretrained()` 的加载顺序是固定的：先读 `config.json` 实例化出空的网络结构，再把 `safetensors` 里的数值按参数名填入对应的 `nn.Parameter`。`config.json` 决定**结构**，`safetensors` 决定**数值**，两者缺一不可。

## config.json：每个字段决定什么

```json title="Qwen2.5-0.5B config.json (核心字段)"
{
  "hidden_size": 896,
  "num_hidden_layers": 24,
  "num_attention_heads": 14,
  "num_key_value_heads": 2,
  "head_dim": 64,
  "intermediate_size": 4864,
  "vocab_size": 151936,
  "rope_theta": 1000000.0,
  "tie_word_embeddings": true
}
```

| Config 字段 | 数学符号 | 决定的算子结构 |
|:---|:---:|:---|
| `hidden_size: 896` | $d_{\text{model}}$ | 残差主干宽度、Embedding 维度、RMSNorm 维度 |
| `num_hidden_layers: 24` | $L$ | `nn.ModuleList` 里 Decoder 层堆叠数量 |
| `num_attention_heads: 14` | $h_q$ | Query 分头数 |
| `num_key_value_heads: 2` | $h_{kv}$ | GQA 的 KV 头数（每份 KV 服务 7 个 Q） |
| `head_dim: 64` | $d_h$ | 单头维度（$h_q \times d_h = 896$） |
| `intermediate_size: 4864` | $d_{ff}$ | SwiGLU 升维宽度（约 $5.4 \times d_{\text{model}}$） |
| `vocab_size: 151936` | $V$ | Embedding 行数、LM Head 分类维度 |
| `rope_theta: 1000000.0` | $\text{base}$ | RoPE 频率基数 $\theta_i = \text{base}^{-2i/d_h}$ |
| `tie_word_embeddings: true` | — | `lm_head.weight` 是否共享 `embed_tokens.weight` |

注意 `head_dim` 是显式字段：它允许 $h_q \times d_h \ne d_{\text{model}}$ 的非常规配置（如 GQA 下 KV 侧 $2 \times 64 = 128 \ne 896$），这也是 02 里 $W_K, W_V$ 形状为 $128 \times 896$ 的来源。

## 权重在文件里长什么样

数学符号里 $y = xW$，PyTorch 的 `nn.Linear` 存的是转置后的形状：

$$
\text{nn.Linear}(\text{in}, \text{out}).\text{weight} \in \mathbb{R}^{\text{out} \times \text{in}}, \qquad
\text{forward: } x @ W^T.
$$

Qwen2-0.5B 单层 7 个权重在 safetensors 里的真实形状：

```python title="weight_shapes.py"
# 1. Attention 投影
model.layers.0.self_attn.q_proj.weight:    torch.Size([896, 896])    # in 896, out 896
model.layers.0.self_attn.k_proj.weight:    torch.Size([128, 896])    # in 896, out 128 (2 头)
model.layers.0.self_attn.v_proj.weight:    torch.Size([128, 896])    # in 896, out 128 (2 头)
model.layers.0.self_attn.o_proj.weight:    torch.Size([896, 896])    # in 896, out 896

# 2. SwiGLU MLP 投影
model.layers.0.mlp.gate_proj.weight:       torch.Size([4864, 896])   # in 896, out 4864
model.layers.0.mlp.up_proj.weight:         torch.Size([4864, 896])   # in 896, out 4864
model.layers.0.mlp.down_proj.weight:       torch.Size([896, 4864])   # in 4864, out 896
```

形状和 01 的参数拓扑表完全一致：$W_Q$ 是 $896 \times 896$，$W_K, W_V$ 是 $128 \times 896$，gate/up 是 $4864 \times 896$，down 是 $896 \times 4864$。数学世界的形状在这里是转置存储的，forward 时统一 `x @ W.T`。

## 源码：模块层级与公式对应

`transformers/models/qwen2/modeling_qwen2.py` 的模块树：

```text title="module_tree.txt"
Qwen2ForCausalLM (外层：挂 lm_head 分类器 + 交叉熵 Loss)
  └── Qwen2Model (Transformer 主干)
        ├── embed_tokens (nn.Embedding: 查表产出 H⁰)
        ├── layers: nn.ModuleList[24 × Qwen2DecoderLayer]
        │     ├── input_layernorm (Qwen2RMSNorm)
        │     ├── self_attn (Qwen2Attention: 封装 RoPE 与 GQA)
        │     ├── post_attention_layernorm (Qwen2RMSNorm)
        │     └── mlp (Qwen2MLP: gate/up/down 三矩阵 SwiGLU)
        └── norm (最终 Qwen2RMSNorm)
```

两个核心 forward 与 02 的公式逐行对应：

```python title="modeling_qwen2.py (核心前向)"
# SwiGLU: 对应 (SiLU(x @ W_gate.T) * (x @ W_up.T)) @ W_down.T
class Qwen2MLP(nn.Module):
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# 单层 Block: 对应 h + f_mlp(RMSNorm(h + f_attn(RMSNorm(h))))
class Qwen2DecoderLayer(nn.Module):
    def forward(self, hidden_states, position_embeddings, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, position_embeddings=position_embeddings, ...)
        hidden_states = residual + hidden_states            # 第一路残差

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states            # 第二路残差
        return hidden_states
```

对照 01 的单层公式 $h^{(l+1)} = h^{(l)} + f_{\text{mlp}}(\text{RMSNorm}(h^{(l)} + f_{\text{attn}}(\text{RMSNorm}(h^{(l)}))))$——代码就是公式的逐行翻译。`residual = hidden_states` 先存、`residual + hidden_states` 后加，就是公式里残差在归一化之外的实现。

## 边界

- **`head_dim` 显式化是 Qwen 的特色，不是所有模型都有**：很多实现（如 LLaMA 早期）用 `hidden_size // num_attention_heads` 隐式计算，无法表达 GQA 下的非整除配置。
- **`tie_word_embeddings` 只对部分小模型开启**：0.5B/1.5B 为了压参数量开启；大模型（7B+）通常关闭，LM Head 独立训练。这是配置项，不是架构定理。
- **safetensors 的形状契约是 PyTorch 约定的，不是数学必然**：数学上 $W$ 写成 $d_{out} \times d_{in}$ 还是 $d_{in} \times d_{out}$ 都行，PyTorch 选了前者配合 `x @ W.T`——换框架（如权重转置存储的某些推理引擎）时形状会变。

至此 Qwen 系列闭环：01 架构主线 → 02 单层算子 → 03 输入训练 → 04 推理采样 → 05 文件代码。总览见 [[论文阅读/qwen/00-qwen阅读地图]]。
