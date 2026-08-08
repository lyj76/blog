---
title: Qwen 模型文件与代码：从目录到投影矩阵
published: 2026-08-08
description: 把 Qwen 模型目录、config.json、safetensors、tokenizer 与 Python 模块对应起来，读懂文件如何变成一个 Decoder。
tags: [论文阅读, Qwen, config, tokenizer, 实现]
category: 论文阅读
---

# Qwen 模型文件与代码：从目录到投影矩阵

> 这一篇解决实际阅读模型时的入口问题：目录里的文件各自负责什么，`config.json` 如何决定网络骨架，数学里的矩阵又如何对应到 Python 模块。

## <<模型目录：参数、骨架与文本契约>>

一个可运行的 Qwen 目录通常包含三类核心信息：

| 信息 | 常见文件 | 负责什么 |
| --- | --- | --- |
| 参数 | `model.safetensors` 或分片 | 给每个层填入训练后的张量 |
| 骨架 | `config.json` | 决定层数、维度、head 数和词表 |
| 文本契约 | `tokenizer.json`、`tokenizer_config.json` 等 | 文本 ↔ token ID |

生成配置 `generation_config.json` 只影响解码参数；README 和 LICENSE 不参与前向，但发布模型时不能因为运行不需要就随意删除许可文件。

## <<config.json：从字段得到网络形状>>

当前配置可以翻译成：

| 字段 | 值 | 产生的结构 |
| --- | ---: | --- |
| `hidden_size` | $896$ | $d_{model}$ |
| `num_hidden_layers` | $24$ | 24 个 Decoder block |
| `num_attention_heads` | $14$ | Q head 数 |
| `num_key_value_heads` | $2$ | K/V head 数，GQA |
| `head_dim` | $64$ | 每个 head 的宽度 |
| `intermediate_size` | $4864$ | SwiGLU 中间维度 |
| `vocab_size` | $151936$ | embedding 与 LM Head 词表维度 |
| `tie_word_embeddings` | `true` | 输入/输出词嵌入共享 |

所以 `config.json` 不是权重，也不包含知识；它是实例化网络所需的结构声明。`AutoModelForCausalLM.from_pretrained(path)` 先读配置建结构，再读权重填参数。

## <<safetensors：把数值填进结构>>

配置决定某个参数应该有多大，权重文件提供这个参数的实际数值。例如配置决定 `q_proj` 是 $896\to896$，safetensors 中保存的张量才决定这层具体做哪个线性变换。

大模型可能拆成多个分片：

```text
model-00001-of-0000N.safetensors
model-00002-of-0000N.safetensors
model.safetensors.index.json
```

index 按参数名记录分片位置。缺少分片通常意味着模型无法完整加载。

## <<tokenizer：模型输入输出的另一半>>

tokenizer 文件不参与神经网络矩阵乘法，却决定字符串如何变成整数：

$$
\text{text}\xrightarrow{\text{tokenizer}}\text{input\_ids}
\xrightarrow{\text{model}}\text{logits}
\xrightarrow{\text{tokenizer.decode}}\text{text}.
$$

常见文件包括 `tokenizer.json`、词表、合并规则、特殊 token 映射和 chat template。模型权重与 tokenizer 必须配套；词表换了，ID 的含义也换了。

## <<代码模块：数学矩阵如何落地>>

数学行向量写法为 $XW$，但 PyTorch `nn.Linear(in,out)` 的 `weight` 保存形状是 $(out,in)$，forward 等价于 $XW^T+b$。当前 Qwen block 中各模块为：

| Python 模块 | 行向量数学 | `weight.shape` |
| --- | --- | --- |
| `q_proj` | $Q=RW_Q$ | $(896,896)$ |
| `k_proj` | $K=RW_K$ | $(128,896)$ |
| `v_proj` | $V=RW_V$ | $(128,896)$ |
| `o_proj` | heads 融合 | $(896,896)$ |
| `gate_proj` | $G=\operatorname{SiLU}(RW_{gate})$ | $(4864,896)$ |
| `up_proj` | $P=RW_{up}$ | $(4864,896)$ |
| `down_proj` | $(G\odot P)W_{down}$ | $(896,4864)$ |

因此一层的代码树可以抽象成：

```text
DecoderLayer
├── input_layernorm
├── self_attn
│   ├── q_proj / k_proj / v_proj
│   └── o_proj
├── post_attention_layernorm
└── mlp
    ├── gate_proj / up_proj
    └── down_proj
```

它和 [[论文阅读/qwen/02-qwen-decoder-block]] 的公式一一对应。

## 从目录读到前向

读一个 Qwen 模型时按这个顺序：

1. 先看 `config.json`，写出 $d_{model}$、层数、head 数和 $d_{ff}$；
2. 再看模块树，确认 attention、RMSNorm、SwiGLU 的位置；
3. 对照 `weight.shape`，检查每个投影的输入输出维度；
4. 最后看 tokenizer 和 generation config，确定输入 ID 与输出解码规则。

这条顺序把“文件名”还原成“数学对象”，也就能从代码回到 Qwen 的完整前向链。
