---
title: HuggingFace generate() 生成机制：自回归循环
published: 2026-08-05
description: generate() 自回归循环本质、greedy vs sampling（do_sample / temperature / top_p）、max_new_tokens 与 eos_token_id 停止条件、inputs 字典解包、outputs 形状与切片剥离 prompt、past_key_values KV 缓存、torch.no_grad 推理
tags: [HuggingFace, transformers, generate, 自回归生成, KV缓存]
category: HuggingFace
---

# HuggingFace generate() 生成机制：自回归循环

`model.generate()` 是 HuggingFace 的文本生成入口。它内部跑一个自回归循环：预测下一个 token，拼回输入，再预测，直到停止。理解这个循环和返回值的结构，才能控制生成行为、正确取出结果。

## 1. generate() 的本质：自回归循环

```python title="autoregressive.py"
outputs = model.generate(**inputs, max_new_tokens=30, eos_token_id=eos_id)
```

内部循环：

1. 把 `input_ids` 喂进模型，拿到 logits（形状 `[batch, seq, vocab]`）
2. 取最后一个位置的 logits，选一个 token（greedy 取 argmax，sampling 按概率采样）
3. 把新 token 拼到 `input_ids` 末尾，序列变长一位
4. 重复步骤 1 到 3，直到遇到 `eos_token_id` 或达到 `max_new_tokens`

- 每一步只生成一个 token，拼回去再预测下一个，这就是「自回归」
- `outputs` 是最终完整序列（prompt + 生成部分），形状 `[batch, prompt_len + new_tokens]`

::::note
**generate() 不是一次出全部 token**：它是一个循环，每次前向只预测一个 token。所以生成 100 个 token 要跑 100 次前向（有 KV 缓存时每次只算新 token，见第 6 节）。
::::

## 2. greedy vs sampling：选 token 的策略

```python title="decoding-strategy.py"
# greedy：每步取概率最高的 token
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=30)

# sampling：按概率分布随机采样
outputs = model.generate(
    **inputs,
    do_sample=True,
    temperature=0.7,    # 缩放 logits，越高越随机
    top_p=0.9,          # 只从累积概率前 90% 的 token 里采样
    max_new_tokens=30,
)
```

| 参数 | 取值 | 作用 |
| --- | --- | --- |
| `do_sample` | `bool` | `False` = greedy（取 argmax），`True` = 按概率采样 |
| `temperature` | `float` | 缩放 logits，`<1` 更确定，`>1` 更随机，`1.0` 不变 |
| `top_p` | `float` | 核采样：只从累积概率达 `top_p` 的 token 集合里采样 |
| `top_k` | `int` | 只从概率最高的 k 个 token 里采样 |

- **greedy**（`do_sample=False`）：每步取 argmax，输出确定性强但容易重复，适合需要稳定结果的场景
- **sampling**（`do_sample=True`）：按概率随机选，输出有多样性
- `temperature` 控制 softmax 分布的尖锐程度：`0.1` 几乎 greedy，`1.0` 原始分布，`2.0` 更平坦更随机
- `top_p=0.9`：把 token 按概率从高到低排序，累积到 90% 就截断，只从这些里采样（去掉长尾噪声）

::::tip
**常见配置**：要确定性输出用 `do_sample=False`；要多样性用 `do_sample=True, temperature=0.7, top_p=0.9`。`temperature=0` 等价于 greedy。
::::

## 3. 停止条件：max_new_tokens 与 eos_token_id

```python title="stop-conditions.py"
outputs = model.generate(
    **inputs,
    max_new_tokens=30,      # 最多生成 30 个新 token
    eos_token_id=eos_id,    # 遇到结束符就停
)
```

| 参数 | 作用 |
| --- | --- |
| `max_new_tokens` | 最多生成多少个**新** token（不含 prompt） |
| `eos_token_id` | 生成这个 id 就停（序列结束符） |
| `stop_strings` | 遇到指定字符串就停（部分模型支持） |

- `max_new_tokens` 控制上限，`eos_token_id` 控制自然停止
- 两个条件满足任一个就停：要么生成够了 30 个，要么遇到了 eos
- `eos_token_id` 通常从 tokenizer 拿：`eos_id = tokenizer.eos_token_id`，或对话模型用 `tokenizer.convert_tokens_to_ids("<|im_end|>")`

::::warning
**max_new_tokens vs max_length**：`max_new_tokens=30` 是「最多生成 30 个新 token」；`max_length=100` 是「总长度（prompt + 生成）不超过 100」。推荐用 `max_new_tokens`，语义更清晰，不会因为 prompt 长度变化而出问题。
::::

## 4. inputs 字典解包

```python title="inputs-unpack.py"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
# inputs = {"input_ids": tensor[1, n], "attention_mask": tensor[1, n]}

outputs = model.generate(**inputs, max_new_tokens=30)
# **inputs 把字典展开成关键字参数：
# model.generate(input_ids=..., attention_mask=..., max_new_tokens=30)
```

- `**inputs` 是 Python 字典解包，把 `BatchEncoding` 的 key-value 展开成关键字参数
- tokenizer 返回的字段名（`input_ids`、`attention_mask`）和 `generate` 的参数名对齐
- 所以 `**inputs` 能直接传，不用手动拆

## 5. outputs 形状与切片：剥离 prompt

```python title="outputs-slice.py"
# inputs.input_ids 形状 [1, prompt_len]
# outputs 形状 [1, prompt_len + new_tokens]

prompt_len = inputs.input_ids.shape[1]        # prompt 的 token 数
generated_tokens = outputs[0][prompt_len:]    # 切出新生成部分
# outputs[0] 取 batch 第 0 条，[prompt_len:] 从第 prompt_len 个切到末尾

text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
```

- `outputs` 形状是 `[batch, prompt_len + new_tokens]`，包含 prompt 和生成部分
- `outputs[0]` 取第一条（batch 第 0 个样本），得到一维张量
- `[prompt_len:]` 切片：跳过前 `prompt_len` 个（prompt），取后面的（新生成）
- `inputs.input_ids.shape[1]` 取形状第 1 维（列数），即 prompt 长度

::::tip
**切片语法回顾**：`tensor[a:b]` 取索引 a 到 b-1 的元素；`tensor[b:]` 从 b 取到末尾；`tensor[:b]` 从头取到 b-1。`outputs[0][prompt_len:]` 就是「第一条样本，从 prompt 之后取到末尾」。
::::

## 6. past_key_values：KV 缓存

```python title="kv-cache.py"
# generate() 内部默认开启 KV 缓存
outputs = model.generate(**inputs, max_new_tokens=30, use_cache=True)
```

- 每次前向传播，self-attention 会算出 K 和 V 矩阵（每个位置一组 key-value 向量）
- **KV 缓存**：把之前所有位置的 K、V 存起来，下一步只算新 token 的 K、V，拼到缓存末尾
- 不用缓存：每生成一个 token，要把整个序列重新算一遍 K、V，复杂度 O(n²)
- 用缓存：每步只算新 token 的 K、V，复杂度 O(n)，大幅加速

| 设置 | 效果 |
| --- | --- |
| `use_cache=True`（默认） | 开启 KV 缓存，生成快，多占显存 |
| `use_cache=False` | 关闭缓存，生成慢，省显存 |

::::note
**KV 缓存的空间换时间**：缓存存了所有历史位置的 K、V 张量，序列越长缓存越大。长文本生成时，KV 缓存的显存可能比模型权重还大。这是生成速度快的代价。
::::

## 7. @torch.no_grad()：推理必用

```python title="no-grad.py"
import torch

@torch.no_grad()
def generate_text(model, tokenizer, prompt, device, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return outputs
```

- `@torch.no_grad()` 是装饰器，包裹后函数内所有张量**不记录计算图**
- 推理时不需要 backward（不更新参数），关掉梯度追踪省显存、省内存、速度快
- 训练（要 backward）不能用，推理（只前向）必须用

::::important
**generate() 内部已经处理了梯度**：`model.generate()` 默认在 `torch.no_grad()` 上下文里运行。但如果你手动调 `model(input_ids)` 做推理，必须自己加 `@torch.no_grad()` 或 `with torch.no_grad():`，否则会白白记录计算图、浪费显存。
::::

## 小结

| 语法 / 概念 | 作用 |
| --- | --- |
| `model.generate(**inputs, max_new_tokens=..., eos_token_id=...)` | 自回归生成 |
| `do_sample=False` | greedy，取 argmax |
| `do_sample=True, temperature=..., top_p=...` | 采样，控制随机性 |
| `max_new_tokens=N` | 最多生成 N 个新 token |
| `eos_token_id=...` | 遇到结束符就停 |
| `outputs[0][prompt_len:]` | 切片剥离 prompt，取新生成部分 |
| `use_cache=True` | KV 缓存，加速生成 |
| `@torch.no_grad()` | 推理关闭梯度追踪 |
