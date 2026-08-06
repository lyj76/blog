---
title: HuggingFace generate() 生成机制：自回归循环
published: 2026-08-05
description: 先讲最基础的输入输出模式 model(input_ids) 返回 logits、一次前向同时预测所有位置的下一个 token、生成 = 把预测拼回输入（输出 = input_ids + answer）、generate() 打包自回归循环、greedy vs sampling、停止条件、切片剥离 prompt、KV 缓存
tags: [HuggingFace, transformers, generate, 自回归生成, KV缓存]
category: HuggingFace
---

# HuggingFace generate() 生成机制：自回归循环

`model.generate()` 是文本生成入口。要理解它，先得明白**最基础的输入输出模式**：`model(input_ids)` 一次前向返回什么。这篇从地基讲起，一路到完整的生成循环。

> 归属：**`transformers` 库** —— 生成方法（`model.generate()`）属于 HF 模型。

## 1. 先看最基础的输入输出：model(input_ids) 一次前向

一切的起点是模型的最原始调用——喂 `input_ids`，拿 `logits`（详见 [[causallm-前向契约]]）：

```python title="basic-forward.py"
inputs = tokenizer("你好世界", return_tensors="pt")   # input_ids: [1, 4]
outputs = model(input_ids=inputs["input_ids"])        # ← 基础模式
logits = outputs.logits                                # [1, 4, vocab]
```

**一次前向返回什么**：`logits`，形状 `[batch, seq_len, vocab_size]`——**每个位置都输出了一个"词表大小的分数向量"**，代表"这个位置该预测的下一个 token 是什么"。

关键洞察——**一次前向 = 同时预测了所有位置的"下一个 token"**：

```
输入：  [t1, t2, t3, t4]         （4 个 token）
输出：  logits [1, 4, vocab]
        │
        ├── 位置 0 的 logits → 预测 t2（"看了 t1，下一个是…"）
        ├── 位置 1 的 logits → 预测 t3（"看了 t1t2，下一个是…"）
        ├── 位置 2 的 logits → 预测 t4
        └── 位置 3 的 logits → 预测 t5（下一个【新】token！）
```

（"为什么每个位置都在预测下一个 token"——因果掩码 + shift 一位，见 [[causallm-前向契约]] 的 shift 小节。）

## 2. 从"预测"到"生成"：把预测拼回输入

现在要生成新内容，做法**朴素而直接**：

1. 取**最后一个位置**的 logits（它在预测"下一个新 token"）
2. 选一个 token（greedy 取分数最高，sampling 按概率随机）
3. **把它拼到 `input_ids` 末尾** → 序列变长一位
4. 重复——这就是自回归循环

```python title="manual-loop.py"
for _ in range(3):                      # 手动生成 3 个 token（示意）
    logits = model(input_ids=input_ids).logits
    next_id = logits[:, -1, :].argmax(dim=-1)   # 取最后位置 + 选分数最高
    input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=-1)  # 拼回
```

用数字看**序列一步步变长**（这就是你记忆里的"返回内容是 input_ids + answer"）：

```
初始 input_ids =  [t1, t2, t3]              （prompt，3 个 token）
① 前向 → 取最后位置 → 采样出 a1 → 拼回
   input_ids = [t1, t2, t3, a1]             （4 个）
② 前向 → 取最后位置 → 采样出 a2 → 拼回
   input_ids = [t1, t2, t3, a1, a2]         （5 个）
③ 前向 → 取最后位置 → 采样出 a3 → 拼回
   input_ids = [t1, t2, t3, a1, a2, a3]     （6 个）

最终 outputs = [t1, t2, t3, a1, a2, a3]     ← 就是 input_ids + answer！
```

**核心认知**：生成的结果**天然包含 prompt**——因为每一轮都把新 token 拼回输入，最终张量 = 原始 `input_ids` 后面接着生成的部分。所以取结果时要"剥掉 prompt"（见第 6 节）。

## 3. generate()：把上面的循环打包

`model.generate()` 就是把这个循环封装成一行（内部做了采样策略、停止判断、KV 缓存等）：

```python title="generate-call.py"
outputs = model.generate(**inputs, max_new_tokens=30, eos_token_id=eos_id)
```

- `**inputs` 是字典解包（`input_ids` + `attention_mask` 展开成关键字参数，见 [[函数参数与解包]]）
- `outputs` 就是第 2 节最终的完整序列 `[batch, prompt_len + new_tokens]`
- 内部默认在 `torch.no_grad()` 下运行（推理不建计算图，见 [[自动求导与梯度]]）

::::note
**generate() 不是一次出全部 token**：它每一步只生成一个 token（每步一次前向）。生成 100 个 token 要跑 100 次前向——KV 缓存让每次前向只算新 token（见第 7 节）。
::::

## 4. greedy vs sampling：选 token 的策略

第 2 步"选一个 token"有两种策略：

```python title="decoding-strategy.py"
# greedy：每步取分数最高的 token（确定性）
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=30)

# sampling：按概率分布随机采样（多样性）
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

- **greedy**：每步取 argmax，输出确定但容易重复，适合要稳定结果的场景
- **sampling**：按概率随机选，输出多样
- `temperature` 缩放 logits 后再 softmax：`0.1` 几乎 greedy，`2.0` 更平坦更随机
- `top_p=0.9`：token 按概率降序排，累积到 90% 截断，只从这些里采样（去掉长尾噪声）

::::tip
**常见配置**：要确定性用 `do_sample=False`；要多样性用 `do_sample=True, temperature=0.7, top_p=0.9`。`temperature=0` 等价于 greedy。
::::

## 5. 停止条件：max_new_tokens 与 eos_token_id

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
| `eos_token_id` | 生成到这个 id 就停（序列结束符） |
| `stop_strings` | 遇到指定字符串就停（部分模型支持） |

- 两个条件满足**任一个**就停：要么生成够 30 个，要么遇到了 eos
- `eos_token_id` 通常从 tokenizer 拿：`tokenizer.eos_token_id`，对话模型用 `tokenizer.convert_tokens_to_ids("<|im_end|>")`（见 [[tokenizer-类型契约]]）

::::warning
**max_new_tokens vs max_length**：`max_new_tokens=30` = 最多生成 30 个**新** token；`max_length=100` = **总长度**（prompt + 生成）≤ 100。推荐 `max_new_tokens`，语义清晰，不受 prompt 长度影响。
::::

## 6. outputs 形状与切片：剥离 prompt

因为"输出 = input_ids + answer"（第 2 节），取结果时要剥掉 prompt 部分：

```python title="outputs-slice.py"
# inputs.input_ids 形状 [1, prompt_len]
# outputs 形状 [1, prompt_len + new_tokens]

prompt_len = inputs.input_ids.shape[1]        # prompt 的 token 数
generated_tokens = outputs[0][prompt_len:]    # 切出新生成部分（跳过 prompt）

text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
```

- `outputs[0]` 取 batch 第 0 条 → 一维张量
- `[prompt_len:]` 跳过前 `prompt_len` 个（prompt），取后面的（answer）——切片语法见 [[张量形状转置切片]]
- 这就是第 2 节洞察的直接应用：**必须知道 prompt 多长，才能把 answer 切出来**

## 7. past_key_values：KV 缓存（加速）

每一轮前向，self-attention 都要算每个位置的 K、V 矩阵。**历史位置的 K、V 其实没变**——缓存它们，每步只算新 token：

```python title="kv-cache.py"
outputs = model.generate(**inputs, max_new_tokens=30, use_cache=True)   # 默认开
```

- **不用缓存**：每生成一个 token，整个序列重新算一遍 K、V → O(n²)
- **用缓存**：每步只算新 token 的 K、V，拼到缓存末尾 → O(n)
- 代价：缓存所有历史 K、V，**序列越长显存占用越大**（长文本生成时可能比权重还大）——空间换时间

## 8. @torch.no_grad()：推理必用

```python title="no-grad.py"
import torch

@torch.no_grad()
def generate_text(model, tokenizer, prompt, device, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return outputs
```

- `@torch.no_grad()` 包裹后函数内所有张量**不记录计算图**——推理不需要 backward，省显存加速（机制见 [[自动求导与梯度]]）
- `generate()` 内部默认已处理；但**手动调 `model(input_ids)` 推理时必须自己加**，否则白白建计算图

## 小结

| 语法 / 概念 | 作用 |
| --- | --- |
| `model(input_ids)` | 一次前向 → logits `[batch, seq, vocab]`，每个位置预测下一个 token |
| 生成循环 | 取最后位置 → 采样 → 拼回输入 → 重复（**输出 = input_ids + answer**） |
| `model.generate(**inputs, ...)` | 把自回归循环打包成一行 |
| `do_sample=False` / `True` | greedy（argmax） vs 按概率采样 |
| `max_new_tokens=N` / `eos_token_id=...` | 停止条件（上限 / 自然停止） |
| `outputs[0][prompt_len:]` | 切片剥离 prompt，取 answer |
| `use_cache=True` | KV 缓存，O(n²) → O(n) |
| `@torch.no_grad()` | 推理关闭梯度追踪 |
