---
title: model.generate() 
published: 2026-08-05
description: 先讲最基础的输入输出模式 model(input_ids) 返回 logits、一次前向同时预测所有位置的下一个 token、生成 = 把预测拼回输入（输出 = input_ids + answer）、generate() 打包自回归循环、greedy vs sampling、停止条件、切片剥离 prompt、KV 缓存
tags: [HuggingFace, transformers, generate, 自回归生成, KV缓存]
category: HuggingFace
---

# model.generate()自回归生成 

`model.generate()` 是文本生成入口。

> 归属：**`transformers` 库** —— 生成方法（`model.generate()`）属于 HF 模型。

## 1. 先看最基础的输入输出：model(input_ids) 一次前向

一切的起点是模型的最原始调用——喂 `input_ids`，拿 `logits`（详见 [[CausalLM 函数model()定义的一次前向过程]]）：

input_ids通过多级musk,反刍输出


比如
```
input[0]输出"input[1]"；

input[0:1]输出"input[2]"；

... 
input[0:n]对应 out[n+1]这个才不是原句子是真正的输出

```


```python title="basic-forward.py"
inputs = tokenizer("你好世界", return_tensors="pt")   # input_ids: [1, 4]
outputs = model(input_ids=inputs["input_ids"])        # ← 基础模式
logits = outputs.logits                                # [1, 4, vocab]
```

### 1.1 logits 是什么：先看懂数据结构

`logits` 是模型输出的原始打分（未归一化），形状 `[batch, seq_len, vocab_size]`。

也就是对应着input大小的每一个位置一个vocab向量

vocab向量的每一个位置就是对应token的概率输出





## 2. model.generate()循环

现在要生成新内容，做法**朴素而直接**：

1. 取**最后一个位置**的 logits（它在预测"下一个新 token"）
2. 选一个 token（greedy 取分数最高，sampling 按概率随机）
3. **把它拼到 `input_ids` 末尾** → 序列变长一位
4. 重复——这就是自回归循环



## 3. model.generate()：把第 2 节的循环打包



**输入**——逐个传关键字参数

```python title="generate-call.py"
outputs = model.generate(
    input_ids=input_ids,              # ① prompt 的 token id 张量（torch.Tensor），形状 [batch, prompt_len]（必传）
    attention_mask=attention_mask,    # ② 0/1 张量，标记真实/pad 位，和 input_ids 同形状（有 padding 必传）
    max_new_tokens=30,                # ③ 最多生成 30 个【新】token
    eos_token_id=eos_id,              # ④ 生成到结束符就停
)
```
### 3.1基础参数
| 参数 | 含义 | 形状 | 训练时传？ |
| --- | --- | --- | --- |
| `input_ids` | 输入 token 序列 | `[batch, seq]` | 必传 |
| `attention_mask` | 01的填充位。$M=-\infty$使得查询权重pad位$softmax (\frac{QK^T}{\sqrt{d}}+M)=0$，则其他向量都不会加入这个向量意义 | `[batch, seq]` | 通常传 |
| `position_ids` | 每个位置的下标编号 | `[batch, seq]` | 少用（默认 |
| `past_key_values` | 历史 KV 缓存（生成加速，在generate中才有意义） | 元组 | 生成时用 |


### 3.2贪心和采样
| 参数 | 含义 | 形状 | 何时开启 |
| --- | --- | --- | --- |
|do_sample|输出的预测向量是否取最高概率的，还是随机采样|bool类型，默认是false|代码数学可以false，写文章可以true|
|temperature|温度控制概率分布形状。原来的预测向量softmax是$P_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$，用T控制会缩放距离变成$P_i = \frac{e^{\frac{z_i}{T}}}{\sum_{j} e^{\frac{z_j}{T}}}$。如果两个token差距是10,T=0.2差距就是10/0.2=50| |当需要采样控制概率的时候 |
|top_k|将topk名之外的采样概率置0|int|截断无效尾|
|top_p|是一个概率，指前n个有效token加起来概率超过topp为止。剩下的全部截断| |截断尾部|
|          


eedy vs sampling：选 token 的策略

第 2 步"选一个 token"有两种策略：




::::tip
**常见配置**：要确定性用 `do_sample=False`；要多样性用 `do_sample=True, temperature=0.7, top_p=0.9`。`temperature=0` 等价于 greedy。
::::

### 3.3. 停止条件：max_new_tokens 与 eos_token_id



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
# input_ids 形状 [1, prompt_len]
# outputs 形状 [1, prompt_len + new_tokens]

prompt_len = input_ids.shape[1]            # prompt 的 token 数（input_ids 就是原始 prompt）
generated_tokens = outputs[0][prompt_len:]  # 切出新生成部分（跳过 prompt）

text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
```

- `outputs[0]` 取 batch 第 0 条 → 一维张量
- `[prompt_len:]` 跳过前 `prompt_len` 个（prompt），取后面的（answer）——切片语法见 [[张量形状转置切片]]
- 这就是第 2 节洞察的直接应用：**必须知道 prompt 多长，才能把 answer 切出来**
