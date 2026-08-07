---
title: generate() 生成机制：自回归循环
published: 2026-08-05
description: 先讲最基础的输入输出模式 model(input_ids) 返回 logits、一次前向同时预测所有位置的下一个 token、生成 = 把预测拼回输入（输出 = input_ids + answer）、generate() 打包自回归循环、greedy vs sampling、停止条件、切片剥离 prompt、KV 缓存
tags: [HuggingFace, transformers, generate, 自回归生成, KV缓存]
category: HuggingFace
---

# generate() 生成机制：自回归循环

`model.generate()` 是文本生成入口。要理解它，先得明白**最基础的输入输出模式**：`model(input_ids)` 一次前向返回什么。这篇从地基讲起，一路到完整的生成循环。

> 归属：**`transformers` 库** —— 生成方法（`model.generate()`）属于 HF 模型。

## 1. 先看最基础的输入输出：model(input_ids) 一次前向

一切的起点是模型的最原始调用——喂 `input_ids`，拿 `logits`（详见 [[causallm-前向契约]]）：

```python title="basic-forward.py"
inputs = tokenizer("你好世界", return_tensors="pt")   # input_ids: [1, 4]
outputs = model(input_ids=inputs["input_ids"])        # ← 基础模式
logits = outputs.logits                                # [1, 4, vocab]
```

### 1.1 logits 是什么：先看懂数据结构

`logits` 是模型输出的原始打分（未归一化），形状 `[batch, seq_len, vocab_size]`。三个维度各是什么：

| 维度 | 含义 | 例子 |
| --- | --- | --- |
| `batch` | 一次处理几条序列 | 1（1 条） |
| `seq_len` | 序列里几个 token | 4（4 个 token，每个位置一行） |
| `vocab_size` | **词表大小**——模型认识的所有 token 的个数 | 5（简化，真实模型是 15 万级别） |

拿一个 vocab=5 的迷你例子看实际样子。假设输入是 4 个 token，id 为 `[0, 2, 3, 1]`，`logits` 形状 `[1, 4, 5]` 展开就是 4 行、每行 5 个分数：

```
             id 0    id 1    id 2    id 3    id 4
位置 0:     [ 0.1,   2.5,    0.3,    0.2,    0.4  ]
位置 1:     [ 0.0,   0.1,    3.1,    0.2,    0.5  ]
位置 2:     [ 0.4,   0.3,    0.2,    3.0,    0.1  ]
位置 3:     [ 0.2,   0.1,    0.3,    0.4,    2.8  ]
```

**每行的 5 个分数 = "下一个 token 是哪个词"的把握**，分数越高越有把握。比如位置 3 那一行，id 4 的分数最高（2.8），说明模型认为"下一个 token 最可能是 id 4"。（分数是演示用随机数。）

### 1.2 每行的含义：看完前缀，预测下一个

**位置 i 的那行分数，是模型"看完前 i+1 个 token 后，猜测下一个该接什么"的打分**：

```
输入：       [0, 2, 3, 1]
             位置 0 只看到 [0]          → 猜"0 后面该接什么"（最高分 2.5 → id 1）
             位置 1 只看到 [0, 2]       → 猜"0,2 后面该接什么"（最高分 3.1 → id 2）
             位置 2 只看到 [0, 2, 3]    → 猜"0,2,3 后面该接什么"（最高分 3.0 → id 3）
             位置 3 看到全部 [0, 2, 3, 1] → 猜"之后该接什么"（最高分 2.8 → id 4，新 token！）
```

用抽象符号写就是原文的图——**一次前向 = 同时预测了所有位置的"下一个 token"**：

```
输入：  [t1, t2, t3, t4]         （4 个 token）
输出：  logits [1, 4, vocab]
        │
        ├── 位置 0 的 logits → 预测 t2（"看了 t1，下一个是…"）
        ├── 位置 1 的 logits → 预测 t3（"看了 t1t2，下一个是…"）
        ├── 位置 2 的 logits → 预测 t4
        └── 位置 3 的 logits → 预测 t5（下一个【新】token！）
```

### 1.3 为什么"看完前缀"就能"预测下一个"：因果掩码 + shift

位置 i 的输出凭什么能当"对下一个 token 的预测"？两个机制保证：

1. **因果掩码**：attention 只允许位置 i 看 0..i，**看不到未来**——位置 i 的输出只携带"前缀 t1..ti"的信息，它的打分就是"基于前缀的猜测"
2. **shift 对齐**：训练时模型内部把"位置 i 的输出"和"位置 i+1 的 token"对齐成一对标准答案，逼着位置 i 学会"猜下一个"——这就是 CausalLM 的训练目标（完整机制见 [[causallm-前向契约]] 第 4 节）

**所以最后一个位置（位置 3）的 logits 就是"下一个新 token 的预测"**——生成循环就从取它开始（见第 2 节）。

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

## 3. generate()：把第 2 节的循环打包

`model.generate()` 就是把**第 2 节那个手写循环**（取最后位置 → 选 token → 拼回 → 重复）打包成一次调用。你只管描述"要什么"，它内部替你跑循环，并处理三件事：**采样策略**（选 token 的方式，第 4 节）、**停止判断**（何时停，第 5 节）、**KV 缓存**（每步只算新 token，第 7 节）。

**输入**——逐个传关键字参数，每个参数是什么一目了然：

```python title="generate-call.py"
outputs = model.generate(
    input_ids=input_ids,              # ① prompt 的 token id 张量（torch.Tensor），形状 [batch, prompt_len]（必传）
    attention_mask=attention_mask,    # ② 0/1 张量，标记真实/pad 位，和 input_ids 同形状（有 padding 必传）
    max_new_tokens=30,                # ③ 最多生成 30 个【新】token
    eos_token_id=eos_id,              # ④ 生成到结束符就停
)
```

**`attention_mask` 是什么**：和 `input_ids` 等长的 0/1 序列，1 = 真实 token、0 = 填充位——由 tokenizer 在 padding 时生成。**生成本体不讲这里**，作用详解见 [[tokenizer-类型契约#8-attention_mask-的两个作用]]；生成时 batch 用左填充（保证真实 token 贴序列末尾）见 [[tokenizer-类型契约#7-padding_side填充方向与生成的意义]]。单条输入没有 padding 时全是 1，传不传都一样，但养成传的习惯更安全。

**输出**——返回一个 `torch.Tensor`，不是文本、也不是带字段的对象：

| 属性 | 值 |
| --- | --- |
| 类型 | `torch.Tensor`（普通张量） |
| 元素 | **token id**（int） |
| 形状 | `[batch, prompt_len + new_tokens]`——**即使 batch=1 也是二维 `[1, ...]`**，不会降成一维（`generate()` 内部按批次处理，batch 维从 tokenizer 的 `[1, n]` 一路保留） |
| 内容 | **prompt + answer**——和第 2 节手写循环完全一致：每轮把新 token 拼回输入，最终 = 原始 input_ids 后面接着生成的部分 |

```python title="generate-output.py"
print(type(outputs))                    # <class 'torch.Tensor'>
print(outputs.shape)                    # torch.Size([1, 7])  ← prompt 4 个 + 新生成 3 个
text = tokenizer.decode(outputs[0], skip_special_tokens=True)   # token id → 文本
```

- `**inputs` 是**捷径写法**：tokenizer 返回的字典里正好有 `input_ids` 和 `attention_mask`，`model.generate(**inputs, max_new_tokens=30)` 等价于把这两个键展开成上面的关键字参数（解包语法见 [[函数参数与解包]]）。想看清每个参数时用展开写；确认行为无误后日常用 `**inputs` 更省事
- 内部默认在 `torch.no_grad()` 下运行（推理不建计算图，见 [[自动求导与梯度]]）

::::note
**generate() 不是一次出全部 token**：它内部仍是"每步一个 token、每步一次前向"的循环，只是你看不见。生成 100 个 token 要跑 100 次前向——KV 缓存让每次前向只算新 token（见第 7 节）。
::::

## 4. greedy vs sampling：选 token 的策略

第 2 步"选一个 token"有两种策略：

```python title="decoding-strategy.py"
# greedy：每步取分数最高的 token（确定性）
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    do_sample=False,
    max_new_tokens=30,
)

# sampling：按概率分布随机采样（多样性）
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
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
- `temperature` 缩放 logits 后再 softmax。**为什么 T 能控制随机性**：softmax 的分布陡峭程度由 logits 之间的差距决定——除以 T 就是把这个差距放大/压缩：`T=0.1` 差距放大 10 倍，softmax 后最高分几乎必胜（≈greedy）；`T=2.0` 差距减半，分布变平，随机性更大
- `top_p=0.9`：token 按概率降序排，累积到 90% 截断，只从这些里采样（去掉长尾噪声）

::::tip
**常见配置**：要确定性用 `do_sample=False`；要多样性用 `do_sample=True, temperature=0.7, top_p=0.9`。`temperature=0` 等价于 greedy。
::::

## 5. 停止条件：max_new_tokens 与 eos_token_id

```python title="stop-conditions.py"
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
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
# input_ids 形状 [1, prompt_len]
# outputs 形状 [1, prompt_len + new_tokens]

prompt_len = input_ids.shape[1]            # prompt 的 token 数（input_ids 就是原始 prompt）
generated_tokens = outputs[0][prompt_len:]  # 切出新生成部分（跳过 prompt）

text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
```

- `outputs[0]` 取 batch 第 0 条 → 一维张量
- `[prompt_len:]` 跳过前 `prompt_len` 个（prompt），取后面的（answer）——切片语法见 [[张量形状转置切片]]
- 这就是第 2 节洞察的直接应用：**必须知道 prompt 多长，才能把 answer 切出来**

## 7. past_key_values：KV 缓存（加速）

### 7.1 背景：self-attention 每轮要算什么

每一轮前向，每层 self-attention 都要给每个位置算三个向量——**Q（Query，我在找什么）、K（Key，我能提供什么）、V（Value，我携带的信息）**，然后"当前位置的 Q 和所有位置的 K 做匹配，按匹配度加权汇总所有 V"（attention 机制详见 [[架构-编码器与解码器]]）。生成 100 个 token 要跑 100 轮前向——如果不做优化，每一轮都得**把整个序列的 K、V 从头重算一遍**。

### 7.2 核心洞察：历史位置的 K、V 永远不变

关键在因果掩码：位置 i 的 hidden state **只依赖它自己和它之前的位置**（位置 i 只能看到 0..i，见 [[架构-编码器与解码器]]）。所以——

- 新 token 加进来时，**旧位置 i 的 hidden state 一个字节都不会变**
- hidden state 不变 → 它算出来的 K、V 也就不变
- 因此历史 K、V **只需要算一次，之后每轮直接复用**

每轮真正要算的只有**新 token 自己的 K、V**。Q 不需要缓存——每个新 token 的 Q 只服务它自己这一次匹配，用完即弃。

### 7.3 具体数字走一遍：缓存怎么长大

用第 2 节的例子 `[t1, t2, t3]` 生成 3 个 token，对比两边的计算量（匹配次数 = attention 里 Q·K 点积数，是计算量主项）：

```
无缓存（每轮把整个序列重算一遍）：
  ① 前向 [t1, t2, t3]           → 3 个位置全算（3² 次匹配）
  ② 前向 [t1, t2, t3, a1]       → 4 个位置全算（4²）——位置 0-2 上轮算过，白算
  ③ 前向 [t1, t2, t3, a1, a2]   → 5 个位置全算（5²）——位置 0-3 白算
  累计：3² + 4² + 5² = 50 次匹配

有缓存（每轮只算新 token）：
  ① 前向 [t1, t2, t3]       → 算 3 份 K、V → 缓存 [(k1,v1),(k2,v2),(k3,v3)]
  ② 前向 [a1] + 缓存         → 只算 a1 的 K、V，拼进缓存（4 项），新位置做 4 次匹配
  ③ 前向 [a2] + 缓存         → 只算 a2 的 K、V，拼进缓存（5 项），新位置做 5 次匹配
  累计：3 + 4 + 5 = 12 次匹配
```

同样的 3 步生成，匹配次数从 50 降到 12——序列越长、生成的 token 越多，差距越大。

### 7.4 复杂度：从 O(N²) 到 O(N)

|  | 每生成一个 token | 生成 N 个 token 的总算力 |
| --- | --- | --- |
| 不用缓存 | 重算整个 k 长序列 → O(k²) | O(N³) |
| 用缓存 | 只对新位置做 O(k) 次匹配 | O(N²) |

- N=1000 时，O(N³) 是 O(N²) 的 1000 倍——**所以 `generate()` 默认 `use_cache=True`**，这是生成速度的生命线

### 7.5 代价：空间换时间

缓存要存**所有层、所有位置**的 K、V。以 7B 规模模型（32 层、32 head、head 维 128）、seq=1024、fp16 估算：

```
32 层 × 1024 位置 × 32 head × 128 维 × 2（K 和 V）× 2 字节 ≈ 512 MB
```

- 缓存大小**随序列长度线性增长**——长文本生成时，KV 缓存可能比模型权重还大（这正是"长上下文"显存吃紧的根本原因）
- 显存不足时的对策：缩短生成长度、换更小的模型、或依赖模型架构层面的 GQA/MQA（多个 head 共享 KV）

### 7.6 与 CausalLMOutputWithPast 的联系

`use_cache=True` 时，每轮前向返回的 `outputs.past_key_values` 就是这份缓存（字段表见 [[causallm-前向契约]]）。`generate()` 内部循环是这样跑的：前一轮的 `past_key_values` 原样传给下一轮 `forward(past_key_values=...)`，新 K、V 拼在后面。这个传参循环**由 generate() 自动管理**，你手动调用时只需知道有这回事。

## 8. @torch.no_grad()：推理必用

```python title="no-grad.py"
import torch

@torch.no_grad()
def generate_text(model, tokenizer, prompt, device, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        input_ids=inputs["input_ids"],        # tokenizer 返回的字典里有这两项
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
    )
    return outputs
```

- `@torch.no_grad()` 包裹后函数内所有张量**不记录计算图**——推理不需要 backward，省显存加速（机制见 [[自动求导与梯度]]）
- `generate()` 内部默认已处理；但**手动调 `model(input_ids)` 推理时必须自己加**，否则白白建计算图

## 小结

| 语法 / 概念 | 作用 |
| --- | --- |
| `model(input_ids)` | 一次前向 → logits `[batch, seq, vocab]`，每个位置预测下一个 token |
| 生成循环 | 取最后位置 → 采样 → 拼回输入 → 重复（**输出 = input_ids + answer**） |
| `model.generate(input_ids=..., max_new_tokens=...)` | 把第 2 节的循环打包；返回 `torch.Tensor`（token id，形状 `[batch, prompt_len+new_tokens]`，含 prompt） |
| `do_sample=False` / `True` | greedy（argmax） vs 按概率采样 |
| `max_new_tokens=N` / `eos_token_id=...` | 停止条件（上限 / 自然停止） |
| `outputs[0][prompt_len:]` | 切片剥离 prompt，取 answer |
| `use_cache=True` | KV 缓存：历史 K/V 因因果掩码永不变，每步只算新 token（O(k)），总算力 O(N³) → O(N²)；代价是显存随序列线性增长 |
| `@torch.no_grad()` | 推理关闭梯度追踪 |
