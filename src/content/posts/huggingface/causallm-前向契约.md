---
title: HuggingFace CausalLM 前向契约：model() 返回什么
published: 2026-08-05
description: forward 参数契约（关键字调用陷阱）、CausalLMOutputWithPast 字段表、labels 触发内部交叉熵的完整流程（logits 对齐 → 展平 → ignore_index 求平均）、shift 一位的位置对齐机制（位置 n 预测 n+1，末位输出丢弃）、-100 在 SFT 里的掩码用法、logits[batch, seq, vocab] 作为 seq 个并列分类任务的视角、train/eval、**inputs 解包
tags: [HuggingFace, transformers, CausalLM, 前向传播, loss]
category: HuggingFace
---

# HuggingFace CausalLM 前向契约：model() 返回什么

调用 `model(input_ids=..., labels=...)` 后，返回的不是张量，而是一个带多个字段的数据包。理解这个数据包的契约，才能知道「loss 从哪来、logits 长什么样、什么时候有 past_key_values」。这篇按 forward 的执行顺序展开：参数进来 → 内部算 loss → 返回字段 → 模式切换。

> 归属：**`transformers` 库**（`from transformers import AutoModelForCausalLM`）——HF 的模型库。

## 1. forward 的参数契约：input_ids / attention_mask / labels

`forward` 是模型接收输入、产出输出的契约边界。所有 HF CausalLM 模型遵循同一签名，关键参数就这几个：

| 参数 | 含义 | 形状 | 训练时传？ |
| --- | --- | --- | --- |
| `input_ids` | 输入 token 序列 | `[batch, seq]` | 必传 |
| `attention_mask` | 标记真实 / 填充位（1=真实，0=填充） | `[batch, seq]` | 通常传 |
| `position_ids` | 每个位置的下标编号 | `[batch, seq]` | 少用（默认自动生成） |
| `past_key_values` | 历史 KV 缓存（生成加速用） | 元组 | 生成时用 |
| `labels` | 目标 token 序列（标准答案） | `[batch, seq]` | **训练时传** |

`input_ids` 是 tokenizer 产出的 token id 序列（来源见 [[tokenizer-类型契约]]），`attention_mask` 标记真实 / padding 位，两者通常成对出现。`labels` 是训练专属——传了它模型才会算 loss（第 3 节展开）。

::::warning
**为什么必须用关键字参数调用**：`forward` 参数多、顺序固定——`labels` 排在第六位，第二位是 `attention_mask`。写 `model(input_ids, labels)` 位置调用会把 labels 错塞给 `attention_mask`，**不报错但行为完全错误**。HF 生态永远写 `model(input_ids=..., labels=...)`（参数语法见 [[函数参数与解包]]）。
::::

两种调用形态对应两种用法：不传 `labels` 是推理形态（只算 logits，见第 6 节）；传 `labels` 是训练形态（内部自动算交叉熵，返回带 `loss` 的对象，见第 3 节）。

## 2. CausalLMOutputWithPast：返回类型

模型直接调用（`__call__` → `forward`）返回的不是张量，而是 `CausalLMOutputWithPast` 对象——一个把多个字段打包在一起的数据类：

```python title="forward-output.py"
outputs = model(input_ids=input_ids, labels=labels)
type(outputs)   # transformers.modeling_outputs.CausalLMOutputWithPast

outputs.loss              # tensor(2.3456)，交叉熵损失
outputs.logits            # tensor[batch, seq, vocab]，每个位置的词表分布
outputs.past_key_values   # tuple of tuples，KV 缓存（推理加速用）
```

这个对象继承自 `ModelOutput`——既能当属性访问（`outputs.loss`），也能当字典访问（`outputs["loss"]`），两种写法等价。字段表如下：

| 字段 | 类型 | 何时有值 |
| --- | --- | --- |
| `loss` | `Optional[torch.Tensor]` | **传了 `labels` 才有**，否则为 `None` |
| `logits` | `torch.Tensor` | 永远有，形状 `[batch, seq, vocab_size]` |
| `past_key_values` | `Optional[Tuple]` | `use_cache=True` 时有，KV 缓存 |
| `hidden_states` | `Optional[Tuple[Tensor]]` | `output_hidden_states=True` 时有 |
| `attentions` | `Optional[Tuple[Tensor]]` | `output_attentions=True` 时有 |

`loss` 的类型标注 `Optional[torch.Tensor]` 拆开看：

| 部分 | 含义 |
| --- | --- |
| `torch.Tensor` | **张量**：带形状、dtype 的多维数组（结构详见 [[tensor-and-parameter]]） |
| `Optional[X]` | Python 类型注解，= `X \| None`：**这个字段可能有值，也可能是空（`None`）** |

直译就是"要么是个张量，要么是 `None`"——`loss` 在没传 `labels` 时就是 `None`，**用之前必须判空**：

```python title="loss-guard.py"
if outputs.loss is not None:      # 训练时才有值
    loss = outputs.loss.item()    # .item()：张量 → Python 浮点数
```

loss 和 logits 都是张量，通用操作如下：

| 操作 | 例子 | 用途 |
| --- | --- | --- |
| 看形状 / 类型 | `t.shape`、`t.dtype` | 调试、确认维度 |
| 转 Python 标量 | `loss.item()` | 打日志、记录训练指标 |
| 反向传播 | `loss.backward()` | 训练（机制见 [[自动求导与梯度]]） |
| 索引 / 切片 | `logits[:, -1, :]` | 取最后一个位置的 logits（生成时） |
| 取最大值位置 | `logits.argmax(dim=-1)` | 得到预测的 token id |
| 转概率分布 | `logits.softmax(dim=-1)` | 采样 / 分析置信度 |

（`.item()` 见 [[设备与精度管理]]，张量的形状与操作见 [[张量形状转置切片]]。）

::::note
**不是所有字段都有值**：`loss` 只在传 `labels` 时计算；`past_key_values` 只在 `use_cache=True` 时返回。不传 labels 时 `outputs.loss` 是 `None`，但 `outputs.logits` 永远有——推理代码可以放心用 logits，用 loss 前要判空。
::::

## 3. labels 与 loss：内部自动算交叉熵

`loss` 度量的是：**模型在每个位置预测的"下一个 token"，和真答案差多远**。传 `labels` 后，模型内部走一整套交叉熵流程，把"差多远"算成一个标量。

```python title="labels-loss.py"
# 传 labels → 模型自动算 loss
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss   # tensor(2.3456)，可以直接 loss.backward()

# 不传 labels → 没有 loss，只有 logits
outputs = model(input_ids=input_ids)
outputs.loss    # None
outputs.logits  # tensor[batch, seq, vocab]
```

`labels` 是和 `input_ids` **同形状**的张量（`[batch, seq]`），内容是目标 token id。拿到 `loss` 后直接 `loss.backward()` 就能反向传播，不需要自己写损失函数。

用一个 vocab=5 的迷你例子走一遍内部流程。假设 `input_ids = [[0, 1, 2]]`（batch=1, seq=3），`labels` 一样（token id 必须在 0~4 范围内）：

```
① 前向算 logits [1, 3, 5]：位置 0 → [0.1, 2.5, 0.3, 0.2, 0.4]（argmax=1），位置 1 → [0.0, 0.1, 3.1, 0.2, 0.5]（argmax=2）
② shift 对齐（第 4 节详解）：logits [:-1]（位置 0,1）↔ labels [1:]（位置 1,2 的 id：1, 2）——位置 0 预测 1、位置 1 预测 2，恰好命中目标
③ 展平：logits → [2, 5]，labels → [2]（CrossEntropyLoss 要求末维是类别数）
④ CrossEntropyLoss(ignore_index=-100)：每位置算 -log(p_目标id)，跳过 -100，有效位置求平均 → 标量
```

第 ④ 步的交叉熵内部机制（LogSoftmax + NLLLoss、数值稳定性、ignore_index 掩码、为什么分母是有效位置数）是 PyTorch 层的知识，完整拆解见 [[损失函数与-100掩码]]。这里只需记住：HF 模型内部把 logits 和 labels 展平后丢给 `CrossEntropyLoss(ignore_index=-100)`，对有效位置求平均，输出一个标量。

::::tip
**训练循环的标准写法**：`loss = model(input_ids=..., labels=...).loss`，一行搞定前向 + 算损失。然后 `loss.backward()` + `optimizer.step()`（完整流程见 [[自动求导与梯度]]）。
::::

## 4. shift 一位逻辑：位置 n 预测 n+1

### 4.1 完整链路：前向 → 对齐 → 比较

shift 不是一个独立概念，它出现在"**训练目标怎么和模型输出对齐**"这一步。训练时一次前向到 loss 的完整链路是：

```
① 前向：模型读 input_ids，每个位置输出一个 logits 向量（形状 [1, 4, vocab]）
② 对齐：把 logits 和 labels 错开一位（这就是 shift）
③ 比较：每个 logits 向量和它对齐的 label 比，算交叉熵
```

用具体数字走一遍。假设输入 `input_ids = [[0, 2, 3, 1]]`（4 个 token，id 在 0~4 范围），`labels` 和它一样（自监督）：

```
位置:         0    1    2    3
input_ids:    0    2    3    1
```

**① 前向**：每个位置拿到一个 vocab 长度的分数向量。简化 vocab=5，`logits` 形状 `[1, 4, 5]`：

```
             id 0    id 1    id 2    id 3    id 4     argmax = 模型的猜测
位置 0:     [ 0.1,   2.5,    0.3,    0.2,    0.4  ]   → 猜 1（"看完 0 之后该接 1"）
位置 1:     [ 0.0,   0.1,    3.1,    0.2,    0.5  ]   → 猜 2（"看完 0,2 之后该接 2"）
位置 2:     [ 0.4,   0.3,    0.2,    3.0,    0.1  ]   → 猜 3（"看完 0,2,3 之后该接 3"）
位置 3:     [ 0.2,   0.1,    0.3,    0.4,    2.8  ]   → 猜 4（"看完 0,2,3,1 之后该接 4"）
```

（分数是演示用随机数；位置 i 只能看 0..i 是因果掩码保证的，见 4.4。）

**② 对齐（shift）**：位置 i 的 logits 在猜测"看完前 i+1 个 token 之后接什么"——它的标准答案就是**下一个位置的 token**。所以把 logits 和 labels 错开一位：logits 丢掉最后一个位置的输出，labels 丢掉第一个位置的 id。

**③ 比较**：配对后的每对 (logits, label) 算交叉熵：

```
logits[0]（猜 1） ↔ labels[1] = 2    ← 位置 0 的输出，目标 = 位置 1 的 token
logits[1]（猜 2） ↔ labels[2] = 3    ← 位置 1 的输出，目标 = 位置 2 的 token
logits[2]（猜 3） ↔ labels[3] = 1    ← 位置 2 的输出，目标 = 位置 3 的 token
logits[3]（猜 4） ↔ 没有位置 4       ← 丢弃（后面没东西可猜）
labels[0] = 0     ↔ 没有位置 -1      ← 丢弃（没人猜它）
```

**这就是"位置 n 预测 n+1"**：位置 n 的 logits 对齐位置 n+1 的 label。因为因果掩码，位置 n 只看到了 t1..tn，所以它的猜测"基于前 n 个 token 的信息"；训练逼它猜中 tn+1——模型学会的正是"看完前缀，预测下一个"。

### 4.2 shift 在代码里是什么

transformers 的 CausalLM 在 forward 末尾就是这两行切片：

```python title="hf-shift.py"
shift_logits = logits[..., :-1, :]    # 丢掉最后一个位置的输出
shift_labels = labels[..., 1:]        # 丢掉第一个位置的 label
```

- `[:-1]` = 去掉最后一项；`[1:]` = 去掉第一项（切片语法见 [[张量形状转置切片]]）
- shift 后的两个张量**长度一致**（都是 seq_len - 1），逐位置配对算交叉熵
- 传 `input_ids=..., labels=...` 后模型内部自动做，**你不用手动 shift**——手动 shift 再传等于 shift 了两次，对齐全错

### 4.3 为什么末位输出、首位 label 被丢弃

| 被丢弃的 | 原因 |
| --- | --- |
| 位置 3 的 logits（末位输出） | 它在猜"位置 4 的 token"，但序列只有 4 个 token，**没有第 5 个当目标** |
| 位置 0 的 label（首位 id） | 它需要"位置 -1 的输出"来预测，**没有谁在猜它** |

- 有效训练位置 = `seq_len - 1`——每一条样本只产生 `seq_len - 1` 次预测
- 4 个 token 的序列 → 3 次有效的"看前缀、猜下一个"

### 4.4 shift 不是因果掩码

两件事，别混：

| 机制 | 作用在哪 | 回答的问题 |
| --- | --- | --- |
| 因果掩码（causal mask） | attention 内部 | 位置 n **能看到**哪些位置（只能看 0..n，下三角，保证不偷看未来，见 [[架构-编码器与解码器]]） |
| shift 一位 | forward 末尾的 loss 计算 | 位置 n 的 logits **对齐**哪个 label（对齐 n+1） |

两者配合才让"一次前向同时训练所有位置"成立：因果掩码保证位置 i 只携带前缀信息，shift 保证它的输出被训练成"猜下一个"。

## 5. -100 掩码：ignore_index 约定

`-100` 的完整机制（为什么是 -100、ignore_index 怎么跳过位置、分母怎么算）是 PyTorch 层的知识，详见 [[损失函数与-100掩码]]。这里只讲 CausalLM 场景下 `-100` 怎么用。

`-100` 标记的位置**跟着 labels 一起 shift**——你在 labels 里哪些位置填 `-100`，shift 后那些位置依然是被忽略的。所以只需按"原始序列里哪个位置不该学"来填 `-100`，不用操心 shift 后的下标变化。

SFT（监督微调）时，一条样本是 `prompt + answer` 拼成的序列。你只想让模型学"答案怎么续写"，不想让它学"复述 prompt"。做法：prompt 段的 label 填 `-100`，answer 段填真 token id。

```python title="ignore-index.py"
# labels 中 -100 的位置不计入损失
labels = torch.tensor([
    [-100, -100, 1234, 5678, 2],   # prompt 部分填 -100，答案部分是真 id
])
```

用一个具体例子看位置标注。假设 prompt = `[1, 234, 5]`（3 个 token），answer = `[67, 89, 2]`（3 个 token，2 是 eos），拼成 6 长序列：

```
位置：       0    1    2    3    4    5
input_ids:   1    234  5    67   89   2
labels:     -100 -100 -100  67   89   2     （eos 也要学，让模型知道何时停）
           忽略  忽略  忽略  学   学   学
```

shift 后：位置 0,1,2 的 logits 对齐 labels 位置 1,2,3（都是 -100，被跳过）；位置 3,4 的 logits 对齐 labels 位置 4,5（真 id，参与 loss）。模型只在"答案段"算损失，prompt 段完全不计入。同理，batch 里短样本被 padding 补齐时，padding 位的 label 也填 `-100`——配合 `attention_mask`（padding 位标 0），padding 位在前向和 loss 计算里都被正确忽略。

::::important
**-100 不是魔法数字**：它是 PyTorch `CrossEntropyLoss` 的 `ignore_index` 默认值。HuggingFace 没有改这个默认值，所以 `-100` 就是"忽略此位置"的约定。如果你把 `ignore_index` 改成别的数，那就要填那个数（机制见 [[损失函数与-100掩码]]）。
::::

## 6. logits 形状：[batch, seq, vocab]

`logits` 是模型对每个位置、每个词的打分（未归一化）。形状 `[batch, seq_len, vocab_size]`——三个维度分别是批次、序列长度、词表大小。

换个角度看这个形状：**`[batch, seq, vocab]` 等于 seq 个并列的"词表分类任务"**。每个位置都是一次 vocab 大小的打分——"这个位置该接哪个 token"。这和第 4 节的 shift 直接对应：位置 n 的打分向量，就是在预测位置 n+1 的 token。用 vocab=5 走一遍：假设位置 0 的 5 个分数是 `[0.1, 2.5, 0.3, 0.2, 0.4]`，最大值 2.5 在 id1 → `argmax = 1` → 模型预测位置 0 之后该接 id 为 1 的 token。`logits.argmax(dim=-1)` 对每个位置取分数最高的 id，`logits.softmax(dim=-1)` 把分数转成概率分布。

同一个 `logits` 张量，训练和推理用法不同。训练时展平成 `[batch×seq, vocab]`，配合展平后的 labels 丢给 CrossEntropyLoss（见第 3 节、[[损失函数与-100掩码]]）。推理时取最后一个位置的 logits，预测下一个新 token：

```python title="logits-shape.py"
outputs = model(input_ids=input_ids)
logits = outputs.logits
print(logits.shape)   # torch.Size([2, 128, 151936])
#                        batch=2, seq_len=128, vocab_size=151936

# 取下一个 token 预测：next_token_logits = logits[:, -1, :]，形状 [batch, vocab_size]
# 转概率：probs = torch.softmax(next_token_logits, dim=-1)
```

**为什么推理用 `logits[:, -1, :]`**：最后一个位置的 logits 看到了整个输入序列（因果掩码保证它只看过去，详见 [[架构-编码器与解码器]]），它在预测"序列之后该接什么"——这正是生成时要的。切片语法见 [[张量形状转置切片]]，完整的生成循环见 [[generate-生成机制]]。

## 7. train() / eval()：模式切换

```python title="train-eval.py"
model.train()   # 训练模式：Dropout 生效、BatchNorm 用 batch 统计量
model.eval()    # 推理模式：Dropout 关闭、BatchNorm 用历史统计量
```

`train()` / `eval()` 是 `nn.Module` 的方法，切换模型的训练 / 推理模式。主要影响 Dropout 和 BatchNorm：训练时 Dropout 随机丢弃，推理时关闭。**训练循环开头调 `model.train()`，推理前调 `model.eval()`**。

::::note
**这是 PyTorch 的机制，不是 HuggingFace 的**：`model.train()` / `model.eval()` 来自 `nn.Module`（详见 [[nn-module]]），所有 PyTorch 模型都有。HuggingFace 的模型继承自 `nn.Module`，所以也有这两个方法。注意 `eval()` 只改模块行为，推理时还要配 `torch.no_grad()` 关闭计算图追踪（见 [[自动求导与梯度]]）。
::::

## 8. inputs 字典解包

```python title="kwargs-unpack.py"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
# inputs 是 BatchEncoding：{"input_ids": ..., "attention_mask": ...}

outputs = model(**inputs)
# 等价于 model(input_ids=..., attention_mask=...)
```

`**inputs` 是 Python 的字典解包：把字典的 key-value 展开成关键字参数（语法见 [[函数参数与解包]]）。tokenizer 返回的 `BatchEncoding` 里有 `input_ids` 和 `attention_mask`（结构见 [[tokenizer-类型契约]]），正好对应 `forward` 的参数名。所以 `model(**inputs)` 等价于 `model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])`。

::::tip
**为什么能直接解包**：tokenizer 的返回 key 名和 model `forward` 的参数名是对齐的。这是 HuggingFace 的设计契约：tokenizer 产出的字段名 = model 接收的参数名。
::::

## 小结

| 知识 | 要点 |
| --- | --- |
| forward 参数契约 | `input_ids` / `attention_mask` / `labels` / `position_ids` / `past_key_values`，必须关键字调用（位置参数会把 labels 错塞给 attention_mask） |
| 返回类型 | `CausalLMOutputWithPast` 数据类，属性和字典两种访问方式等价；`loss` 传 labels 才有（用前判空），`logits` 永远有，`past_key_values` 需 `use_cache=True` |
| labels 触发 loss | 内部走 logits → shift 对齐 → 展平 → CrossEntropyLoss(ignore_index=-100) → 有效位置求平均 → 标量 |
| shift 机制 | 位置 n 的 logits 对齐位置 n+1 的 label；末位输出和首位 label 被丢弃；HF 内部自动做，不用手动 shift；与因果掩码（attention 机制）是两件事（见 [[架构-编码器与解码器]]） |
| -100 用法 | SFT 时 prompt 段和 padding 位填 -100，answer 段填真 id；机制见 [[损失函数与-100掩码]] |
| logits 视角 | seq 个并列的词表分类任务，每位置一个 vocab 维打分；推理取 `logits[:, -1, :]`（见 [[generate-生成机制]]） |
| `model(**inputs)` | 字典解包，tokenizer 字段名对齐 forward 参数 |
| `model.train()` / `model.eval()` | 切换训练 / 推理模式（来自 [[nn-module]]） |
