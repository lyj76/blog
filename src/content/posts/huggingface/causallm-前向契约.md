---
title: HuggingFace CausalLM 前向契约：model() 返回什么
published: 2026-08-05
description: forward 参数契约（input_ids / attention_mask / labels）、CausalLMOutputWithPast 字段表（loss / logits / past_key_values / hidden_states / attentions）、labels 传入才返回 loss、shift 一位逻辑、-100 掩码与 ignore_index、logits 形状、train/eval 模式、inputs 字典解包
tags: [HuggingFace, transformers, CausalLM, 前向传播, loss]
category: HuggingFace
---

# HuggingFace CausalLM 前向契约：model() 返回什么

调用 `model(input_ids=..., labels=...)` 后，返回的不是张量，而是一个带多个字段的对象。理解这个对象的契约，才能知道「loss 从哪来、logits 长什么样、什么时候有 past_key_values」。

## 1. CausalLMOutputWithPast：返回类型

模型直接调用（`__call__` → `forward`）返回 `CausalLMOutputWithPast` 对象：

```python title="forward-output.py"
outputs = model(input_ids=input_ids, labels=labels)
type(outputs)   # transformers.modeling_outputs.CausalLMOutputWithPast

outputs.loss              # tensor(2.3456)，交叉熵损失
outputs.logits            # tensor[batch, seq, vocab]，每个位置的词表分布
outputs.past_key_values   # tuple of tuples，KV 缓存（推理加速用）
```

这个对象是数据类，字段表如下：

| 字段 | 类型 | 何时有值 |
| --- | --- | --- |
| `loss` | `Optional[torch.Tensor]` | **传了 `labels` 才有**，否则为 `None` |
| `logits` | `torch.Tensor` | 永远有，形状 `[batch, seq, vocab_size]` |
| `past_key_values` | `Optional[Tuple]` | `use_cache=True` 时有，KV 缓存 |
| `hidden_states` | `Optional[Tuple[Tensor]]` | `output_hidden_states=True` 时有 |
| `attentions` | `Optional[Tuple[Tensor]]` | `output_attentions=True` 时有 |

- `loss` 和 `logits` 是最常用的两个字段
- `past_key_values`、`hidden_states`、`attentions` 默认不返回，需要显式开启
- 这个对象可以当命名元组用：`outputs.logits` 或 `outputs["logits"]` 都行

::::note
**不是所有字段都有值**：`loss` 只在传 `labels` 时计算；`past_key_values` 只在 `use_cache=True` 时返回。不传 labels 时 `outputs.loss` 是 `None`，但 `outputs.logits` 永远有。
::::

## 2. forward 的参数契约：input_ids / attention_mask / labels

`forward` 接收的关键参数就这几个（所有 HF CausalLM 模型遵循同一签名）：

| 参数 | 含义 | 形状 | 训练时传？ |
| --- | --- | --- | --- |
| `input_ids` | 输入 token 序列 | `[batch, seq]` | 必传 |
| `attention_mask` | 标记真实 / 填充位（1=真实，0=填充） | `[batch, seq]` | 通常传 |
| `position_ids` | 每个位置的下标编号 | `[batch, seq]` | 少用（默认自动生成） |
| `past_key_values` | 历史 KV 缓存（生成加速用） | 元组 | 生成时用 |
| `labels` | 目标 token 序列（标准答案） | `[batch, seq]` | **训练时传** |

- **为什么必须用关键字参数调用**：`forward` 参数多、顺序固定——`labels` 排在第六位，第二位是 `attention_mask`。写 `model(input_ids, labels)` 位置调用会把 labels 错塞给 `attention_mask`，**不报错但行为完全错误**。所以 HF 生态永远写 `model(input_ids=..., labels=...)`（参数语法见 Python「函数参数与解包」篇）
- **两种调用形态**：不传 `labels` 是推理形态（只算 logits）；传 `labels` 是训练形态（内部自动算交叉熵，返回带 `loss` 的对象）——详见下一节

## 3. labels 与 loss：内部自动算交叉熵

```python title="labels-loss.py"
# 传 labels → 模型自动算 loss
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss   # tensor(2.3456)，可以直接 loss.backward()

# 不传 labels → 没有 loss，只有 logits
outputs = model(input_ids=input_ids)
outputs.loss    # None
outputs.logits  # tensor[batch, seq, vocab]
```

- 传 `labels` 后，模型内部自动调用交叉熵损失函数，返回值带 `.loss`
- `labels` 是和 `input_ids` **同形状**的张量（`[batch, seq]`），内容是目标 token id
- 拿到 `loss` 后直接 `loss.backward()` 就能反向传播，不需要自己写损失函数

::::tip
**训练循环的标准写法**：`loss = model(input_ids=..., labels=...).loss`，一行搞定前向 + 算损失。然后 `loss.backward()` + `optimizer.step()`。
::::

## 4. shift 一位逻辑：位置 n 预测 n+1

CausalLM 的训练目标是「用前 n 个 token 预测第 n+1 个」。模型内部对 logits 和 labels 做了 shift：

```python title="shift-logic.py"
# 输入序列：  [A, B, C, D]   （input_ids）
# 目标序列：  [A, B, C, D]   （labels，和 input_ids 一样）

# 模型内部做 shift：
# logits 用 [:-1]：位置 0,1,2 的输出 → 预测 B,C,D
# labels  用 [1:]：  位置 1,2,3 的 id   → 作为 B,C,D 的目标

# 等价于：
# 位置 0 的 logits 预测位置 1 的 label
# 位置 1 的 logits 预测位置 2 的 label
# 位置 n 的 logits 预测位置 n+1 的 label
```

- 你传进去的 `labels` 和 `input_ids` 通常是**同一个序列**（自监督训练）
- 模型内部自动 shift：logits 取 `[..., :-1, :]`，labels 取 `[..., 1:]`
- 所以位置 n 的 hidden state 预测位置 n+1 的 token，这就是「因果」的含义

::::note
**你不用手动 shift**：HuggingFace 的 CausalLM 在 `forward` 内部自动做 shift。你只要把 `input_ids` 和 `labels` 传成同一个序列（或 labels 是目标序列），模型自己处理对齐。
::::

## 5. -100 掩码：ignore_index 约定

```python title="ignore-index.py"
# labels 中 -100 的位置不计入损失
labels = torch.tensor([
    [-100, -100, 1234, 5678, 2],   # prompt 部分填 -100，答案部分是真 id
])
```

- PyTorch 的 `CrossEntropyLoss` 有一个 `ignore_index` 参数，默认值是 `-100`
- HuggingFace 沿用这个约定：`labels` 中值为 `-100` 的位置**不计入损失**
- 用途：SFT（监督微调）时，prompt 部分填 `-100`，只有答案部分是真 token id，模型只学答案不学 prompt

::::important
**-100 不是魔法数字**：它是 PyTorch `CrossEntropyLoss` 的 `ignore_index` 默认值。HuggingFace 没有改这个默认值，所以 `-100` 就是「忽略此位置」的约定。如果你把 `ignore_index` 改成别的数，那就要填那个数。
::::

## 6. logits 形状：[batch, seq, vocab]

```python title="logits-shape.py"
outputs = model(input_ids=input_ids)
logits = outputs.logits
print(logits.shape)   # torch.Size([2, 128, 151936])
#                        batch=2, seq_len=128, vocab_size=151936
```

- `logits` 是模型对每个位置、每个词的打分（未归一化）
- 形状 `[batch, seq_len, vocab_size]`：三个维度分别是批次、序列长度、词表大小
- 取下一个 token 预测：`next_token_logits = logits[:, -1, :]`，形状 `[batch, vocab_size]`
- 转概率：`probs = torch.softmax(next_token_logits, dim=-1)`

## 7. train() / eval()：模式切换

```python title="train-eval.py"
model.train()   # 训练模式：Dropout 生效、BatchNorm 用 batch 统计量
model.eval()    # 推理模式：Dropout 关闭、BatchNorm 用历史统计量
```

- `train()` / `eval()` 是 `nn.Module` 的方法，切换模型的训练 / 推理模式
- 主要影响 Dropout 和 BatchNorm：训练时 Dropout 随机丢弃，推理时关闭
- **训练循环开头调 `model.train()`，推理前调 `model.eval()`**

::::note
**这是 PyTorch 的机制，不是 HuggingFace 的**：`model.train()` / `model.eval()` 来自 `nn.Module`，所有 PyTorch 模型都有。HuggingFace 的模型继承自 `nn.Module`，所以也有这两个方法。
::::

## 8. inputs 字典解包

```python title="kwargs-unpack.py"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
# inputs 是 BatchEncoding：{"input_ids": ..., "attention_mask": ...}

outputs = model(**inputs)
# 等价于 model(input_ids=..., attention_mask=...)
```

- `**inputs` 是 Python 的字典解包：把字典的 key-value 展开成关键字参数
- tokenizer 返回的 `BatchEncoding` 里有 `input_ids` 和 `attention_mask`，正好对应 `forward` 的参数名
- 所以 `model(**inputs)` 等价于 `model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])`

::::tip
**为什么能直接解包**：tokenizer 的返回 key 名和 model `forward` 的参数名是对齐的。这是 HuggingFace 的设计契约：tokenizer 产出的字段名 = model 接收的参数名。
::::

## 小结

| 字段 / 语法 | 类型 / 作用 |
| --- | --- |
| `forward` 参数 | `input_ids` / `attention_mask` / `labels` / `position_ids` / `past_key_values`，必须关键字调用 |
| `outputs.loss` | `Optional[Tensor]`，传 labels 才有 |
| `outputs.logits` | `Tensor[batch, seq, vocab]`，永远有 |
| `outputs.past_key_values` | `Optional[Tuple]`，`use_cache=True` 时有 |
| `model(input_ids=..., labels=...)` | 前向 + 自动算交叉熵 |
| `labels` 中 `-100` | 忽略此位置（PyTorch `ignore_index` 约定） |
| shift 逻辑 | logits `[:-1]` 预测 labels `[1:]`，位置 n 预测 n+1 |
| `model(**inputs)` | 字典解包，tokenizer 字段名对齐 forward 参数 |
| `model.train()` / `model.eval()` | 切换训练 / 推理模式 |
