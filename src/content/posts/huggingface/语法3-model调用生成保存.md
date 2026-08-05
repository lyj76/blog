---
title: HuggingFace 语法③：Model 调用、生成与保存
published: 2026-08-04
description: model(input_ids, labels).loss 前向调用、model.train()、**kwargs 解包、model.generate()、save_pretrained、@torch.no_grad() 装饰器
tags: [HuggingFace, transformers, 语法]
category: HuggingFace
---

# HuggingFace 语法③：Model 调用、生成与保存

模型对象的使用语法：前向计算、训练/推理模式、文本生成、权重保存。

## 1. 前向调用：`model(input_ids=..., labels=...)`

```python title="forward-example.py"
loss = model(input_ids=input_ids, labels=labels).loss
```

- 模型对象**直接调用**就会执行前向传播（`__call__` → `forward`）
- 传入关键字参数：`input_ids` 是输入 token 张量，`labels` 是目标 token 张量
- **`labels` 传了之后**，模型内部自动算交叉熵损失，返回值（对象）带 `.loss` 属性
- 重点：`labels` 里值为 `-100` 的位置**不计入损失**（这就是 SFT 只让模型学答案、不学 prompt 的原理）

::::note
**`-100` 掩码约定**：PyTorch 的交叉熵把 `-100` 视为「忽略此位置」。HuggingFace 的 `labels` 沿用这个约定——prompt 部分填 `-100`，只有答案部分是真 token。
::::

## 2. 训练/推理模式：`model.train()` / `model.eval()`

```python title="mode-example.py"
model.train()   # 训练模式：Dropout 生效、BatchNorm 用 batch 统计量
model.eval()    # 推理模式：Dropout 关闭、BatchNorm 用历史统计量
```

## 3. 文本生成：`model.generate()`

```python title="generate-example.py"
outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, eos_token_id=eos_id)
```

- `**inputs` 是**字典解包**：把 `inputs = {"input_ids": ..., "attention_mask": ...}` 的键值展开成关键字参数，等价于 `model.generate(input_ids=..., attention_mask=...)`
- `max_new_tokens=30`：最多新生成 30 个 token
- `eos_token_id=eos_id`：遇到结束符就停

**剥离 prompt，只取新生成的 token**：

```python title="strip-prompt.py"
prompt_len = inputs.input_ids.shape[1]      # prompt 的 token 数量（形状第 1 维）
generated_tokens = outputs[0][prompt_len:]  # 张量切片：跳过前 prompt_len 个
```

- `outputs` 是张量，`outputs[0]` 取第 1 条（batch 第 0 个样本），`[prompt_len:]` 从第 prompt_len 个元素切到末尾
- `shape[1]` 取形状第 2 个数字（列数），因为 `inputs.input_ids` 形状是 `[batch, seq_len]`

## 4. 保存模型：`model.save_pretrained()`

```python title="save-model.py"
model.save_pretrained(merged_dir)   # 保存权重到文件夹
```

把模型权重（`pytorch_model.bin` / `model.safetensors`）和配置写进目录，之后 `from_pretrained` 可以直接读回来。

## 5. 关闭梯度：`@torch.no_grad()` 装饰器

```python title="no-grad-example.py"
@torch.no_grad()
def generate_title(model, tokenizer, article, device, max_new_tokens=30):
    ...
```

- `@装饰器` 是 Python 语法：给函数「加一层包装」
- `torch.no_grad()` 包裹后，函数内所有张量**不记录计算图**——推理时省显存、省内存、速度快
- 训练（要 backward）**不能用**，推理（只前向）**必须用**

## 小结

| 语法 | 作用 |
| --- | --- |
| `model(input_ids=..., labels=...).loss` | 前向 + 自动算损失 |
| `model.train()` / `model.eval()` | 切换训练/推理模式 |
| `model.generate(**inputs, max_new_tokens=..., eos_token_id=...)` | 自回归生成 |
| `outputs[0][prompt_len:]` | 剥离 prompt 取新生成部分 |
| `model.save_pretrained(dir)` | 保存权重 |
| `@torch.no_grad()` | 推理时关闭梯度追踪 |
