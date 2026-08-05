---
title: HuggingFace 语法⑥：SFT 数据构造与损失掩码
published: 2026-08-04
description: prompt 模板、tokenizer.encode、-100 损失掩码、截断与 padding —— 监督微调数据怎么从文本变成训练张量
tags: [HuggingFace, SFT, 数据构造, 损失掩码]
category: HuggingFace
---

# HuggingFace 语法⑥：SFT 数据构造与损失掩码

SFT（监督微调）数据构造的核心：**让模型只学答案、不学 prompt**。实现手段是 `-100` 损失掩码。

## 1. 完整流程：文本 → 训练张量

```python title="sft-data-overview.py"
# 1. 拼 prompt 模板（对话格式）
prompt = build_prompt(article)                    # str

# 2. 文本 → id 列表
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)          # list[int]
target_ids = tokenizer.encode(f"{summary}<|im_end|>", add_special_tokens=False)

# 3. 拼接
input_ids = prompt_ids + target_ids               # list + list = 拼成一个大 list

# 4. 构造 labels（prompt 部分填 -100）
labels = [-100] * len(prompt_ids) + target_ids

# 5. 转张量
torch.tensor(input_ids, dtype=torch.long)
```

## 2. prompt 模板（f-string 拼对话）

```python title="prompt-template.py"
def build_prompt(article):
    return (
        f"<|im_start|>system\n你是一个优秀的新闻编辑，请为以下正文拟定一个简洁精炼的标题。<|im_end|>\n"
        f"<|im_start|>user\n文章正文：{article}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
```

- 用 f-string 拼**对话格式**：system（角色设定）→ user（输入）→ assistant（待生成）
- `<|im_start|>` / `<|im_end|>` 是 Qwen 的对话标记 token
- 最后停在 `<|im_start|>assistant\n`，模型从这里开始生成

::::note
**为什么要模板**：基座模型只会续写文本，模板把「任务描述 + 输入」组织成它能理解的形式。**模板必须和训练/推理时完全一致**，否则效果崩。
::::

## 3. 损失掩码：`-100` 的妙用

```python title="loss-mask-example.py"
labels = [-100] * len(prompt_ids) + target_ids
# [-100, -100, ..., -100,    <-- prompt 部分：不计算损失
#  target_token_1, ...]       <-- 答案部分：计算损失
```

- **列表乘法**：`[-100] * n` 生成 n 个 `-100` 的列表
- **`-100` 约定**：PyTorch 交叉熵把 `-100` 视为「忽略此位置」，不参与损失计算
- 效果：模型**只从答案部分学**，prompt 部分不产生损失

::::important
**为什么要掩码**：不掩码的话，模型会同时学「预测 prompt 的下一个词」——但 prompt 是给定的输入，不该被学习。掩码后 loss 只衡量「生成答案」的对错，这才是微调的目标。
::::

## 4. 截断与 padding：等长化

```python title="pad-truncate.py"
if len(input_ids) > self.max_len:
    input_ids = input_ids[:self.max_len]      # 超长截断
    labels = labels[:self.max_len]
else:
    pad_len = self.max_len - len(input_ids)
    input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len   # 用 pad_token 补
    labels = labels + [-100] * pad_len                                # 补的位置也不算损失
```

- **为什么必须等长**：DataLoader 要把一批样本堆叠成 `[batch, seq]`，长度不一无法堆叠
- **input 用 `pad_token_id` 补**、**labels 用 `-100` 补**（补的位置也忽略损失）
- 截断用 `[:max_len]`，padding 用列表加法 + 列表乘法

## 5. 返回给 DataLoader

```python title="return-example.py"
return {
    "input_ids": torch.tensor(input_ids, dtype=torch.long),
    "labels": torch.tensor(labels, dtype=torch.long),
}
```

- 返回**字典**，DataLoader 堆叠后 `batch["input_ids"]` / `batch["labels"]`
- `torch.tensor(list, dtype=torch.long)`：list → 张量，`torch.long` = 整数类型（token id 必须是整数）

## 小结

| 语法 | 作用 |
| --- | --- |
| `f"..."` 拼 prompt 模板 | 组织成对话格式 |
| `encode(text, add_special_tokens=False)` | 文本 → id 列表 |
| `[-100] * len(prompt_ids)` | 列表乘法生成掩码 |
| `input_ids + target_ids` | 列表拼接 |
| `[:max_len]` 截断 / `[pad_token_id] * n` 补齐 | 等长化 |
| `torch.tensor(x, dtype=torch.long)` | list → 整数张量 |
