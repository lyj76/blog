---
title: HuggingFace 语法②：Tokenizer 用法全解
published: 2026-08-04
description: tokenizer.encode / 直接调用 / pad_token_id / convert_tokens_to_ids / decode —— 文本与 token id 互转的所有写法
tags: [HuggingFace, transformers, 语法]
category: HuggingFace
---

# HuggingFace 语法②：Tokenizer 用法全解

Tokenizer 负责「文本 ↔ token id」互转，是训练与推理的必经之路。以下全部语法都来自 LoRA 训练代码。

## 1. 文本 → id 列表：`tokenizer.encode()`

```python title="encode-example.py"
prompt_ids = tokenizer.encode(build_prompt(article), add_special_tokens=False)
# 返回一个 list[int]：如 [151644, 8948, ...]
```

- `encode(文本)` 把字符串转成 **token id 列表**
- `add_special_tokens=False`：**不自动加** 开始符/结束符（拼接 prompt+target 时手动控制，避免重复加）

## 2. 文本 → 张量：直接调用 tokenizer

```python title="call-example.py"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
```

- **对象可以直接当函数调用**（Python 的 `__call__`）：`tokenizer(文本, 参数)` 等价于更丰富的 `encode`
- `return_tensors="pt"`：返回 PyTorch 张量（`"pt"` = PyTorch，`"tf"` = TensorFlow）
- 返回值是个**类似字典的对象**，用 `.input_ids` 或 `["input_ids"]` 取张量
- 拿到后立刻 `.to(device)` 搬到 GPU

::::note
**`encode` vs 直接调用**：`encode(文本)` 返回纯 list（轻量）；`tokenizer(文本, return_tensors="pt")` 返回带 `input_ids`/`attention_mask` 的张量字典（用于模型输入）。训练数据构造用前者，推理喂模型用后者。
::::

## 3. 特殊 token id：`pad_token_id` / `eos_token_id`

```python title="special-ids.py"
tokenizer.pad_token_id    # 填充符 id（可能为 None）
tokenizer.eos_token_id    # 结束符 id
```

| 属性 | 含义 |
| --- | --- |
| `pad_token_id` | 填充（padding）时用的 id |
| `eos_token_id` | 结束（end of sequence）时用的 id |
| `bos_token_id` | 开始（begin of sequence）时用的 id |

## 4. 任意 token 字符串 → id：`convert_tokens_to_ids()`

```python title="convert-example.py"
eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
# 把任意 token 字符串转成 id，比如对话模板的结束标记
```

生成时告诉模型「遇到 `<|im_end|>` 就停」，需要先拿到它的 id。

## 5. id → 文本：`tokenizer.decode()`

```python title="decode-example.py"
result_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
result_text.strip()
```

- `decode(id列表)` 把 id 还原成文本
- `skip_special_tokens=True`：**去掉** `<|im_end|>` 这类特殊 token，只留正文
- 配合 `.strip()` 去掉首尾空白

## 6. 保存：`tokenizer.save_pretrained()`

```python title="save-tokenizer.py"
tokenizer.save_pretrained(merged_dir)
```

把分词器的配置文件写进文件夹，和 `model.save_pretrained()` 成对使用——模型文件里没有分词器，部署/再次加载需要单独存。

## 小结

| 语法 | 作用 |
| --- | --- |
| `tokenizer.encode(文本, add_special_tokens=False)` | 文本 → `list[int]` |
| `tokenizer(文本, return_tensors="pt")` | 文本 → 张量字典（喂模型） |
| `tokenizer.pad_token_id` / `.eos_token_id` | 特殊 token id |
| `tokenizer.convert_tokens_to_ids("<|im_end|>")` | 任意 token → id |
| `tokenizer.decode(ids, skip_special_tokens=True)` | id → 文本 |
| `tokenizer.save_pretrained(dir)` | 保存分词器 |
