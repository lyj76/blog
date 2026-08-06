---
title: HuggingFace Tokenizer 类型契约：输入输出结构全解
published: 2026-08-05
description: BatchEncoding 结构与字典继承、encode vs 直接调用 vs return_tensors 返回类型对比、padding/truncation/max_length 参数体系、特殊 token 与 padding_side、attention_mask 双重角色
tags: [HuggingFace, transformers, Tokenizer, BatchEncoding]
category: HuggingFace
---

# HuggingFace Tokenizer 类型契约：输入输出结构全解

Tokenizer 负责「文本 ↔ token id」互转。理解它「怎么加载、每种调用返回什么类型、有哪些字段、内部怎么处理」，是看懂任何 transformers 代码的第一步。

> 归属：**`transformers` 库**（`from transformers import AutoTokenizer`）——HF 的模型库，和 `datasets`（数据）、PyTorch（训练引擎）平级。

## 1. 加载：AutoTokenizer.from_pretrained()

训练脚本里拿到 tokenizer 的唯一入口——**和模型共用同一个路径**：

```python title="tokenizer-load.py"
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)   # 本地路径或 Hub id
```

- `MODEL_PATH` 可以是**本地文件夹**（如 `"./Qwen2.5-0.5B-Instruct"`），也可以是 **HF Hub 仓库 id**（如 `"Qwen/Qwen2.5-0.5B-Instruct"`）——和 `AutoModelForCausalLM.from_pretrained` 用同一个值，两者**成对出现**
- **它读的是词表文件，不是权重**：`tokenizer_config.json`（配置）+ `vocab.json` / `merges.txt` / `tokenizer.json`（词表与合并规则）
- 所以加载**又轻又快**（几 MB，没有权重）——先加载 tokenizer 检查数据、再加载模型，是排查问题的标准姿势

**加载后立刻做的三件事**：

```python title="post-load.py"
tokenizer.pad_token_id is None          # ① 检查有没有 pad_token（很多 LLM 没有）
if tokenizer.pad_token_id is None:      # ② 没有就用 eos 兜底（见第 5 节）
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(save_dir)     # ③ 需要时保存（和模型成对，见下）
```

**保存是加载的逆操作**：`tokenizer.save_pretrained(dir)` 把词表文件写进文件夹，和 `model.save_pretrained(dir)` **成对使用**——**模型文件里不包含分词器**，部署/迁移时要单独带上。

::::note
**Auto 机制细节见「Auto 机制与模型加载」篇**：`AutoTokenizer` 按仓库里的配置自动选对具体实现类（如 Qwen 的 `Qwen2Tokenizer`）——你不需要知道用哪个类，只传路径即可。
::::

## 2. BatchEncoding：tokenizer 调用的返回类型

直接调用 tokenizer 时，返回值不是普通字典，而是 `BatchEncoding` 对象：

```python title="batch-encoding.py"
out = tokenizer("hello world")
type(out)               # transformers.tokenization_utils_base.BatchEncoding
out.keys()              # dict_keys(['input_ids', 'attention_mask'])
out["input_ids"]        # list[int]，如 [1, 2345, 678, 2]
out.input_ids           # 等价写法，属性访问也行
out.to("cuda")          # 当 value 是张量时，一次性搬到 GPU
```

- `BatchEncoding` 继承自 `UserDict`（字典子类），所以 `out["input_ids"]` 和 `out.input_ids` 两种写法都能访问
- 常见 key 有三个：`input_ids`（token id 序列）、`attention_mask`（标记真实 token 位置）、`token_type_ids`（句子编号，仅 BERT 系有，LLM 通常没有）
- 额外方法 `.to(device)`：当 value 是张量时，把所有张量一次性搬到目标设备

::::note
**`BatchEncoding` 不是普通 `dict`**：它继承 `UserDict`，行为像字典，但多了 `.to(device)`、`.convert_to_tensors()` 等方法。当字典用没问题，但要知道它有额外能力。
::::

## 3. 三种调用的返回类型对比

Tokenizer 有三种常见调用方式，返回类型完全不同：

```python title="three-calls.py"
# 写法一：encode → 纯 list[int]
ids = tokenizer.encode("hello world")
# ids = [1, 2345, 678, 2]，类型是 list[int]

# 写法二：直接调用 → BatchEncoding（value 是 list）
out = tokenizer("hello world")
out["input_ids"]        # list[int]，和 encode 结果一样
out["attention_mask"]   # list[int]，encode 不返回这个

# 写法三：直接调用 + return_tensors="pt" → BatchEncoding（value 是 tensor）
inputs = tokenizer("hello world", return_tensors="pt")
inputs["input_ids"]        # tensor([[1, 2345, 678, 2]])，形状 [1, 4]
inputs["attention_mask"]   # tensor([[1, 1, 1, 1]])
```

| 调用方式 | 返回类型 | input_ids 形状 | 有 attention_mask |
| --- | --- | --- | --- |
| `tokenizer.encode(text)` | `list[int]` | `[n]`（一维列表） | 否 |
| `tokenizer(text)` | `BatchEncoding` | `list[int]`，长度 `n` | 是 |
| `tokenizer(text, return_tensors="pt")` | `BatchEncoding` | `tensor[1, n]`（二维） | 是 |

- 一句话关系：`tokenizer.encode(text)` ≈ `tokenizer(text)["input_ids"]`，两者走同一套处理流程，区别只在返回什么
- **形状变化要注意**：列表是一维 `[n]`，转成张量后自动加 batch 维变成 `[1, n]`，模型按批次处理数据，第 0 维是 batch
- `return_tensors` 取值：`"pt"`（PyTorch）、`"tf"`（TensorFlow）、`"np"`（NumPy）、`None`（默认，返回列表）

::::tip
**什么时候用哪个**：训练数据构造用 `encode`，你要的是裸 id，方便自己拼接 prompt + target、做 `-100` 掩码；推理或喂模型用直接调用 + `return_tensors="pt"`，模型需要张量输入和 `attention_mask`。
::::

## 4. 参数体系：padding / truncation / max_length

直接调用 tokenizer 时，常用参数控制输出结构：

```python title="params.py"
inputs = tokenizer(
    ["短句", "这是一个稍微长一点的句子"],
    padding=True,              # 批内补齐到等长
    truncation=True,           # 超长就截断
    max_length=128,            # 截断/填充的目标长度
    add_special_tokens=True,   # 自动加 bos/eos（默认 True）
    return_tensors="pt",       # 返回张量
)
inputs["input_ids"]        # tensor([[1, 234, 5, 2, pad, pad],
                           #          [1, 23, 45, 67, 89, 2]])
inputs["attention_mask"]   # tensor([[1, 1, 1, 1, 0, 0],
                           #          [1, 1, 1, 1, 1, 1]])
```

| 参数 | 类型 | 作用 |
| --- | --- | --- |
| `padding` | `bool` / `"max_length"` / `"longest"` | 是否补齐；`True` = 补到批内最长 |
| `truncation` | `bool` | 超长是否截断 |
| `max_length` | `int` | 截断和填充的长度上限 |
| `add_special_tokens` | `bool` | 是否自动加 bos/eos，默认 `True` |
| `return_tensors` | `str` / `None` | 返回张量类型，`"pt"` / `"tf"` / `"np"` |

- `padding=True`：批内多条不等长时，短的补 `pad_token_id`，同时 `attention_mask` 对应位置标 `0`
- `add_special_tokens=False`：拼接 prompt + target 时手动控制，避免重复加特殊 token
- `max_length` 和 `truncation` 配合：模型有最大序列长度（如 2048 / 4096），超了必须截断

## 5. 特殊 token 体系

每个 tokenizer 持有四个特殊 token 的 id：

```python title="special-tokens.py"
tokenizer.bos_token_id    # 序列开始符
tokenizer.eos_token_id    # 序列结束符
tokenizer.pad_token_id    # 填充符
tokenizer.unk_token_id    # 未知词符
```

| 属性 | 含义 | 常见情况 |
| --- | --- | --- |
| `bos_token_id` | begin of sequence | GPT 系通常有 |
| `eos_token_id` | end of sequence | 几乎都有 |
| `pad_token_id` | padding 用 | **很多 LLM 没定义，值为 `None`** |
| `unk_token_id` | unknown token | BERT 系有，LLM 通常用 byte-level 无 unk |

::::warning
**很多 decoder-only 模型没有 `pad_token`**：Llama、Qwen 等模型出厂时 `pad_token_id` 为 `None`。训练需要 padding 成等长批次，必须手动补一个：

```python title="pad-fallback.py"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    # 或 tokenizer.pad_token_id = tokenizer.eos_token_id
```

把结束符当填充符用，是 LLM 训练脚本的标配写法。
::::

## 6. padding_side：填充方向与生成的意义

```python title="padding-side.py"
tokenizer.padding_side = "left"   # 左填充：pad 加在序列开头
tokenizer.padding_side = "right"  # 右填充：pad 加在序列末尾（默认）
```

- **右填充**（默认）：真实 token 在前，pad 在后。适合训练（attention_mask 标掉 pad 位即可）
- **左填充**：pad 在前，真实 token 在后。**生成时必须用左填充**

::::important
**为什么生成要左填充**：decoder-only 模型从序列末尾往后生成新 token。如果右填充，真实文本后面跟着 pad，模型会在 pad 后面生成，位置错乱。左填充保证真实 token 紧贴序列末尾，新生成的 token 自然接在后面。
::::

## 7. attention_mask 的两个作用

`attention_mask` 是一个全 0/1 的序列，长度和 `input_ids` 相同：

```python title="attention-mask.py"
input_ids       = [1, 234, 5, 2, pad, pad]
attention_mask  = [1,   1, 1, 1,   0,   0]
# 1 = 真实 token，参与注意力计算
# 0 = 填充位，注意力计算时忽略
```

两个作用：

1. **填充掩码**：标记 padding 位置，模型算注意力时跳过这些位置（不 attend 到 pad 上）
2. **配合因果掩码**：在 CausalLM 中，模型内部还有一层因果掩码（下三角，防止看未来 token）。attention_mask 和因果掩码组合，共同决定「哪些位置能被哪些位置看到」

::::note
**attention_mask 不是因果掩码**：attention_mask 由 tokenizer 生成，标记填充位；因果掩码由模型内部生成，防止看未来。两者是独立的机制，在前向传播中组合使用。详见「架构地图」篇。
::::

## 小结

| 语法 | 返回类型 | 用途 |
| --- | --- | --- |
| `AutoTokenizer.from_pretrained(path)` | `Tokenizer` 实例 | 加载（读词表文件，非权重） |
| `tokenizer.save_pretrained(dir)` | 写词表文件 | 保存（和模型成对） |
| `tokenizer.encode(text)` | `list[int]` | 拿裸 id 做数据构造 |
| `tokenizer(text)` | `BatchEncoding`（list 值） | 完整处理结果 |
| `tokenizer(text, return_tensors="pt")` | `BatchEncoding`（tensor 值） | 喂模型 |
| `padding=True` | 补齐 + attention_mask | 批次等长 |
| `tokenizer.pad_token_id` | `int` 或 `None` | 填充符（None 要兜底） |
| `tokenizer.padding_side = "left"` | 设置属性 | 生成时左填充 |
| `out.to("cuda")` | `BatchEncoding` | 张量版搬到 GPU |
