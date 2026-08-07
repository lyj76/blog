---
title: Tokenizer 类型契约：输入输出结构全解
published: 2026-08-05
description: BatchEncoding 结构与字典继承、encode vs 直接调用 vs return_tensors 返回类型对比、padding/truncation/max_length 参数体系、特殊 token 与 padding_side、attention_mask 双重角色
tags: [HuggingFace, transformers, Tokenizer, BatchEncoding]
category: HuggingFace
---

# Tokenizer 类型契约：输入输出结构全解

Tokenizer 负责「文本 ↔ token id」互转。这篇按**使用顺序**展开：先怎么拿到它（加载）→ 调用它返回什么（BatchEncoding）→ 三种调用方式 → **互转的方法**（encode / convert_tokens_to_ids / decode）→ 参数与特殊 token。

> 归属：**`transformers` 库**（`from transformers import AutoTokenizer`）——HF 的模型库，和 `datasets`（数据）、PyTorch（训练引擎）平级。

## 1. 加载：AutoTokenizer.from_pretrained()

一切从「拿到 tokenizer」开始——它是训练脚本里创建 tokenizer 的唯一入口，**和模型共用同一个路径**：

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
if tokenizer.pad_token_id is None:      # ② 没有就用 eos 兜底（见第 6 节）
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(save_dir)     # ③ 需要时保存（和模型成对，见下）
```

**保存是加载的逆操作**：`tokenizer.save_pretrained(dir)` 把词表文件写进文件夹，和 `model.save_pretrained(dir)` **成对使用**——**模型文件里不包含分词器**，部署/迁移时要单独带上。

::::note
**Auto 机制细节见 [[auto-机制与模型加载]]**：`AutoTokenizer` 按仓库里的配置自动选对具体实现类（如 Qwen 的 `Qwen2Tokenizer`）——你不需要知道用哪个类，只传路径即可。
::::

## 2. BatchEncoding：tokenizer 调用返回什么

### 2.1 一句话模型

`tokenizer(text)` 返回**一个对象** `BatchEncoding`——概念上就是一个**字典**：键是"处理结果的名称"，值是"对应的 token id 序列"。

```python title="batch-encoding.py"
out = tokenizer("hello world")
# 概念上 out 长这样：
# {
#     "input_ids":      [1, 2345, 678, 2],   # 文本 → token id 列表
#     "attention_mask": [1, 1, 1, 1],        # 1 = 真实 token；单条文本不填充所以全 1
# }
```

**什么时候 `attention_mask` 会出现 `0`**：只有**多条文本一起、传了 `padding=True`** 时——短的序列补了 pad 位，对应位置标 `0`（模型忽略它）。单条文本永远全 `1`。完整机制见第 5 节。

### 2.2 两种访问方式（同一个东西）

```python title="access-ways.py"
out["input_ids"]        # 字典式访问（本质）
out.input_ids           # 属性式访问（便捷写法，等价）
```

### 2.3 为什么能当属性访问：UserDict 子类

`BatchEncoding` 继承自 `collections.UserDict`（一种"行为像字典"的类），所以 `out["input_ids"]` 和 `out.input_ids` 都行。它还额外提供字典没有的方法：`.to("cuda")`——当值都是张量时，一键把整个字典搬到 GPU。

::::note
**`BatchEncoding` 不是普通 `dict`**：当字典用没问题，但它多了 `.to(device)`、`.convert_to_tensors()` 等能力。不必记全，知道"字典 + 一点点增强"即可。
::::

### 2.4 常见的 key

| key | 值是什么 | 什么时候有 |
| --- | --- | --- |
| `input_ids` | token id 序列 | **总有** |
| `attention_mask` | 0/1：1=真实 token、0=填充位 | 总有（无 padding 时全 1；作用见 [[tokenizer-类型契约#8-attention_mask-的两个作用]]） |
| `token_type_ids` | 句子编号 | 仅 BERT 系，LLM 通常没有 |

## 3. 三种调用方式：差别只有两点

三种写法走**同一套处理流程**（分词 → 转 id），返回值的差别只有两点：

1. **包什么壳**：`encode` 给裸列表；直接调用给字典（BatchEncoding）
2. **值是什么类型**：`list[int]` 还是 `tensor`（由 `return_tensors` 决定）

### 3.1 同一个文本的三种真实返回值

```python title="three-calls.py"
# 写法一：encode → 裸列表（就是 input_ids 那一个列表）
ids = tokenizer.encode("hello world")
# ids = [1, 2345, 678, 2]     ← 注意两头：1 = bos、2 = eos，是默认 add_special_tokens=True 加的

# 写法二：直接调用 → 字典，值是列表
out = tokenizer("hello world")
# out = {"input_ids": [1, 2345, 678, 2], "attention_mask": [1, 1, 1, 1]}

# 写法三：直接调用 + return_tensors="pt" → 字典，值是张量
out = tokenizer("hello world", return_tensors="pt")
# out = {"input_ids": tensor([[1, 2345, 678, 2]]), "attention_mask": tensor([[1, 1, 1, 1]])}
```

**看到 `[1, ..., 2]` 的两头了吗**：`1`（bos）和 `2`（eos）不是 "hello world" 的一部分——是 `add_special_tokens=True`（**默认开启**）自动加的包裹。不想要就显式关掉：

```python title="no-special-tokens.py"
ids = tokenizer.encode("hello world", add_special_tokens=False)
# ids = [2345, 678]          ← 只剩正文两个 token，没有 bos/eos
```

**什么时候必须关**：自己拼接 prompt + target 做 SFT 时——否则每个片段都被包一层 bos/eos，拼起来就乱了（详见第 6 节）。

### 3.2 encode 和直接调用是什么关系

`tokenizer.encode(text)` ≈ `tokenizer(text)["input_ids"]`——**encode 就是"只取 input_ids、丢掉其余键"的便捷写法**，处理流程完全一样。所以：训练数据构造用 `encode`（你要裸 id 自己拼接），喂模型用直接调用（要 `input_ids` + `attention_mask` 一起）。

### 3.3 形状：list [n] vs tensor [1, n]

列表是一维 `[n]`——4 个 token 排成一排：

```
[1, 2345, 678, 2]
```

转成张量后是二维 `[1, n]`——前面多了一层：

```
[[1, 2345, 678, 2]]
```

**为什么多这层**：模型一次处理一个**批次**，张量的第 0 维约定是 batch 大小——即使只有 1 条文本也占住这个维度（这里 = 1）。列表是"裸数据"、没有这个约定，转张量时自动补上。

### 3.4 对比表（速查）

| 调用方式 | 返回 | input_ids 形状 | 有 attention_mask |
| --- | --- | --- | --- |
| `tokenizer.encode(text)` | `list[int]` | `[n]` | 否 |
| `tokenizer(text)` | `BatchEncoding`（值 list） | `[n]` | 是 |
| `tokenizer(text, return_tensors="pt")` | `BatchEncoding`（值 tensor） | `[1, n]` | 是 |

- `return_tensors` 取值：`"pt"`（PyTorch）、`"tf"`（TensorFlow）、`"np"`（NumPy）、`None`（默认，返回列表）

### 3.5 调用时的完整参数（重要输入，一次列全）

`tokenizer(text, ...)` 和 `encode(text, ...)` 共享同一套参数——**每个都要知道默认值和副作用**：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `add_special_tokens` | `True` | **自动加 bos/eos 包裹**（第 3.1 节那个 `[1, ..., 2]` 就是它干的）；不想要设 `False` |
| `return_tensors` | `None` | 值返回 list 还是 tensor：`"pt"` / `"tf"` / `"np"`（见 3.3 形状） |
| `padding` | `False` | 批量时短的补长：`True` / `"longest"` / `"max_length"`（见第 5 节） |
| `truncation` | `False` | 超长截断（见第 5 节） |
| `max_length` | `None` | 截断 / 填充的目标长度（见第 5 节） |
| `return_attention_mask` | `True` | 返回里带不带 `attention_mask` |

**完整调用示例**（训练构造 SFT 样本的典型形态）：

```python title="full-call.py"
inputs = tokenizer(
    ["样本一", "样本二"],
    add_special_tokens=False,      # 关包裹：自己拼 prompt + target，不要每段都带 bos/eos
    padding="longest",             # 批量对齐到最长
    truncation=True,
    max_length=512,                # 超 512 截断
    return_tensors="pt",           # 值转张量
)
```

::::tip
**什么时候用哪个**：训练数据构造用 `encode`（裸 id，方便拼接 prompt + target、做 `-100` 掩码）；推理或喂模型用直接调用 + `return_tensors="pt"`（模型要张量输入和 `attention_mask`）。
::::

## 4. 常用方法：convert_tokens_to_ids / decode（互转闭环）

第 3 节的 `encode` 管「文本 → id 列表」。互转的**另一半**和特殊场景在这里：

```python title="token-methods.py"
# 任意 token 字符串 → id（比如对话模板的结束标记）
eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
# generate 时告诉模型"遇到这个就停"（见 [[generate-生成机制]]）

# id 列表 → 文本（生成结果还原成文字）
text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
text.strip()
```

| 方法 | 方向 | 返回 |
| --- | --- | --- |
| `encode(text)` | 文本 → id 列表 | `list[int]`（见第 3 节） |
| `convert_tokens_to_ids("token")` | 单个 token 字符串 → id | `int` |
| `decode(ids)` | id 列表 → 文本 | `str` |

- `convert_tokens_to_ids` 只认**单个 token**（如 `<|im_end|>`），不是整段文本——整段文本用 `encode`
- `decode` 是 `encode` 的**逆操作**：生成完拿到 token id，`decode(skip_special_tokens=True)` 还原成文字并去掉 `<|im_end|>` 这类标记
- 三者合起来是完整的「文本 ↔ id」闭环：`encode` 进去、`decode` 出来（切片语法见 [[张量形状转置切片]]）

::::note
**decode 和 generate 是一对**：`generate()` 产出的是 token id（见 [[generate-生成机制]]），`decode` 把它们变回人能读的文本——生成后必有的一步。
::::

### decode：id 怎么变回文本（详解）

`decode` **不是"把每个 id 对回一个字符再拼起来"**——它要**反向走一遍分词器的合并规则**：id → token 词条（可能带 BPE/byte-level 的合并后缀）→ 按规则拼回原始文本。

**关键参数 `skip_special_tokens`**：

```python title="decode-special.py"
# 假设 ids = [1, 2345, 678, 2]，其中 1 是 bos、2 是 eos，中间是 "hello world"

tokenizer.decode(ids, skip_special_tokens=True)    # "hello world"（干净正文）
tokenizer.decode(ids, skip_special_tokens=False)   # 保留特殊 token（如 "hello world<|endoftext|>"）
```

- `True`：把 `<|im_end|>`、`<pad>`、bos/eos 这类特殊 token 从结果里拿掉，**只留正文**——生成后用 `True` 是标配
- `False`：原样保留它们——调试时想确认"模型有没有输出结束符"才用

**`batch_decode`：批量解码**——`decode` 只吃**一维** id 列表；二维 `[batch, seq]` 用 `batch_decode`（内部对每一行调一次 decode）：

```python title="batch-decode.py"
outputs.shape                       # [4, 20]：4 条生成结果
texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# texts = ["...", "...", "...", "..."]   4 条字符串
```

**生成完整链路**（decode 是最后一环，配套 generate）：

```python title="generate-decode-chain.py"
outputs = model.generate(input_ids=input_ids, max_new_tokens=30)   # [1, prompt_len+30]
answer_ids = outputs[0][prompt_len:]                               # 剥 batch + 切 prompt
text = tokenizer.decode(answer_ids, skip_special_tokens=True)      # id → 文本
text.strip()                                                       # 去首尾空白
```

**解码的常见坑**：decode 出来的文本可能带多余的空白/换行（`strip()` 处理）；中文等 CJK 文本正常解码，但若分词器是 byte-level（如 Llama），个别生僻字可能出现字符级分裂——一般不影响阅读。

## 5. 参数体系：padding / truncation / max_length

**先搞懂 padding 是什么**：模型一次吃一个**批次**，要求输入是"矩形"（批次里每条等长，形状 `[batch, len]`）。但真实文本长短不一——所以短的补长，这就是 **padding**。

```
批次两条（长 3 和 长 5）：
"短句"（3 个 token）              → [5, 100, 2]
"这是一个稍微长一点的句子"（5 个）→ [6, 20, 30, 40, 50]

不填充 → [3] 和 [5] 拼不成一个批次（模型要等长的矩形，直接报错）
填充后 → [[5, 100, 2, pad, pad],      ← 短句补 2 个 pad
          [6, 20, 30, 40, 50]]         ← 统一成 [2, 5]
```

**两个要点**：
- **单条文本永远不填充**——没有"和谁对齐"的问题，`attention_mask` 全 1
- **填充要两个条件缺一不可**：① 有 pad token（很多模型出厂 `pad_token_id = None`，要手动补，见第 6 节）；② 传 `padding=True`（注册了 pad token 但不传参数也不会触发）

参数控制输出结构：

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

## 6. 特殊 token 体系

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

### add_special_tokens：包裹行为与对话 token

**它到底加了什么**：`add_special_tokens=True`（默认）编码时给序列包上 bos/eos：

```
add_special_tokens=True   → [bos, ...正文..., eos]
add_special_tokens=False  → [...正文...]
```

**对话 token（im_start/im_end）不是它加的**：`<|im_start|>`、`<|im_end|>` 这类对话格式标记由 **chat template** 添加（`tokenizer.apply_chat_template(...)` 按模型的对话模板拼），与 `add_special_tokens` 是**两套独立机制**：

- 裸调用 `tokenizer("你好")` → `[bos, 你好, eos]`——只包 bos/eos，**不会**冒出 im_start/im_end
- 对话模板调用 → 按模板拼出 `<|im_start|>user\n你好<|im_end|>\n...` 再编码

**什么时候必须设 `False`**：自己拼接 SFT 样本（prompt + target 手动拼）时，每段都被包一层 bos/eos 会造成重复包裹、位置错乱——拼接的各段全部 `add_special_tokens=False`，只在需要处手动加 eos。

::::tip
**经验法则**：整段文本直接喂模型 → 默认 `True` 没问题；**自己手工拼接多段文本 → 一律 `False`**。
::::

## 7. padding_side：填充方向与生成的意义

```python title="padding-side.py"
tokenizer.padding_side = "left"   # 左填充：pad 加在序列开头
tokenizer.padding_side = "right"  # 右填充：pad 加在序列末尾（默认）
```

- **右填充**（默认）：真实 token 在前，pad 在后。适合训练（attention_mask 标掉 pad 位即可）
- **左填充**：pad 在前，真实 token 在后。**生成时必须用左填充**

::::important
**为什么生成要左填充**：decoder-only 模型从序列末尾往后生成新 token（见 [[generate-生成机制]]）。如果右填充，真实文本后面跟着 pad，模型会在 pad 后面生成，位置错乱。左填充保证真实 token 紧贴序列末尾，新生成的 token 自然接在后面。
::::

## 8. attention_mask 的两个作用

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
**attention_mask 不是因果掩码**：attention_mask 由 tokenizer 生成，标记填充位；因果掩码由模型内部生成，防止看未来。两者是独立的机制，在前向传播中组合使用（详见 [[架构-编码器与解码器]]）。
::::

## 小结

| 语法 | 返回类型 | 用途 |
| --- | --- | --- |
| `AutoTokenizer.from_pretrained(path)` | `Tokenizer` 实例 | 加载（读词表文件，非权重） |
| `tokenizer.save_pretrained(dir)` | 写词表文件 | 保存（和模型成对） |
| `tokenizer.encode(text)` | `list[int]` | 文本 → id 列表（数据构造） |
| `tokenizer(text)` | `BatchEncoding`（list 值） | 完整处理结果 |
| `tokenizer(text, return_tensors="pt")` | `BatchEncoding`（tensor 值） | 喂模型 |
| `tokenizer.convert_tokens_to_ids("token")` | `int` | 单个 token → id（生成停止标记） |
| `tokenizer.decode(ids)` | `str` | id 列表 → 文本（生成后还原） |
| `padding=True` | 补齐 + attention_mask | 批次等长 |
| `tokenizer.pad_token_id` | `int` 或 `None` | 填充符（None 要兜底） |
| `tokenizer.padding_side = "left"` | 设置属性 | 生成时左填充 |
| `out.to("cuda")` | `BatchEncoding` | 张量版搬到 GPU |
