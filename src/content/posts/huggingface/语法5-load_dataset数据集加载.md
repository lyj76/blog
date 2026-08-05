---
title: HuggingFace 语法⑤：load_dataset 数据集加载
published: 2026-08-04
description: datasets 库的 load_dataset 用法：split 划分、streaming 流式、行迭代、截断取子集、转成 PyTorch Dataset
tags: [HuggingFace, datasets, 语法, 数据]
category: HuggingFace
---

# HuggingFace 语法⑤：load_dataset 数据集加载

`datasets` 库（HuggingFace 出品）负责加载公开数据集，是训练数据的源头。

## 1. 一句话理解

```python title="load-basic.py"
from datasets import load_dataset

dataset = load_dataset("hugcyp/LCSTS")   # 从 HuggingFace Hub 下载数据集
```

`load_dataset("作者/数据集名")` 从 **HuggingFace Hub** 下载并缓存数据集，返回一个 `DatasetDict`（按 split 分的字典）。

## 2. `split` 参数：选哪个划分

```python title="split-example.py"
load_dataset("hugcyp/LCSTS", split="train")        # 只要训练集
load_dataset("hugcyp/LCSTS", split="validation")   # 验证集
load_dataset("hugcyp/LCSTS", split="test")         # 测试集
```

- 不传 `split` 返回所有划分（`train` / `validation` / `test`）
- 传 `split` 直接返回对应划分的 `Dataset`（不用再取字典）

## 3. `streaming=True`：流式加载

```python title="streaming-example.py"
stream = load_dataset("hugcyp/LCSTS", split="train", streaming=True)
for i, row in enumerate(stream):
    if i >= 3000:
        break
    raw_data.append(row)
```

- **`streaming=True`**：边下载边迭代，不用等整个数据集下完
- 大模型数据动辄几百 GB，流式是标准做法
- `enumerate(stream)` 拿到序号 + 数据行，`break` 提前截断（只取前 3000 条）

::::note
**流式 vs 普通**：普通模式 `load_dataset` 会阻塞到**全部下载完**；流式模式拿到迭代器立即返回，用多少取多少。小数据集用普通模式即可，大的一定要 `streaming=True`。
::::

## 4. 行的结构：row 是字典

```python title="row-access.py"
row["text"]     # 按列名取值，如文章的正文
row["summary"]  # 摘要/标签列
```

数据集的每一行是一个 **dict**，按列名取值。列名由数据集决定——先 `print(stream)` 或 `dataset.column_names` 查看有什么列。

## 5. 转成 PyTorch Dataset

`load_dataset` 得到的是 HF 的 `Dataset`，**不能直接喂 DataLoader**（没有 `__getitem__` 的返回约定）。标准做法是自己包一层：

```python title="wrap-dataset.py"
from torch.utils.data import Dataset

class TitleDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_len=256):
        self.data = raw_data          # raw_data = 从 stream 取的行列表
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]         # item 就是一行 dict
        ...
        return {"input_ids": ..., "labels": ...}

raw_data = []
for i, row in enumerate(stream):     # 先取原始行
    if i >= 3000:
        break
    raw_data.append(row)

dataset = TitleDataset(raw_data, tokenizer, max_len=256)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

数据流：**HF Hub → load_dataset 流式取行 → 包进自定义 Dataset → DataLoader 分批**。

## 6. 其他常用

```python title="other-usages.py"
# 本地文件（json / csv / parquet）
load_dataset("json", data_files="data.jsonl")

# 从 Hub 仓库里的文件
load_dataset("mydataset/demo", data_files="train.jsonl", split="train")

# 划分训练/验证
split_dataset = dataset.train_test_split(test_size=0.1)
```

| 写法 | 用途 |
| --- | --- |
| `load_dataset("作者/名", split="train")` | 加载 Hub 数据集某划分 |
| `streaming=True` | 流式（大数据集） |
| `row["列名"]` | 按列取值 |
| `for i, row in enumerate(stream): break` | 截断取子集 |
| 自定义 Dataset 包一层 | 转成 DataLoader 可用 |

## 小结

- `load_dataset("作者/名", split="train", streaming=True)` 是万能开头
- 大数据集必须 `streaming=True` + `enumerate` 截断
- 拿到行列表后，自己写 `Dataset` 子类（`__len__` / `__getitem__`）转给 `DataLoader`
