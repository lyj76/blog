---
title: PyTorch 核心：Dataset 与 DataLoader 数据管线
published: 2026-08-04
description: 从 Dataset 三方法到 DataLoader 批量加载 —— 完整数据管线的构建、参数详解与训练循环模板
tags: [PyTorch, Dataset, DataLoader, 数据管线]
category: PyTorch
---

# PyTorch 核心：Dataset 与 DataLoader 数据管线

数据管线由两层组成：**`Dataset` 定义「怎么取一个样本」，`DataLoader` 负责「批量、打乱、并行取」**。两者配合构成完整的训练数据流。

## 1. 一句话理解 Dataset

`Dataset` 是 PyTorch 对「数据集」的抽象：**通过索引访问样本**——`dataset[i]` 返回第 `i` 个样本，通常是 `(输入, 标签)` 元组。

```python title="dataset-indexing.py"
x, y = dataset[0]   # 索引一个样本，返回 (输入, 标签)
```

这层抽象让上层组件不关心数据存在哪里（内存、磁盘、数据库），只需知道「给我第 i 个样本」。

## 2. 三个必须实现的方法

| 方法 | 作用 | 谁调用 |
| --- | --- | --- |
| `__len__` | 返回样本总数，供打乱与分批使用 | `DataLoader` 内部的采样器 |
| `__getitem__(idx)` | 根据索引返回单个样本 `(x, y)` | 取批次时按索引逐个取 |

```python title="dataset-example.py"
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, file_path, label_path, transform=None):
        # 1. 初始化：加载样本路径、标签与预处理函数
        self.file_paths = [line.strip() for line in open(file_path)]
        self.labels = [int(line.strip()) for line in open(label_path)]
        self.transform = transform

    def __len__(self):
        # 2. 返回总样本数：DataLoader 靠它知道数据边界、做随机采样
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 3. 核心！根据 idx 返回第 idx 个样本 (x, y)
        x = load_sample(self.file_paths[idx])
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
```

::::note
**Map-style 与 Iterable-style**：`Dataset` 默认是 Map-style（可通过索引随机取）；Iterable-style 只能顺序取，用于流式加载。绝大多数场景用 Map-style。
::::

## 3. DataLoader：批量加载器

```python title="dataloader-basic.py"
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

`DataLoader` 拿着 `Dataset`，自动完成：**按 batch_size 分批 → 每批调用 `dataset[i]` → 堆叠成批次张量**。

**遍历得到什么**：Dataset 返回字典 → batch 是字典；返回元组 → batch 是元组。堆叠规则：多个 `[seq]` 叠成 `[batch, seq]`——**第 0 维永远是 batch_size**：

```python title="iterate-batch.py"
for batch in dataloader:
    input_ids = batch["input_ids"]   # [4, seq_len]
    labels = batch["labels"]         # [4, seq_len]
```

**常用参数详解**：

| 参数 | 作用 | 例 |
| --- | --- | --- |
| `batch_size` | 每批几个样本 | `4` |
| `shuffle` | 每轮打乱顺序（防过拟合） | `True` |
| `num_workers` | 并行加载的进程数（0=主进程） | `4` |
| `drop_last` | 最后一批不满 batch_size 时丢弃 | `True` |
| `collate_fn` | 自定义「如何把样本堆成批次」 | 自定义函数 |
| `pin_memory` | 锁页内存，加速 CPU→GPU 拷贝 | `True` |

**总批数**：`len(dataloader)` = `ceil(样本数 / batch_size)`（向上取整）。

## 4. 完整的训练循环模板

```python title="train-loop.py"
for epoch in range(1, 4):                      # 外层：多轮
    for step, batch in enumerate(dataloader):  # 内层：遍历所有批次
        input_ids = batch["input_ids"].to(device)   # 每批都要搬到 GPU！
        labels = batch["labels"].to(device)

        loss = model(input_ids=input_ids, labels=labels).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch}/3] | Step [{step+1}/{len(dataloader)}] | Loss: {loss.item():.4f}")
```

## 5. 常见问题

- **`shuffle=True` 只对训练集用**：验证/测试集保持顺序，否则指标不可比
- **`num_workers > 0` 在 Windows 报错**：需配合 `if __name__ == "__main__":` 保护；`__getitem__` 里别做重活
- **样本长度不一报错**：必须自己 padding 成等长（`input_ids` 用 `pad_token_id` 补、`labels` 用 `-100` 补）
- **最后一批不满**：默认保留（`drop_last=False`），代码要容忍 batch 大小不一致

## 小结

| 层 | 职责 | 关键点 |
| --- | --- | --- |
| `Dataset` | 索引 → 样本 | `__len__` + `__getitem__` |
| `DataLoader` | 分批/打乱/并行 | `batch_size` / `shuffle` / `num_workers` |
| 训练循环 | 取批 → 前向 → 反向 | `.to(device)` + `zero_grad/backward/step` |
