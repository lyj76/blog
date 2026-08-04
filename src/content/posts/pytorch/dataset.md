---
title: PyTorch 核心：Dataset 与数据加载管线
published: 2026-08-04
description: 深入理解 torch.utils.data.Dataset —— 数据集的抽象、三个必须实现的方法，以及与 DataLoader 的协作方式
tags: [PyTorch, 深度学习, Dataset, DataLoader]
category: PyTorch
---

# PyTorch 核心：Dataset 与数据加载管线

## 一句话理解

`Dataset` 是 PyTorch 对「数据集」的抽象，来自 `torch.utils.data.dataset`。它最核心的能力是：**通过索引访问样本**——`dataset[i]` 返回第 `i` 个样本，通常是 `(输入, 标签)` 的元组。

```python title="dataset-indexing.py"
dataset = MyDataset(...)
x, y = dataset[0]   # 索引一个样本，返回 (输入, 标签)
```

这层抽象让上层组件（`DataLoader`）不需要关心数据存在哪里（内存、磁盘、数据库），只需要知道「给我第 i 个样本」。

## 为什么需要 Dataset？

训练循环需要反复地：取样本 → 组成批次 → 喂给模型。`DataLoader` 负责「批量、打乱、多进程预加载」，而它依赖 Dataset 的两个约定：

| 方法 | 作用 | 谁调用 |
| --- | --- | --- |
| `__len__` | 返回样本总数，供打乱与分批使用 | `DataLoader` 内部的采样器 |
| `__getitem__(idx)` | 根据索引返回单个样本 `(x, y)` | 取批次时按索引逐个取 |

::::note
**Map-style 与 Iterable-style**：`Dataset` 默认是 Map-style（可通过索引随机取任意样本）；还有 Iterable-style（类似迭代器，只能顺序取），用于流式加载无法全部放进内存的数据。绝大多数场景用 Map-style 就够了。
::::

## 三个必须实现的方法

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

::::important
**`__getitem__` 的返回值形式很灵活**：可以是 `(x, y)` 元组，也可以是字典（HuggingFace 风格），甚至可以是多个输入的元组。`DataLoader` 会自动把批次内同位置的元素堆叠成批次张量（如 `[batch, ...]`）。
::::

## 一个更真实的例子：图像分类

```python title="image-dataset.py"
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        # 假设文件名带标签：cat_001.jpg / dog_002.jpg
        label = 0 if "cat" in os.path.basename(path) else 1
        return img, label
```

## 与 DataLoader 协作

```python title="train-loop.py"
from torch.utils.data import DataLoader

dataset = ImageDataset("data/train", transform=transforms.ToTensor())
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for epoch in range(10):
    for x_batch, y_batch in loader:
        # x_batch: [32, 3, H, W]，y_batch: [32]
        ...
```

::::tip
**Transform 放哪里？** 把 `transform` 作为参数传入 `__init__` 是标准做法（而不是在 `__getitem__` 里写死）。这样训练集 / 验证集可以使用不同的预处理，同一个 Dataset 类也能复用。
::::

::::warning
**`__len__` 必须准确**：`shuffle=True` 时 DataLoader 依赖它生成随机排列，长度错误会导致采样越界或部分样本永远取不到。
::::

## 小结

- `Dataset` 本质是「索引 → 样本」的映射：`dataset[i] -> (x, y)`
- 必须实现 `__len__` 与 `__getitem__`；`__init__` 负责加载数据、准备预处理
- 与 `DataLoader(batch_size, shuffle, num_workers)` 组合，构成完整的训练数据管线
