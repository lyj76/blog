---
title: PyTorch 核心：Dataset 与 DataLoader 数据管线
published: 2026-08-04
description: 从 Dataset 三方法到 DataLoader 批量加载 —— 完整数据管线的构建、参数详解与训练循环模板
tags: [PyTorch, Dataset, DataLoader, 数据管线]
category: PyTorch
---

# PyTorch 核心：Dataset 与 DataLoader 数据管线

数据管线由两层组成：**`Dataset` 定义「怎么取一个样本」，`DataLoader` 负责「批量、打乱、并行取」**。两者配合构成完整的训练数据流。

> 归属：**PyTorch · `torch.utils.data`** —— 数据接口层（`from torch.utils.data import Dataset, DataLoader`）。

## 1. 一句话理解 Dataset

`Dataset` 是 PyTorch 对「数据集」的抽象：**通过索引访问样本**——`dataset[i]` 返回第 `i` 个样本，通常是 `(输入, 标签)` 元组。

```python title="dataset-indexing.py"
x, y = dataset[0]   # 索引一个样本，返回 (输入, 标签)
```

这层抽象让上层组件不关心数据存在哪里（内存、磁盘、数据库），只需知道「给我第 i 个样本」。

::::note
**数据从哪来（来源层）**：PyTorch 只管「怎么取样本」，不管「数据在哪」。从 HuggingFace Hub / 文件拿数据是 `load_dataset` 的事（HF `datasets` 库，见「load_dataset 数据加载」篇）——它产出原始数据，你把它包进 `__getitem__` 就成了这里的 `Dataset`。
::::

## 2. 三个必须实现的方法

| 方法 | 作用 | 谁调用 |
| --- | --- | --- |
| `__len__` | 返回样本总数，供打乱与分批使用 | `DataLoader` 内部的采样器 |
| `__getitem__(idx)` | 根据索引返回**单个样本，返回什么由你定**（元组 / 字典 / 张量都行） | 取批次时按索引逐个取 |

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

### 返回什么由你定：元组 vs 字典

`__getitem__` 的返回值**没有固定格式**——你返回什么，DataLoader 就收集什么、堆叠什么。两种写法完全等价，没有对错；字典更自解释（不用记 `batch[0]` 到底是输入还是标签），是 LLM 训练的标准写法。

**元组形式：按位置堆叠**（假设 `batch_size=2`）：

```python title="tuple-stack.py"
# 两个样本
x0, y0 = torch.tensor([1, 2, 3]), torch.tensor([0])   # 样本 0：形状 [3] 和 []
x1, y1 = torch.tensor([4, 5, 6]), torch.tensor([1])   # 样本 1

# ① 收集：DataLoader 把 2 个样本放进一个列表
batch_list = [(x0, y0), (x1, y1)]

# ② 按位置堆叠：所有"第 0 个元素"叠一起，所有"第 1 个元素"叠一起
stacked_x = torch.stack([x0, x1])   # tensor([[1,2,3],[4,5,6]])  形状 [2,3]
stacked_y = torch.stack([y0, y1])   # tensor([0,1])               形状 [2]

batch = (stacked_x, stacked_y)      # 还是元组
x_batch, y_batch = batch            # 解包拿到
```

**字典形式：按键堆叠**（你代码里的写法）：

```python title="dict-stack.py"
# 你的 __getitem__ 返回字典：
def __getitem__(self, idx):
    ...
    return {"input_ids": ..., "labels": ...}

# ① 收集：同样放进列表
batch_list = [
    {"input_ids": x0, "labels": y0},
    {"input_ids": x1, "labels": y1},
]

# ② 按键堆叠：所有 "input_ids" 叠一起，所有 "labels" 叠一起
stacked_ids  = torch.stack([d["input_ids"] for d in batch_list])   # tensor([[1,2,3],[4,5,6]])  [2,3]
stacked_labs = torch.stack([d["labels"]    for d in batch_list])   # tensor([0,1])              [2]

batch = {"input_ids": stacked_ids, "labels": stacked_labs}
batch["input_ids"]   # tensor([[1,2,3],[4,5,6]])
```

**本质：同一个动作，不同的"分组方式"**：

- 元组 → **按位置分组**：所有样本的第 0 个元素一组、第 1 个元素一组……
- 字典 → **按键分组**：所有样本的 `"input_ids"` 一组、`"labels"` 一组……

每组内 `torch.stack` 把组里的 N 个张量叠起来，**在最前面插一个新维度**——这就是 batch 第 0 维永远是 batch 大小的原因。

### DataLoader 是怎么"统一"堆叠的：结构递归

为什么同一个 DataLoader 能不加区分地处理元组和字典？因为它不关心你的样本是什么语义，只关心**结构**，然后**递归**处理（简化版）：

```python title="collate-recursion.py"
def default_collate(batch):           # batch = [样本0, 样本1, ..., 样本N-1]
    elem = batch[0]                    # 只看第一个样本的结构

    if isinstance(elem, dict):         # 字典 → 按键分组，递归
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple):      # 元组 → 按位置分组，递归
        return tuple(default_collate(s) for s in zip(*batch))
    elif isinstance(elem, torch.Tensor):  # 张量 → 叶子，直接堆叠
        return torch.stack(batch)
```

::::note
**背后的两个抽象（通用知识）**：
1. **对象协议统一访问方式**：`tuple[0]` 和 `dict["input_ids"]` 走的是同一个 `__getitem__` 协议（key 只是类型不同），通用代码不需要知道结构细节
2. **结构递归**：按容器类型分组 → 逐槽位递归 → 叶子堆叠，任意嵌套结构一份代码处理

这套"按类型分组 + 递归 + 叶子操作"的模式，`pickle`、`copy.deepcopy`、JSON 序列化全在用。**写任何"把 N 个结构合成一个"的代码时，直接套这个模板**。
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

### 自定义堆叠方式：`collate_fn`

默认 `collate_fn` 只做一件事：把 N 个样本**堆叠成批次**（list 变张量）。样本形状不一致、或要动态 padding 时，就得自己写：

```python title="collate-example.py"
def my_collate(batch):                    # batch = [样本1, 样本2, ...]，每批 N 个
    input_ids = [item["input_ids"] for item in batch]   # 每个是 [seq_i] 长度不一
    labels    = [item["labels"]    for item in batch]

    # 动态 padding：本批内统一到最长长度
    max_len = max(len(x) for x in input_ids)
    padded = []
    for x in input_ids:
        padded.append(x + [pad_id] * (max_len - len(x)))   # 短的补 pad_id
    return {
        "input_ids": torch.tensor(padded),
        "labels":    torch.tensor(labels),                 # labels 同样处理
    }

dataloader = DataLoader(dataset, batch_size=4, collate_fn=my_collate)
```

- `collate_fn` 收到的 `batch` 是 `__getitem__` 返回值的**列表**（本批 N 个）
- 默认行为 = `torch.stack` 直接堆叠——要求所有样本形状一致，否则报错
- 训练 LLM 时样本长度天然不一，**几乎总是需要自定义 collate 做动态 padding**（不 padding 到固定最大长度、只到本批最长，省内存）

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

::::note
**想实时看训练进度？** 用 `tqdm` 包住 dataloader——`for batch in tqdm(dataloader, desc="...")` 自动显示进度 / 耗时 / 剩余时间，比 `print` 清晰得多，还能做 epoch + batch 双层进度条（用法见 [[tqdm-进度条]]）。
::::

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
