---
title: PyTorch DataLoader：批量数据加载全解
published: 2026-08-04
description: DataLoader 的 batch_size / shuffle / num_workers / collate_fn 参数、与 Dataset 的配合、批次遍历与设备搬移
tags: [PyTorch, DataLoader, 数据加载, 基础]
category: PyTorch
---

# PyTorch DataLoader：批量数据加载全解

`Dataset` 定义「怎么取单个样本」，`DataLoader` 负责「批量、打乱、并行取」。两者配合构成完整数据管线。

## 1. 一句话理解

```python title="dataloader-basic.py"
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

`DataLoader` 拿着你的 `Dataset`，自动完成：**按 batch_size 分批 → 每批自动调用 `dataset[i]` → 堆叠成批次张量**。

## 2. 遍历 DataLoader 得到什么

```python title="iterate-dataloader.py"
for batch in dataloader:
    # batch 的形状：第 0 维永远是 batch_size
    input_ids = batch["input_ids"]   # [4, seq_len]
    labels = batch["labels"]         # [4, seq_len]
```

- Dataset 的 `__getitem__` 返回什么「形状」，DataLoader 就把它**堆叠**成什么
- 返回字典 → batch 是字典；返回元组 → batch 是元组
- 堆叠规则：多个样本的 `[seq]` 叠成 `[batch, seq]`

::::note
**为什么第 0 维永远是 batch**：DataLoader 把 batch_size 个样本沿新维度堆叠。所以 `batch["input_ids"].shape[0]` 永远是批次大小。
::::

## 3. 常用参数详解

| 参数 | 作用 | 例 |
| --- | --- | --- |
| `batch_size` | 每批几个样本 | `4` |
| `shuffle` | 每轮打乱顺序（防过拟合） | `True` |
| `num_workers` | 并行加载的进程数（0=主进程） | `4` |
| `drop_last` | 最后一批不满 batch_size 时丢弃 | `True` |
| `collate_fn` | 自定义「如何把样本堆成批次」 | 自定义函数 |
| `pin_memory` | 锁页内存，加速 CPU→GPU 拷贝 | `True` |

```python title="dataloader-full.py"
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,    # 多进程预取，加速数据读取
    pin_memory=True,  # GPU 训练常用
)
```

## 4. 与 Dataset 的协作

DataLoader 内部只依赖 Dataset 的两个方法：

| Dataset 方法 | DataLoader 何时调用 |
| --- | --- |
| `__len__` | 计算总批数、生成打乱索引 |
| `__getitem__(idx)` | 每取一个样本调一次 |

**总批数**：`len(dataloader)` = `ceil(样本数 / batch_size)`（向上取整）。

## 5. 完整的 epoch 循环模板

```python title="epoch-loop.py"
for epoch in range(1, 4):                      # 3 个 epoch
    for step, batch in enumerate(dataloader):  # 遍历所有批次
        input_ids = batch["input_ids"].to(device)   # 每批都要搬到 GPU！
        labels = batch["labels"].to(device)

        loss = model(input_ids=input_ids, labels=labels).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch}/3] | Step [{step+1}/{len(dataloader)}] | Loss: {loss.item():.4f}")
```

- `for epoch in range(...)`：外层循环多轮
- `enumerate(dataloader)`：内层拿到 `(step, batch)`
- `.to(device)`：数据初始在 CPU，**每次取出来都要搬**到 GPU

::::warning
**`shuffle=True` 只对训练集用**：验证/测试集保持顺序（`shuffle=False`），否则每次评估的数据顺序都不同，指标不可比。
::::

## 6. 常见问题

- **`num_workers` 多进程报错**：Windows 下 `num_workers > 0` 需配合 `if __name__ == "__main__":` 保护；`__getitem__` 里别做重活
- **batch 最后一个不满**：默认保留（`drop_last=False`），需处理 batch 大小不一致
- **`__getitem__` 返回长度不一**：必须自己 padding 成等长（否则堆叠报错），见 SFT 数据构造笔记

## 小结

| 语法 | 作用 |
| --- | --- |
| `DataLoader(dataset, batch_size=4, shuffle=True)` | 分批 + 打乱 |
| `for batch in dataloader` | 遍历批次 |
| `batch["input_ids"]` | 取批次张量 |
| `.to(device)` | 搬到 GPU |
| `len(dataloader)` | 总批数 |
