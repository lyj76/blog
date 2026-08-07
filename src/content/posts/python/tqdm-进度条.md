---
title: Python tqdm 进度条：训练循环的可视化
published: 2026-08-07
description: tqdm 给任意可迭代对象加进度条、最常用形态（包住 dataloader 看训练进度）、desc/total/leave 参数、手动 update、嵌套进度条（epoch + batch）、与 f-string 日志配合
tags: [Python, tqdm, 进度条, 训练循环]
category: Python
---

# Python tqdm 进度条：训练循环的可视化

训练跑起来最想知道的是"到哪了、还剩多久"。`tqdm` 就是干这个的：把一个可迭代对象包起来，自动显示进度条、已用时间、预计剩余时间、速度。

> 归属：**`tqdm` 库** —— 第三方进度条工具（`pip install tqdm`），不是 Python 标准库，也不是 PyTorch 专属。

## 概念本体：tqdm 在度量什么

`tqdm` 的核心就一句话：**给任何可迭代对象加进度条**。

```python title="tqdm-basic.py"
from tqdm import tqdm

for i in tqdm(range(100)):
    ...   # 每个元素做完，进度条自动前进一格
```

它不关心你迭代的是什么——列表、生成器、`range`、`dataloader` 都行，只要能被 `for` 遍历。**所以它不是 dataloader 专属**；只是 ML 训练里最常见的就是包住 `dataloader` 看训练进度。

## 最常用形态：包住 dataloader

训练循环里最标准的用法——把 `for batch in dataloader` 改成 `for batch in tqdm(dataloader)`，进度条自动显示：已处理批数 / 总数、耗时、每批速度、剩余时间。

```python title="train-with-tqdm.py"
from tqdm import tqdm

for epoch in range(epochs):
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        loss = model(batch["input_ids"], labels=batch["labels"]).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**为什么能包住 dataloader**：`DataLoader` 实现了 `__len__`（总批数）和 `__iter__`（逐批产出）——tqdm 靠 `__len__` 知道总进度、靠 `__iter__` 逐格前进（协议机制见 [[类与属性访问]]、[[迭代与可变集合]]）。**能 `len()` 又能 `for` 的对象，tqdm 都能包**——这正是它"通用"的根源。

## 关键参数

| 参数 | 作用 |
| --- | --- |
| `desc` | 进度条左侧的文字，区分不同循环（如 `"Epoch 0"`） |
| `total` | 总进度数（可迭代对象没有 `__len__` 时手动指定） |
| `leave` | 循环结束后进度条是否保留（内层循环用 `False` 避免刷屏） |
| `unit` | 进度单位名称（如 `unit="batch"` → 显示 "batch"） |

::::note
**内层循环 `leave=False`**：外层 epoch + 内层 batch 双进度条时，内层结束后若保留会叠一堆进度条——设 `leave=False` 让内层结束即消失，只剩外层的。
::::

## 手动进度：不基于迭代器

有些场景不是 `for` 循环（比如手动逐块处理、流式下载），用 `total` + `update` 手动控制：

```python title="manual-progress.py"
p = tqdm(total=100, desc="处理中")
for chunk in chunks:
    process(chunk)
    p.update(len(chunk))   # 前进 n 格
p.close()                  # 收尾
```

- `update(n)` 手动前进 `n` 格（配合 `total` 用）
- 结束记得 `close()`（或 `with tqdm(total=...) as p:` 自动关闭）

## 嵌套进度条：epoch + batch 双层

训练的标准双条形态：外层是 epoch，内层是 batch。

```python title="nested-tqdm.py"
for epoch in tqdm(range(epochs), desc="Epoch"):
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
        loss = model(batch["input_ids"], labels=batch["labels"]).loss
        loss.backward()
        optimizer.step()
```

- 外层 `desc="Epoch"`；内层 `desc=f"Epoch {epoch}"` 带上当前轮次
- 内层 `leave=False` 结束后自动消失，界面干净
- `tqdm` 自动识别嵌套层级，两个条上下排布

## 与日志配合：f-string 不冲突

进度条和 `print` 日志可以共存——`tqdm` 的输出走自己的行，普通 `print` 也能正常显示（见 [[f-string格式化]]）：

```python title="tqdm-and-log.py"
for batch in tqdm(dataloader, desc="Train"):
    ...
    if step % 100 == 0:
        print(f"step {step}: loss={loss.item():.4f}")   # 日志照常打
```

## 小结

| 语法 | 作用 |
| --- | --- |
| `tqdm(iterable)` | 给任意可迭代对象加进度条（包住 dataloader 是 ML 标配） |
| `tqdm(dataloader, desc="...")` | 包 dataloader + 左侧文字 |
| `total` / `update(n)` | 手动进度（不基于迭代器） |
| `leave=False` | 内层循环结束即消失 |
| 嵌套 | 外层 epoch + 内层 batch，desc 区分 |

相关：[[dataset]]（训练循环）· [[f-string格式化]]（日志输出）· [[迭代与可变集合]]（可迭代协议）
