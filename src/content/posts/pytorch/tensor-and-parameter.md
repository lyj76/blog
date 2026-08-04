---
title: PyTorch 核心：Tensor 与 nn.Parameter
published: 2026-08-04
description: torch.Tensor 的六大属性、自动求导计算图与叶子结点，以及 nn.Parameter 的本质与注册机制
tags: [PyTorch, 深度学习, Tensor, 自动求导]
category: PyTorch
---

# PyTorch 核心：Tensor 与 nn.Parameter

## 1. torch.Tensor 的核心属性

| 属性 | 类型 | 解释 | 默认值 |
| --- | --- | --- | --- |
| `A.data` | Tensor | 承载实际数据与运算 | — |
| `A.grad` | Tensor 或 `None` | 保存的梯度（与 `A` 同形状） | `None` |
| `A.requires_grad` | bool | 是否追踪梯度、参与自动求导 | `False` |
| `A.grad_fn` | Function 或 `None` | 生成该张量的运算节点（计算图来源） | `None` |
| `A.is_leaf` | bool | 是否是计算图的叶子结点 | — |
| `A.dtype` | torch.dtype | 元素类型（`float32` / `int64` 等） | — |

::::note
**两处容易记错**：`A.grad` 的类型是**张量**（与 `A` 同形状），不是浮点数；`A.is_leaf` 是布尔值（`True` / `False`），不是指针。
::::

## 2. 计算图与叶子结点

PyTorch 用**动态计算图**记录运算历史。以 $W = 2A$ 为例：

$$W = 2 \cdot A$$

```python title="graph-example.py"
import torch

A = torch.tensor([3.0], requires_grad=True)  # 用户直接创建 → 叶子结点
W = 2 * A                                     # 由 A 运算得到 → 非叶子结点
L = W ** 2                                    # 再往后一层

L.backward()
print(A.grad)   # tensor([24.])
```

用链式法则手工验证：

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial W} \cdot \frac{\partial W}{\partial A} = 2W \cdot 2 = 4W = 8A = 24$$

关键规则：

- **叶子结点（leaf）**：由用户直接创建、不依赖其他张量的张量（如 `A`）。`requires_grad=True` 时，`backward()` 后它的 `.grad` 会被填充
- **非叶子结点**：由运算产生的张量（如 `W`、`L`），有 `grad_fn` 指向产生它的运算，默认**不保留** `.grad`
- 叶子结点是「前向传播的起点、梯度更新的终点」。这种设计让自动求导只需沿计算图反向走一遍，就能把梯度回传到所有叶子

::::tip
**查看中间量的梯度**：对非叶子结点调用 `W.retain_grad()`，`backward()` 后 `W.grad` 就会被保留下来——调试复杂计算图时非常有用。
::::

## 3. torch.nn.Parameter

`nn.Parameter` 是 `Tensor` 的子类，专为「可学习的模型参数」设计：

```python title="parameter-example.py"
import torch
import torch.nn as nn

param = nn.Parameter(torch.randn(3, 4))
print(isinstance(param, torch.Tensor))   # True —— 本质还是 Tensor
print(param.requires_grad)               # True —— 默认开启梯度
```

与普通 Tensor 的区别：

| 特性 | `torch.Tensor` | `nn.Parameter` |
| --- | --- | --- |
| `requires_grad` 默认值 | `False` | `True` |
| 作为 `nn.Module` 属性时 | 不注册 | **自动注册进 `_parameters`** |
| 被 `model.parameters()` 访问 | 否 | 是 |
| 跟随 `model.to("cuda")` 迁移 | 否 | 是 |
| 被 `model.state_dict()` 保存 | 否 | 是 |
| `is_leaf` | 取决于创建方式 | 默认 `True` |

::::important
**`nn.Parameter` 与 `nn.Module` 的绑定**：只有把 `Parameter` 赋值给 `Module` 的属性（如 `self.weight = nn.Parameter(...)`），它才会进入 `_parameters` 字典，从而被 `model.parameters()`、`model.to("cuda")`、`model.state_dict()` 统一管理。
::::

**专有方法**：`param.numel()` 返回参数的元素总数（`int`），常用于统计模型参数量：

```python title="numel-example.py"
total = sum(p.numel() for p in model.parameters())
print(f"参数量: {total / 1e6:.2f}M")
```

## 4. 矩阵运算

`Tensor` 和 `Parameter` 都能直接参与矩阵运算，本质是对 `A.data` 的计算（梯度通过计算图自动回传）：

```python title="matmul-example.py"
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = torch.randn(4, 5)

result = A @ B @ C        # 形状 [2, 5]
```

常用运算速查：

| 运算 | 写法 |
| --- | --- |
| 矩阵乘法 | `A @ B` |
| 逐元素乘法 | `A * B` |
| 转置 | `A.T` / `A.transpose(0, 1)` |
| 求和 | `A.sum()` |
| 展平 | `A.flatten()` / `A.view(-1)` |

## 5. 创建张量

```python title="create-example.py"
torch.randn(2, 3)        # 标准正态分布 [2, 3]
torch.rand(2, 3)         # 均匀分布 [0, 1)
torch.zeros(2, 3)        # 全 0
torch.ones(2, 3)         # 全 1
torch.full((2, 3), 7)    # 全 7
torch.arange(0, 10)      # [0, 1, ..., 9]
torch.eye(3)             # 3x3 单位矩阵
```

::::tip
**`dtype` 参数可选**：`torch.randn(2, 3, dtype=torch.float64)` 可指定精度。默认 `float32` 满足绝大多数训练需求；**模型的输入张量建议与参数保持同一 `dtype`**，否则会触发隐式类型转换，甚至直接报错。
::::

## 小结

- `Tensor` 是数据与自动求导的载体，把 `data / grad / requires_grad / grad_fn / is_leaf / dtype` 六个属性理解透，计算图就通了一半
- `nn.Parameter` = 默认 `requires_grad=True` 的 Tensor + 注册进 Module 被统一管理
- 叶子结点是梯度更新的目标，链式法则沿计算图自动完成
