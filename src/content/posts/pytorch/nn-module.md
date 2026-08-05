---
title: PyTorch 核心：nn.Module 与参数管理
published: 2026-08-04
description: nn.Module 的自动注册机制、forward 与 __call__、常用属性，以及 modules / parameters / state_dict 等迭代器家族
tags: [PyTorch, 深度学习, nn.Module, 神经网络]
category: PyTorch
---

# PyTorch 核心：nn.Module 与参数管理

`nn.Module` 是所有神经网络层与模型的基类。理解它的「自动注册」机制，是掌握 PyTorch 模型管理的一把钥匙。

## 1. 自动注册机制

自定义层时**必须**调用 `super().__init__()`：

```python title="auto-register.py"
import torch
import torch.nn as nn

class MyLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()        # 必须！初始化内部的 _parameters / _modules 字典
        self.fc = nn.Linear(in_dim, out_dim)              # nn.Module 子类 → 注册进 _modules
        self.scale = nn.Parameter(torch.ones(out_dim))    # nn.Parameter → 注册进 _parameters

    def forward(self, x):
        return self.fc(x) * self.scale
```

`super().__init__()` 会创建两个内部 `OrderedDict`：

- `_parameters`：存放所有 `nn.Parameter` 类型的属性
- `_modules`：存放所有 `nn.Module` 类型的属性

只要在 `__init__` 里给 `self` 赋值为 `nn.Module` 或 `nn.Parameter`，就会**自动注册**到对应字典。这带来两个能力：

```python title="auto-register-result.py"
model = MyLayer(4, 8)

# 1. 所有参数可被迭代器访问
for name, param in model.named_parameters():
    print(name, param.shape)   # fc.weight [8,4] / fc.bias [8] / scale [8]

# 2. 一键移动设备、切换训练模式
model.to("cuda")
model.train()   # 或 model.eval()
```

::::warning
**参数不能包在普通容器里**：把 `nn.Parameter` 赋给 Python 的 `list` / `dict` 不会被注册（`model.parameters()` 里找不到它）。需要动态参数列表时，请用 `nn.ParameterList` / `nn.ModuleList` / `nn.ModuleDict`。
::::

## 2. 重写 forward（call 方法）

`model(x)` 能完成调用，是因为 `nn.Module` 重写了 `__call__`——它会先执行已注册的钩子，再调用你重写的 `forward`：

```python title="forward-example.py"
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.relu(self.fc(x))   # 前向逻辑写在这里

y1 = model(x)            # 推荐：走 __call__，钩子生效
y2 = model.forward(x)    # 不推荐：跳过钩子
```

::::important
**始终使用 `model(x)` 而不是 `model.forward(x)`**。`__call__` 除了调用 `forward`，还会执行 `register_forward_hook` 注册的前向钩子、`register_full_backward_hook` 注册的反向钩子等。直接调 `forward` 会绕过这些机制，破坏依赖钩子的功能（特征提取、可视化、剪枝等）。
::::

## 3. 常用属性：以 nn.Linear 为例

`nn.Linear` 是最常用的层，它的属性很有代表性：

| 属性 | 类型 | 解释 |
| --- | --- | --- |
| `.weight` | `nn.Parameter` | 权重矩阵，形状 `(out_features, in_features)` |
| `.bias` | `nn.Parameter` 或 `None` | 偏置向量，形状 `(out_features,)`；`bias=False` 时为 `None` |
| `.in_features` | `int` | 输入维度 |
| `.out_features` | `int` | 输出维度 |

`nn.Module` 本身还提供这些常用成员：

| 成员 | 说明 |
| --- | --- |
| `.training` | 布尔值，由 `train()` / `eval()` 切换，影响 BatchNorm / Dropout 的行为 |
| `.state_dict()` | 返回 `_parameters` 与缓冲区（buffer）的键值快照，用于保存与加载 |
| `.device` | ⚠️ 没有这个属性；设备信息要通过参数的 `.device` 获取 |

### train() / eval() 到底做了什么

```python title="train-eval.py"
model.train()   # 递归把所有子模块的 training 标志设为 True
model.eval()    # 递归把所有子模块的 training 标志设为 False
```

- **机制**：`train()` / `eval()` 不是"魔法开关"，而是**递归遍历整棵模块树，把每个模块的 `.training` 属性设为 `True` / `False`**
- **谁在读这个标志**：`Dropout` 在 `forward` 里查 `self.training`——`True` 时随机丢弃神经元，`False` 时原样通过；`BatchNorm` 也查——`True` 时用当前 batch 的统计量并更新 running 均值，`False` 时用历史 running 统计量
- **`nn.Linear` 不关心 training**：它没有任何行为差异——所以只用 Linear 的模型，train/eval 切换看不出区别
- **为什么推理前必须 `eval()`**：忘记切回 `eval()`，Dropout 还在随机丢神经元，输出不稳定；BatchNorm 用 batch 统计量导致结果偏差

::::warning
**`eval()` 不关梯度**：`eval()` 只改 `training` 标志，**不**停止梯度计算。推理时省显存要靠 `with torch.no_grad():`（见"自动求导与梯度"篇）——两者配合才是完整推理姿势。
::::

**按参数重建线性层**（LoRA 合并权重时常用）：用已有层的维度/设备/精度创建同构新层：

```python title="recreate-linear.py"
new_linear = nn.Linear(
    in_features=orig_linear.in_features,       # 输入维度保持一致
    out_features=orig_linear.out_features,     # 输出维度保持一致
    bias=orig_linear.bias is not None,         # 是否带偏置
    device=W0.device,                          # 设备一致（cuda/cpu）
    dtype=W0.dtype                             # 精度一致（float32 / bfloat16）
)
```

## 4. 几个重要的迭代器

`nn.Module` 提供了一整套「树形遍历」迭代器，按粒度从大到小：

### modules() / named_modules() —— 遍历子模块树

```python title="modules-example.py"
for name, sub_module in model.named_modules():
    print(name, type(sub_module).__name__)
# 输出示例："" MyLayer / "fc" Linear / "block" Sequential / "block.0" Linear ...
```

- `model.modules()`：遍历包含的所有 `nn.Module` 子类（**包含自身**）
- `model.named_modules()`：同时返回模块对象和**属性路径**（如 `"block.0"`）
- `model.children()` / `model.named_children()`：只遍历**直接**子模块（不递归）

### parameters() / named_parameters() —— 遍历叶子参数

```python title="parameters-example.py"
for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)
# 输出示例：fc.weight [8, 4] True / fc.bias [8] True / scale [8] True
```

- `model.parameters()`：遍历所有 `nn.Parameter` 参数（递归进入子模块）
- `model.named_parameters()`：同时返回参数对象和属性路径
- **与 optimizer 的关系**：`torch.optim.Adam(model.parameters())` 就是靠它拿到全部可学习参数

::::tip
**遍历范围的区别**：`modules()` 遍历的是「模块」（`nn.Module` 实例，包含容器层），`parameters()` 遍历的是「参数」（`nn.Parameter` 实例，只含真正的权重）。前者偏结构、后者偏数值。
::::

### get_submodule() —— 按属性路径获取子模块

`model.get_submodule("block.0.linear")` 按**点分路径**取子模块，是遍历与修改模型结构的常用工具（替换层时先拿父模块）：

```python title="get-submodule-example.py"
# 模型结构: model -> encoder.layers.0.self_attn.q_proj
parent = model.get_submodule("encoder.layers.0.self_attn")  # 拿到父模块
child  = model.get_submodule("encoder.layers.0.self_attn.q_proj")  # 拿到子模块
```

配合 `setattr` 可以实现「替换子模块」（LoRA 注入的核心操作）：

```python title="replace-submodule.py"
parent = model.get_submodule(parent_name)
setattr(parent, child_name, new_layer)   # 用 new_layer 替换掉旧的子模块
```

### state_dict() —— 保存与加载

```python title="save-load.py"
torch.save(model.state_dict(), "model.pt")          # 保存
model.load_state_dict(torch.load("model.pt"))       # 加载
```

`state_dict()` 返回参数与缓冲区的**键值快照**（不包含计算图），是官方推荐的模型持久化方式。加载时 `strict=True`（默认）会检查键是否完全匹配，常用于迁移学习时先加载再替换分类头。

## 小结

- 继承 `nn.Module` + 调用 `super().__init__()` → 获得自动注册与全套管理能力
- `model(x)` 内部走 `__call__` → `forward`，钩子在 `__call__` 层生效
- 用 `named_parameters()` 对接 optimizer、用 `state_dict()` 做持久化、用 `modules()` 做结构遍历
