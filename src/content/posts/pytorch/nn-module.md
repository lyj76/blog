---
title: PyTorch 核心：nn.Module 与参数管理
published: 2026-08-04
description: nn.Module 的自动注册机制、forward 与 __call__、常用属性，以及 modules / parameters / state_dict 等迭代器家族
tags: [PyTorch, 深度学习, nn.Module, 神经网络]
category: PyTorch
---

# PyTorch 核心：nn.Module 与参数管理

`nn.Module` 是所有神经网络层与模型的基类。理解它的「自动注册」机制，是掌握 PyTorch 模型管理的一把钥匙。

> 归属：**PyTorch · `torch.nn`** —— 模型构建层（`import torch.nn as nn`）。

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
| `.device` / `.dtype` | ⚠️ **都没有**；设备与精度是**张量的属性**，要通过参数获取（`linear.weight.device` / `linear.weight.dtype`） |

**按参数重建线性层**（LoRA 合并权重时常用）：用已有层的维度/设备/精度创建同构新层：

```python title="recreate-linear.py"
W0 = orig_linear.weight.detach()   # ① 先取出参数张量（W0 是 Tensor）

new_linear = nn.Linear(
    in_features=orig_linear.in_features,       # 输入维度保持一致（模块属性：有）
    out_features=orig_linear.out_features,     # 输出维度保持一致（模块属性：有）
    bias=orig_linear.bias is not None,         # 是否带偏置
    device=W0.device,                          # ② 设备：从张量 W0 上取（模块没有 .device）
    dtype=W0.dtype                             # ③ 精度：从张量 W0 上取（模块没有 .dtype）
)
```

::::warning
**`.dtype` / `.device` 在张量上，不在模块上**：`nn.Linear` 本身**没有** `.dtype` 和 `.device` 属性（`orig_linear.dtype` 会报 `AttributeError`）。它们属于**参数张量**——正确写法是 `orig_linear.weight.dtype`。示例里写成 `W0.dtype`，因为 `W0 = orig_linear.weight.detach()` 是张量。模块能直接访问的只有维度这类配置属性（`in_features` / `out_features`）。
::::

## 4. train() / eval() 到底做了什么

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

## 5. 几个重要的迭代器

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

### 实战原则：遍历 + 按名匹配 + 修改（LoRA 注入必读）

**原则一：名字是点分属性路径，遍历会走到每个节点（含最内层叶子）**

`named_modules()` 产出的 `name` 是**从根到该模块的属性路径**（如 `layers.0.self_attn.q_proj`），而且**树上每个有名字的模块都是一个节点**——包括被替换进去的新模块的内部。比如把 `q_proj` 换成 `LoRALinear`（里面装着原始 `Linear` 作为 `W_new`）：

```
layers.0.self_attn.q_proj        → LoRALinear（盒子）
layers.0.self_attn.q_proj.W_new  → Linear（盒子里的原始层）
```

**两个都会被遍历到**——`q_proj.W_new` 是"盒子里的叶子"，但它同样是一个有名字的模块节点。

**原则二：按名判断"要不要处理"必须精确匹配，不能用子串 `in`**

```python title="match-precise.py"
# ❌ 子串匹配：q_proj.W_new 也含 "q_proj" → 把"盒子里的叶子"误判成"待处理的盒子"
if "q_proj" in name:

# ✅ 精确叶子匹配：只认"以 .q_proj 结尾"的节点
if name.endswith(".q_proj"):
```

子串 `in` 的三个坑（全部可复现）：

| 坑 | 触发条件 | 结果 |
| --- | --- | --- |
| 误匹配叶子 | 遍历到 `...q_proj.W_new`（含 `q_proj`） | 把原始 `Linear` 当 `LoRALinear` 取 `.W_new` → `'Linear' object has no attribute 'W_new'` |
| 链式双匹配 | 名字 `...q_proj.k_proj` 同时含两个目标词 | 内层循环匹配两次，`setattr` 互相覆盖 |
| 顶层裸名 | 目标词就在根上、名字没点 | `rsplit(".", 1)` 拆不出父名 → `ValueError` |

**原则三：遍历中改结构，生成器是懒的，内容会变**

`named_modules()` 返回的是**生成器**（惰性）——遍历过程中 `setattr` 替换模块，生成器"看到的结构"和替换前不一样（和 [[迭代与可变集合]] 讲的"边遍历边改"是同一个坑的 module 版本）。**改前先固化快照**：

```python title="snapshot-first.py"
for name, module in list(model.named_modules()):   # list() 先全部取出来
    ...   # 再随便改
```

**LoRA 注入的标准安全姿势**（三条原则合起来）：

```python title="safe-inject.py"
for name, module in list(model.named_modules()):          # ① 快照
    if name.endswith(".q_proj") or name.endswith(".v_proj"):   # ② 精确叶子匹配
        if "." in name:                                   # ③ 顶层裸名特判
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent, child_name = model, name
        setattr(parent, child_name, LoRALinear(module, r, alpha))
```

- ① `list()` 快照：遍历期间随便改结构，不依赖生成器状态
- ② `endswith(".xxx")`：只认叶子本身，`q_proj.W_new` 不会误入
- ③ `"." in name` 特判：目标在根上时父模块就是 `model` 自己

**按名字匹配 vs 按类型匹配**：上面用 `endswith` 是按**名字**选目标——适合"只替换特定层"（如只对 q/v 注入）。另一种思路是**按类型**选：`isinstance(module, nn.Linear)` 匹配所有线性层，完全绕开名字匹配的坑（`isinstance` 原理见 [[类与属性访问#6-isinstance判断类型含子类]]）：

```python title="inject-by-type.py"
for name, module in list(model.named_modules()):
    if isinstance(module, nn.Linear):      # 所有 Linear 都换（天然只命中叶子）
        parent_name, child_name = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), child_name, LoRALinear(module, r, alpha))
```

- **按类型**（isinstance）：适合"全部替换同类层"——不用管名字，`isinstance` 只命中真正的 `nn.Linear`，`q_proj.W_new` 这类"盒子里的叶子"问题根本不存在（新替换进去的 `LoRALinear` 不是 `nn.Linear` 子类，不会二次命中）
- **按名字**（endswith）：适合"只动特定层"——精确后缀匹配

::::warning
**"运气型正确"要不得**：子串 `in` + 直接遍历改结构，在 Qwen/Llama 这种"目标全是深层叶子、名字互不包含"的架构上恰好能跑通——换个架构（目标词是容器模块、或名字拼一起）立刻炸。匹配用精确后缀、修改前先快照，是唯一稳的写法。
::::

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
