---
title: HuggingFace 语法⑦：LoRA 注入 —— 替换模型层
published: 2026-08-04
description: list(model.named_modules())、isinstance 过滤、rsplit 定位父模块、setattr 替换 —— 把目标 nn.Linear 换成 LoRALinear 的完整套路
tags: [HuggingFace, LoRA, 语法, 模型改造]
category: HuggingFace
---

# HuggingFace 语法⑦：LoRA 注入 —— 替换模型层

LoRA 注入 = 遍历模型结构，把目标 `nn.Linear` 层**替换**成 `LoRALinear` 包裹层。这是一套组合拳语法。

## 1. 完整注入函数

```python title="inject-lora.py"
def inject_lora(model, target_modules=["q_proj", "v_proj"], r=8, alpha=16):
    # 注意：必须转成 list！遍历过程中会修改模型结构
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # 取层名（路径最后一段），判断是否目标层
            layer_name = name.split(".")[-1]
            if layer_name in target_modules:
                # 定位父模块和子模块名
                if "." in name:
                    parent_name, child_name = name.rsplit(".", 1)
                    parent_module = model.get_submodule(parent_name)
                else:
                    parent_module = model
                    child_name = name

                # 用 LoRA 层包裹原层
                lora_layer = LoRALinear(module, r=r, alpha=alpha)

                # 替换父模块中的原层
                setattr(parent_module, child_name, lora_layer)
```

## 2. 逐个语法拆解

### `list(model.named_modules())`

```python title="named-modules.py"
for name, module in model.named_modules():   # 遍历所有子模块（含自身）
    name     # "model.layers.0.self_attn.q_proj"
    module   # 对应的 nn.Linear / nn.Sequential 等
```

- `named_modules()` 返回 `(路径, 模块)` 迭代器，**递归遍历**所有层
- **必须 `list()` 转快照**：循环里 `setattr` 改了模型结构，遍历一个正在变化的迭代器会出问题

### `isinstance(module, nn.Linear)`

过滤出真正的线性层（`nn.Embedding`、`nn.LayerNorm` 等不是 `Linear` 的会被跳过）。

### `name.split(".")[-1]`

取路径最后一段 = 层名（`q_proj`），用它和 `target_modules` 列表比对。

### `name.rsplit(".", 1)` + `get_submodule` + `setattr`

```python title="locate-parent.py"
parent_name, child_name = name.rsplit(".", 1)   # 拆成 父路径 + 子名
parent_module = model.get_submodule(parent_name) # 按路径取父模块
setattr(parent_module, child_name, lora_layer)   # 替换子模块
```

三步是「替换任意层」的固定套路：**定位父模块 → setattr 换子模块**。

## 3. 注入后模型结构的变化

```python title="before-after.py"
# 注入前
model.model.layers.0.self_attn.q_proj     # nn.Linear

# 注入后（q_proj 变成了包裹层）
model.model.layers.0.self_attn.q_proj     # LoRALinear
    ├── original_linear                    # 原来的 nn.Linear（已冻结）
    ├── LoRA_A                             # 可训练
    └── LoRA_B                             # 可训练
```

::::note
**为什么是「包裹」而不是改权重**：`LoRALinear` 内部持有原层 + A/B 两个小矩阵，前向时 `out = 原层(x) + 旁路(x)`。原层冻结不动，只训练 A/B——这就是「适配器」的含义。
::::

## 4. 注入后的冻结（组合拳）

```python title="freeze-after-inject.py"
# 除了 LoRA 的 A/B，其余全部冻结
for name, param in model.named_parameters():
    if "lora" not in name.lower():
        param.requires_grad = False
```

`LoRALinear.__init__` 内部也会冻结自己包裹的原层：

```python title="freeze-inside.py"
for para in self.original_linear.parameters():
    para.requires_grad = False
```

## 5. 与 `nn.ModuleList` 等容器的区别

- `named_modules()` 能遍历到 `nn.Sequential` / `nn.ModuleList` **内部**的层（路径含 `.0.`、`.1.`）
- 只遍历**注册过的**模块——裸 Python list 里的模块不会被遍历到

## 小结

| 语法 | 作用 |
| --- | --- |
| `list(model.named_modules())` | 结构遍历（快照） |
| `isinstance(module, nn.Linear)` | 类型过滤 |
| `name.split(".")[-1]` | 取层名 |
| `name.rsplit(".", 1)` | 拆父路径 + 子名 |
| `model.get_submodule(parent)` | 按路径取父模块 |
| `setattr(parent, child, new)` | 替换子模块 |
| `param.requires_grad = False` | 冻结原层 |
