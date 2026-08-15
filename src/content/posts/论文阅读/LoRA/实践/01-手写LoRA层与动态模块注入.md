---
title: 手写 LoRA 层与动态模块注入：从算子实现到反射热插拔
published: 2026-08-07
description: 纯手工实现 LoRALinear 自定义算子，解决 dtype 对齐与梯度冻结，通过 PyTorch 反射机制实现非侵入式全线性层动态替换与参数量验证。
tags: [论文阅读, LoRA, 实战, PyTorch, 算子实现]
category: 论文阅读
---

# 手写 LoRA 层与动态模块注入：从算子实现到反射热插拔

> 承接 [[论文阅读/lora/实践/00-从零手写lora实战总览]]：要实现 LoRA，核心只有两件事：一是构建一个**双分支复合线性层**，二是把大模型中原本的普通 `nn.Linear` **无缝偷换**成这个复合层。本篇手写 `lora.py` 与 `load.py`，并剖析其中的工程暗坑。

---

## 一、 手写 LoRALinear 复合层 (`lora.py`)

### 1. 物理心智模型

`LoRALinear` 本质上是一个“装饰器/外挂组件”：它包裹住原有的 `nn.Linear`，在输入到来时，左路走冻结的原矩阵计算，右路走 $A \to B$ 的低秩降维升维，最后将两路结果按缩放因子累加输出。

```
                  ┌───────────────────────┐
                  │ 输入向量 x (如 4096维)│
                  └──────────┬────────────┘
                             │ (分流成两路)
               ┌─────────────┴─────────────┐
               ▼                           ▼
      【左路：冻结的原始权重】       【右路：低秩旁路适配器】
       原版 nn.Linear (W0)          x @ A.T @ B.T * (alpha/r)
       (requires_grad = False)     (A: 降维至 r, B: 升维至 dout)
               │                           │
               └─────────────┬─────────────┘
                             │ (逐元素相加)
                             ▼
                  ┌───────────────────────┐
                  │ 输出向量 h (如 4096维)│
                  └───────────────────────┘
```

### 2. 完整算子实现

```python title="lora.py"
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, alpha: int = 16, r: int = 8):
        super().__init__()
        
        # 1. 托管原始线性层，并锁死全部参数（禁止反向传播求梯度）
        self.original_linear = original_linear
        for param in self.original_linear.parameters():
            param.requires_grad = False

        in_features = self.original_linear.in_features
        out_features = self.original_linear.out_features
        
        # 2. 关键：A/B 必须显式对齐被包裹层的 dtype（如 bfloat16）
        param_dtype = self.original_linear.weight.dtype

        # 3. 初始化低秩矩阵 A 和 B
        # 矩阵 A：形状 (r, in_features)，Kaiming 均匀分布初始化
        self.LoRA_A = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(r, in_features, dtype=param_dtype), 
                a=math.sqrt(5)
            )
        )
        # 矩阵 B：形状 (out_features, r)，全零初始化（保证初始时刻增量为 0）
        self.LoRA_B = nn.Parameter(
            torch.zeros(out_features, r, dtype=param_dtype)
        )

        # 4. 缩放系数 alpha / r
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 左路：基础模型前向
        out_base = self.original_linear(x)
        # 右路：低秩旁路前向 (注意 PyTorch 行向量输入的转置契约)
        out_lora = x @ self.LoRA_A.T @ self.LoRA_B.T
        
        return out_base + out_lora * self.scaling
```

---

## 二、 核心工程细节与避坑指南

### 1. 为什么 A/B 矩阵必须显式指定 `dtype`？
* **陷阱**：如果不传 `dtype=param_dtype`，PyTorch 的 `torch.empty` 默认创建 `torch.float32`。
* **后果**：现代大模型（如 Qwen2.5）加载时通常是 `bfloat16`。当 `x`（bf16）与 `LoRA_A`（fp32）做矩阵乘法时，PyTorch 会在 GPU 上直接抛出运行时错误：
  ```
  RuntimeError: expected scalar type BFloat16 but found Float
  ```
* **对策**：必须从 `original_linear.weight.dtype` 获取底层精度，并在构造张量时显式传入。

### 2. 为什么前向乘法写成 `x @ A.T @ B.T`？
* 在数学符号中，列向量变换写作 $W x + \frac{\alpha}{r} B A x$；
* 在 PyTorch 中，数据是 Batch 优先的行向量 $x \in \mathbb{R}^{B \times L \times d_{\text{in}}}$；
* 对行向量右乘矩阵映射：
  $$
  x A^T \in \mathbb{R}^{B \times L \times r}, \qquad (x A^T) B^T \in \mathbb{R}^{B \times L \times d_{\text{out}}}.
  $$
* 这与数学上的 $B A$ 线性变换完全等价，且在计算图构建中无需多余的转置开销。

---

## 三、 动态模块注入器 (`load.py`)

我们不需要手动去修改 Qwen 官方的 Python 模型源码。借助 PyTorch 的反射机制，我们可以遍历模型的命名子模块树，定位目标层并完成动态替换。

### 1. 动态注入实现

```python title="load.py"
import torch
import torch.nn as nn
from lora import LoRALinear

def inject_lora(
    model: nn.Module, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    r=8, 
    alpha=16
):
    """
    遍历 model 中所有子模块，将名字匹配 target_modules 的 nn.Linear 动态替换为 LoRALinear
    """
    # ⚠️ 关键：必须显式转为 list，防止在遍历字典/生成器过程中修改模型结构导致 RuntimeError
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # 获取叶子节点名称，例如 "model.layers.0.self_attn.q_proj" -> "q_proj"
            layer_name = name.split(".")[-1]
            if layer_name in target_modules:
                # 递归拆解父模块路径与子节点属性名
                if "." in name:
                    parent_name, child_name = name.rsplit(".", 1)
                    parent_module = model.get_submodule(parent_name)
                else:
                    parent_module = model
                    child_name = name

                # 用手写的 LoRALinear 包裹原有的 nn.Linear
                lora_layer = LoRALinear(module, r=r, alpha=alpha)

                # 将父模块中的原零件替换为 LoRA 复合零件
                setattr(parent_module, child_name, lora_layer)
                print(f"✅ 成功注入 LoRA 算子: {name}")
```

### 2. 关键陷阱：遍历与设备生命周期

1. **`list(model.named_modules())` 的必要性**：
   `model.named_modules()` 返回的是一个生成器。如果不用 `list()` 提前固化快照，当你在循环体内使用 `setattr` 替换子模块时，会导致迭代器底层的模块树哈希表在遍历中被修改，引发 Python 报错。
2. **设备加载顺序（Device Placement）**：
   - ❌ **错误做法**：`model.to('cuda')` $\to$ `inject_lora(model)`。这样会导致新初始化的 `LoRA_A` 和 `LoRA_B` 默认落在 CPU 上，产生 CPU/GPU 跨设备计算错误。
   -  **正确做法**：在 CPU 上构建模型 $\to$ `inject_lora(model)` 注入完成 $\to$ 最后统一调用 `model.to('cuda')`。

---

## 四、 参数量与显存验证

为了直观验证 LoRA 是否成功冻结了基座并削减了参数量，编写参数量分析函数：

```python title="load.py (续)"
def print_trainable_parameters(model: nn.Module):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    percentage = 100 * trainable_params / all_param
    print("=" * 50)
    print(f"模型总参数量 (Total Params):     {all_param:,}")
    print(f"可训练参数量 (Trainable):       {trainable_params:,}")
    print(f"可训练占比   (Trainable %):      {percentage:.4f}%")
    print("=" * 50)
```

### 运行实测输出（Qwen2.5-0.5B 实测）：

```text
==================================================
模型总参数量 (Total Params):     494,032,768
可训练参数量 (Trainable):       2,031,616
可训练占比   (Trainable %):      0.4112%
==================================================
```

**验证结论**：原模型近 5 亿参数全部被成功冻结，仅有通过全线性层注入的 200 万个低秩参数参与反向传播，参数量直降 **$99.59\%$**。

---

下一篇我们将构建 SFT 训练数据管线，重点解析大模型微调中最为关键的 **Loss Masking（`-100` 掩码）机制**：[[论文阅读/lora/实践/02-sft数据构建与loss掩码机制]]。
