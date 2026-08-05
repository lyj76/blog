---
title: HuggingFace 语法⑧：LoRA 权重合并与保存
published: 2026-08-04
description: detach 取数据、B@A 计算增量、copy_ 写入、setattr 还原标准层、getattr 处理 tie_word_embeddings、save_pretrained 完整流程
tags: [HuggingFace, LoRA, 语法, 权重合并]
category: HuggingFace
---

# HuggingFace 语法⑧：LoRA 权重合并与保存

训练完的 LoRA 是「原模型 + A/B 旁路」。合并 = 把旁路增量写回原权重，并**还原成标准 `nn.Linear`**，方便部署。

## 1. 数学原理

训练时前向是：

$$y = W_0 x + (B \cdot A \cdot x) \cdot \frac{\alpha}{r}$$

合并后等价为一个权重：

$$W_{new} = W_0 + (B \cdot A) \cdot \frac{\alpha}{r}$$

合并前是「两层结构」，合并后是「一个权重」，前向完全等价但更简单、更快。

## 2. 完整合并函数

```python title="merge-lora.py"
def merge_lora_into_model(model):
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            orig_linear = module.original_linear

            # 1. 算合并权重：W_new = W0 + (B @ A) * scalar
            W0 = orig_linear.weight.detach()                    # 拿数据，切断梯度
            delta_W = (module.LoRA_B.detach() @ module.LoRA_A.detach()) * module.scalar
            merged_weight = W0 + delta_W

            # 2. 新建标准 Linear 层（同维度/设备/精度）
            new_linear = nn.Linear(
                in_features=orig_linear.in_features,
                out_features=orig_linear.out_features,
                bias=orig_linear.bias is not None,
                device=W0.device,
                dtype=W0.dtype
            )

            # 3. 写入合并权重
            new_linear.weight.data.copy_(merged_weight)
            if orig_linear.bias is not None:
                new_linear.bias.data.copy_(orig_linear.bias.detach())

            # 4. 替换回标准层
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent_module = model.get_submodule(parent_name)
                setattr(parent_module, child_name, new_linear)
            else:
                setattr(model, name, new_linear)

    # 5. 处理词嵌入绑定
    if getattr(model.config, "tie_word_embeddings", False):
        model.config.tie_word_embeddings = False
```

## 3. 逐个语法拆解

### `detach()`：切断计算图

```python title="detach-merge.py"
W0 = orig_linear.weight.detach()   # 只拿数据，不再追踪梯度
```

合并是「数值操作」，不是「训练操作」——`detach()` 防止这些运算被记进计算图。

### `@` 矩阵乘 + `*` 标量

```python title="delta-calc.py"
delta_W = (module.LoRA_B.detach() @ module.LoRA_A.detach()) * module.scalar
# 形状: [out, r] @ [r, in] = [out, in]，再乘标量 α/r
```

### `nn.Linear(...)` 同构重建 + `copy_` 写入

```python title="copy-into.py"
new_linear = nn.Linear(
    in_features=orig_linear.in_features,    # 维度一致
    out_features=orig_linear.out_features,
    bias=orig_linear.bias is not None,      # 原层有无偏置
    device=W0.device,                       # 设备一致
    dtype=W0.dtype                          # 精度一致
)
new_linear.weight.data.copy_(merged_weight)   # 原地写入合并权重
```

`copy_`（原地复制）保证 `new_linear.weight` 对象不变、只改数值。

### `rsplit` + `get_submodule` + `setattr`：替换回标准层

和注入时完全相同的套路，只是把 `LoRALinear` 换回 `nn.Linear`。

### `getattr(config, "tie_word_embeddings", False)`

```python title="tie-embeddings.py"
getattr(model.config, "tie_word_embeddings", False)
```

- 有些模型「词嵌入 = 输出层」共享权重（tied embeddings），合并后可能不兼容
- `getattr(对象, 属性, 默认值)`：属性存在用属性，不存在用默认值，**不报错**
- 存在就关掉：`model.config.tie_word_embeddings = False`

## 4. 合并后保存

```python title="save-merged.py"
model.save_pretrained(merged_dir)        # 保存合并后的模型
tokenizer.save_pretrained(merged_dir)    # 分词器成对保存
print(f"模型已保存到 {merged_dir}")
```

合并后的模型是**标准 Transformer**（没有 LoRALinear 了），可以当普通模型加载、部署、继续全量微调。

## 5. 为什么 `list()` 又出现

```python title="why-list.py"
for name, module in list(model.named_modules()):
```

合并也在遍历中**修改模型结构**（`setattr` 换层），所以同样需要 `list()` 快照。

## 小结

| 语法 | 作用 |
| --- | --- |
| `W.detach()` | 数值操作时切断梯度 |
| `B.detach() @ A.detach()` | 算 LoRA 增量 ΔW |
| `nn.Linear(同参重建)` | 造标准层 |
| `weight.data.copy_(新权重)` | 原地写入 |
| `rsplit/get_submodule/setattr` | 换回标准层 |
| `getattr(config, "tie_word_embeddings", False)` | 安全读配置 |
| `model.save_pretrained(dir)` | 保存合并结果 |
