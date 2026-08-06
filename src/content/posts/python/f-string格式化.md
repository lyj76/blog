---
title: Python f-string 格式化：日志输出的正确姿势
published: 2026-08-05
description: f-string 嵌入变量与任意表达式、格式说明符 :.4f 保留小数、:, 千分位、:.2% 百分比、:.2e 科学计数、多行日志模板拼接、print 的 flush=True 不缓冲
tags: [Python, f-string, 格式化, 日志]
category: Python
---

# Python f-string 格式化：日志输出的正确姿势

训练脚本里打印 loss、参数量、进度，几乎都用 f-string。这篇讲清楚 `{}` 里能放什么、`:` 后面的格式说明符怎么写，以及 `flush=True` 在服务端日志里的作用。

> 归属：**Python 标准库** —— 语言本身。

## 1. 基础：`f"..."` + `{变量}`

```python title="fstring-basic.py"
epoch, step, loss = 2, 20, 1.2345678

print(f"Epoch [{epoch}/3] | Step [{step}] | Loss: {loss}")
# 输出: Epoch [2/3] | Step [20] | Loss: 1.2345678
```

- 字符串前加 `f`，`{}` 里直接放变量或任意表达式
- 比 `"Epoch [" + str(epoch) + "/3]"` 拼接干净，比 C 的 `sprintf` 安全
- `{}` 里可以是变量、算式、函数调用、属性访问

::::note
**C++ 类比**：f-string 类似 `printf` 的格式串，但类型安全、不用记 `%d %f %s`。`{loss}` 自动按变量类型输出，不会因为类型符写错崩掉。
::::

## 2. `{}` 里放任意表达式

```python title="expression.py"
total = sum(p.numel() for p in model.parameters())

print(f"参数量: {total / 1e6:.2f}M")     # 参数量: 7.68M
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"形状: {tensor.shape}")           # 直接打印属性
```

- `{}` 不限于变量，任何合法 Python 表达式都能放
- `{total / 1e6:.2f}` 先算除法再格式化，`:` 后是格式说明符
- 打印张量形状、设备名、函数返回值都很方便

## 3. 格式说明符：`:.4f` / `:,` / `:.2%`

```python title="format-spec.py"
loss = 1.2345678
n = 1234567
rate = 0.8923

print(f"loss: {loss:.4f}")    # loss: 1.2346 —— 4 位小数
print(f"n: {n:,}")            # n: 1,234,567 —— 千分位
print(f"acc: {rate:.2%}")     # acc: 89.23% —— 百分比
print(f"loss: {loss:.2e}")   # loss: 1.23e+00 —— 科学计数
```

- `:` 后跟格式说明符，控制输出样式
- `.4f` = 浮点数保留 4 位小数；`.2f` = 2 位
- `,` = 千分位分隔符，大数字易读
- `.2%` = 百分比，自动乘 100 加 `%`
- `.2e` = 科学计数法

| 说明符 | 效果 | 示例 |
| --- | --- | --- |
| `:.4f` | 4 位小数 | `1.2346` |
| `:,` | 千分位 | `1,234,567` |
| `:.2%` | 百分比 | `89.23%` |
| `:.2e` | 科学计数 | `1.23e+00` |

## 4. 多行日志模板

```python title="multi-line.py"
print(
    f"Epoch [{epoch}/{epochs}] "
    f"Step [{step}/{total_steps}] "
    f"Loss: {loss:.4f} "
    f"LR: {lr:.2e}"
)
```

- 多个 `f"..."` 字符串字面量挨着写，Python 自动拼接成一个
- 每行一个字段，加字段只改一行，diff 干净
- 比用 `+` 拼接省事，比单行超长字符串易读

## 5. `print(..., flush=True)`：不缓冲

```python title="flush.py"
import sys

for step, loss in enumerate(losses):
    print(f"step {step}: loss={loss:.4f}", flush=True)
    # 服务端日志立刻刷出，不卡在缓冲区
```

- `print` 默认带缓冲，输出可能攒一批才显示
- `flush=True` 强制立刻写出，服务端训练日志能实时看到进度
- 等价于 `print(..., end="\n", file=sys.stdout, flush=True)`

::::tip
**什么时候要 flush**：本地终端一般不用，标准输出够快；SSH 远程跑训练、Docker 容器、nohup 重定向到文件时，缓冲会让日志延迟几十秒甚至更久，加 `flush=True` 才能实时跟踪。
::::

## 小结

| 语法 | 作用 |
| --- | --- |
| `f"{var}"` | 嵌入变量 |
| `f"{expr}"` | 嵌入任意表达式 |
| `{x:.4f}` | 4 位小数 |
| `{n:,}` | 千分位 |
| `{x:.2%}` | 百分比 |
| `{x:.2e}` | 科学计数 |
| 多个 `f"..."` 挨着 | 自动拼接 |
| `print(..., flush=True)` | 强制立刻输出 |
