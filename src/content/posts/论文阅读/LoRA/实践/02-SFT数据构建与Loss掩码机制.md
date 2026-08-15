---
title: SFT 数据构建与 Loss 掩码机制：-100 忽略索引的本质
published: 2026-08-07
description: 深入解析大模型 SFT 微调中的 Loss Masking 机制，手写 PyTorch Dataset，利用 -100 忽略索引实现仅对 Target 区域计算交叉熵损失。
tags: [论文阅读, LoRA, 实战, SFT, LossMasking, 数据管线]
category: 论文阅读
---

# SFT 数据构建与 Loss 掩码机制：-100 忽略索引的本质

> 承接 [[论文阅读/lora/实践/01-手写lora层与动态模块注入]]：模型结构就绪后，最关键的一步是准备训练数据。在指令微调（SFT）中，如果简单把整段文本直接丢给模型算 Loss，会导致模型死记硬背提示词，甚至破坏基础语言能力。本篇手写 `data.py`，彻底讲透 **SFT Loss Masking（`-100` 掩码）** 的底层机制。

---

## 一、 SFT 的本质与提示词模板

自回归语言模型（Causal LM）在预训练时，是在整段语料上进行无差别的 **Next-Token Prediction**（下一个 Token 预测）。但在监督微调（SFT）阶段，数据的语义结构被显式分成了两半：

1. **Prompt 区域（题干）**：包含 System 指令与 User 输入，模型只需要理解并作为条件，**不需要预测**；
2. **Target 区域（答案）**：模型真正需要生成的回答与结束符，**需要重点优化**。

### Qwen 官方 ChatML 格式化

我们遵循 Qwen 官方的 ChatML 标记规范构建 Prompt：

```python title="data.py (Prompt 构建)"
def build_prompt(article: str) -> str:
    """构建符合 Qwen 对话格式的标准 Prompt"""
    return (
        f"<|im_start|>system\n你是一个优秀的新闻编辑，请为以下正文拟定一个简洁精炼的标题。<|im_end|>\n"
        f"<|im_start|>user\n文章正文：{article}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
```

模型的生成目标是紧随其后的新闻标题，并以 `<|im_end|>` 终结：

```
Target 文本: f"{summary}<|im_end|>"
```

---

## 二、 核心机制：为什么必须做 Loss Masking？

在训练自回归模型时，前向计算会输出每个位置的 Logits：$\hat{y} \in \mathbb{R}^{B \times L \times V}$。若直接拿整个序列与输入做交叉熵损失计算：

### 1. 算全量 Loss 的三大严重缺陷
* **容量浪费**：Prompt 区域往往占了序列长度的 $70\% \sim 90\%$。强制模型去预测 Prompt 会把有限的 LoRA 低秩容量耗费在“死记硬背用户输入正文”上；
* **逻辑倒错**：System 提示词和正文是外部给定的输入，模型在实际推理时根本不需要自回归地生成它们；
* **破坏先验**：强行拟合 Prompt 中的固定句式，容易导致模型出现严重的过拟合与复读现象。

### 2. `-100` 的底层数学契约

在 PyTorch 的 `nn.CrossEntropyLoss`（以及 HuggingFace CausalLM 的内部实现）中，默认设置了参数：

$$
\text{ignore\_index} = -100.
$$

其内部计算公式为：

$$
\mathcal{L} = \frac{1}{\sum_{i=1}^N \mathbb{I}(y_i \ne -100)} \sum_{i=1}^N \mathbb{I}(y_i \ne -100) \cdot \left(-\log P(y_i \mid x_{<i})\right).
$$

**只要将某个 Token 的目标 Label 设为 `-100`，PyTorch 在反向传播时就会在计算图里直接将该位置的梯度置为零，不产生任何回传更新**。

```
输入序列 (input_ids):
[<|im_start|>, system, ..., <|im_start|>, assistant, \n, 央行, 宣布, 降准, <|im_end|>, <pad>, <pad>]
                                                          ▲
                                                 从这里开始计算 Loss
目标标签 (labels):
[    -100,      -100,  ...,     -100,      -100,    -100, 央行, 宣布, 降准, <|im_end|>, -100,  -100 ]
 └──────────────────────┬──────────────────────────────┘ └───────────┬─────────────┘ └──────┬──────┘
             Prompt 区域全部填 -100 (不产生梯度)               Target 区域正常计分        Padding 填 -100
```

---

## 三、 数据集实现 (`data.py`)

结合 Tokenizer 编码、Loss Masking、截断与 Padding，手写标准的 PyTorch `Dataset`：

```python title="data.py (完整数据集)"
import torch
from torch.utils.data import Dataset

class TitleDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_len: int = 256):
        self.data = raw_data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 分别编码 Prompt 与 Target（不自动加特殊 token，手动精确拼接）
        prompt_ids = self.tokenizer.encode(
            build_prompt(item["text"]), 
            add_special_tokens=False
        )
        target_ids = self.tokenizer.encode(
            f"{item['summary']}<|im_end|>", 
            add_special_tokens=False
        )

        # 2. 拼接完整的模型输入
        input_ids = prompt_ids + target_ids
        
        # 3. 构造 Loss Mask：Prompt 部分填 -100，Target 部分填真实 token id
        labels = [-100] * len(prompt_ids) + target_ids

        # 4. 长度对齐：超长截断，不足补齐 (Padding)
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            labels = labels[:self.max_len]
        else:
            pad_len = self.max_len - len(input_ids)
            # input_ids 填充 pad_token_id
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            # labels 填充 -100 (Padding 部分同样不参与计算 Loss)
            labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
```

---

## 四、 单元验证与数据对齐检查

编写测试代码验证单个 Sample 的张量输出：

```python title="test_data.py"
# 验证代码片段
dataset = TitleDataset(raw_data, tokenizer, max_len=256)
sample = dataset[0]

print("input_ids shape:", sample["input_ids"].shape)
print("labels shape:   ", sample["labels"].shape)

# 打印非 -100 的有效 Label 数量（即真正参与计算 Loss 的 Token 数）
valid_tokens = (sample["labels"] != -100).sum().item()
print(f"有效计算 Loss 的 Token 数: {valid_tokens} / {len(sample['labels'])}")
```

**输出预期**：
- `input_ids` 与 `labels` 长度严格一致（均为 `max_len=256`）；
- `labels` 前半段与后半 Padding 段全为 `-100`，有效 Token 数正好对应新闻标题与 `<|im_end|>` 的长度。

---

下一篇我们将串联模型与数据，编写纯 PyTorch 训练循环，并实现**零延迟的 LoRA 权重无损合并与实测**：[[论文阅读/lora/实践/03-原生训练循环与权重合并落地]]。
