---
title: CausalLM 函数model()一次前向和SFT 
published: 2026-08-05
description: forward 参数契约（关键字调用陷阱）、CausalLMOutputWithPast 字段表、labels 触发内部交叉熵的完整流程（logits 对齐 → 展平 → ignore_index 求平均）、shift 一位的位置对齐机制（位置 n 预测 n+1，末位输出丢弃）、-100 在 SFT 里的掩码用法、logits[batch, seq, vocab] 作为 seq 个并列分类任务的视角、train/eval、**inputs 解包
tags: [HuggingFace, transformers, CausalLM, 前向传播, loss]
category: HuggingFace
---

# CausalLM 函数model()一次前向和SFT 

调用 `model(input_ids=..., labels=...)` 一次前向传播后，返回的不是张量，而是一个带多个字段的数据包。
> 归属：**`transformers` 库**（`from transformers import AutoModelForCausalLM`）——HF 的模型库。

## 1. model()中forward一次前向的输入参数 

所有 HF CausalLM 模型遵循同一签名，关键参数如下：
    ·
| 参数 | 含义 | 形状 | 训练时传？ |
| --- | --- | --- | --- |
| `input_ids` | 输入 token 序列 | `[batch, seq]` | 必传 |
| `attention_mask` | 01的填充位。$M=-\infty$使得查询权重pad位$softmax (\frac{QK^T}{\sqrt{d}}+M)=0$，则其他向量都不会加入这个向量意义 | `[batch, seq]` | 通常传 |
| `position_ids` | 每个位置的下标编号 | `[batch, seq]` | 少用（默认 |
| `past_key_values` | 历史 KV 缓存（生成加速，在generate中才有意义） | 元组 | 生成时用 |
| `labels` | 目标 token 大小和输入完全一样。通过label向左移位来对齐输入计算loss。比如input[0]对应label[1]；input[0:1]对应label[2]；input[0:n]对应 label[n]这个才不是原句子| `[batch, seq]` | **训练时传** |

`input_ids` 是 tokenizer 产出的 token id 序列（来源见 [[tokenizer-类型契约]]）



两种调用形态对应两种用法：不传 `labels` 是推理形态（只算 logits，见第 6 节）；传 `labels` 是训练形态（内部自动算交叉熵，返回带 `loss` 的对象，见第 3 节）。

## 2. CausalLM：返回类型

模型直接调用Model() forward返回的不是张量，而是 `CausalLMOutputWithPast` 对象——一个把多个字段打包在一起的数据类：



CausalLM：返回类型对象继承自 `ModelOutput`——既能当属性访问（`outputs.loss`），也能当字典访问（`outputs["loss"]`），两种写法等价。字段表如下：

| 字段 | 类型 | 何时有值 大小|含义|
| --- | --- | --- |---|
| `loss` | `Optional[torch.Tensor]` | **传了 `labels` 才有**，否则为 `None` |交叉熵，每一个输出vocab向量P和label向量Y,计算Yi标签每一个维度的加权重要性的 -logPi输出信息量（模型惊讶程度比如一个true量给了0.01，则-log(0.01)非常大同时权会进一步放大，所以要最小化这个）$\arg\min_{\theta} \sum_{i} Y_i \cdot \big(-\log P(x_i ; \theta)\big)$  具体看下面|
| `logits` | `torch.Tensor` | 永远有，`[batch, seq, vocab_size]` | 也就是每一个输出位置都有一个归一vocab向量    |
| `past_key_values` | `Optional[Tuple]` | `use_cache=True` 时有，KV 缓存 | 重复词的key和value向量   |
| `hidden_states` | `Optional[Tuple[Tensor]]` | `output_hidden_states=True` 时有 | 每一层隐藏输出Tensor然后打包tuple |
| `attentions` | `Optional[Tuple[Tensor]]` | `output_attentions=True` 时有 | 每一层的$softmax(\frac{QK^T}{\sqrt{d}})$ att矩阵打包为tuple  |

`loss` 的类型标注 `Optional[torch.Tensor]` 拆开看：

| 部分 | 含义 |
| --- | --- |
| `torch.Tensor` | **张量**：带形状、dtype 的多维数组（结构详见 [[tensor-and-parameter]]） |
| `Optional[X]` | Python 类型注解，= `X \| None`：**这个字段可能有值，也可能是空（`None`）** |



```python title="loss-guard.py"
if outputs.loss is not None:      # 训练时才有值
    loss = outputs.loss.item()    # .item()：张量 → Python 浮点数
```



## 3. Loss：输出token和label的交叉熵


`loss` 度量的是：**模型在每个位置预测的"下一个 token"，和真答案label差多远**。

交叉熵来衡量的loss。
### 3.1模型输出的向量组
模型输出的是vocab向量$\vec P$向量组，维度$$batch\times seq\times \dim v ob $$

也就是和inputs token一样的规模，每一个位置都有一个预测向量

### 3.2 label向量组，和token代表序列
label向量组$\vec Y$,维度也是$$batch\times seq \times dim vob$$每一个都是一个onehot向量。所以label token 中，一般用序列 [i:i in range(0:50000)]来代表这个label的onehot向量组。

也就是比如token：56 对应的就是onehot向量[0000...1000] 1的就是index55位置上面


### 3.3 计算交叉熵
计算Yi标签每一个维度数值加权重要性的 -logPi输出信息量

信息量是（模型惊讶程度。比如一个true量给了0.01，则-log(0.01)非常大同时权会进一步放大，所以要最小化这个）


$$\arg\min_{\theta} \sum_{i} Y_i \cdot \big(-\log P(x_i ; \theta)\big)$$


但是因为label Y是onehot而且01的向量，则这个求和可以化简
$$\arg \min_{\theta}(-\log P(token:\theta))$$

`labels` 是和 `input_ids` **同形状**的张量（`[batch, seq]`），内容是目标 token id。拿到 `loss` 后直接 `loss.backward()` 就能反向传播，不需要自己写损失函数。




::::tip
**训练循环的标准写法**：`loss = model(input_ids=..., labels=...).loss`，一行搞定前向 + 算损失。然后 `loss.backward()` + `optimizer.step()`（完整流程见 [[自动求导与梯度]]）。
::::

## 4. shift 和输出逻辑

### 4.1 完整链路：前向 → 对齐 → 比较

shift 的含义：目标 token 大小和输入完全一样。通过label向左移位来对齐输入计算loss。

比如
```
input[0]对应label[1]；

input[0:1]对应label[2]；

... 
input[0:n]对应 label[n]这个才不是原句子

```








## 5.SFT和训练逻辑


### 5.1 非训练标识
由前面可知label向量的每一位都是token标量，代表了一个onehot向量。

则如果label中的某位是-100的话，意味着映射不到一个有效的onehot vocab向量 

则对应的；label向量就是[0000..00]全0向量

这些位置计算出来的loss也就是0


### 5.2 SFT过程

比如我需要训练的是：输入：你是谁？

输出：我是ai

拼接为["你", "是", "谁", "？", "我", "是", "AI"]

对应的input_ids: [ 12, 45, 99, 14, 66, 45, 88 ]

可以设置label_token为labels: [-100, -100, -100, -100, 66, 45, 88 ]

也就是
| 输入|预测|label|loss|
|---|---|---|--|
|你| 是|-100| 0|
|你是|谁|-100|0|
|你是谁|？|-100|0|
|你是谁？|我|66|loss3|
|你是谁？我|是|45|loss4|
|你是谁？我是|AI|88|loss5|


计算总的loss然后backward
$$Loss=loss_3+loss_4+loss_5$$

利用计算图backward然后更新

