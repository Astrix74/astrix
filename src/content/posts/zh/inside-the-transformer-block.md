---
title: Transformer 块的内部：残差流、MLP，以及参数都去了哪里
description: 注意力、MLP 和归一化如何围绕残差流组合在一起，为什么一个块大约有 12d² 个参数，以及怎样随手粗算出一个模型的训练计算量。
date: 2026-10-22T15:11:00Z
tags: [transformers, architecture, math]
series: transformers
chapter: 2
references: [vaswani2017, elhage2021, xiong2020, hendrycks2016, shazeer2020, kaplan2020, brown2020]
---

[第 1 章](/zh/writing/attention-as-a-differentiable-lookup/)构建了注意力，也就是让 token 能够互相读取信息的那个操作。Transformer 由一模一样的**块**（block）堆叠而成，而注意力只占每个块的一半。本章组装出一个完整的块，介绍“残差流”这个让整个架构更容易思考的视角，然后算一些账。读完之后，你应该能看着一个模型的形状（层数和宽度），估算出它的参数量和训练计算量，误差在百分之几以内。

## 块

现代 Transformer 块有两个子层，每个子层都以同样的方式包装起来：

$$
\begin{aligned}
x &\leftarrow x + \operatorname{Attention}\big(\operatorname{LN}(x)\big) \\
x &\leftarrow x + \operatorname{MLP}\big(\operatorname{LN}(x)\big)
\end{aligned}
$$

$\operatorname{LN}$ 是层归一化（layer normalization），两次加法都是残差连接，这些都出自[《深度学习基础》系列](/zh/writing/making-depth-trainable/)。第一个块的输入是 token 嵌入；最后一个块的输出还要再经过一次归一化和一个线性层，得到词表上的 logits。

### 归一化放在哪里

最初的 Transformer 把归一化放在每次残差加法之后：$x \leftarrow \operatorname{LN}(x + \operatorname{Sublayer}(x))$。这种做法叫作 **Post-LN**。此后的大多数模型都把它挪进了分支内部，也就是上面的写法。这种做法叫作 **Pre-LN**，这样一来，从输入到输出的残差路径上就只剩下加法。

Xiong 等人（2020）分析了两者的区别。采用 Post-LN 时，在初始化阶段“靠近输出层的参数，其梯度的期望值很大”，这使得大学习率会导致训练不稳定。这正是 Post-LN Transformer 需要学习率**预热**（warm-up）的原因，也就是在训练开始时让学习率缓慢爬升。他们发现，不做预热的 Pre-LN Transformer“能取得与基线相当的结果”。这又是残差网络的那个[恒等路径论证](/zh/writing/making-depth-trainable/)：保持主路径干净，让各个分支往上叠加修正。

## 残差流

有一种看待这个块的方式，能让整个架构更容易推理。跟着单个 token 的向量 $x$ 走遍整个网络：它从未被替换，每个子层都只是在它上面做加法。所以，不妨把 $x$ 看成一条贯穿模型全部深度的共享通道。每个子层都从中读取（经过一次归一化和一次线性投影），再写回去（把自己的输出加上去）。

Elhage 等人（2021）把这条通道称为**残差流**（residual stream），并围绕它建立了一套解读 Transformer 的框架。这个视角有两个推论，值得一开始就掌握：

- **注意力是唯一在 token 之间搬运信息的操作**。每个头从其他位置的流中读取，再写入当前位置的流。块里的其他一切都是对每个 token 独立起作用的。
- **层与层之间通过流来通信，而不是直接相连**。第 3 层的一个头可以读取第 1 层某个头写下的内容，因为它们用的是同一条通道。一层看到的不只是“上一层的输出”，而是迄今为止写入的所有内容之和。

## MLP

第二个子层是一个小型的两层网络，分别作用于每个 token：

$$
\operatorname{MLP}(x) = \phi(x W_1 + b_1)\, W_2 + b_2
$$

$W_1$ 把宽度从 $d$ 扩展到 $4d$ 的隐藏宽度，$\phi$ 是一个非线性函数，$W_2$ 再投影回 $d$。最初的 Transformer 用 ReLU 作为 $\phi$；后来的许多模型改用 **GELU**（Hendrycks 和 Gimpel，2016），这是一条光滑曲线，它用标准高斯变量小于输入值的概率给每个输入加权，即 $x \, \Phi(x)$。许多近期的模型更进一步，采用门控（gated）MLP。一个流行的选择是 SwiGLU，出自 Shazeer 2020 年对各种门控变体所做的比较：其中一个投影与另一个经过激活的投影逐元素相乘。门控 MLP 有三个权重矩阵而不是两个，所以隐藏宽度通常要削减约三分之一，以保持参数量不变。

如果说注意力是 token 之间交换信息的地方，那么 MLP 就是每个 token 的表示得到加工的地方：特征在这里被组合、变换，再写回流中。下面的计数会表明，它也是大部分参数所在的地方。

## 用代码写一个

借助第 1 章的 `causal_self_attention` 函数，一个完整的 Pre-LN 块大约只要十五行。参数按名称列出，这样我们就能把它们精确地数清楚：

```python
import numpy as np
from attention import causal_self_attention   # from chapter 1

def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

def gelu(x):                                   # the tanh approximation of GELU
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def block_shapes(d):
    """Every parameter in one pre-LN transformer block, by name."""
    return {
        "Wq": (d, d), "Wk": (d, d), "Wv": (d, d), "Wo": (d, d),   # attention
        "W1": (d, 4 * d), "b1": (4 * d,), "W2": (4 * d, d), "b2": (d,),  # MLP
        "ln1_g": (d,), "ln1_b": (d,), "ln2_g": (d,), "ln2_b": (d,),  # two LayerNorms
    }

def block(x, p, n_heads):
    attn, _ = causal_self_attention(layer_norm(x, p["ln1_g"], p["ln1_b"]),
                                    p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads)
    x = x + attn                               # attention reads the stream, writes back
    h = gelu(layer_norm(x, p["ln2_g"], p["ln2_b"]) @ p["W1"] + p["b1"])
    return x + h @ p["W2"] + p["b2"]           # so does the MLP, one token at a time

# A small block, run once.
rng = np.random.default_rng(0)
d, n_heads, n_tokens = 64, 4, 10
shapes = block_shapes(d)
p = {name: rng.normal(size=s) / np.sqrt(s[0]) for name, s in shapes.items()}
for name in ["b1", "b2", "ln1_b", "ln2_b"]:
    p[name] = np.zeros_like(p[name])
for name in ["ln1_g", "ln2_g"]:
    p[name] = np.ones_like(p[name])
x = rng.normal(size=(n_tokens, d))
print("output shape:", block(x, p, n_heads).shape)

# Parameter count against the 12 d^2 rule.
for d in [768, 4096, 12288]:
    exact = sum(np.prod(shape) for shape in block_shapes(d).values())
    print(f"d={d:<6} parameters per block {exact:>13,}   12*d^2 = {12 * d * d:>13,}")
```

```text
output shape: (10, 64)
d=768    parameters per block     7,084,800   12*d^2 =     7,077,888
d=4096   parameters per block   201,363,456   12*d^2 =   201,326,592
d=12288  parameters per block 1,812,049,920   12*d^2 = 1,811,939,328
```

输出与输入形状相同，块之所以能堆叠，靠的正是这一点：每个块不过是又一轮对流的读取和写入。

## 参数都去了哪里

上面的精确计数与一条简单规则相差不到 0.1%，值得看看这条规则从何而来：

- **注意力**：四个 $d \times d$ 矩阵（$W_Q, W_K, W_V, W_O$），所以是 $4d^2$。拆分成多个头并不改变这个数，各个头只是把列分了分。
- **MLP**：向上投影 $d \times 4d$，向下投影 $4d \times d$，所以是 $8d^2$。
- **偏置和归一化**：$d$ 的几倍而已，与 $d^2$ 相比可以忽略。

$$
N_\text{block} \approx 12 d^2, \qquad N \approx 12\, L\, d^2
$$

其中 $L$ 是模型的块数，不计嵌入矩阵。三分之二的参数都在 MLP 里。Kaplan 等人在他们 2020 年那篇缩放定律论文的表 1 中，用的也是这个近似：$N \approx 12\, n_\text{layer}\, d_\text{model}^2$。

拿一个真实的模型验证一下。根据 GPT-3 论文（Brown 等人，2020）的表 2.1，它最大的版本有 96 层，宽度为 12,288。于是 $12 \times 96 \times 12288^2 \approx 173.9$ 十亿，而论文报告的是 1750 亿。差额的大部分是嵌入，偏置和归一化参数再补上一小部分。知道一个模型的深度和宽度，就足以知道它的规模。

## 计算量都去了哪里

同样的算术也能算出计算量。一个长度为 $d$ 的向量乘以一个 $d \times k$ 矩阵，需要 $dk$ 次乘法和差不多同样多次加法，所以大约是 $2dk$ 次浮点运算（FLOPs）。在前向传播中，权重矩阵里的每个参数对每个 token 都恰好用到一次：一次乘法加一次加法。所以：

$$
C_\text{forward} \approx 2N \quad \text{FLOPs per token}
$$

反向传播的计算量大约是前向的两倍，因为它既要计算对激活值的梯度，也要计算对权重的梯度。因此，训练时每个 token 大约要花 $6N$ FLOPs，而在 $D$ 个 token 上完成一次完整训练的开销是

$$
C_\text{train} \approx 6\, N\, D
$$

GPT-3 在 3000 亿个 token 上训练，算下来是 $6 \times 175 \times 10^9 \times 300 \times 10^9 \approx 3.15 \times 10^{23}$ FLOPs。凡是讨论训练一个模型要花多少成本，这个公式都是起点；第 4 章的[缩放定律](/zh/writing/scaling-laws/)也从它出发。

### 那注意力本身呢？

$2N$ 这个估计只算了权重矩阵。注意力还做了一些不涉及参数的工作：用每个查询给之前各个 token 的键打分，再对它们的值求平均。对于一个关注之前 $n$ 个位置的 token，打分大约需要 $2nd$ FLOPs，求平均也要 $2nd$。在长度为 $n_\text{ctx}$ 的因果序列上取平均，每个 token 平均只关注一半的上下文，于是每层每个 token 大约是 $2\, n_\text{ctx}\, d$。Kaplan 等人的估计里恰好包含这一项：$C_\text{forward} \approx 2N + 2\, n_\text{layer}\, n_\text{ctx}\, d$。

把它和每层的权重项 $2 \times 12d^2 = 24d^2$ 比较一下：

$$
\frac{2\, n_\text{ctx}\, d}{24\, d^2} = \frac{n_\text{ctx}}{12\, d}
$$

对 GPT-3（上下文 2,048，宽度 12,288）来说，这个比值约为 1.4%，权重完全占主导。宽度为 4,096、上下文为 8,192 个 token 时，它约为 17%。只有当上下文长度是模型宽度的许多倍时，注意力的平方开销才会占上风，这也正是长上下文模型需要特殊处理的原因。

> [!TIP] 要点
> - 一个块就是 $x \leftarrow x + \operatorname{Attn}(\operatorname{LN}(x))$，后面再接 $x \leftarrow x + \operatorname{MLP}(\operatorname{LN}(x))$。Pre-LN 让残差路径保持干净，不用预热就能训练。
> - 残差流是一条共享通道：每个子层都从中读取，再往里叠加。只有注意力会在位置之间搬运信息。
> - 一个块大约有 $12d^2$ 个参数，其中 $4d^2$ 在注意力里，$8d^2$ 在 MLP 里。整个模型大约有 $12Ld^2$ 个参数，外加嵌入。
> - 训练时每个 token 大约花 $6N$ FLOPs，总计 $6ND$。注意力自身的开销会在此之上再增加约 $n_\text{ctx}/12d$ 的比例。

我们搭出的这个块有一个明显的缺失：其中没有任何东西知道 token 的顺序。[下一章](/zh/writing/position-sinusoids-and-rope/)讲 Transformer 如何找回位置信息，从最初的正弦编码，一直讲到如今大多数模型采用的旋转位置编码。
