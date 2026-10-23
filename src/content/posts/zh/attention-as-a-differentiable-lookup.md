---
title: 注意力：一次可微分的查找
description: 从第一性原理讲清查询、键和值，解释注意力分数为什么要除以键维度的平方根，再用 NumPy 实现一个因果多头注意力层。
date: 2026-10-22T15:10:00Z
tags: [transformers, attention, math]
series: transformers
chapter: 1
references: [bahdanau2014, vaswani2017]
---

语言里到处都是长程依赖。比如这句英文：“The keys that I left on the kitchen table this morning **are** missing”，其中的动词要和十个词之前的名词保持一致。一个逐个 token 读取文本的模型，需要某种机制，让每个 token 都能把所需的信息取过来，不管这些信息恰好位于序列中的什么位置。

在 Transformer 出现之前，这件事由循环神经网络承担：它把迄今读到的所有内容压缩进一个固定大小的状态向量，再一路向前传递。模型将来用得上的每一个事实，都必须一次又一次地挤过这个向量，还得完好无损。2014 年，Bahdanau、Cho 和 Bengio 为机器翻译提出了一种绕开这个瓶颈的办法：让解码器回头查看编码器的所有状态，并为每个输出词决定该给哪些状态更大的权重。三年后，Vaswani 等人提出了一个问题：如果彻底抛弃循环结构，整个模型只用这种“回头看”的操作搭起来，会怎样？他们的答案就是 Transformer，而这种操作就是**注意力**（attention）。

本章从一个想法出发构建注意力：一张可以求导的查找表。

## 从字典到软字典

Python 字典把键（key）映射到值（value）。你给它一个查询（query），它找出恰好等于这个查询的键，返回该键对应的值。有两点让它在神经网络里毫无用处：匹配是非此即彼的，而“相等”给不出梯度。把查询稍微挪动一点，结果要么纹丝不动，要么全盘改变。

那就把这两点都软化。不再只找那一个匹配的键，而是按相似度给查询和每一个键打分，把分数变成总和为一的权重，再返回所有值的加权平均：

$$
\text{output} = \sum_j w_j \, v_j, \qquad w_j = \frac{\exp(\text{score}(q, k_j))}{\sum_m \exp(\text{score}(q, k_m))}
$$

如果某个键的匹配程度远远胜过其余的键，它的权重就趋近于一，我们又回到了普通的查找；如果有好几个键都匹配，得到的就是它们的混合。每一步都是光滑的，所以梯度能流向查询、键和值。网络因此能够自己学会该查找什么。

## 自注意力

在**自注意力**（self-attention）中，序列里的每个 token 同时扮演这三种角色。每个 token 的向量 $x_i$（大小为 $d$）经过三个学到的矩阵，被投影成三份：

$$
q_i = x_i W_Q, \qquad k_i = x_i W_K, \qquad v_i = x_i W_V
$$

查询表示这个 token 在寻找什么，键表示它凭什么被别人找到，值则是它被找到时交出去的内容。把三者分开很重要：一个代词要找的可能是名词，而它自己对外提供的却是自身的语法角色，这是两份不同的信息。

相似度分数是点积，再乘以 $1/\sqrt{d_k}$ 做缩放，其中 $d_k$ 是查询向量和键向量的大小。（原因见下一节。）把所有查询、键和值分别按行堆成矩阵 $Q$、$K$ 和 $V$，整个序列的注意力只需一行：

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V
$$

$QK^\top$ 是一个 $n \times n$ 的矩阵，存放着每一对查询–键的分数。softmax 沿每一行进行，于是第 $i$ 行就成了 token $i$ 在整个序列上的权重。再乘以 $V$，就得到加权平均。

## 为什么要除以键维度的平方根？

假设 $q$ 和 $k$ 的各个分量相互独立，均值为零、方差为一，这大致就是初始化良好的网络里的情形。它们的点积是 $d_k$ 个独立项之和，每一项的方差都是一：

$$
\operatorname{Var}(q \cdot k) = \sum_{m=1}^{d_k} \operatorname{Var}(q_m k_m) = d_k
$$

所以分数的典型大小是 $\sqrt{d_k}$，并且会随模型宽度增长。分数一大就会出问题，原因在于 softmax 对它们的处理：指数运算会放大差距，于是最大的那个分数几乎拿走全部权重。softmax 的输出变得近乎 one-hot，而它的导数

$$
\frac{\partial w_i}{\partial s_j} = w_i \, (\delta_{ij} - w_j)
$$

在权重全是零和一时会趋于零。饱和的 softmax 几乎传不回任何梯度，这和[《深度学习基础》系列](/zh/writing/making-depth-trainable/)里饱和的 tanh 单元是同一个问题。除以 $\sqrt{d_k}$ 能把分数的方差拉回到一，无论宽度是多少。Transformer 论文给出的正是这个理由。

这一点很容易测量。让一个查询给 128 个随机键打分，看看排在第一的键能拿到多少权重：

```python
import numpy as np

rng = np.random.default_rng(0)

def softmax(s):
    s = s - s.max(axis=-1, keepdims=True)       # for numerical stability
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)

n_keys = 128
for d_k in [16, 64, 256, 1024]:
    q = rng.normal(size=(500, d_k))                 # 500 queries
    k = rng.normal(size=(500, n_keys, d_k))         # 128 keys for each
    scores = np.einsum("qd,qnd->qn", q, k)          # dot products
    raw = softmax(scores).max(axis=-1).mean()
    scaled = softmax(scores / np.sqrt(d_k)).max(axis=-1).mean()
    print(f"d_k={d_k:<5} score std {scores.std():5.1f}   "
          f"largest weight: unscaled {raw:.2f}, scaled {scaled:.2f}")
```

```text
d_k=16    score std   4.0   largest weight: unscaled 0.53, scaled 0.07
d_k=64    score std   8.0   largest weight: unscaled 0.78, scaled 0.07
d_k=256   score std  15.9   largest weight: unscaled 0.88, scaled 0.07
d_k=1024  score std  31.9   largest weight: unscaled 0.94, scaled 0.07
```

分数的标准差与 $\sqrt{d_k}$ 完全吻合（4、8、16、32）。不做缩放时，一个随机的、未经训练的查询就已经把大部分权重压在单个键上，而且模型越宽，情况越糟。做了缩放，无论 $d_k$ 是多少，注意力一开始都分散在许多键上，网络可以在学习中自由地把它变得尖锐。论文[《注意力何时冻结》](/zh/papers/when-attention-freezes/)沿着这条线继续深入：权重如何分布在全部 $n$ 个键上，以及上下文变长时会发生什么变化。

## 因果掩码

语言模型的训练目标，是根据前面的 token 预测每一个 token。如果训练时第 5 个 token 能关注到第 6 个 token，它直接读答案就行了。**因果掩码**（causal mask）正是为了防止这一点：在 softmax 之前，把所有 $j > i$ 的分数 $s_{ij}$ 都设为 $-\infty$。由于 $e^{-\infty} = 0$，这些位置的权重恰好为零，注意力矩阵也就成了下三角矩阵。

也正是掩码，让 Transformer 能在一个序列的所有位置上并行训练。一次前向传播就在每个位置上各产生一个对下一个 token 的预测，而每个预测只用到它被允许看到的那些 token。

## 多头

一次注意力操作为每个 token 产生一组权重，也就是一种对“什么是相关的”的判断。而语言同时存在许多种相关性，比如句法、指代和话题。**多头注意力**（multi-head attention）并排运行 $h$ 个注意力操作，每个都有自己的投影，并且 $d_k = d/h$。它把这些输出拼接起来，再用另一个矩阵 $W_O$ 混合：

$$
\operatorname{MultiHead}(X) = \big[\, \text{head}_1 ; \dots ; \text{head}_h \,\big] \, W_O
$$

由于每个头都窄了 $h$ 倍，总开销与一个全宽的头大致相当。花一次查找的代价，就得到几次相互独立的查找。

## 一个因果多头注意力层

下面用 NumPy 把上面的内容全部实现出来。函数先把投影拆成多个头，然后依次打分、加掩码、做 softmax、混合值，最后把各个头合并起来：

```python
import numpy as np

def softmax(s):
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)

def causal_self_attention(X, Wq, Wk, Wv, Wo, n_heads):
    n, d = X.shape
    d_k = d // n_heads
    def split(M):                                   # (n, d) -> (heads, n, d_k)
        return M.reshape(n, n_heads, d_k).transpose(1, 0, 2)
    Q, K, V = split(X @ Wq), split(X @ Wk), split(X @ Wv)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)     # (heads, n, n)
    future = np.triu(np.ones((n, n), dtype=bool), k=1)   # True where j > i
    scores = np.where(future, -np.inf, scores)            # no peeking ahead
    weights = softmax(scores)                             # each row sums to 1
    heads = weights @ V                                   # (heads, n, d_k)
    return heads.transpose(1, 0, 2).reshape(n, d) @ Wo, weights

rng = np.random.default_rng(0)
n, d, n_heads = 6, 16, 4
Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4))
X = rng.normal(size=(n, d))

out, weights = causal_self_attention(X, Wq, Wk, Wv, Wo, n_heads)
print("head 0 attention weights:")
print(weights[0].round(2))

X_changed = X.copy()
X_changed[4:] = rng.normal(size=(2, d))              # change only the last two tokens
out_changed, _ = causal_self_attention(X_changed, Wq, Wk, Wv, Wo, n_heads)
print("first four outputs unchanged:", np.allclose(out[:4], out_changed[:4]))
print("last two outputs unchanged: ", np.allclose(out[4:], out_changed[4:]))
```

```text
head 0 attention weights:
[[1.   0.   0.   0.   0.   0.  ]
 [0.65 0.35 0.   0.   0.   0.  ]
 [0.73 0.19 0.07 0.   0.   0.  ]
 [0.21 0.19 0.44 0.16 0.   0.  ]
 [0.06 0.08 0.46 0.1  0.31 0.  ]
 [0.06 0.04 0.32 0.13 0.15 0.29]]
first four outputs unchanged: True
last two outputs unchanged:  False
```

权重是下三角的，每一行的和都是一。第一个 token 只能关注它自己。最后那个测试值得你在自己的代码里也留一份：改变未来，检查过去有没有察觉。掩码泄漏是最容易写出来的 Transformer bug 之一，也是只看损失最难发现的 bug 之一，因为模型只会显得好得可疑。

## 需要记住的两个性质

**注意力的开销与序列长度成平方关系**。$QK^\top$ 有 $n^2$ 个元素，所以上下文长度翻倍，计算分数的开销和存放分数所需的内存都会变成原来的四倍。长上下文模型背后的大量工程，都是在想办法绕开这个平方。

**注意力完全不知道 token 的顺序**。再看一遍公式：其中没有任何东西依赖位置。打乱输入 token 的顺序，除了掩码带来的影响，输出也会以同样的方式被打乱。在注意力看来，序列就是一个集合。本系列第 3 章讲的就是 Transformer 怎样把词序找回来。

> [!TIP] 要点
> - 注意力是一种软性的字典查找：用查询给每个键打分，用 softmax 把分数变成权重，再对值求加权平均。
> - 查询、键和值是对每个 token 分别学到的三个投影：它在找什么，它凭什么被找到，它交出什么。
> - 随机的 $d_k$ 维向量做点积，标准差为 $\sqrt{d_k}$。除以它可以防止 softmax 饱和，梯度才得以保留。
> - 因果掩码把未来位置的分数设为 $-\infty$，从而在不作弊的前提下实现并行训练。测试方法：改变未来，检查过去。
> - 多头带来几次相互独立的查找，开销却和一次差不多。注意力的开销随序列长度的平方增长。

注意力让 token 之间可以交换信息，但它只是 Transformer 层的一半。[下一章](/zh/writing/inside-the-transformer-block/)会组装出一个完整的块（注意力、MLP、归一化，以及把它们串在一起的残差流），并清点它的参数和计算量都花在了哪里。
