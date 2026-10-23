---
title: 位置：正弦编码、RoPE，以及顺序为什么不是免费的
description: 注意力分不清“狗咬人”和“人咬狗”。Transformer 如何把词序找回来：从最初的正弦编码讲到旋转位置编码，并验证让 RoPE 成立的那一条关键性质。
date: 2026-10-22T15:12:00Z
tags: [transformers, attention, math]
series: transformers
chapter: 3
references: [vaswani2017, shaw2018, su2021, press2021, touvron2023]
---

[第 1 章](/zh/writing/attention-as-a-differentiable-lookup/)结尾留下了一个警告：注意力公式里没有任何东西依赖 token 在序列中的位置。这听起来像个无关紧要的技术细节，直到你看清它意味着什么。下面是单个注意力头，先给它“dog bites man”，再给它“man bites dog”：

```python
import numpy as np

def softmax(s):
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def attention(X, Wq, Wk, Wv):                 # one head, no mask, no positions
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    return softmax(Q @ K.T / np.sqrt(K.shape[1])) @ V

rng = np.random.default_rng(0)
d = 8
Wq, Wk, Wv = (rng.normal(size=(d, d)) for _ in range(3))
emb = {word: rng.normal(size=d) for word in ["dog", "bites", "man"]}

a = attention(np.stack([emb[w] for w in ["dog", "bites", "man"]]), Wq, Wk, Wv)
b = attention(np.stack([emb[w] for w in ["man", "bites", "dog"]]), Wq, Wk, Wv)
print("output for 'dog' is the same in both sentences:", np.allclose(a[0], b[2]))
print("output for 'man' is the same in both sentences:", np.allclose(a[2], b[0]))
```

```text
output for 'dog' is the same in both sentences: True
output for 'man' is the same in both sentences: True
```

谁咬了谁，完全看不出来。打乱输入，只会以同样的方式打乱输出，用术语来说，这叫**置换等变性**（permutation equivariance）。因果掩码会稍稍打破这种对称，因为每个 token 看到的前缀各不相同；但一个只能靠掩码来推断词序的模型，等于被捆住一只手干活。Transformer 需要有意地注入位置信息。本章按出现的先后，介绍三种主要做法。

## 绝对位置：每个位置加一个向量

最简单的补救办法，是给每个位置 $p$ 分配一个专属向量 $\text{PE}(p)$，并在第一层之前把它加到 token 嵌入上。这样一来，位置 5 上的 token 和位置 9 上的同一个 token 看起来就不一样了。

这些向量可以直接学出来，每个位置一个，就像 token 嵌入那样。最初的 Transformer 论文用的则是一个固定公式：一组频率各不相同的正弦和余弦，

$$
\text{PE}(p, 2i) = \sin\!\left( \frac{p}{10000^{2i/d}} \right), \qquad
\text{PE}(p, 2i+1) = \cos\!\left( \frac{p}{10000^{2i/d}} \right)
$$

每一对维度 $(2i, 2i+1)$ 都是一个点，随着 $p$ 增大，以各自的速度绕圆周转动。波长构成一个从 $2\pi$ 到 $10000 \cdot 2\pi$ 的等比数列，所以转得快的维度对用来区分相邻的位置，转得慢的则区分相距较远的区域。这很像钟表上的几根指针。

Vaswani 等人给出了这样选择的具体理由：“对任意固定的偏移量 $k$，$\text{PE}_{pos+k}$ 都可以表示为 $\text{PE}_{pos}$ 的线性函数。”原因在于和角公式。向前移动 $k$ 个位置，相当于把每一对维度旋转一个固定的角度，而旋转是线性映射。可以直接验证：

```python
import numpy as np

def sinusoidal(n_positions, d):
    pos = np.arange(n_positions)[:, None]
    freq = 10000.0 ** (-np.arange(0, d, 2) / d)  # one frequency per pair of dimensions
    pe = np.zeros((n_positions, d))
    pe[:, 0::2] = np.sin(pos * freq)
    pe[:, 1::2] = np.cos(pos * freq)
    return pe, freq

pe, freq = sinusoidal(200, 16)

def shift(k):
    """The matrix that maps PE(pos) to PE(pos + k): one 2x2 rotation per frequency."""
    M = np.zeros((16, 16))
    for i, w in enumerate(freq):
        c, s = np.cos(k * w), np.sin(k * w)
        M[2*i:2*i+2, 2*i:2*i+2] = [[c, -s], [s, c]]
    return M

M = shift(7)
print("PE(pos + 7) == PE(pos) @ M for every pos:", np.allclose(pe[:-7] @ M, pe[7:]))
```

```text
PE(pos + 7) == PE(pos) @ M for every pos: True
```

同一个矩阵在每个位置上都成立。当时的期望是，这能让注意力很容易学会“往回看三个 token”。论文也试过学出来的位置嵌入，“发现两种版本产生的结果几乎完全相同”。作者最终保留了正弦编码，因为它“或许能让模型外推到比训练中遇到的更长的序列长度”。先记住这个期望，我们后面还会回到它。

加法式的做法本身就带着一个弱点。位置向量在网络的最底层就混进了 token 的表示，之后的每一层都得一边处理内容，一边惦记着它。而注意力比较的是 $q \cdot k$，两边都是内容和绝对位置的混合。相对距离通常才是语言真正在意的东西，在这里却只是间接地体现出来。

## 相对位置：直接告诉注意力距离

Shaw、Uszkoreit 和 Vaswani（2018）正面解决了这个问题。他们不再把位置加到输入上，而是修改注意力本身。当查询 $i$ 查看键 $j$ 时，一个对应于距离 $j - i$ 的学习向量会被加到键上，也会加到值上。距离在某个最大值 $k$ 处截断，因为“超过一定距离之后，精确的相对位置信息就不再有用”。在英德翻译上，这种方法比绝对位置高出 1.3 BLEU，而把两种位置信息结合起来，也没有带来进一步的提升。

代价是复杂度。现在注意力的计算要为每一对查询–键额外加上一项，这让高效实现变得麻烦。我们想要的是相对位置，同时又完全不改动注意力公式。

## 旋转位置编码：把位置看作旋转

旋转位置编码 RoPE（Su 等人，2021）做的正是这件事。这个想法顺着正弦编码而来：既然沿着序列移动就是一次旋转，那就干脆真的去旋转查询向量和键向量来编码位置，剩下的交给点积。

取一个位于位置 $m$ 的 2 维查询 $q$，和一个位于位置 $n$ 的键 $k$。把它们各自旋转一个与位置成正比的角度，得到 $R(m\theta)\,q$ 和 $R(n\theta)\,k$，其中 $R(\alpha)$ 是转角为 $\alpha$ 的旋转矩阵。旋转矩阵是正交的，$R(\alpha)^\top = R(-\alpha)$，而且复合时角度相加，所以

$$
\big(R(m\theta)\, q\big)^\top \big(R(n\theta)\, k\big)
= q^\top R(-m\theta)\, R(n\theta)\, k
= q^\top R\big((n - m)\theta\big)\, k
$$

绝对位置被消掉了。分数取决于 $q$ 和 $k$ 的内容，而与位置有关的部分只剩下两者的距离 $n - m$。在 $d$ 维的情形下，RoPE 把每个查询和键拆成 $d/2$ 对，第 $i$ 对以自己的频率 $\theta_i = 10000^{-2(i-1)/d}$ 旋转，与正弦编码的等比分布完全相同。

有三个细节值得注意。旋转发生在每一个注意力层内部，而且只作用于 $q$ 和 $k$。值和残差流从来不接触位置。它也不增加任何参数。下面是实现，并用数值验证了这种抵消：

```python
import numpy as np

def rope(x, pos, base=10000.0):
    """Rotate each pair of dimensions (x[2i], x[2i+1]) by the angle pos * theta_i."""
    d = x.shape[-1]
    theta = base ** (-np.arange(0, d, 2) / d)
    angle = pos * theta
    cos, sin = np.cos(angle), np.sin(angle)
    x_even, x_odd = x[0::2], x[1::2]
    out = np.empty_like(x)
    out[0::2] = x_even * cos - x_odd * sin
    out[1::2] = x_even * sin + x_odd * cos
    return out

rng = np.random.default_rng(0)
q, k = rng.normal(size=64), rng.normal(size=64)

for m, n in [(5, 2), (105, 102), (1005, 1002), (5, 4)]:
    score = rope(q, m) @ rope(k, n)
    print(f"query at {m:>4}, key at {n:>4} (distance {m - n}):  score {score:+.6f}")
```

```text
query at    5, key at    2 (distance 3):  score -8.446152
query at  105, key at  102 (distance 3):  score -8.446152
query at 1005, key at 1002 (distance 3):  score -8.446152
query at    5, key at    4 (distance 1):  score -8.519140
```

同一对向量在距离为 3 时得分完全相同，无论它们靠近序列开头，还是已经在一千个 token 之后。改变距离，分数也随之改变。论文还表明，在这样选择频率的情况下，分数往往会随距离增大而衰减，相当于内置了一种对邻近 token 的温和偏好。

RoPE 已经成为一个标准组件。例如，Meta 的 LLaMA 模型（Touvron 等人，2023）去掉了绝对位置嵌入，改用旋转位置编码，同时还采用了 RMSNorm，以及[上一章](/zh/writing/inside-the-transformer-block/)讲过的预归一化和 SwiGLU MLP。

## 外推问题

回到原论文的那个期望：一种光滑的、由公式定义的编码，能让模型处理比训练时见过的任何序列都长的序列。Press、Smith 和 Lewis（2021）检验了这一点。他们的图 1 显示，当输入长度超过训练长度时，正弦编码、旋转编码和 T5 风格的相对位置方法都会退化。一个公式对任意位置都有定义，并不等于模型已经学会了怎样处理它从未见过的位置。

他们提出的替代方案 ALiBi 彻底去掉了位置嵌入，改为从每个注意力分数中减去一个与距离成正比的惩罚项，每个头的斜率各不相同。一个在 1,024 个 token 的输入上训练的 13 亿参数模型，“能外推到长度为 2048 的输入序列，困惑度与在长度为 2048 的输入上训练的正弦位置嵌入模型相同”，同时训练速度快 11%，内存占用少 11%。

长上下文扩展是一个活跃的领域，细节变化很快。经得起时间考验的教训还是上面那一条：位置的编码方式决定了模型能泛化到哪里，更长的输入必须实测，不能想当然。

> [!TIP] 要点
> - 没有位置信息，注意力就把序列当成集合：置换输入，输出也随之置换。
> - 正弦编码把固定的正弦和余弦加到输入上。平移 $k$ 个位置是一次固定的旋转，也就是一个线性映射。在原论文里，学出来的绝对位置嵌入效果差不多一样好。
> - 相对位置编码（Shaw 等人）把距离直接放进注意力，代价是注意力的计算更复杂。
> - RoPE 按与位置相关的角度旋转查询和键。绝对位置在点积中相互抵消，所以分数只取决于距离，而且不增加任何参数。
> - 对任意长度有定义，不等于在任意长度上都管用。外推能力必须实测。

至此，Transformer 的核心就讲完了：注意力、围绕它搭起的块，以及位置。[下一章](/zh/writing/scaling-laws/)从架构转向规模：给定预算，模型该多大，又该看多少数据。
