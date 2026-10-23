---
title: 一个公式讲清 softmax 温度
date: 2026-10-22T15:40:00Z
references: [hinton2015]
---

语言模型采样下一个 token 时，会用 softmax 把 logits $z_i$ 变成概率。温度 $T$ 就是插在 softmax 之前的一次除法：

$$
p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

除以 $T$ 会按比例缩放 logits 之间的每一个差值。softmax 只取决于这些差值（给每个 logit 都加上同一个常数，结果完全不变），所以温度控制的是最大的那个 logit 能占多大优势：

```python
import numpy as np

logits = np.array([2.0, 1.0, 0.1])
for T in [0.25, 0.5, 1.0, 2.0, 10.0]:
    p = np.exp(logits / T)
    p /= p.sum()
    print(f"T={T:<5} probabilities {p.round(3)}")
```

```text
T=0.25  probabilities [0.982 0.018 0.   ]
T=0.5   probabilities [0.864 0.117 0.019]
T=1.0   probabilities [0.659 0.242 0.099]
T=2.0   probabilities [0.502 0.304 0.194]
T=10.0  probabilities [0.366 0.331 0.303]
```

- **$T = 1$** 就是模型自己的分布，也就是它在训练中学着输出的那个分布。
- **$T \to 0$** 会把全部概率都推给最大的 logit。采样退化成贪心解码：永远选排在第一的 token。
- **$T \to \infty$** 会把一切压平、趋向均匀分布，模型自身的偏好也就不再起作用。

温度越低，生成的文本越可预测、越容易重复；温度越高，文本越多样，也越容易出错。温度从不改变 token 之间的排名先后，它改变的只是模型对排在前面的那几个 token 偏爱得有多鲜明。

训练中也有同一个旋钮。[知识蒸馏](https://arxiv.org/abs/1503.02531)（Hinton、Vinyals 和 Dean，2015）会调高大模型 softmax 的温度，让那些很小的概率变得足够大，好让一个更小的模型从中学习；这些小概率透露的是，哪些错误答案其实几乎是对的。注意力里的 $1/\sqrt{d_k}$ 实际上也是一个固定的温度，选它是为了防止 softmax 饱和，详见[《注意力：一次可微分的查找》](/zh/writing/attention-as-a-differentiable-lookup/)。当注意力同时面对许多键（key）时，温度决定了它是分散开来，还是冻结在少数几个键上，这正是论文[《注意力何时冻结》](/zh/papers/when-attention-freezes/)的主题。
