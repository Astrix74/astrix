---
title: LayerNorm 与 RMSNorm
date: 2026-10-22T15:41:00Z
references: [ba2016, zhang2019, touvron2023]
---

层归一化（layer normalization）对一个激活向量做两件事：**重新中心化**（减去均值）和**重新缩放**（除以标准差），然后再乘上一个可学习的增益、加上一个可学习的偏置：

$$
\operatorname{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

[RMSNorm](https://arxiv.org/abs/1910.07467)（Zhang 和 Sennrich，2019）只保留了重新缩放。它除以均方根（root mean square），并去掉了均值和偏置：

$$
\operatorname{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}}
$$

作者“假设 LayerNorm 中的重新中心化不变性并非必不可少”。剩下的部分依然具备“重新缩放不变性”和一种“隐式的学习率自适应能力”，而且“在不同模型上把运行时间缩短了 7%~64%”。

```python
import numpy as np

def layer_norm(x, eps=1e-6):
    return (x - x.mean()) / np.sqrt(x.var() + eps)

def rms_norm(x, eps=1e-6):
    return x / np.sqrt((x ** 2).mean() + eps)

x = np.array([1.0, 2.0, 3.0, 6.0])
print("layer norm:", layer_norm(x).round(3))
print("rms norm:  ", rms_norm(x).round(3))
centered = x - x.mean()
same = np.allclose(layer_norm(centered), rms_norm(centered))
print("same once the mean is zero:", same)
```

```text
layer norm: [-1.069 -0.535  0.     1.604]
rms norm:   [0.283 0.566 0.849 1.697]
same once the mean is zero: True
```

对于均值为零的向量，两者完全相同，因为这时 RMS 恰恰就是标准差。否则，RMSNorm 会保留向量的方向（这个例子里结果全是正数），只修正它的长度；LayerNorm 则还会把它平移。

改动虽小，但深层模型的每一个块里都有它，累积起来就很可观，如今它也已相当普遍。例如 [LLaMA](https://arxiv.org/abs/2302.13971) 就在预归一化（pre-normalization）布局中使用 RMSNorm。归一化在 Transformer 块中处于什么位置，参见[《Transformer 块的内部》](/zh/writing/inside-the-transformer-block/)。
