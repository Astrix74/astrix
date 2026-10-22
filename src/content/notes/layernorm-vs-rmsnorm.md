---
title: LayerNorm vs RMSNorm
date: 2026-10-22T15:41:00Z
references: [ba2016, zhang2019, touvron2023]
---

Layer normalization does two things to a vector of activations: it *re-centers* (subtracts the mean) and *re-scales* (divides by the standard deviation), then applies a learned gain and bias:

$$
\operatorname{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

[RMSNorm](https://arxiv.org/abs/1910.07467) (Zhang and Sennrich, 2019) keeps only the re-scaling. It divides by the root mean square and drops the mean and the bias:

$$
\operatorname{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}}
$$

The authors "hypothesize that re-centering invariance in LayerNorm is dispensable". What remains still gives "re-scaling invariance" and an "implicit learning rate adaptation ability", and it "reduces the running time by 7%~64% on different models".

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

For a vector with zero mean the two are identical, since the RMS *is* the standard deviation then. Otherwise RMSNorm keeps the vector's direction, all positive here, and only fixes its length. LayerNorm also moves it.

It is a small change, but it adds up across every block of a deep model, and it has become common. [LLaMA](https://arxiv.org/abs/2302.13971), for example, uses RMSNorm in a pre-normalization layout. For where normalization sits in a transformer block, see [Inside the transformer block](/writing/inside-the-transformer-block/).
