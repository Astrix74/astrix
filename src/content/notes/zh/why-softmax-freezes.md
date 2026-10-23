---
title: "softmax 为什么会冻结：那些并不存在的键"
date: 2026-10-22T15:45:00Z
references: [derrida1981, giorlandino2025]
---

给 softmax 输入 $n$ 个随机分数 $s_j \sim \mathcal{N}(0, \sigma^2)$。$\sigma$ 小的时候，权重会分散到其中许多分数上；$\sigma$ 大的时候，权重会堆到少数几个上。两种状态之间的切换，一句话就能解释清楚。

归一化因子 $Z = \sum_j e^{s_j}$ 是 $n$ 乘以一个平均值，而按照大数定律，这个平均值应该等于 $\mathbb{E}[e^{s}] = e^{\sigma^2/2}$。但要看看这个期望究竟是由哪些分数构成的。被积函数 $e^{s} e^{-s^2/2\sigma^2}$ 在 $s = \sigma^2$ 处取得峰值，所以撑起这个平均值的，是 $\sigma^2$ 附近的分数。而 $n$ 个高斯样本中，几乎不会有哪个超过大约 $\sigma\sqrt{2 \ln n}$。所以：

$$
\sigma^2 < \sigma \sqrt{2 \ln n} \iff \sigma < \sqrt{2 \ln n}
$$

低于这个点时，撑起平均值的那些分数就在样本里，$\ln Z \approx \ln n + \sigma^2/2$。高于这个点时，撑起平均值的是那些并不存在的键。这时 $Z$ 由样本中确实存在的少数几个最大分数构成，softmax 也就冻结在它们身上：

```python
import numpy as np

rng = np.random.default_rng(0)
n = 10_000
g = rng.standard_normal((200, n))                 # 200 rows of n standard normal scores
print(f"n = {n:,}, sqrt(2 ln n) = {np.sqrt(2 * np.log(n)):.2f}")
print("sigma   log Z   law of large numbers   largest score")
for sigma in [1, 2, 3, 4, 5, 6, 8]:
    s = sigma * g
    top = s.max(axis=1)
    log_z = top + np.log(np.exp(s - top[:, None]).sum(axis=1))
    print(f"{sigma:<7} {log_z.mean():>5.1f} {np.log(n) + sigma**2 / 2:>16.1f} {top.mean():>17.1f}")
```

```text
n = 10,000, sqrt(2 ln n) = 4.29
sigma   log Z   law of large numbers   largest score
1         9.7              9.7               3.8
2        11.2             11.2               7.7
3        13.6             13.7              11.5
4        16.8             17.2              15.4
5        20.2             21.7              19.2
6        23.8             27.2              23.0
8        31.3             41.2              30.7
```

直到 $\sigma \approx 3$，大数定律都是准的。从 $\sigma = 5$ 开始，$\ln Z$ 始终与单个最大分数相差不到大约一个单位，而大数定律给出的公式值则离它越来越远。在 $n$ 很大的极限下，这一切换是陡峭的，它就是 Derrida 随机能量模型中的冻结相变。在有限的 $n$ 下，切换是渐进的，$\sigma = 4.29$ 附近的那几行就说明了这一点。

把同样的论证用在 $e^{2s}$ 上（它的被积函数在 $2\sigma^2$ 处取得峰值），会得到第二个阈值，恰好是第一个的一半；越过它之后，$Z$ 的涨落就不再是高斯型的。论文[《注意力何时冻结》](/zh/papers/when-attention-freezes/)详细讨论了这两个阈值对注意力意味着什么：为什么除以 $\sqrt{d_k}$ 能让一个刚初始化的注意力头处在分散的那一侧，以及随着上下文变长，单个键的权重会怎样变化。
