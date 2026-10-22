---
title: "Why softmax freezes: the keys that aren't there"
date: 2026-10-22T15:45:00Z
references: [derrida1981, giorlandino2025]
---

Give a softmax $n$ random scores $s_j \sim \mathcal{N}(0, \sigma^2)$. Small $\sigma$ spreads the weight across many of them; large $\sigma$ piles it onto a few. The switch between the two has a one-line explanation.

The normalizer $Z = \sum_j e^{s_j}$ is $n$ times an average, and the law of large numbers says that average should be $\mathbb{E}[e^{s}] = e^{\sigma^2/2}$. But look at *which* scores make up that expectation. The integrand $e^{s} e^{-s^2/2\sigma^2}$ peaks at $s = \sigma^2$, so the average is carried by scores near $\sigma^2$. And $n$ Gaussian draws almost never include one above about $\sigma\sqrt{2 \ln n}$. So:

$$
\sigma^2 < \sigma \sqrt{2 \ln n} \iff \sigma < \sqrt{2 \ln n}
$$

Below that point, the scores that carry the average are in the sample, and $\ln Z \approx \ln n + \sigma^2/2$. Above it, the average is carried by keys that aren't there. $Z$ is then made of the few largest scores the sample does contain, and the softmax freezes onto them:

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

Up to $\sigma \approx 3$, the law of large numbers is right. From $\sigma = 5$ on, $\ln Z$ stays within about one unit of the single largest score, while the law-of-large-numbers formula runs away from it. In the limit of large $n$ the switch is sharp, and it is the freezing transition of Derrida's random energy model. At finite $n$ it is gradual, as the rows near $\sigma = 4.29$ show.

The same argument with $e^{2s}$, whose integrand peaks at $2\sigma^2$, puts a second threshold at half the first, beyond which $Z$ stops having Gaussian fluctuations. The paper [When attention freezes](/papers/when-attention-freezes/) works through what both thresholds mean for attention: why dividing by $\sqrt{d_k}$ keeps a fresh head on the diffuse side, and what happens to a single key's weight as contexts grow.
