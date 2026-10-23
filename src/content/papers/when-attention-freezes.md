---
title: "When attention freezes: softmax, the random energy model, and long contexts"
description: Why scaled attention starts out nearly uniform, why it would start out frozen without the square root of the key dimension, and what the same picture says about rescaling attention for long contexts.
abstract: "At initialization, the scores a query gives to n keys behave like independent Gaussians, and the softmax over them is Derrida's random energy model, a mapping identified by Lucibello and Mézard and used by Giorlandino and Goldt to analyze signal propagation in transformers. This paper works through that picture from first principles, with every number reproduced by NumPy code in the text. In the limit of long sequences, the softmax has a diffuse regime and a frozen one, separated at a score spread of √(2 ln n); dividing scores by √d_k keeps attention diffuse, while without it a 64-dimensional head would start frozen at any realistic length. At practical lengths the transition is a slow crossover: the limiting formulas hold up to half the critical spread and drift for a long way beyond it. The same picture separates two families of length-dependent logit scaling. Multiplying logits by √(ln n) keeps the regime fixed; multiplying by ln n largely keeps a well-separated key's weight but pushes noisy heads toward freezing. Neither can rescue a key whose score no longer beats the largest distractor, which grows like √(2 ln n)."
date: 2026-10-22T15:50:00Z
references: [derrida1980, derrida1981, benarous2005, lucibello2023, giorlandino2025, levien2026, alemanno2026, ramsauer2020, zhai2023, vaswani2017, chiang2022, jin2023, peng2023, nakanishi2025]
---

## 1. The question

[Attention as a differentiable lookup](/writing/attention-as-a-differentiable-lookup/) gave the standard reason for dividing attention scores by $\sqrt{d_k}$: dot products of random $d_k$-dimensional vectors have standard deviation $\sqrt{d_k}$, and large scores saturate the softmax. That is true, but it only describes one key at a time. A query attends over $n$ keys at once, and what matters is the *distribution* of weight across all of them: how many keys share it, and how that changes as $n$ grows from a thousand tokens to a million.

That distribution has an exact counterpart in statistical physics. A softmax over independent Gaussian scores is the **random energy model** (REM), which Derrida introduced in 1980 as a minimal model of a disordered system. The REM has two phases, a diffuse one and a frozen one, with a sharp transition between them. Lucibello and Mézard (2023) used this mapping for dense associative memories, and Giorlandino and Goldt (2025) built a full theory of signal propagation through transformers at initialization on it, including the "entropy collapse" failure mode and how to initialize to avoid it.

This paper is a worked tour of that picture, aimed at readers of the [transformers series](/series/transformers/). It adds three things:

1. A derivation of both regimes that needs nothing beyond the law of large numbers and one Gaussian identity, including a one-line reason for where the transition sits.
2. Measurements of how far practical sequence lengths are from the infinite-length limit. They are far: the transition is a slow crossover at every length a model actually sees.
3. A side-by-side reading of two families of length-dependent logit scaling used in practice, which the REM picture separates cleanly: one keeps the regime fixed, the other keeps a retrieved key's weight fixed, and the second pushes noisy heads toward freezing.

Every number in the paper comes from the code in it, which runs in about ten seconds with NumPy.

> [!NOTE] Not new physics
> The mapping, the phases and the frozen-phase results are known; sections 3 and 4 credit them. What is offered here is the derivation in plain terms, the finite-size measurements, and the comparison of scaling rules.

## 2. Setup

A query scores $n$ keys, $s_j = q \cdot k_j / \tau$, and attention gives key $j$ the weight

$$
w_j = \frac{e^{s_j}}{Z}, \qquad Z = \sum_{j=1}^{n} e^{s_j}.
$$

Standard attention uses $\tau = \sqrt{d_k}$. At initialization, with random projection weights and token representations that are not strongly correlated, the scores of one query are close to independent draws from $\mathcal{N}(0, \sigma^2)$. If the entries of $q$ and $k$ have unit variance, then $\sigma^2 = d_k / \tau^2$: exactly 1 for scaled attention, and $d_k$ without the scaling. Smaller initial weights give a smaller $\sigma$; correlated tokens change the picture in ways Giorlandino and Goldt treat in full. Everything below takes the independent Gaussian case at face value.

Two numbers summarize how weight is spread:

- the **entropy** $H = -\sum_j w_j \ln w_j$, between 0 (all weight on one key) and $\ln n$ (uniform). Below it is reported as the share $H / \ln n$.
- the **inverse participation ratio** $Y = \sum_j w_j^2$, roughly one over the number of keys that matter.

```python
import numpy as np
from statistics import NormalDist

def softmax_stats(scores):
    """Entropy, inverse participation ratio and largest weight of softmax(scores), row by row."""
    top = scores.max(axis=1, keepdims=True)
    e = np.exp(scores - top)                       # shift by the max so exp never overflows
    z = e.sum(axis=1, keepdims=True)
    w = e / z
    log_z = np.log(z[:, 0]) + top[:, 0]
    entropy = log_z - (w * scores).sum(axis=1)     # H = log Z - sum_j w_j s_j
    return entropy, (w ** 2).sum(axis=1), w.max(axis=1)

def gaussian_rows(n, rows, rng, batch=2**22):
    """Standard normal rows of length n, about `batch` numbers at a time."""
    per = max(1, batch // n)
    for start in range(0, rows, per):
        yield rng.standard_normal((min(per, rows - start), n))
```

## 3. Two regimes

### The diffuse regime

If $Z$ is a sum of many comparable terms, the law of large numbers replaces it by $n$ times the mean of one term. For a Gaussian score, $\mathbb{E}[e^{s}] = e^{\sigma^2/2}$, so

$$
\ln Z \approx \ln n + \frac{\sigma^2}{2}.
$$

The entropy is $H = \ln Z - \sum_j w_j s_j$, and the second term is a weighted average of the scores. The same law of large numbers turns it into $\mathbb{E}[s\,e^{s}] / \mathbb{E}[e^{s}]$, and Stein's identity for Gaussians, $\mathbb{E}[s\,f(s)] = \sigma^2\,\mathbb{E}[f'(s)]$, makes that exactly $\sigma^2$. So

$$
H \approx \ln n + \frac{\sigma^2}{2} - \sigma^2 = \ln n - \frac{\sigma^2}{2}.
$$

It is convenient to measure the score spread against the length, with

$$
x = \frac{\sigma}{\sqrt{2 \ln n}}, \qquad \frac{H}{\ln n} \approx 1 - x^2 .
$$

Jin et al. (2023) derive the same entropy formula when they adapt image models to new resolutions.

### Where it breaks: the freezing point

The formula reaches zero entropy at $x = 1$, and the reason is visible in the integral behind $\mathbb{E}[e^{s}]$. The integrand $e^{s}\,e^{-s^2/2\sigma^2}$ peaks at $s = \sigma^2$: the average is carried by scores near $\sigma^2$. But a sample of $n$ Gaussians almost never contains a score above about $\sigma\sqrt{2 \ln n}$. As long as $\sigma^2 < \sigma\sqrt{2\ln n}$, that is $x < 1$, the scores that carry the average are present in the sample and the law of large numbers works. Beyond it, the average is carried by keys that are not there, and $Z$ is dominated by the few largest scores that are. The softmax **freezes** onto them.

In the limit $n \to \infty$ this is a sharp transition, Derrida's freezing transition. Below it, $H / \ln n \to 1 - x^2$. Above it, $H / \ln n \to 0$, the weight sits on a handful of keys, and the inverse participation ratio tends to $1 - 1/x$ (the REM result, stated for attention by Giorlandino and Goldt).

The same argument applied to $\mathbb{E}[e^{2s}]$, whose integrand peaks at $2\sigma^2$, gives a second threshold at $x = 1/2$. Below it, $Z$ also has Gaussian fluctuations around its mean; above it, it does not (Ben Arous, Bogachev and Molchanov, 2005; Giorlandino and Goldt point out its role for finite lengths). That turns out to be where practical lengths part company with the limit.

### What finite lengths actually do

Here is the entropy share at four lengths, against the limit:

```python
xs = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
sizes = [2**6, 2**10, 2**14, 2**17]
rng = np.random.default_rng(0)
curves = {}
for n in sizes:
    g = rng.standard_normal((max(16, 2**22 // n), n))   # one set of draws, reused for every x
    sigma_c = np.sqrt(2 * np.log(n))
    curves[n] = [softmax_stats(x * sigma_c * g)[0].mean() / np.log(n) for x in xs]

print("x     " + "".join(f"n = 2^{int(np.log2(n)):<4}" for n in sizes) + "limit")
for i, x in enumerate(xs):
    print(f"{x:<5} " + "".join(f"{curves[n][i]:<10.3f}" for n in sizes) + f"{max(0.0, 1 - x * x):.3f}")
```

```text
x     n = 2^6   n = 2^10  n = 2^14  n = 2^17  limit
0.25  0.939     0.938     0.938     0.938     0.938
0.5   0.779     0.758     0.753     0.750     0.750
0.75  0.597     0.537     0.507     0.490     0.438
1.0   0.454     0.373     0.329     0.314     0.000
1.25  0.354     0.273     0.229     0.222     0.000
1.5   0.286     0.210     0.172     0.171     0.000
```

![Line chart of entropy divided by ln n against x from 0 to 2, for n = 64, 1,024, 16,384 and 131,072, with the limit 1 minus x squared that reaches zero at x = 1. All curves coincide up to x = 0.5; beyond that the finite-n curves stay well above the limit, the shorter lengths highest, and none reaches zero by x = 2.](./when-attention-freezes-phase.svg "Figure 1. Entropy share against the scaled spread x. Up to x = 0.5 every length follows the limit 1 − x²; beyond it, finite lengths stay far more diffuse than the limit, which freezes at x = 1. Computed with the code in this section on a finer grid.")

Up to $x = 0.5$ all four lengths are within 0.03 of the formula, and $n = 2^{17}$ matches it to three decimals. Past it, finite lengths stay far more diffuse than the limit. At $x = 1$, where the limit has frozen completely, a sequence of 131,072 keys still keeps 31% of the maximum entropy.

The frozen side approaches its limit just as slowly:

```python
xs = [1.0, 1.5, 2.0, 3.0]
rng = np.random.default_rng(1)
print("n       H / ln n at x = 1, 1.5, 2, 3    Y at x = 1, 1.5, 2, 3")
for n, rows in [(2**10, 4000), (2**13, 2000), (2**16, 600), (2**19, 150)]:
    h = {x: [] for x in xs}
    y = {x: [] for x in xs}
    for g in gaussian_rows(n, rows, rng):
        for x in xs:
            hx, yx, _ = softmax_stats(x * np.sqrt(2 * np.log(n)) * g)
            h[x].append(hx / np.log(n))
            y[x].append(yx)
    mean = lambda v: np.concatenate(v).mean()
    print(f"2^{int(np.log2(n)):<5} " + " ".join(f"{mean(h[x]):.3f}" for x in xs)
          + "     " + " ".join(f"{mean(y[x]):.3f}" for x in xs))
print("limit   " + " ".join(f"{0:.3f}" for x in xs) + "     " + " ".join(f"{1 - 1 / x:.3f}" for x in xs))
```

```text
n       H / ln n at x = 1, 1.5, 2, 3    Y at x = 1, 1.5, 2, 3
2^10    0.373 0.210 0.142 0.085     0.238 0.448 0.576 0.713
2^13    0.333 0.176 0.115 0.068     0.211 0.433 0.567 0.708
2^16    0.302 0.149 0.096 0.057     0.196 0.427 0.564 0.705
2^19    0.287 0.133 0.084 0.049     0.159 0.404 0.546 0.690
limit   0.000 0.000 0.000 0.000     0.000 0.333 0.500 0.667
```

The inverse participation ratio approaches $1 - 1/x$ from above, and the entropy share falls toward zero, but over a 512-fold increase in length they move by only 0.02 to 0.09. The distance to the limit depends on $n$ mainly through $\ln n$, which only grows from 6.9 to 13.2 across this table. (Standard errors, from the spread across rows, are between 0.001 and 0.02, largest for $Y$ at $2^{19}$.)

The practical reading: **the phase diagram is the right map, but at the lengths models use, the border is a wide band, not a line.** Between $x \approx 0.5$ and well past $x = 1$, attention is neither diffuse nor frozen.

## 4. Why the square root of $d_k$ is a phase choice

Without scaling, a unit-variance query and key give $\sigma^2 = d_k$, so

$$
x_{\text{unscaled}} = \sqrt{\frac{d_k}{2 \ln n}}, \qquad x_{\text{scaled}} = \frac{1}{\sqrt{2\ln n}}.
$$

For $d_k = 64$, unscaled attention is past the freezing point unless $2 \ln n > 64$, that is, unless the sequence holds more than $e^{32} \approx 8 \times 10^{13}$ tokens. Scaled attention sits at $x \approx 0.2$–$0.3$ for any realistic length, deep in the diffuse regime. Measured with random queries and keys:

```python
rng = np.random.default_rng(2)
print("d_k  n      unscaled: x     H/ln n  top weight   scaled: x     H/ln n  top weight")
for d_k in [64, 128]:
    for k in [10, 17]:
        n = 2**k
        rows = max(8, 2**21 // n)
        # For random keys K and a query q, the scores K q are exactly Gaussian with variance |q|^2.
        q_norm = np.sqrt(rng.chisquare(d_k, size=(rows, 1)))
        scores = q_norm * rng.standard_normal((rows, n))
        line = f"{d_k:<4} 2^{k:<4}"
        for scale in [1.0, 1 / np.sqrt(d_k)]:
            h, _, top = softmax_stats(scale * scores)
            x = scale * np.sqrt(d_k) / np.sqrt(2 * np.log(n))
            line += f" {x:>14.2f} {h.mean() / np.log(n):>9.3f} {top.mean():>11.4f}"
        print(line)
```

```text
d_k  n      unscaled: x     H/ln n  top weight   scaled: x     H/ln n  top weight
64   2^10             2.15     0.134      0.7007           0.27     0.929      0.0162
64   2^17             1.65     0.132      0.5985           0.21     0.955      0.0004
128  2^10             3.04     0.084      0.7962           0.27     0.928      0.0163
128  2^17             2.33     0.056      0.7833           0.21     0.958      0.0004
```

Without the scaling, a fresh 64-dimensional head over 1,024 keys puts 70% of its weight on a single key, chosen by the random initialization rather than by anything learned. That is entropy collapse at initialization. Zhai et al. (2023) link low attention entropy to training instability, and Giorlandino and Goldt show that transformers initialized in the frozen regime fail to train. With the scaling, the same head starts nearly uniform, with an entropy share of 0.93–0.96, and has to *learn* to concentrate.

So the $\sqrt{d_k}$ in the transformer paper is more than a fix for saturating gradients. It chooses the phase attention starts in. Giorlandino and Goldt go a step further and suggest initializing query and key weights so that $\sigma$ grows like $\sqrt{\ln n}$ for the training length, which places a model at a chosen point of the phase diagram rather than at whatever $x$ the length implies.

## 5. One key among many

Attention is often asked to find one key: the noun a pronoun refers to, or the line of code a question is about. Model this as a **needle**: one key scored $c\mu$ among $n - 1$ distractors scored $c\,g_j$ with $g_j \sim \mathcal{N}(0, 1)$. Here $\mu$ is the needle's margin in units of the distractors' spread, and $c$ is the head's sharpness, a multiplier on all logits.

Two separate things decide whether the needle gets the weight.

**Ranking: is the needle the highest score?** That depends only on $\mu$ and the largest of $n - 1$ Gaussians, never on $c$: multiplying every logit by the same positive number changes no order. The largest distractor is below $t$ with probability $\Phi(t)^{n-1}$, so its median has a closed form:

```python
print("n       median largest distractor   sqrt(2 ln n)")
for k in [10, 12, 16, 20]:
    n = 2**k
    median_max = NormalDist().inv_cdf(0.5 ** (1 / (n - 1)))   # P(max of n - 1 draws < t) = 1/2
    print(f"2^{k:<5} {median_max:>15.2f} {np.sqrt(2 * np.log(n)):>22.2f}")
```

```text
n       median largest distractor   sqrt(2 ln n)
2^10               3.20                   3.72
2^12               3.58                   4.08
2^16               4.25                   4.71
2^20               4.84                   5.27
```

The asymptotic $\sqrt{2 \ln n}$ overstates it by about half a standard deviation at practical lengths, but the growth is what matters. To stay on top as the context grows from 4K to 1M keys, a needle needs about 35% more margin, from 3.58 to 4.84 spreads. That margin has to come from the queries and keys, that is, from the representations. No choice of temperature can supply it.

**Allocation: how much weight does a top-ranked needle get?** In the diffuse regime, the distractors contribute about $(n - 1)\,e^{c^2/2}$ to $Z$, so

$$
w_{\text{needle}} \approx \frac{1}{1 + \exp\!\big(\ln(n-1) + c^2/2 - c\mu\big)} .
$$

The $\ln(n - 1)$ is a tax on context length. Each doubling of the context costs the needle $\ln 2 \approx 0.69$ of logit margin, whatever the needle is.

```python
def needle_and_rival(c, mu, g):
    """Weights on the needle (logit c*mu) and on the strongest distractor (logits c*g), per row."""
    s = c * g
    top = np.maximum(s.max(axis=1), c * mu)
    log_z = top + np.log(np.exp(s - top[:, None]).sum(axis=1) + np.exp(c * mu - top))
    return np.exp(c * mu - log_z), np.exp(s.max(axis=1) - log_z)

n0 = 2**12                                          # the length the head was trained at
scalings = {"none": lambda n: 1.0,
            "sqrt-log": lambda n: np.sqrt(np.log(n) / np.log(n0)),
            "log": lambda n: np.log(n) / np.log(n0)}

def calibrate(c0, target=0.5):
    """The needle margin that gives it `target` weight at length n0, found by bisection."""
    g = np.random.default_rng(99).standard_normal((2**22 // n0, n0 - 1))
    lo, hi = 0.0, 20.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if needle_and_rival(c0, mid, g)[0].mean() < target else (lo, mid)
    return (lo + hi) / 2

def needle_run(n, c, mu, seed):
    """Mean needle weight, mean strongest-distractor weight, and how often the needle is on top."""
    rows = max(32, 2**22 // n)
    parts = [(*needle_and_rival(c, mu, g), g.max(axis=1) < mu)
             for g in gaussian_rows(n - 1, rows, np.random.default_rng(seed))]
    return [np.concatenate(p).mean() for p in zip(*parts)]
```

## 6. Rescaling logits for longer contexts

Several methods change the logit scale with sequence length, usually to let a model trained at length $n_0$ work at length $n$. Written as the multiplier $c$ applied at length $n$ to a head trained with sharpness $c_0$, they fall into two families:

| Rule | Multiplier at length $n$ | What the REM says it keeps | Examples |
| --- | --- | --- | --- |
| none | $c_0$ | nothing | standard attention |
| square-root-log | $c_0\sqrt{\ln n / \ln n_0}$ | the regime $x$ | Jin et al. (2023), and the initialization of Giorlandino and Goldt |
| log | $c_0 \ln n / \ln n_0$ | roughly, a top needle's weight | Chiang and Cholak (2022); SSMax (Nakanishi, 2025) |

Chiang and Cholak scale attention logits by $\log n$ so that a transformer's confidence does not decay with length. SSMax replaces the softmax's $e^{z_i}$ by $n^{s z_i}$, a factor of $s \ln n$ on the logits, after finding that a learned length-dependent factor grew logarithmically with $n$. YaRN (Peng et al., 2023) fits a temperature for extending LLaMA's context, $\sqrt{1/t} = 0.1 \ln s + 1$ for an extension factor $s$, which also grows with the logarithm of the length. The table's labels follow from $x = c\,\sigma/\sqrt{2 \ln n}$: only the square-root rule holds $x$ fixed.

### The regime

First, distractors alone, starting at $x = 0.5$ at a training length of $2^{12}$:

```python
print("distractors only, x = 0.5 at n0 = 2^12")
print("n       x: none  sqrt-log  log     H / ln n: none  sqrt-log  log")
c0 = 0.5 * np.sqrt(2 * np.log(n0))
for k in [12, 14, 16, 18, 20]:
    n = 2**k
    xs_k, hs_k = [], []
    for f in scalings.values():
        c = c0 * f(n)
        h = np.concatenate([softmax_stats(c * g)[0] for g in gaussian_rows(n, max(32, 2**22 // n), np.random.default_rng(k))])
        xs_k.append(c / np.sqrt(2 * np.log(n)))
        hs_k.append(h.mean() / np.log(n))
    print(f"2^{k:<5} " + "  ".join(f"{v:>6.2f}" for v in xs_k) + "        " + "  ".join(f"{v:>6.3f}" for v in hs_k))
```

```text
distractors only, x = 0.5 at n0 = 2^12
n       x: none  sqrt-log  log     H / ln n: none  sqrt-log  log
2^12      0.50    0.50    0.50         0.755   0.755   0.755
2^14      0.46    0.50    0.54         0.787   0.752   0.713
2^16      0.43    0.50    0.58         0.811   0.748   0.665
2^18      0.41    0.50    0.61         0.833   0.751   0.636
2^20      0.39    0.50    0.65         0.851   0.753   0.601
```

The square-root rule holds the entropy share at 0.75 out to a million keys. With no rescaling, attention drifts toward uniform. The log rule drifts the other way, sharpening as $x$ climbs from 0.5 to 0.65, toward the freezing band.

### A soft head with a clear needle

Now add a needle. In a soft head, with $c_0 = 1$, the needle's margin is calibrated so it gets half the weight at the training length. That takes a margin of 8.82 spreads, far above any distractor, so ranking is never in doubt and only allocation matters:

```python
c0 = 1.0
mu = calibrate(c0)
print(f"soft head: c0 = {c0}, needle margin = {mu:.2f}")
print("n       measured: none  sqrt-log  log     predicted: none  sqrt-log  log")
for k in [12, 14, 16, 18, 20]:
    n = 2**k
    measured, predicted = [], []
    for f in scalings.values():
        c = c0 * f(n)
        measured.append(needle_run(n, c, mu, seed=k)[0])
        predicted.append(1 / (1 + np.exp(np.log(n - 1) + c * c / 2 - c * mu)))
    print(f"2^{k:<5} " + "  ".join(f"{v:>6.3f}" for v in measured)
          + "          " + "  ".join(f"{v:>6.3f}" for v in predicted))
```

```text
soft head: c0 = 1.0, needle margin = 8.82
n       measured: none  sqrt-log  log     predicted: none  sqrt-log  log
2^12     0.500   0.500   0.500           0.500   0.500   0.500
2^14     0.200   0.318   0.476           0.200   0.318   0.475
2^16     0.059   0.171   0.444           0.059   0.171   0.444
2^18     0.015   0.081   0.407           0.015   0.081   0.407
2^20     0.004   0.035   0.364           0.004   0.035   0.364
```

The formula from section 5 predicts every measurement to within 0.001, because this head never leaves the diffuse regime. With no rescaling, the needle's weight falls from 0.5 to 0.004 by a million keys: the length tax, unpaid. The square-root rule pays part of it, and the log rule pays nearly all of it, keeping 0.36. The log rule works because it grows $c\mu$ in proportion to $\ln n$, which is exactly how the tax grows. What it does not quite cover is the $c^2/2$ term, which grows faster.

### A sharp head with a marginal needle

A sharper head, $c_0 = 3$, needs a margin of only 4.23 spreads to give its needle half the weight at the training length. That is barely above the largest distractor at $2^{12}$, whose median is 3.58:

```python
c0 = 3.0
mu = calibrate(c0)
print(f"sharp head: c0 = {c0}, needle margin = {mu:.2f}")
print("n       needle on top   needle weight: none   log   strongest distractor: none   log")
for k in [12, 14, 16, 18, 20]:
    n = 2**k
    none = needle_run(n, c0, mu, seed=k)
    log = needle_run(n, c0 * scalings["log"](n), mu, seed=k)
    print(f"2^{k:<5} {none[2]:>12.2f} {none[0]:>20.3f} {log[0]:>6.3f} {none[1]:>25.3f} {log[1]:>6.3f}")
```

```text
sharp head: c0 = 3.0, needle margin = 4.23
n       needle on top   needle weight: none   log   strongest distractor: none   log
2^12            0.96                0.497  0.497                     0.109  0.109
2^14            0.81                0.190  0.303                     0.117  0.163
2^16            0.34                0.052  0.122                     0.102  0.250
2^18            0.03                0.014  0.041                     0.048  0.234
2^20            0.00                0.003  0.011                     0.027  0.251
```

![Two line charts of the weight on the needle against context length from 4K to 1M keys, for no scaling, square-root-log scaling and log scaling. Left, a soft head: the log scaling keeps the weight between 0.5 and 0.36 while the others fall toward zero. Right, a sharp head: all three fall to nearly zero by 256K keys.](./when-attention-freezes-needle.svg "Figure 2. Weight on the needle as the context grows. Left: in a soft head with a well-separated needle, the log rule keeps most of the weight. Right: when the needle's margin is small, a distractor overtakes it as the context grows, and no rescaling helps. Same code as the tables, every power of two from 2¹² to 2²⁰.")

By 256K keys the needle is on top in only 3% of contexts, and every rule fails, because the failure is in ranking, not allocation. The log rule fails in the most misleading way: it gives the strongest *distractor* a quarter of the weight. Sharpening the softmax commits the head to whatever scores highest. When that is the needle, sharpening is what you want. When it is a distractor, the head becomes confidently wrong.

### Reading the two families

The REM picture separates what the two families do:

- **The square-root rule preserves the regime.** It suits heads whose job is to pool or average, where the share of context consulted should stay the same at every length. It does not stop a specific key's weight from fading.
- **The log rule preserves focus.** It nearly pays the length tax for a key that clearly stands out, which suits retrieval. The price is a steady move toward the frozen regime, where a head commits to its top score whether or not that score is right.
- **Neither creates margin.** Whether the right key still ranks first at a longer length is decided by the representations, and the bar rises like $\sqrt{2 \ln n}$.

This suggests that the best length rule may differ by head, depending on whether the head pools or retrieves. That is a hypothesis, not something these experiments test.

## 7. Limitations

- **Independent Gaussian scores.** Trained heads have structured, correlated and heavy-tailed scores, positional effects such as the long-range decay of [rotary embeddings](/writing/position-sinusoids-and-rope/), and attention sinks. The REM is exact at initialization in the independent case. After training it is only a null model for distractors that are unrelated to the query.
- **One query, one head, one layer.** Nothing here follows a signal through depth. Rank collapse, residual connections and normalization are treated by Giorlandino and Goldt.
- **The needle model is a caricature.** One relevant key, Gaussian distractors, a fixed margin.
- **No trained models.** Every experiment is synthetic. Whether real heads sort into pooling and retrieval types that prefer different length rules is untested here.
- **Finite range.** Measurements stop at $2^{19}$ to $2^{20}$ keys. The trends shown are slow enough that extrapolating them much further is risky.

## 8. Related work

The random energy model is due to Derrida (1980, 1981); the law-of-large-numbers and central-limit thresholds for sums of random exponentials are worked out by Ben Arous, Bogachev and Molchanov (2005). Lucibello and Mézard (2023) used the REM to compute the capacity of dense associative memories, the model that Ramsauer et al. (2020) identified with transformer attention. Giorlandino and Goldt (2025) built on the same mapping a theory of signal propagation through transformers at initialization, with exact phase diagrams for rank collapse and entropy collapse, finite-size corrections, and an analysis of vanishing gradients. More recently, Levien and GrandPre (2026) found a REM-type freezing transition and a retrieval transition in the training dynamics of softmax pooling, and Alemanno (2026) found freezing of the same kind in an exponential Hopfield network that stores noisy examples. Zhai et al. (2023) connected low attention entropy to training instability. On length, Chiang and Cholak (2022), Jin et al. (2023), Peng et al. (2023) and Nakanishi (2025) proposed or fitted the length-dependent scalings compared in section 6.

## 9. Conclusion

A softmax over random scores has two regimes. Scaled attention starts deep in the diffuse one, and without the $\sqrt{d_k}$ it would start frozen. At the lengths models use, the border between the regimes is a broad band, and the infinite-length formulas are reliable only on its diffuse side. Context length enters in two separate ways: as a tax on the weight of any single key, which logit rescaling can pay, and as a rising bar on how far that key must outscore its distractors, which only better representations can clear.

**Reproducing.** Every table above is the unedited output of the code before it, run with NumPy 2.5 and fixed seeds; the whole paper runs in about ten seconds. The figures use the same functions on finer grids.
