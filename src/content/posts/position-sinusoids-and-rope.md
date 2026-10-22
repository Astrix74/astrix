---
title: "Position: sinusoids, RoPE, and why order isn't free"
description: Attention can't tell "dog bites man" from "man bites dog". How transformers get word order back, from the original sinusoids to rotary embeddings, with a check of the one property that makes RoPE work.
date: 2026-10-22T15:12:00Z
tags: [transformers, attention, math]
series: transformers
chapter: 3
references: [vaswani2017, shaw2018, su2021, press2021, touvron2023]
---

At the end of [chapter 1](/writing/attention-as-a-differentiable-lookup/) there was a warning: nothing in the attention formula depends on where a token sits in the sequence. That sounds like a technicality until you see what it means. Here is a single attention head, given "dog bites man" and then "man bites dog":

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

Who bit whom is invisible. Shuffling the inputs just shuffles the outputs the same way; the technical term is *permutation equivariance*. A causal mask breaks the symmetry a little, since each token sees a different prefix, but a model that has to infer word order from the mask alone is working with one hand tied. Transformers need position information injected deliberately. This chapter covers the three main ways of doing that, in the order they appeared.

## Absolute positions: add a vector per position

The simplest fix is to give every position $p$ its own vector $\text{PE}(p)$ and add it to the token embedding before the first layer. The token at position 5 now looks different from the same token at position 9.

Those vectors can simply be learned, one per position, like the token embeddings. The original transformer paper instead used a fixed formula: sines and cosines at a range of frequencies,

$$
\text{PE}(p, 2i) = \sin\!\left( \frac{p}{10000^{2i/d}} \right), \qquad
\text{PE}(p, 2i+1) = \cos\!\left( \frac{p}{10000^{2i/d}} \right)
$$

Each pair of dimensions $(2i, 2i+1)$ is a point going around a circle as $p$ increases, at its own speed. The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$, so the fast pairs distinguish neighbouring positions while the slow ones distinguish distant regions. It works a lot like the hands of a clock.

Vaswani et al. gave a specific reason for the choice: "for any fixed offset $k$, $\text{PE}_{pos+k}$ can be represented as a linear function of $\text{PE}_{pos}$." The reason is the angle-addition formulas. Moving $k$ positions forward rotates each pair by a fixed angle, and a rotation is a linear map. You can check it directly:

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

The same matrix works at every position. The hope was that this would make it easy for attention to learn "look three tokens back". The paper also tried learned position embeddings and "found that the two versions produced nearly identical results". The authors kept the sinusoids because they "may allow the model to extrapolate to sequence lengths longer than the ones encountered during training". Keep that hope in mind; we'll come back to it.

There is a weakness built into the additive approach. The position vector is mixed into the token's representation at the bottom of the network, and every later layer has to keep track of it alongside the content. And what attention compares is $q \cdot k$, where both sides are blends of content and absolute position. Relative distance, usually what matters for language, is in there only indirectly.

## Relative positions: tell attention the distance

Shaw, Uszkoreit and Vaswani (2018) went straight at that. Instead of adding positions to the input, they modified attention itself. When query $i$ looks at key $j$, a learned vector for the *distance* $j - i$ is added to the key, and also to the value. Distances are clipped at a maximum $k$, because "precise relative position information is not useful beyond a certain distance". On English-to-German translation this beat absolute positions by 1.3 BLEU, and combining both kinds gave no further gain.

The cost is complexity. The attention computation now needs an extra term for every query–key pair, which complicates efficient implementations. What we want is relative position with no change to the attention formula at all.

## Rotary embeddings: position as a rotation

That is what RoPE (Su et al., 2021) does. The idea follows from the sinusoids. If moving along the sequence is a rotation, then encode position by actually *rotating* the query and key vectors, and let the dot product do the rest.

Take a 2-D query $q$ at position $m$ and key $k$ at position $n$. Rotate each by an angle proportional to its position, $R(m\theta)\,q$ and $R(n\theta)\,k$, where $R(\alpha)$ is the rotation matrix for angle $\alpha$. Rotations are orthogonal, $R(\alpha)^\top = R(-\alpha)$, and compose by adding angles, so

$$
\big(R(m\theta)\, q\big)^\top \big(R(n\theta)\, k\big)
= q^\top R(-m\theta)\, R(n\theta)\, k
= q^\top R\big((n - m)\theta\big)\, k
$$

The absolute positions cancel. The score depends on the content of $q$ and $k$ and *only* on their distance $n - m$. In $d$ dimensions, RoPE splits each query and key into $d/2$ pairs and rotates pair $i$ at its own frequency, $\theta_i = 10000^{-2(i-1)/d}$, the same geometric spread as the sinusoids.

Three details are worth noticing. The rotation is applied inside every attention layer, to $q$ and $k$ only. The values and the residual stream never see positions. And it adds no parameters. Here it is, with the cancellation checked numerically:

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

The same pair of vectors scores identically at distance 3 whether they are near the start of the sequence or a thousand tokens in. Change the distance and the score changes. The paper also shows that, with this choice of frequencies, the scores tend to decay as the distance grows, a mild built-in preference for nearby tokens.

RoPE has become a standard component. Meta's LLaMA models (Touvron et al., 2023), for example, removed absolute position embeddings and used rotary embeddings instead, alongside pre-normalization, RMSNorm and the SwiGLU MLP from [the previous chapter](/writing/inside-the-transformer-block/).

## The extrapolation question

Back to the original paper's hope: that a smooth, formula-based encoding would let a model handle sequences longer than any it was trained on. Press, Smith and Lewis (2021) tested it. Their Figure 1 shows the sinusoidal, rotary and T5-style relative methods all degrading as inputs grow past the training length. A formula that is *defined* for any position is not the same as a model that has *learned* what to do with positions it has never seen.

Their alternative, ALiBi, drops position embeddings entirely and subtracts a penalty proportional to distance from each attention score, with a different slope for each head. A 1.3-billion-parameter model trained on 1,024-token inputs "extrapolates to input sequences of length 2048, achieving the same perplexity as a sinusoidal position embedding model trained on inputs of length 2048", while training 11% faster and using 11% less memory.

Long-context extension is an active area, and the details move fast. The lasting lesson is the one above. How positions are encoded determines what a model can generalize to, and longer inputs have to be tested, not assumed.

> [!TIP] What to remember
> - Without position information, attention treats a sequence as a set: permuting the input permutes the output.
> - Sinusoidal encodings add fixed sines and cosines to the input. Shifting by $k$ positions is a fixed rotation, which is a linear map. Learned absolute embeddings worked about as well in the original paper.
> - Relative encodings (Shaw et al.) put the distance into attention directly, at the cost of a more complicated attention computation.
> - RoPE rotates queries and keys by position-dependent angles. Absolute positions cancel in the dot product, so scores depend only on distance, with no parameters added.
> - Defined for any length is not the same as works at any length. Extrapolation has to be measured.

That completes the core of the transformer: attention, the block around it, and positions. The [next chapter](/writing/scaling-laws/) moves from architecture to scale: how big a model should be, and how much data it should see, for a given budget.
