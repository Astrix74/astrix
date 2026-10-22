---
title: Attention as a differentiable lookup
description: Queries, keys and values from first principles, why the scores are divided by the square root of the key dimension, and a causal multi-head attention layer in NumPy.
date: 2026-10-22T15:10:00Z
tags: [transformers, attention, math]
series: transformers
chapter: 1
references: [bahdanau2014, vaswani2017]
---

Language is full of long-range dependencies. In "The keys that I left on the kitchen table this morning *are* missing", the verb agrees with a noun ten words earlier. A model that reads text one token at a time needs some way for each token to pull in information from wherever it happens to be in the sequence.

Before transformers, that job fell to recurrent networks, which compress everything read so far into a single fixed-size state vector and carry it forward. Every fact the model will ever need has to survive being squeezed through that vector, over and over. In 2014, Bahdanau, Cho and Bengio proposed a way around the bottleneck for machine translation: let the decoder look back at *all* the encoder's states and decide, for each output word, which ones to weight. Three years later, Vaswani et al. asked what happens if you drop the recurrence entirely and build the whole model out of that looking-back operation. Their answer was the transformer, and the operation is **attention**.

This chapter builds attention from one idea: a lookup table you can differentiate.

## From a dictionary to a soft dictionary

A Python dictionary maps keys to values. You give it a query, it finds the key that *equals* the query, and returns that key's value. Two things make this useless inside a neural network: matching is all-or-nothing, and "equal" gives no gradient. Nudging the query slightly either changes nothing or changes everything.

Soften both. Instead of finding the one matching key, score the query against *every* key by similarity, turn the scores into weights that sum to one, and return the weighted average of all the values:

$$
\text{output} = \sum_j w_j \, v_j, \qquad w_j = \frac{\exp(\text{score}(q, k_j))}{\sum_m \exp(\text{score}(q, k_m))}
$$

If one key matches far better than the rest, its weight approaches one and we recover an ordinary lookup. If several match, we get a blend. Every step is smooth, so gradients flow to the query, the keys and the values. The network can *learn* what to look up.

## Self-attention

In **self-attention**, every token in a sequence plays all three roles. Each token's vector $x_i$ (of size $d$) is projected three ways by learned matrices:

$$
q_i = x_i W_Q, \qquad k_i = x_i W_K, \qquad v_i = x_i W_V
$$

The query is what this token is looking for, the key is what it offers to be found by, and the value is what it hands over when found. Separating them matters: a pronoun might *look for* a noun while *offering* its own grammatical role, and those are different pieces of information.

The similarity score is a dot product, scaled by $1/\sqrt{d_k}$ where $d_k$ is the size of the query and key vectors. (The next section explains why.) Stacking all the queries, keys and values as rows of matrices $Q$, $K$ and $V$, attention for the whole sequence is one line:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V
$$

$QK^\top$ is an $n \times n$ matrix holding every query–key score. The softmax runs along each row, so row $i$ becomes token $i$'s weights over the sequence. Multiplying by $V$ takes the weighted averages.

## Why divide by the square root of the key dimension?

Suppose the entries of $q$ and $k$ are independent with mean zero and variance one, roughly what you get from a well-initialized network. Their dot product is a sum of $d_k$ independent terms, each with variance one:

$$
\operatorname{Var}(q \cdot k) = \sum_{m=1}^{d_k} \operatorname{Var}(q_m k_m) = d_k
$$

So the typical score has size $\sqrt{d_k}$, and it grows with the width of the model. Large scores are a problem because of what softmax does with them: exponentiation exaggerates gaps, so the largest score takes nearly all the weight. The softmax output becomes almost one-hot, and its derivative,

$$
\frac{\partial w_i}{\partial s_j} = w_i \, (\delta_{ij} - w_j)
$$

goes to zero when the weights are all zeros and ones. A saturated softmax passes back almost no gradient, the same problem as the saturated tanh units in [the foundations series](/writing/making-depth-trainable/). Dividing by $\sqrt{d_k}$ brings the score variance back to one, whatever the width. The transformer paper gives exactly this reason.

It is easy to measure. Score one query against 128 random keys and look at how much weight the top key gets:

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

The score's standard deviation tracks $\sqrt{d_k}$ exactly (4, 8, 16, 32). Without scaling, a random, untrained query already puts most of its weight on a single key, and the wider the model the worse it gets. With scaling, attention starts out spread across many keys, whatever $d_k$ is, and the network is free to sharpen it through learning. The paper [When attention freezes](/papers/when-attention-freezes/) follows this further: how the weight is spread across all $n$ keys, and what changes as contexts grow.

## Causal masking

A language model is trained to predict each token from the ones before it. If token 5 could attend to token 6 during training, it could simply read the answer. A **causal mask** prevents that by setting every score $s_{ij}$ with $j > i$ to $-\infty$ before the softmax. Since $e^{-\infty} = 0$, those positions get exactly zero weight, and the attention matrix becomes lower-triangular.

The mask is also what lets a transformer train on every position of a sequence in parallel. One forward pass produces a next-token prediction at each position, each one using only the tokens it is allowed to see.

## Multiple heads

A single attention operation produces one set of weights per token: one notion of "what is relevant". Language has many kinds of relevance at once, such as syntax, coreference and topic. **Multi-head attention** runs $h$ attention operations side by side, each with its own projections and with $d_k = d/h$. It concatenates their outputs and mixes them with one more matrix $W_O$:

$$
\operatorname{MultiHead}(X) = \big[\, \text{head}_1 ; \dots ; \text{head}_h \,\big] \, W_O
$$

Because each head is $h$ times narrower, the total cost is about the same as one full-width head. You get several independent lookups for the price of one.

## A causal multi-head attention layer

Here is all of the above in NumPy. The function splits the projections into heads, scores, masks, applies softmax, mixes the values and merges the heads:

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

The weights are lower-triangular and every row sums to one. The first token can only attend to itself. The test at the end is the one worth keeping in your own code: change the future and check that the past doesn't notice. A leaky mask is one of the easiest transformer bugs to write and one of the hardest to spot from the loss alone, because the model just looks suspiciously good.

## Two properties to keep in mind

**Attention is quadratic in sequence length.** $QK^\top$ has $n^2$ entries, so doubling the context quadruples the cost of computing the scores and the memory needed to hold them. Much of the engineering behind long-context models is about working around that square.

**Attention has no idea what order the tokens are in.** Look at the formula again: nothing in it depends on positions. Shuffle the input tokens and, apart from the mask, the outputs are shuffled the same way. To attention, a sequence is a set. Chapter 3 of this series is about how transformers get word order back.

> [!TIP] What to remember
> - Attention is a soft dictionary lookup: score a query against every key, softmax the scores into weights, and average the values.
> - Queries, keys and values are separate learned projections of each token: what it looks for, what it can be found by, and what it hands over.
> - Dot products of random $d_k$-dimensional vectors have standard deviation $\sqrt{d_k}$. Dividing by it keeps the softmax from saturating, so gradients survive.
> - A causal mask sets future scores to $-\infty$, which gives parallel training without cheating. Test it by changing the future and checking the past.
> - Multiple heads give several independent lookups at roughly the cost of one. The cost of attention grows with the square of the sequence length.

Attention lets tokens exchange information, but it is only half of a transformer layer. The [next chapter](/writing/inside-the-transformer-block/) assembles a full block (attention, an MLP, normalization and the residual stream that ties them together) and counts where its parameters and compute go.
