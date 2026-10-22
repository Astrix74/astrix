---
title: "Inside the transformer block: the residual stream, MLPs, and where the parameters go"
description: How attention, MLPs and normalization fit together around the residual stream, why a block has about 12d² parameters, and how to estimate a model's training compute on the back of an envelope.
date: 2026-10-22T15:11:00Z
tags: [transformers, architecture, math]
series: transformers
chapter: 2
references: [vaswani2017, elhage2021, xiong2020, hendrycks2016, shazeer2020, kaplan2020, brown2020]
---

[Chapter 1](/writing/attention-as-a-differentiable-lookup/) built attention, the operation that lets tokens read from each other. A transformer is a stack of identical **blocks**, and attention is only half of each one. This chapter assembles a complete block, explains the "residual stream" view that makes the architecture easier to think about, and then does some arithmetic. By the end you should be able to look at a model's shape (its number of layers and its width) and estimate its parameter count and its training compute to within a few percent.

## The block

A modern transformer block has two sublayers, each wrapped the same way:

$$
\begin{aligned}
x &\leftarrow x + \operatorname{Attention}\big(\operatorname{LN}(x)\big) \\
x &\leftarrow x + \operatorname{MLP}\big(\operatorname{LN}(x)\big)
\end{aligned}
$$

$\operatorname{LN}$ is layer normalization and both additions are residual connections, all from [the foundations series](/writing/making-depth-trainable/). The input to the first block is the token embeddings; the output of the last block goes through one more normalization and a linear layer that produces logits over the vocabulary.

### Where the normalization goes

The original transformer put normalization *after* each residual addition, $x \leftarrow \operatorname{LN}(x + \operatorname{Sublayer}(x))$. This is called **Post-LN**. Most models since have moved it *inside* the branch, as above. This is **Pre-LN**, and the residual path from the input to the output then passes through nothing but additions.

Xiong et al. (2020) analysed the difference. With Post-LN, at initialization "the expected gradients of the parameters near the output layer are large", which makes a large learning rate unstable. That is why Post-LN transformers need a learning-rate *warm-up*, a slow ramp at the start of training. They found that Pre-LN transformers trained without warm-up "can reach comparable results with baselines". This is the [identity-path argument](/writing/making-depth-trainable/) for residual networks again: keep the main path clean and let the branches add corrections.

## The residual stream

Here is a way of looking at the block that makes the whole architecture easier to reason about. Follow the vector $x$ for a single token through the network. It never gets replaced; every sublayer *adds* to it. So think of $x$ as a shared channel that runs the full depth of the model. Each sublayer reads from it (through a normalization and a linear projection) and writes back to it (by adding its output).

Elhage et al. (2021) call this channel the **residual stream** and build a framework for interpreting transformers around it. Two consequences of the view are worth having from the start:

- **Attention is the only operation that moves information between tokens.** Each head reads from the streams of other positions and writes into this one. Everything else in the block acts on each token independently.
- **Layers communicate through the stream, not directly.** A head in layer 3 can read something written by a head in layer 1, because both use the same channel. A layer doesn't only see "the previous layer's output"; it sees the sum of everything written so far.

## The MLP

The second sublayer is a small two-layer network applied to each token separately:

$$
\operatorname{MLP}(x) = \phi(x W_1 + b_1)\, W_2 + b_2
$$

$W_1$ expands from width $d$ to a hidden width of $4d$, $\phi$ is a nonlinearity, and $W_2$ projects back down to $d$. The original transformer used ReLU for $\phi$; many later models use **GELU** (Hendrycks and Gimpel, 2016), a smooth curve that weights each input by the probability that a standard Gaussian falls below it, $x \, \Phi(x)$. Many recent models go further and use a *gated* MLP. A popular choice is SwiGLU, from Shazeer's 2020 comparison of gated variants, in which one projection is multiplied elementwise by an activated second projection. Gated MLPs have three weight matrices instead of two, so the hidden width is usually cut by about a third to keep the parameter count unchanged.

If attention is where tokens exchange information, the MLP is where each token's representation gets *processed*: features combined, transformed, and written back to the stream. As the count below shows, it is also where most of the parameters are.

## Here is one in code

Using the `causal_self_attention` function from chapter 1, a complete Pre-LN block is about fifteen lines. The parameters are listed by name, so we can count them exactly:

```python
import numpy as np
from attention import causal_self_attention   # from chapter 1

def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

def gelu(x):                                   # the tanh approximation of GELU
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def block_shapes(d):
    """Every parameter in one pre-LN transformer block, by name."""
    return {
        "Wq": (d, d), "Wk": (d, d), "Wv": (d, d), "Wo": (d, d),   # attention
        "W1": (d, 4 * d), "b1": (4 * d,), "W2": (4 * d, d), "b2": (d,),  # MLP
        "ln1_g": (d,), "ln1_b": (d,), "ln2_g": (d,), "ln2_b": (d,),  # two LayerNorms
    }

def block(x, p, n_heads):
    attn, _ = causal_self_attention(layer_norm(x, p["ln1_g"], p["ln1_b"]),
                                    p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads)
    x = x + attn                               # attention reads the stream, writes back
    h = gelu(layer_norm(x, p["ln2_g"], p["ln2_b"]) @ p["W1"] + p["b1"])
    return x + h @ p["W2"] + p["b2"]           # so does the MLP, one token at a time

# A small block, run once.
rng = np.random.default_rng(0)
d, n_heads, n_tokens = 64, 4, 10
shapes = block_shapes(d)
p = {name: rng.normal(size=s) / np.sqrt(s[0]) for name, s in shapes.items()}
for name in ["b1", "b2", "ln1_b", "ln2_b"]:
    p[name] = np.zeros_like(p[name])
for name in ["ln1_g", "ln2_g"]:
    p[name] = np.ones_like(p[name])
x = rng.normal(size=(n_tokens, d))
print("output shape:", block(x, p, n_heads).shape)

# Parameter count against the 12 d^2 rule.
for d in [768, 4096, 12288]:
    exact = sum(np.prod(shape) for shape in block_shapes(d).values())
    print(f"d={d:<6} parameters per block {exact:>13,}   12*d^2 = {12 * d * d:>13,}")
```

```text
output shape: (10, 64)
d=768    parameters per block     7,084,800   12*d^2 =     7,077,888
d=4096   parameters per block   201,363,456   12*d^2 =   201,326,592
d=12288  parameters per block 1,812,049,920   12*d^2 = 1,811,939,328
```

The output has the same shape as the input, which is what lets blocks stack: each one is just another round of reading from and writing to the stream.

## Where the parameters go

The exact counts above sit within 0.1% of a simple rule, and it is worth seeing where the rule comes from:

- **Attention:** four $d \times d$ matrices ($W_Q, W_K, W_V, W_O$), so $4d^2$. Splitting into heads doesn't change this; the heads just partition the columns.
- **MLP:** $d \times 4d$ up and $4d \times d$ down, so $8d^2$.
- **Biases and normalization:** a few multiples of $d$, negligible next to $d^2$.

$$
N_\text{block} \approx 12 d^2, \qquad N \approx 12\, L\, d^2
$$

for a model with $L$ blocks, not counting the embedding matrix. Two-thirds of the parameters are in the MLPs. Kaplan et al. use this same approximation, $N \approx 12\, n_\text{layer}\, d_\text{model}^2$, in Table 1 of their 2020 scaling-law paper.

Check it against a real model. GPT-3's largest version has 96 layers and width 12,288, according to Table 2.1 of its paper (Brown et al., 2020). Then $12 \times 96 \times 12288^2 \approx 173.9$ billion, against the 175.0 billion the paper reports. The remaining billion or so are the embeddings, biases and normalization parameters. Knowing a model's depth and width is enough to know its size.

## Where the compute goes

The same arithmetic gives compute. Multiplying a vector of length $d$ by a $d \times k$ matrix takes $dk$ multiplications and about as many additions, so roughly $2dk$ floating-point operations (FLOPs). Every parameter in a weight matrix is used once per token in the forward pass, with one multiply and one add. So:

$$
C_\text{forward} \approx 2N \quad \text{FLOPs per token}
$$

The backward pass needs about twice as much, because it computes gradients both for the activations and for the weights. Training therefore costs about $6N$ FLOPs per token, and a full training run on $D$ tokens costs

$$
C_\text{train} \approx 6\, N\, D
$$

For GPT-3, trained on 300 billion tokens, that is $6 \times 175 \times 10^9 \times 300 \times 10^9 \approx 3.15 \times 10^{23}$ FLOPs. This formula is the starting point for every conversation about the cost of training a model, and for the [scaling laws](/writing/scaling-laws/) in chapter 4.

### What about attention itself?

The $2N$ estimate counts only the weight matrices. Attention also does work that involves no parameters: scoring each query against the previous tokens' keys, then averaging their values. For a token that attends to $n$ earlier positions, that is about $2nd$ FLOPs for the scores and $2nd$ for the averaging. Averaged over a causal sequence of length $n_\text{ctx}$, each token attends to half the context on average, which gives about $2\, n_\text{ctx}\, d$ per token per layer. Kaplan et al.'s estimate includes exactly this term: $C_\text{forward} \approx 2N + 2\, n_\text{layer}\, n_\text{ctx}\, d$.

Compare it with the weight term per layer, $2 \times 12d^2 = 24d^2$:

$$
\frac{2\, n_\text{ctx}\, d}{24\, d^2} = \frac{n_\text{ctx}}{12\, d}
$$

For GPT-3 (context 2,048, width 12,288) that ratio is about 1.4%, so the weights dominate completely. At width 4,096 with an 8,192-token context it is about 17%. Attention's quadratic cost only takes over when the context is many times longer than the model is wide, and that is why long-context models need special treatment.

> [!TIP] What to remember
> - A block is $x \leftarrow x + \operatorname{Attn}(\operatorname{LN}(x))$ followed by $x \leftarrow x + \operatorname{MLP}(\operatorname{LN}(x))$. Pre-LN keeps the residual path clean and trains without warm-up.
> - The residual stream is a shared channel: every sublayer reads from it and adds to it. Only attention moves information between positions.
> - A block has about $12d^2$ parameters, $4d^2$ in attention and $8d^2$ in the MLP. A model has about $12Ld^2$ plus embeddings.
> - Training costs about $6N$ FLOPs per token, so $6ND$ in total. Attention's own cost adds a fraction of about $n_\text{ctx}/12d$.

The block we built has one glaring omission. Nothing in it knows what order the tokens are in. The [next chapter](/writing/position-sinusoids-and-rope/) is about how transformers recover position, from the original sinusoids to the rotary embeddings most models use today.
