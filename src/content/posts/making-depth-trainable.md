---
title: "Making depth trainable: initialization, residuals and normalization"
description: Why deep networks used to be nearly impossible to train, and the three ideas that changed that, each shown with a short experiment you can run.
date: 2026-10-22T15:02:00Z
tags: [deep-learning, math, architecture]
series: deep-learning-foundations
chapter: 3
references: [glorot2010, he2015delving, he2015resnet, he2016identity, ioffe2015, ba2016, santurkar2018]
---

After the [previous chapter](/writing/backpropagation-is-reverse-mode-differentiation/) we can compute the gradient of any network, however deep. That turns out not to be enough. For a long time, networks deeper than a handful of layers were notoriously hard to train: the loss would stall, or explode, or go down far more slowly than a shallower network's. The gradient was correct; it just wasn't useful.

The problem is multiplication. A deep network is a long chain of layers, and both the forward signal and the backward gradient get multiplied by something at every layer. Multiply fifty numbers slightly smaller than one and you get almost zero; slightly larger, and you get an overflow. This chapter covers the three ideas that tamed that chain: initializing weights at the right scale, adding residual connections, and normalizing activations. Every large model today uses all three.

## The scale of a random layer

Start with one linear layer, $y = W x$, where $x$ has $n$ entries. Assume the entries of $x$ and $W$ are independent with mean zero, and call their variances $\operatorname{Var}(x)$ and $\sigma^2$. Each output is a sum of $n$ independent products, so the variances add:

$$
\operatorname{Var}(y_i) = \sum_{j=1}^{n} \operatorname{Var}(W_{ij} x_j) = n \, \sigma^2 \operatorname{Var}(x)
$$

Each layer multiplies the variance by $n\sigma^2$. After $L$ layers that factor is raised to the $L$th power, so unless $n\sigma^2 \approx 1$, the signal either vanishes or explodes exponentially with depth. The same argument run backwards says the gradient is multiplied by a similar factor at each layer on the way down.

The fix is to choose the scale of the initial weights so the factor is one: $\sigma^2 = 1/n$. That is the core of the **Glorot (or Xavier) initialization**. Glorot and Bengio (2010) also balanced the forward pass, which depends on the number of inputs, against the backward pass, which depends on the number of outputs, and proposed $\sigma^2 = 2/(n_\text{in} + n_\text{out})$.

The activation function changes the arithmetic. A ReLU, $\max(0, a)$, sets roughly half of its inputs to zero, which halves the second moment of the signal at every layer. He et al. (2015) corrected for exactly that factor and got **He initialization**, $\sigma^2 = 2/n$. It is the standard choice for ReLU networks.

Here is what those choices do to a 50-layer network of width 512, measured as the root-mean-square size of the activations:

```python
import numpy as np

rng = np.random.default_rng(0)
width, depth = 512, 50
relu = lambda a: np.maximum(a, 0)

def activation_scale(act, weight_std):
    """Push random inputs through `depth` layers; report the size of the activations."""
    x = rng.normal(size=(256, width))
    sizes = {}
    for layer in range(1, depth + 1):
        W = rng.normal(size=(width, width)) * weight_std
        x = act(x @ W)
        if layer in (1, 10, 50):
            sizes[layer] = np.sqrt((x ** 2).mean())    # root-mean-square activation
    return sizes

for name, act, std in [
    ("tanh, std 0.01      ", np.tanh, 0.01),
    ("tanh, std 1         ", np.tanh, 1.0),
    ("tanh, std 1/sqrt(n) ", np.tanh, 1 / np.sqrt(width)),
    ("relu, std 1/sqrt(n) ", relu, 1 / np.sqrt(width)),
    ("relu, std sqrt(2/n) ", relu, np.sqrt(2 / width)),
]:
    s = activation_scale(act, std)
    print(f"{name} layer 1: {s[1]:.3f}   layer 10: {s[10]:.3g}   layer 50: {s[50]:.3g}")
```

```text
tanh, std 0.01       layer 1: 0.217   layer 10: 3.37e-07   layer 50: 5.32e-33
tanh, std 1          layer 1: 0.982   layer 10: 0.982   layer 50: 0.982
tanh, std 1/sqrt(n)  layer 1: 0.630   layer 10: 0.228   layer 50: 0.103
relu, std 1/sqrt(n)  layer 1: 0.708   layer 10: 0.0349   layer 50: 3.56e-08
relu, std sqrt(2/n)  layer 1: 0.999   layer 10: 0.971   layer 50: 0.651
```

Read it row by row:

- **Too small** (std 0.01): $n\sigma^2 = 0.05$, and the signal is gone after ten layers. Every layer's output, and so every gradient, is effectively zero.
- **Too large** (std 1): the activations look healthy at about $0.98$, but that is a trap. Nearly every tanh unit is pinned at $\pm 1$, where its derivative $1 - \tanh^2$ is almost zero, so the *gradient* vanishes even though the forward signal doesn't. This is called saturation.
- **$1/\sqrt{n}$ with tanh**: close to right. The signal shrinks slowly because tanh squashes, but it survives 50 layers.
- **$1/\sqrt{n}$ with ReLU**: the halving compounds. Over 50 layers the mean square shrinks by $0.5^{50}$, so the root-mean-square size shrinks by $0.5^{25} \approx 3 \times 10^{-8}$, which is what the table shows.
- **He initialization with ReLU**: the scale stays within a factor of two over 50 layers.

Initialization fixes the scale *at the start* of training. It doesn't stop the scale from drifting as the weights change, and it doesn't make very deep networks easy to optimize. Two more ideas handle that.

## Residual connections

Later in 2015, the same group reported something odd. With good initialization and batch normalization, they could train plain networks of 20 and 56 layers, but the 56-layer one ended up with *higher training error*. That is not overfitting, which would show up as worse test error with better training error. The deeper network was simply harder to optimize, even though it could in principle copy the shallower one and set its extra layers to the identity. Gradient descent couldn't find that solution. They called it the **degradation problem**.

Their fix was to make the identity the default. Instead of asking each block to compute a new representation, $x_{l+1} = F(x_l)$, let it compute a *correction* that is added to its input:

$$
x_{l+1} = x_l + F(x_l)
$$

If a block has nothing useful to add, it only has to learn $F \approx 0$, which is easy. The bigger effect is on the gradient. Unroll the recursion from any layer $l$ to the last layer $L$:

$$
x_L = x_l + \sum_{i=l}^{L-1} F(x_i)
\qquad\Longrightarrow\qquad
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left( I + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} F(x_i) \right)
$$

The identity term $I$ means part of the gradient arrives at layer $l$ *unchanged*, however deep the network is. The gradient is no longer a product of fifty factors that can each shrink it; it is a sum with a direct path in it. He et al. made this argument in their 2016 follow-up paper on identity mappings. It is why residual networks with hundreds of layers train, and why every transformer is built around a "residual stream".

There is a catch, and it is easy to see in an experiment. The residual sum keeps adding, so without some control the signal and the gradient *grow* with depth. Here are three 50-layer tanh networks, each with a random gradient sent backwards from the output: a plain stack, a residual stack, and a residual stack whose branches are scaled down by $1/\sqrt{L}$:

```python
import numpy as np

rng = np.random.default_rng(0)
width, depth = 256, 50
Ws = [rng.normal(size=(width, width)) / np.sqrt(width) for _ in range(depth)]

def gradient_sizes(kind):
    """Backpropagate a random output gradient; report its size at a few depths."""
    alpha = 1 / np.sqrt(depth) if kind == "scaled" else 1.0   # branch multiplier
    x = rng.normal(size=(64, width))
    hs = []
    for W in Ws:                                   # forward, keeping activations
        h = np.tanh(x @ W)
        hs.append(h)
        x = h if kind == "plain" else x + alpha * h
    g = rng.normal(size=x.shape)                   # stand-in for dL/d(output)
    sizes = {depth: np.linalg.norm(g)}
    for layer in range(depth - 1, -1, -1):         # backward through each layer
        branch = (g * (1 - hs[layer] ** 2)) @ Ws[layer].T
        g = branch if kind == "plain" else g + alpha * branch
        if layer in (25, 0):
            sizes[layer] = np.linalg.norm(g)
    return sizes

for kind in ["plain", "residual", "scaled"]:
    s = gradient_sizes(kind)
    print(f"{kind:<8}  layer 50: {s[50]:.0f}   layer 25: {s[25]:.0f}"
          f"   layer 0: {s[0]:.0f}")
```

```text
plain     layer 50: 128   layer 25: 94   layer 0: 16
residual  layer 50: 128   layer 25: 434   layer 0: 4955
scaled    layer 50: 128   layer 25: 142   layer 0: 161
```

The plain stack loses most of its gradient on the way down, shrinking eightfold over 50 layers. The raw residual stack has the opposite problem: every layer adds its branch's contribution on top of the identity, and the gradient grows almost fortyfold. With the branches scaled down, the identity path dominates and the gradient arrives at the first layer at about the size it started.

Scaling branches is one way to keep a residual stack under control. The more common way is normalization.

## Normalization

The idea is blunt: rather than hoping activations stay at a reasonable scale, force them to. Take a vector of activations, subtract its mean, divide by its standard deviation, then let the network rescale it with two learned parameters $\gamma$ and $\beta$ in case the normalized scale isn't what it wants:

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y = \gamma \odot \hat{x} + \beta
$$

The small $\epsilon$ avoids dividing by zero. The two common variants differ only in which numbers $\mu$ and $\sigma^2$ are computed over.

- **Batch normalization** (Ioffe and Szegedy, 2015) computes them per feature, across the examples in a minibatch. It made very deep convolutional networks much easier to train. But it couples examples together: the output for one image depends on the other images in its batch. It misbehaves with small batches, and it needs running averages to behave the same at inference time.
- **Layer normalization** (Ba et al., 2016) computes them per example, across that example's features. No batch dependence, identical at training and inference time, and natural for sequences of different lengths. That is why transformers use it, or its simplified cousin RMSNorm.

### Why does it work?

The original BatchNorm paper explained its success as reducing *internal covariate shift*: the distribution of each layer's inputs keeps changing as earlier layers learn, and normalization holds it steady. That explanation has not aged well. Santurkar et al. (2018) tested it directly and found that the stability of input distributions "has little to do with the success of BatchNorm". Instead, they argue, it "makes the optimization landscape significantly smoother".

That connects back to [chapter 1](/writing/learning-as-optimization/). A smoother landscape means lower curvature, and the largest curvature is what limits the learning rate, $\eta < 2/\lambda_1$. Normalized networks tolerate larger learning rates and are less sensitive to initialization, which is much of their practical value. It is a useful reminder that a technique can work for reasons other than the ones in its paper.

> [!TIP] What to remember
> - Each layer multiplies the signal's variance by about $n\sigma^2$, so the weight scale must be chosen to make that factor one: $1/n$ for Glorot-style initialization, $2/n$ for He initialization with ReLU.
> - Saturated activations kill gradients even when the forward signal looks fine.
> - Residual connections, $x_{l+1} = x_l + F(x_l)$, give the gradient a direct identity path through the network. Without control, though, the residual sum grows with depth.
> - Normalization holds activations at a fixed scale. BatchNorm normalizes across the batch, LayerNorm across features. Its benefit is better explained by a smoother loss landscape than by "internal covariate shift".

That completes the core of the foundations: a loss to minimize, a way to compute its gradient, and an architecture whose gradient is usable at depth. Two series build on it. *Under the hood* implements the same machinery yourself, and *Transformers from first principles* puts these pieces together into the architecture behind modern language models, starting with attention.
