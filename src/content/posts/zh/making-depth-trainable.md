---
title: "让深度可训练：初始化、残差与归一化"
description: 深度网络为什么一度几乎无法训练，以及改变这一局面的三个想法，每个都配有一个可以自己动手运行的小实验。
date: 2026-10-22T15:02:00Z
tags: [deep-learning, math, architecture]
series: deep-learning-foundations
chapter: 3
references: [glorot2010, he2015delving, he2015resnet, he2016identity, ioffe2015, ba2016, santurkar2018]
---

有了[上一章](/zh/writing/backpropagation-is-reverse-mode-differentiation/)的内容，任何网络无论多深，我们都能求出梯度。可事实证明，这还不够。在很长一段时间里，深度超过寥寥几层的网络就出了名地难训练：损失要么停滞不前，要么爆炸，要么下降得比浅一些的网络慢得多。梯度本身没有算错，只是派不上用场。

问题出在乘法上。深度网络是一条由许多层组成的长链，前向的信号和反向的梯度在每一层都要被乘上某个东西。把五十个略小于一的数连乘，结果几乎为零；略大于一，结果就会溢出。本章介绍驯服这条长链的三个想法：以合适的尺度初始化权重、加入残差连接、对激活值做归一化。如今每一个大模型都同时用着这三样。

## 随机层的尺度

先看一个线性层 $y = W x$，其中 $x$ 有 $n$ 个分量。假设 $x$ 和 $W$ 的各个元素相互独立、均值为零，把它们的方差分别记作 $\operatorname{Var}(x)$ 和 $\sigma^2$。每个输出都是 $n$ 个独立乘积之和，所以方差直接相加：

$$
\operatorname{Var}(y_i) = \sum_{j=1}^{n} \operatorname{Var}(W_{ij} x_j) = n \, \sigma^2 \operatorname{Var}(x)
$$

每经过一层，方差就乘上 $n\sigma^2$。经过 $L$ 层后，这个因子就变成了 $L$ 次方，所以除非 $n\sigma^2 \approx 1$，否则信号会随深度呈指数级消失或爆炸。把同样的论证倒过来走一遍就会发现，梯度在往回传的途中，每一层也会被乘上一个类似的因子。

解决办法是选好初始权重的尺度，让这个因子等于一：$\sigma^2 = 1/n$。这就是 **Glorot 初始化**（又称 Xavier 初始化）的核心。Glorot 和 Bengio（2010）还在前向传播和反向传播之间做了平衡，前者取决于输入的数量，后者取决于输出的数量，最终提出了 $\sigma^2 = 2/(n_\text{in} + n_\text{out})$。

激活函数会改变这笔账。ReLU，即 $\max(0, a)$，会把大约一半的输入置零，于是每一层都让信号的二阶矩减半。He 等人（2015）恰好针对这个因子做了修正，得到 **He 初始化**：$\sigma^2 = 2/n$。它是 ReLU 网络的标准选择。

下面看看这些选择对一个 50 层、宽度为 512 的网络有什么影响，衡量指标是激活值的均方根大小：

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

逐行来看：

- **太小**（std 0.01）：$n\sigma^2 = 0.05$，十层之后信号就没了。每一层的输出，进而每一个梯度，实际上都是零。
- **太大**（std 1）：激活值在 $0.98$ 左右，看起来很健康，但这是个陷阱。几乎每个 tanh 单元都被钉死在 $\pm 1$，那里的导数 $1 - \tanh^2$ 几乎为零，所以即使前向信号没有消失，梯度也消失了。这叫作饱和。
- **$1/\sqrt{n}$ 配 tanh**：接近正确。由于 tanh 会压缩数值，信号会慢慢缩小，但能撑过 50 层。
- **$1/\sqrt{n}$ 配 ReLU**：减半效应层层累积。经过 50 层，均方值变为原来的 $0.5^{50}$，所以均方根大小变为原来的 $0.5^{25} \approx 3 \times 10^{-8}$，正是表中显示的结果。
- **He 初始化配 ReLU**：经过 50 层，尺度的变化仍在两倍以内。

初始化管的只是训练开始那一刻的尺度。它挡不住尺度随权重的变化而漂移，也不能让非常深的网络变得容易优化。这些问题要靠另外两个想法来解决。

## 残差连接

同样在 2015 年，稍晚些时候，同一个研究组报告了一个怪现象。有了好的初始化和批归一化，他们能训练 20 层和 56 层的普通网络，但 56 层的那个，训练误差反而更高。这不是过拟合：过拟合的表现是训练误差更低、测试误差更高。更深的网络只是更难优化而已，尽管从原理上说，它完全可以照搬那个较浅的网络，再把多出来的层设成恒等映射。梯度下降就是找不到这个解。他们把这称为**退化问题**（degradation problem）。

他们的解决办法是把恒等映射设为默认行为。不再要求每个块计算一个全新的表示 $x_{l+1} = F(x_l)$，而是让它计算一个修正量，加到自己的输入上：

$$
x_{l+1} = x_l + F(x_l)
$$

如果某个块没有什么有用的东西可加，它只需学成 $F \approx 0$，这很容易。更大的影响在梯度上。把递推式从任意一层 $l$ 展开到最后一层 $L$：

$$
x_L = x_l + \sum_{i=l}^{L-1} F(x_i)
\qquad\Longrightarrow\qquad
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left( I + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} F(x_i) \right)
$$

恒等项 $I$ 意味着，无论网络有多深，总有一部分梯度会原封不动地到达第 $l$ 层。梯度不再是五十个各自都可能让它缩小的因子的乘积，而是一个含有直达路径的和。He 等人在 2016 年关于恒等映射的后续论文中给出了这个论证。正因如此，几百层的残差网络才训练得动，每个 Transformer 也都围绕着一条“残差流”来搭建。

不过这里有个隐患，用实验很容易看出来。残差求和会不断累加，所以如果不加控制，信号和梯度反而会随深度增长。下面是三个 50 层的 tanh 网络，各自从输出端反向送入一个随机梯度：一个普通堆叠，一个残差堆叠，还有一个分支按 $1/\sqrt{L}$ 缩小的残差堆叠：

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

普通堆叠在往回传的途中丢掉了大部分梯度，经过 50 层缩小到原来的八分之一。未加处理的残差堆叠则恰好相反：每一层都在恒等路径之上再叠加自己分支的贡献，梯度涨了将近四十倍。把分支缩小之后，恒等路径占了主导，梯度到达第一层时，大小和出发时差不多。

缩放分支是控制残差堆叠的一种方法。更常见的方法是归一化。

## 归一化

这个想法简单粗暴：与其指望激活值自己保持在合理的尺度上，不如强制它们这样。取一个激活值向量，减去它的均值，除以它的标准差，再让网络用两个可学习的参数 $\gamma$ 和 $\beta$ 重新缩放，以防归一化后的尺度不是它想要的：

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y = \gamma \odot \hat{x} + \beta
$$

那个很小的 $\epsilon$ 是为了避免除以零。两种常见的变体，区别只在于 $\mu$ 和 $\sigma^2$ 是在哪些数上计算的。

- **批归一化**（BatchNorm；Ioffe 和 Szegedy，2015）按特征计算它们，统计范围是一个小批次里的所有样本。它让非常深的卷积网络容易训练得多。但它把样本耦合在了一起：一张图像的输出，取决于同一批次里的其他图像。批次很小时它会出问题，而且要在推理时保持同样的行为，还得依赖滑动平均。
- **层归一化**（LayerNorm；Ba 等人，2016）按样本计算它们，统计范围是该样本自身的所有特征。它不依赖批次，训练和推理时完全一致，处理长度不一的序列也很自然。正因如此，Transformer 用的都是它，或者它的简化版近亲 RMSNorm。

### 它为什么有效？

最初的 BatchNorm 论文把它的成功解释为减少了“内部协变量偏移”（internal covariate shift）：前面的层在学习，每一层输入的分布就跟着不断变化，而归一化能让这个分布保持稳定。这个解释没能经受住时间的考验。Santurkar 等人（2018）直接检验了它，发现输入分布是否稳定，“与 BatchNorm 的成功关系不大”。他们认为，BatchNorm 真正的作用是“使优化地形显著变得更平滑”。

这又回到了[第 1 章](/zh/writing/learning-as-optimization/)。地形更平滑意味着曲率更低，而限制学习率的正是最大曲率：$\eta < 2/\lambda_1$。经过归一化的网络能承受更大的学习率，对初始化也没那么敏感，这正是它实用价值的很大一部分。这也提醒我们：一项技术之所以有效，原因可能并不是它的论文里所说的那些。

> [!TIP] 要点
> - 每一层大约会把信号的方差乘上 $n\sigma^2$，所以权重的尺度必须选得让这个因子等于一：Glorot 式初始化取 $1/n$，配合 ReLU 的 He 初始化取 $2/n$。
> - 饱和的激活值会扼杀梯度，即使前向信号看起来一切正常。
> - 残差连接 $x_{l+1} = x_l + F(x_l)$ 为梯度提供了一条贯穿网络的恒等直达路径。不过如果不加控制，残差求和会随深度增长。
> - 归一化把激活值维持在固定的尺度上。BatchNorm 在批次维度上做归一化，LayerNorm 在特征维度上做。它的好处，用损失地形变得更平滑来解释，比用“内部协变量偏移”来解释更合理。

基础的核心部分到此就齐了：一个要最小化的损失，一种计算其梯度的方法，以及一种在网络很深时梯度依然可用的架构。有两个系列以此为基础：《深入底层》带你亲手实现同样的机制，《从第一性原理理解 Transformer》则把这些部件组装成现代语言模型背后的架构，从注意力讲起。
