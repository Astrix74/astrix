---
title: 五行讲清 Adam 与 AdamW
date: 2026-10-22T15:42:00Z
references: [kingma2014, loshchilov2017, touvron2023]
---

[Adam](https://arxiv.org/abs/1412.6980)（Kingma 和 Ba，2014）为每个参数维护两个滑动平均：一个是梯度 $g_t$ 的平均（即动量），另一个是梯度平方的平均（即每个参数各自的尺度）：

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\
\hat m_t &= m_t / (1 - \beta_1^t), \quad \hat v_t = v_t / (1 - \beta_2^t) \\
\theta_t &= \theta_{t-1} - \eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)
\end{aligned}
$$

带“帽子”的两项用来修正偏差，这个偏差来自两个平均值都从零起步。除以 $\sqrt{\hat v_t}$ 让每一步的更新幅度大致与梯度的尺度无关。这是 Adam 最主要的长处，也是一个隐蔽 bug 的根源。

这个 bug 就是权重衰减。对普通的 SGD 来说，在损失里加一个 L2 惩罚项 $\tfrac{\lambda}{2}\lVert\theta\rVert^2$，等价于每一步都把权重按一个固定比例缩小。[Loshchilov 和 Hutter](https://arxiv.org/abs/1711.05101) 指出，“对标准随机梯度下降而言，$L_2$ 正则化与权重衰减正则化是等价的（在按学习率重新缩放之后）”，但对 Adam 这样的自适应方法并不成立。在 Adam 中，惩罚项的梯度 $\lambda\theta$ 和其余各项一样，也要除以 $\sqrt{\hat v_t}$。于是梯度大的参数几乎得不到衰减，梯度小的参数却被大幅衰减。

**AdamW** 把两者解耦：它只用损失的梯度来跑 Adam，然后直接缩小权重。这就是第五行：

$$
\theta_t \leftarrow \theta_t - \eta\, \lambda\, \theta_{t-1}
$$

下面拿一个损失根本不在意的参数来看两者的区别，它的梯度是纯噪声，取两种不同的尺度：

```python
import numpy as np

def run(decoupled, grad_scale, steps=2000, lr=1e-3, wd=0.1, b1=0.9, b2=0.999, eps=1e-8):
    rng = np.random.default_rng(0)
    theta, m, v = 1.0, 0.0, 0.0
    for t in range(1, steps + 1):
        g = grad_scale * rng.normal()    # pure noise: the loss ignores theta
        if decoupled:
            theta -= lr * wd * theta     # AdamW: decay applied to the weight itself
        else:
            g += wd * theta              # Adam + L2: decay inside the gradient
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)
        theta -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta

for name, decoupled in [("Adam + L2", False), ("AdamW    ", True)]:
    small, large = run(decoupled, 0.01), run(decoupled, 10.0)
    print(f"{name}  weight after 2000 steps: small gradients {small:.3f},"
          f" large gradients {large:.3f}")
```

```text
Adam + L2  weight after 2000 steps: small gradients 0.029, large gradients 1.033
AdamW      weight after 2000 steps: small gradients 0.868, large gradients 0.868
```

把 L2 放进 Adam 内部时，同样的惩罚会抹掉梯度小的那个权重，对梯度大的那个却毫无作用。换成 AdamW，两者的衰减完全一样。单靠衰减本身会剩下 $(1 - 10^{-4})^{2000} \approx 0.82$，其余的差别来自噪声的随机游走；经过 Adam 归一化之后，这种随机游走在任何梯度尺度下看起来都一样。如今，解耦的权重衰减是 Transformer 的常见选择。例如 [LLaMA](https://arxiv.org/abs/2302.13971) 就是用 AdamW 训练的，权重衰减为 0.1。
