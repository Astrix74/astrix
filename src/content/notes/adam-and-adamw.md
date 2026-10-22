---
title: Adam and AdamW in five lines
date: 2026-10-22T15:42:00Z
references: [kingma2014, loshchilov2017, touvron2023]
---

[Adam](https://arxiv.org/abs/1412.6980) (Kingma and Ba, 2014) keeps two running averages per parameter: of the gradient $g_t$ (momentum) and of its square (a per-parameter scale):

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\
\hat m_t &= m_t / (1 - \beta_1^t), \quad \hat v_t = v_t / (1 - \beta_2^t) \\
\theta_t &= \theta_{t-1} - \eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)
\end{aligned}
$$

The hats correct the bias from starting both averages at zero. Dividing by $\sqrt{\hat v_t}$ makes the step size roughly independent of the gradient's scale. That is Adam's main strength, and the source of one subtle bug.

The bug is weight decay. With plain SGD, adding an L2 penalty $\tfrac{\lambda}{2}\lVert\theta\rVert^2$ to the loss is the same as shrinking the weights by a constant factor each step. [Loshchilov and Hutter](https://arxiv.org/abs/1711.05101) point out that "$L_2$ regularization and weight decay regularization are equivalent for standard stochastic gradient descent (when rescaled by the learning rate)", but not for adaptive methods like Adam. In Adam, the penalty's gradient $\lambda\theta$ is divided by $\sqrt{\hat v_t}$ like everything else. Parameters with large gradients get almost no decay, and parameters with small gradients get a lot.

**AdamW** decouples the two. It runs Adam on the loss gradient alone, then shrinks the weights directly. That is the fifth line:

$$
\theta_t \leftarrow \theta_t - \eta\, \lambda\, \theta_{t-1}
$$

Here is the difference on a parameter the loss doesn't care about, with gradients of pure noise at two different scales:

```python
import numpy as np

def run(decoupled, grad_scale, steps=2000, lr=1e-3, wd=0.1, b1=0.9, b2=0.999, eps=1e-8):
    rng = np.random.default_rng(0)
    theta, m, v = 1.0, 0.0, 0.0
    for t in range(1, steps + 1):
        g = grad_scale * rng.normal()    # pure noise: the loss ignores theta
        if not decoupled:
            g += wd * theta              # Adam + L2: decay inside the gradient
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)
        theta -= lr * m_hat / (np.sqrt(v_hat) + eps)
        if decoupled:
            theta -= lr * wd * theta     # AdamW: decay applied to the weight
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

With L2 inside Adam, the same penalty wipes out the weight with small gradients and does nothing to the one with large gradients. With AdamW, both decay identically. The decay itself would leave $(1 - 10^{-4})^{2000} \approx 0.82$, and the rest is the noise's random walk, which looks the same at any gradient scale once Adam has normalized it. Decoupled decay is now the common choice for transformers. [LLaMA](https://arxiv.org/abs/2302.13971), for example, was trained with AdamW and a weight decay of 0.1.
