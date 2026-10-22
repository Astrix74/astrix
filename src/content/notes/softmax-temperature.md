---
title: Softmax temperature in one formula
date: 2026-10-22T15:40:00Z
references: [hinton2015]
---

When a language model samples its next token, it turns its logits $z_i$ into probabilities with a softmax. Temperature $T$ is a single division inserted before it:

$$
p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

Dividing by $T$ scales every gap between logits. The softmax depends only on those gaps, since adding a constant to every logit changes nothing, so temperature controls how much the largest logit dominates:

```python
import numpy as np

logits = np.array([2.0, 1.0, 0.1])
for T in [0.25, 0.5, 1.0, 2.0, 10.0]:
    p = np.exp(logits / T)
    p /= p.sum()
    print(f"T={T:<5} probabilities {p.round(3)}")
```

```text
T=0.25  probabilities [0.982 0.018 0.   ]
T=0.5   probabilities [0.864 0.117 0.019]
T=1.0   probabilities [0.659 0.242 0.099]
T=2.0   probabilities [0.502 0.304 0.194]
T=10.0  probabilities [0.366 0.331 0.303]
```

- **$T = 1$** is the model's own distribution, the one it was trained to produce.
- **$T \to 0$** sends all the probability to the largest logit. Sampling becomes greedy decoding: always pick the top token.
- **$T \to \infty$** flattens everything toward uniform, and the model's preferences stop mattering.

Lower temperatures give more predictable, repetitive text; higher ones give more varied and more error-prone text. Temperature never changes the *ranking* of tokens, only how sharply the top ones are preferred.

The same knob shows up in training. [Knowledge distillation](https://arxiv.org/abs/1503.02531) (Hinton, Vinyals and Dean, 2015) raises the temperature of a large model's softmax so its small probabilities, which say which wrong answers are *almost* right, become large enough for a smaller model to learn from. And the $1/\sqrt{d_k}$ in attention is effectively a fixed temperature, chosen to stop the softmax from saturating; see [Attention as a differentiable lookup](/writing/attention-as-a-differentiable-lookup/). Over many keys at once, temperature decides whether attention is spread out or frozen onto a few keys, which is the subject of the paper [When attention freezes](/papers/when-attention-freezes/).
