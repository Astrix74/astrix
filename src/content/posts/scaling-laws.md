---
title: "Scaling laws: how big a model, and how much data"
description: Loss falls as a power law in parameters, data and compute. How Kaplan and Chinchilla turned that into a rule for spending a training budget, why one of Chinchilla's fits gives the wrong answer, and why models are now trained far past the "optimum".
date: 2026-10-22T15:13:00Z
tags: [transformers, scaling, math]
series: transformers
chapter: 4
references: [kaplan2020, hoffmann2022, besiroglu2024, touvron2023, sardana2023]
---

[Chapter 2](/writing/inside-the-transformer-block/) ended with a formula for what training costs: about $6ND$ floating-point operations for a model with $N$ parameters trained on $D$ tokens. That turns a budget into a constraint, but not into a decision. With $10^{24}$ FLOPs to spend, should you train a 100-billion-parameter model on 1.7 trillion tokens, or a 20-billion-parameter model on 8 trillion? Both cost about the same.

Until 2020 the answer was mostly folklore. Then two papers, two years apart, turned it into arithmetic, and reached opposite conclusions. This chapter works through both, a replication that settled the disagreement, and why the models people actually train ignore the "optimal" answer anyway.

## Loss is a power law

Kaplan et al. (2020) trained a large family of language models and found that test loss falls smoothly as a power law in each resource, "with some trends spanning more than seven orders of magnitude". For model size, with enough data and training:

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad \alpha_N \approx 0.076, \quad N_c \approx 8.8 \times 10^{13}
$$

where $N$ counts non-embedding parameters and the loss is in nats per token. They report matching laws for data, $\alpha_D \approx 0.095$, and for compute, $\alpha_C^{\min} \approx 0.050$. A power law means a fixed *multiple* of resources buys a fixed *fraction* of loss:

```python
alpha_N, N_c = 0.076, 8.8e13            # Kaplan et al. (2020), eq. 1.1: L(N) = (N_c / N) ** alpha_N

def kaplan_loss(N):
    return (N_c / N) ** alpha_N

for N in [1e8, 1e9, 1e10, 1e11]:
    print(f"N = {N:.0e} parameters: loss {kaplan_loss(N):.3f}")
print(f"each doubling of N multiplies the loss by {2 ** -alpha_N:.3f}")
print(f"each 10x multiplies it by {10 ** -alpha_N:.3f}")
```

```text
N = 1e+08 parameters: loss 2.830
N = 1e+09 parameters: loss 2.376
N = 1e+10 parameters: loss 1.994
N = 1e+11 parameters: loss 1.674
each doubling of N multiplies the loss by 0.949
each 10x multiplies it by 0.839
```

Two things follow. Progress is slow: each doubling of model size removes about 5% of the loss. And progress is predictable: the line can be extrapolated, which is what makes it worth fitting. You can try a recipe on small models and predict what it will do at a scale you can only afford once.

### Kaplan's prescription: bigger models, fewer tokens

Kaplan et al. then asked how to split a fixed compute budget. Their answer was to put most of it into model size: the compute-optimal model size grows as $N \propto C^{0.73}$, while "data requirements" grow "very slowly as $D \sim C^{0.27}$". They concluded that the best use of compute is "training very large models and stopping significantly short of convergence". In the summary Hoffmann et al. later gave, a 10× larger budget should buy a model 5.5× bigger but only 1.8× more tokens.

The field followed. Hoffmann et al. list GPT-3 (175B parameters) and Gopher (280B) among the large models of the time trained on about 300 billion tokens each, "in line with the approach of predominantly increasing model size when increasing compute".

## Chinchilla: scale both equally

Hoffmann et al. (2022) revisited the question with over 400 models, from 70 million to over 16 billion parameters, and three independent methods for finding the compute-optimal split. Writing $N_{\text{opt}} \propto C^{a}$ and $D_{\text{opt}} \propto C^{b}$:

| Approach | $a$ (parameters) | $b$ (tokens) |
| --- | --- | --- |
| 1. Minimum over training curves | 0.50 | 0.50 |
| 2. IsoFLOP profiles | 0.49 | 0.51 |
| 3. Parametric fit of the loss | 0.46 | 0.54 |

Their conclusion was that "model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled." Their Table 3 turns this into numbers that come out to about 20 tokens per parameter. For example, a 67-billion-parameter model should see 1.5 trillion tokens.

By that measure, the 300-billion-token models were badly undertrained. Hoffmann et al. tested it directly. With the same compute budget as Gopher, they trained **Chinchilla**: 70 billion parameters on 1.4 trillion tokens, a quarter of the size and nearly five times the data. Chinchilla "uniformly and significantly outperforms" Gopher, GPT-3 and other much larger models.

Why did Kaplan get a different answer? Hoffmann et al. point mainly to the training setup. Kaplan et al. "use a fixed number of training tokens and learning rate schedule for all models", so a model checked partway through a long run is judged by a loss its schedule wasn't designed to reach. In Hoffmann et al.'s words, using those intermediate losses "results in underestimating the effectiveness of training models on less data than 130B tokens, and eventually contributes to the conclusion that model size should increase faster than training data size as compute budget increases."

## The optimal split, derived

Approach 3 fits a formula for the loss of a model with $N$ parameters trained on $D$ tokens:

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

Each term has a meaning. $E$ is the loss no model can remove: the entropy of text itself. $A / N^{\alpha}$ is the cost of a finite model, and $B / D^{\beta}$ the cost of finite data. Hoffmann et al. report $E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$, $\beta = 0.28$.

To spend a budget $C = 6ND$ optimally, substitute $D = C / 6N$ and set the derivative with respect to $N$ to zero:

$$
\frac{\partial L}{\partial N} = -\alpha A N^{-\alpha - 1} + \beta B \left(\frac{C}{6}\right)^{-\beta} N^{\beta - 1} = 0
$$

which solves to

$$
N_{\text{opt}} = G \left(\frac{C}{6}\right)^{\frac{\beta}{\alpha + \beta}}, \qquad
D_{\text{opt}} = G^{-1} \left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha + \beta}}, \qquad
G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha + \beta}}
$$

The exponents give $a = \beta / (\alpha + \beta)$, and whether model size or data should grow faster depends only on which of $\alpha$ and $\beta$ is bigger. Here is the formula applied at Chinchilla's budget, with a brute-force search over model sizes as a check:

```python
import numpy as np

fits = {   # L(N, D) = E + A / N**alpha + B / D**beta
    "printed": dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28),           # Hoffmann et al., eq. 10
    "source": dict(E=1.6934, A=406.4, B=410.7, alpha=0.3392, beta=0.2849),      # the same fit, unrounded
    "refit": dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658),    # Besiroglu et al.
}

def loss(fit, N, D):
    return fit["E"] + fit["A"] / N ** fit["alpha"] + fit["B"] / D ** fit["beta"]

def compute_optimal(fit, C):
    """Closed form: minimize loss(N, D) subject to C = 6 N D."""
    a, b = fit["alpha"], fit["beta"]
    G = (a * fit["A"] / (b * fit["B"])) ** (1 / (a + b))
    N = G * (C / 6) ** (b / (a + b))
    return N, C / (6 * N)

def brute_force(fit, C):
    """The same optimum by searching a fine grid of model sizes."""
    N = np.logspace(7, 14, 200_001)
    best = N[np.argmin(loss(fit, N, C / (6 * N)))]
    return best, C / (6 * best)

C = 5.76e23   # Gopher's training budget, the budget Chinchilla was trained with (Hoffmann et al., Table 3)
print("fit       N        D          tokens/param   grid search")
for name, fit in fits.items():
    N, D = compute_optimal(fit, C)
    N_grid, _ = brute_force(fit, C)
    print(f"{name:<8} {N / 1e9:5.1f}B   {D / 1e12:5.2f}T   {D / N:12.1f}   N = {N_grid / 1e9:.1f}B")
print(f"Chinchilla as trained: 70B on 1.4T tokens, {1.4e12 / 70e9:.0f} tokens per parameter")
```

```text
fit       N        D          tokens/param   grid search
printed   32.2B    2.98T           92.6   N = 32.2B
source    40.3B    2.38T           59.1   N = 40.3B
refit     72.2B    1.33T           18.4   N = 72.2B
Chinchilla as trained: 70B on 1.4T tokens, 20 tokens per parameter
```

The closed form and the grid search agree, so the algebra is right. The answer isn't. The fit as printed in the paper recommends a model under half Chinchilla's size, trained on more than twice the data: 93 tokens per parameter where Chinchilla used 20. The paper's own fit contradicts the model it was used to justify.

## A replication

Besiroglu et al. (2024) noticed. They reconstructed the data behind Approach 3 from the paper's figures and refit the same formula. They found that the published estimates "are inconsistent with their first two estimation methods, fail at fitting the extracted data, and report implausibly narrow confidence intervals".

They traced it to two causes, both visible in the table above.

**Rounding.** The paper's TeX source holds more precise values, for example $\beta \approx 0.2849$ rather than 0.28. Because $\beta$ is an exponent applied to numbers around $10^{11}$, the difference matters: Besiroglu et al. estimate it biases the data term by about 13% at that scale. In the table, the unrounded "source" values move the optimum from 93 to 59 tokens per parameter.

**An optimizer that stopped early.** Even unrounded, the fit is worse than the best fit of the same formula. As one of the original authors later clarified, in a note the replication cites, the fitting procedure averaged its loss over examples instead of summing it, and the optimizer stopped too soon. That also explains the confidence intervals, which were too narrow because the parameters had barely moved from their starting values.

The refit, with $a = 0.5126$, lands on 72 billion parameters and 1.3 trillion tokens at Chinchilla's budget, "around 20 tokens per parameter", consistent with Approaches 1 and 2 and with how Chinchilla was actually trained. The difference grows with scale:

```python
print("optimal tokens per parameter")
print("compute     " + "".join(f"{name:>10}" for name in fits))
for C in [1e21, 1e23, 1e25, 1e27]:
    ratios = [D / N for N, D in (compute_optimal(fit, C) for fit in fits.values())]
    print(f"{C:<10.0e}  " + "".join(f"{r:>10.1f}" for r in ratios))
```

```text
optimal tokens per parameter
compute        printed    source     refit
1e+21             50.1      34.0      21.6
1e+23             78.2      50.7      19.2
1e+25            122.1      75.7      17.1
1e+27            190.7     113.1      15.2
```

Under the printed fit, the ideal ratio climbs steadily with compute. Under the refit it stays between 15 and 22 across six orders of magnitude, which is what "scale both equally" means.

The lesson isn't about Chinchilla, whose main result survived. It is about reading fitted constants. A fit is a summary of experiments, and it can be wrong in ways the experiments are not. When a paper gives several estimates of the same quantity, check that they agree before trusting any one of them.

## Training past the optimum

"Compute-optimal" answers one question: the lowest loss for a fixed *training* budget. But a model is trained once and then run, possibly on trillions of tokens. Each token of inference costs about $2N$ FLOPs, the forward pass from chapter 2, so a smaller model is cheaper for its entire working life.

That is why the [LLaMA](https://arxiv.org/abs/2302.13971) models were trained far past the Chinchilla ratio, 1 trillion tokens for the 7B model. The paper puts the reasoning plainly: "although Hoffmann et al. (2022) recommends training a 10B model on 200B tokens, we find that the performance of a 7B model continues to improve even after 1T tokens", and a smaller model trained longer "will ultimately be cheaper at inference". Sardana et al. (2023) made the trade-off quantitative and found that with large expected demand, models should be trained "smaller and longer than Chinchilla-optimal".

The fit makes this easy to compute. Fix the quality you want, here the loss of the compute-optimal model at Chinchilla's budget, and find the model that reaches it at the lowest *total* cost, training plus serving:

```python
fit = fits["refit"]
N0, D0 = compute_optimal(fit, 5.76e23)
target = loss(fit, N0, D0)              # the quality of the compute-optimal model

def cheapest(served_tokens):
    """Model size and training tokens that reach `target` loss at the lowest total cost."""
    N = np.logspace(9.5, 12, 200_001)
    gap = target - fit["E"] - fit["A"] / N ** fit["alpha"]          # what data must still buy
    N = N[gap > 0]
    D = (fit["B"] / gap[gap > 0]) ** (1 / fit["beta"])                # tokens that close the gap
    cost = 6 * N * D + 2 * N * served_tokens                         # training + inference FLOPs
    i = np.argmin(cost)
    return N[i], D[i], cost[i]

print(f"target loss {target:.4f}")
for served in [0, 1e12, 1e13, 1e14]:
    N, D, cost = cheapest(served)
    print(f"serving {served:.0e} tokens: N = {N / 1e9:5.1f}B, D = {D / 1e12:5.2f}T, "
          f"{D / N:6.1f} tokens/param, total {cost:.2e} FLOPs")
```

```text
target loss 1.9744
serving 0e+00 tokens: N =  72.2B, D =  1.33T,   18.4 tokens/param, total 5.76e+23 FLOPs
serving 1e+12 tokens: N =  57.0B, D =  1.72T,   30.2 tokens/param, total 7.03e+23 FLOPs
serving 1e+13 tokens: N =  34.1B, D =  3.73T,  109.4 tokens/param, total 1.44e+24 FLOPs
serving 1e+14 tokens: N =  20.9B, D = 13.07T,  624.5 tokens/param, total 5.83e+24 FLOPs
```

With no inference, the answer is the Chinchilla-style optimum. Serve ten trillion tokens over the model's lifetime, and the cheapest model with the same quality is half the size, trained on nearly three times the data at 109 tokens per parameter. At a hundred trillion tokens, the best ratio is over 600. Training costs more in every one of these cases, but the total cost is lower.

One caution belongs with these numbers. They extrapolate the fit far beyond the ratios it was fitted on, and Sardana et al. found that fits built from typical ratios overestimate the benefit of extra tokens at extreme ones. Treat the direction as solid and the exact figures as rough.

> [!TIP] What to remember
> - Loss falls as a power law in parameters, data and compute: a fixed multiple of resources buys a fixed fraction of loss. That makes scale predictable.
> - Kaplan et al. put most of a budget into model size. Chinchilla showed that parameters and tokens should grow together, at about 20 tokens per parameter.
> - The compute-optimal split follows from $L = E + A/N^{\alpha} + B/D^{\beta}$ and $C = 6ND$ in a few lines. The exponents decide which should grow faster.
> - Chinchilla's printed parametric fit contradicts its own experiments. A replication traced this to rounding and an optimizer that stopped early. Check fitted constants against the paper's other evidence.
> - Compute-optimal ignores inference. When a model will serve many tokens, train it smaller and longer.

The next chapter turns to inference itself: generating text token by token, the KV cache that makes it affordable, and the tricks that make it fast. For a preview of the memory arithmetic, see the note on [KV-cache memory](/notes/kv-cache-arithmetic/).
