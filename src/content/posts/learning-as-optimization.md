---
title: "Learning as optimization: loss, likelihood and gradient descent"
description: What it means for a neural network to learn, where the loss function comes from, and why a single learning-rate threshold decides whether training works at all.
date: 2026-10-22T15:00:00Z
tags: [deep-learning, math, optimization]
series: deep-learning-foundations
chapter: 1
references: [robbins1951, bottou2016]
---

"The model learned to recognize cats" sounds like something mysterious happened. It didn't. A neural network is a function with adjustable numbers in it, and learning means searching for the numbers that make one particular quantity, the loss, small. Everything else in deep learning is detail about which function, which loss and which search.

This chapter sets up those three pieces. By the end you will know where the standard loss functions come from (they are not arbitrary), why following the gradient works, and the one inequality that tells you whether a learning rate will converge or explode.

## The setup

We have a dataset of $n$ examples $(x_i, y_i)$: an input and the answer we want for it. We have a model $f_\theta$, a function whose behaviour depends on a vector of parameters $\theta$. For a large language model, $\theta$ has billions of entries; for the example at the end of this chapter it has nine.

A **loss function** $\ell(f_\theta(x), y)$ scores one prediction: zero when it is perfect, larger when it is worse. Averaging it over the dataset gives the **empirical risk**:

$$
\hat{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell\big(f_\theta(x_i),\, y_i\big)
$$

Training is the problem $\min_\theta \hat{L}(\theta)$. That's the whole definition.

One caveat is worth stating now, because it never goes away. We do not actually care about the examples we have; we care about the ones we will see later. The quantity we want small is the *expected* risk over the whole data distribution, and $\hat L$ is only an estimate of it from a sample. The gap between the two is called generalization, and it gets its own chapter later in this series.

## Where loss functions come from

You could invent any loss you like. In practice almost everyone uses one of two, and both fall out of the same principle: **maximum likelihood**.

Suppose the model doesn't output a single answer but a probability distribution over answers, $p_\theta(y \mid x)$. A good model should assign high probability to the answers that actually occurred. The probability of the whole dataset, treating examples as independent, is the product $\prod_i p_\theta(y_i \mid x_i)$. Products of many small numbers underflow, and logs turn products into sums, so we maximize the log instead, and flip the sign to get something to minimize:

$$
\hat{L}(\theta) = -\frac{1}{n} \sum_{i=1}^{n} \log p_\theta(y_i \mid x_i)
$$

This is the **negative log-likelihood**. Now pick a form for $p_\theta$ and see which loss falls out.

**Classification.** With $k$ classes, the network produces $k$ real numbers $z_1, \dots, z_k$ called logits, and the softmax function turns them into probabilities:

$$
p_j = \frac{e^{z_j}}{\sum_{m=1}^{k} e^{z_m}}
$$

The negative log-probability of the correct class $y$ is

$$
\ell = -\log p_y = -z_y + \log \sum_{m=1}^{k} e^{z_m}
$$

which is the **cross-entropy loss**. It is what every image classifier minimizes, and also what a language model minimizes: predicting the next token is classification over the vocabulary.

**Regression.** If the model predicts a number $\mu_\theta(x)$ and we assume the true value is that prediction plus Gaussian noise of fixed variance $\sigma^2$, then

$$
-\log p_\theta(y \mid x) = \frac{\big(y - \mu_\theta(x)\big)^2}{2\sigma^2} + \text{const}
$$

and minimizing it is exactly minimizing **squared error**.

So the choice of loss is really a choice of noise model. Squared error says "errors are Gaussian". Cross-entropy says "the answer is one of $k$ categories". When a loss behaves strangely, asking which noise model it implies is often the quickest diagnosis.

### The gradient you will see everywhere

Differentiate the cross-entropy with respect to the logits. For the correct class, $\partial \ell / \partial z_y = -1 + p_y$; for any other class $j$, $\partial \ell / \partial z_j = p_j$. Written as one vector:

$$
\frac{\partial \ell}{\partial z} = p - e_y
$$

where $e_y$ is the one-hot vector for the correct class. The gradient is just *predicted probabilities minus target probabilities*. It is bounded and well behaved, and it pushes hardest exactly where the model is most confidently wrong. This one line is the reason softmax and cross-entropy are almost always used together, and it will show up again in the next chapter as the starting point of backpropagation.

## Gradient descent

We want the $\theta$ that makes $\hat L$ smallest, and $\hat L$ is a function of possibly billions of variables. We can't search that space. What we can compute cheaply is the **gradient** $g = \nabla_\theta \hat{L}(\theta)$: the direction in which the loss increases fastest.

A first-order Taylor expansion says what happens if we take a small step of size $\eta$ against it:

$$
\hat{L}(\theta - \eta g) \approx \hat{L}(\theta) - \eta \, \lVert g \rVert^2
$$

For small enough $\eta > 0$ the loss goes down, unless the gradient is already zero. That gives the update rule

$$
\theta \leftarrow \theta - \eta \, \nabla_\theta \hat{L}(\theta)
$$

and $\eta$ is the **learning rate**. The "small enough" in that argument is doing a lot of work, so let's make it exact.

### How big can a step be?

Take the simplest possible loss, a one-dimensional bowl: $L(\theta) = \tfrac{a}{2}\theta^2$ with $a > 0$. Its gradient is $a\theta$, so one step of gradient descent gives

$$
\theta_{t+1} = \theta_t - \eta a \theta_t = (1 - \eta a)\, \theta_t
$$

After $t$ steps, $\theta_t = (1 - \eta a)^t \theta_0$. This shrinks to the minimum at zero exactly when $|1 - \eta a| < 1$, that is when

$$
0 < \eta < \frac{2}{a}
$$

Above that threshold every step overshoots by more than it corrects, and the iterate grows without bound. Here it is with $a = 4$, so the threshold is $0.5$:

```python
def descend(eta, a=4.0, theta=1.0, steps=10):
    """Gradient descent on L(theta) = a/2 * theta^2, whose gradient is a * theta."""
    for _ in range(steps):
        theta -= eta * a * theta
    return theta

for eta in [0.1, 0.4, 0.49, 0.6]:
    print(f"eta={eta:<4}  theta after 10 steps: {descend(eta):+.4f}")
```

```text
eta=0.1   theta after 10 steps: +0.0060
eta=0.4   theta after 10 steps: +0.0060
eta=0.49  theta after 10 steps: +0.6648
eta=0.6   theta after 10 steps: +28.9255
```

Two things are worth noticing. A learning rate of $0.4$ converges just as fast as $0.1$, even though every step overshoots the minimum and flips the sign: $(1 - 1.6) = -0.6$ has the same magnitude as $(1 - 0.4) = 0.6$. And the boundary is sharp. At $0.49$ the iterate barely moves; at $0.6$ it has grown almost thirtyfold.

Real losses are not one-dimensional bowls, but near a minimum they look like many bowls at once, one along each eigenvector of the Hessian (the matrix of second derivatives), with curvatures equal to its eigenvalues $\lambda_1 \ge \dots \ge \lambda_d$. The same analysis applies in each direction, so:

- the *largest* curvature sets the speed limit, $\eta < 2/\lambda_1$
- the *smallest* curvature, at that speed, sets how slowly the flattest direction converges.

Their ratio $\lambda_1 / \lambda_d$, the condition number, measures how hard the problem is for plain gradient descent. Much of the rest of this series, including normalization layers, careful initialization and adaptive optimizers, can be read as attempts to make that ratio smaller.

## Stochastic gradient descent

Computing $\nabla \hat L$ exactly means a pass over all $n$ examples for every single step. With millions of examples that is far too slow. The fix is to estimate the gradient from a small random **minibatch** $B$ instead:

$$
g_B = \frac{1}{|B|} \sum_{i \in B} \nabla_\theta \, \ell\big(f_\theta(x_i), y_i\big)
$$

If the minibatch is drawn uniformly at random, $g_B$ is an unbiased estimate of the full gradient: right on average, noisy on any single step. The cost of a step now depends on the batch size, not the dataset size.

This idea is older than neural networks. Robbins and Monro studied it in 1951 as *stochastic approximation*: finding the root of a function you can only measure with noise. Their analysis gives the classic conditions on a decaying step size $\eta_t$ for the noise to average out:

$$
\sum_{t=1}^{\infty} \eta_t = \infty, \qquad \sum_{t=1}^{\infty} \eta_t^2 < \infty
$$

The first says the steps never become so small that progress stops; the second says they shrink fast enough that the noise does not keep kicking the iterate around forever. $\eta_t = 1/t$ satisfies both. Modern training rarely follows that schedule literally, but the trade-off it describes, large steps for progress and small steps for precision, is why learning rates are usually warmed up and then decayed.

Why does such a noisy method work so well at scale? Bottou, Curtis and Nocedal's 2016 survey makes the case carefully. The short version: when data is plentiful, a cheap noisy step that you can take a thousand times beats an exact step that you can take once.

## Putting it together

Here is the whole chapter in about thirty lines: a softmax classifier on three noisy clusters of 2-D points, trained with minibatch SGD. The model is linear, with logits $z = xW + b$, so it has $2 \times 3 + 3 = 9$ parameters, and the gradient uses the $p - e_y$ result from above.

```python
import numpy as np

rng = np.random.default_rng(0)
n, d, k = 600, 2, 3                       # examples, input features, classes
centers = np.array([[0.0, 2.0], [-2.0, -1.0], [2.0, -1.0]])
y = rng.integers(0, k, n)
X = centers[y] + rng.normal(size=(n, d))  # three noisy blobs

W, b = np.zeros((d, k)), np.zeros(k)

def loss_and_grads(X, y, W, b):
    z = X @ W + b                              # logits, shape (batch, k)
    z -= z.max(axis=1, keepdims=True)          # softmax ignores a constant shift
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)          # predicted probabilities
    rows = np.arange(len(y))
    loss = -np.log(p[rows, y]).mean()          # cross-entropy
    p[rows, y] -= 1                            # dL/dz = p - onehot(y) ...
    p /= len(y)                                # ... averaged over the batch
    return loss, X.T @ p, p.sum(axis=0)

eta, batch = 0.5, 32
for step in range(301):
    if step % 100 == 0:                        # measure on all the data
        loss, _, _ = loss_and_grads(X, y, W, b)
        accuracy = ((X @ W + b).argmax(axis=1) == y).mean()
        print(f"step {step:3d}  loss {loss:.3f}  accuracy {accuracy:.2f}")
    idx = rng.integers(0, n, batch)            # a random minibatch
    _, dW, db = loss_and_grads(X[idx], y[idx], W, b)
    W -= eta * dW
    b -= eta * db
```

```text
step   0  loss 1.099  accuracy 0.31
step 100  loss 0.130  accuracy 0.95
step 200  loss 0.126  accuracy 0.96
step 300  loss 0.126  accuracy 0.95
```

The starting loss is no accident: with all weights at zero, every class gets probability $1/3$, and $-\log(1/3) = \log 3 \approx 1.099$. Checking that the initial loss equals $\log k$ is a cheap sanity test for any classifier. (The starting accuracy of $0.31$ is just the share of the first class, because ties in `argmax` go to index 0.)

The accuracy wobbles between $0.95$ and $0.96$ after the loss has settled. That is the minibatch noise from the previous section, still nudging the parameters at a fixed learning rate.

Try breaking it: set `eta` to `50` and watch the loss, or set `batch` to `1` and watch how much the accuracy jumps around.

> [!TIP] What to remember
> - Learning is minimizing an average loss over the data. The standard losses come from maximum likelihood: cross-entropy for categories, squared error for Gaussian noise.
> - The gradient of softmax cross-entropy with respect to the logits is $p - e_y$.
> - Gradient descent on a bowl of curvature $a$ converges only if $\eta < 2/a$. In many dimensions, the sharpest direction sets the limit.
> - Minibatch gradients are unbiased but noisy, and their cost does not grow with the dataset. That trade is what makes training at scale possible.

The model here was linear, so its gradient could be written by hand. The next chapter covers what happens when the model is a deep stack of layers: how to get the gradient of anything, automatically, for roughly the cost of running the model once.
