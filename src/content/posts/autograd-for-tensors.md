---
title: An autograd engine for tensors
description: The same algorithm as a scalar autograd engine, applied to whole arrays. Build it, learn the one new rule that broadcasting adds to the backward pass, gradient-check every operation, and train a network with it.
date: 2026-10-23T08:21:00Z
tags: [under-the-hood, autograd, numpy]
series: under-the-hood
chapter: 2
references: [rumelhart1986, baydin2015, yang2019]
---

[Backpropagation is reverse-mode differentiation](/writing/backpropagation-is-reverse-mode-differentiation/) built a forty-line autograd engine in which every node held a single number. PyTorch, JAX and every other framework run the same algorithm, with two differences: each node holds a whole tensor, and each operation's backward step is itself a tensor computation. This chapter makes that change. The algorithm stays the same; what is new is a handful of array rules, the most important of which comes from broadcasting.

## One new rule: broadcasting runs backwards as a sum

In the forward pass, NumPy broadcasting lets a bias of shape $(4,)$ be added to every row of a $(3, 4)$ matrix. As the [previous chapter](/writing/tensors-memory-shapes-and-strides/) showed, the bias is read three times through a stride of zero. In the backward pass, the multivariable chain rule says that a value used in several places receives the **sum** of the gradients from all of them. So the bias's gradient is the incoming $(3, 4)$ gradient summed over the rows:

$$
y_{ij} = x_{ij} + b_j \quad\Longrightarrow\quad \frac{\partial L}{\partial b_j} = \sum_i \frac{\partial L}{\partial y_{ij}}
$$

In general: sum the gradient over every dimension that broadcasting added or stretched, until it has the input's shape again. Forgetting this is the most common bug in hand-written backward passes, and it often fails silently, because NumPy will broadcast a wrongly shaped gradient without complaint.

## The engine

Each `Tensor` keeps its data, its gradient, the tensors it was computed from, and a function that turns its own gradient into gradients for those parents. Five operations are enough for a neural network:

```python
import numpy as np

def unbroadcast(grad, shape):
    """Sum a gradient back down to `shape`, undoing NumPy broadcasting."""
    while grad.ndim > len(shape):                 # dimensions broadcasting added on the left
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:  # dimensions stretched from size 1
            grad = grad.sum(axis=axis, keepdims=True)
    return grad

class Tensor:
    def __init__(self, data, parents=(), backward=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.parents = parents
        self._backward = backward             # turns self.grad into contributions to the parents

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))
        out._backward = lambda: (unbroadcast(out.grad, self.data.shape),
                                 unbroadcast(out.grad, other.data.shape))
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))
        out._backward = lambda: (unbroadcast(out.grad * other.data, self.data.shape),
                                 unbroadcast(out.grad * self.data, other.data.shape))
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other))
        out._backward = lambda: (out.grad @ other.data.T, self.data.T @ out.grad)
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), (self,))
        out._backward = lambda: (out.grad * (self.data > 0),)
        return out

    def sum(self, axis=None):
        out = Tensor(self.data.sum(axis=axis), (self,))
        def backward():
            g = out.grad if axis is None else np.expand_dims(out.grad, axis)
            return (np.broadcast_to(g, self.data.shape).copy(),)
        out._backward = backward
        return out

    def backward(self):
        order, seen = [], set()
        def visit(t):
            if id(t) not in seen:
                seen.add(id(t))
                for p in t.parents:
                    visit(p)
                order.append(t)
        visit(self)
        for t in order:
            t.grad = np.zeros_like(t.data)
        self.grad = np.ones_like(self.data)
        for t in reversed(order):
            if t._backward is not None:
                for parent, g in zip(t.parents, t._backward()):
                    parent.grad += g
```

The rules inside are the array versions of familiar derivatives. For `a @ b`, the gradient with respect to `a` is the incoming gradient times $b^\top$, and with respect to `b` it is $a^\top$ times the gradient. As before, the shapes force these. For `relu`, the gradient passes through where the input was positive. For `sum`, the gradient is broadcast back to the input's shape: summing is the reverse of broadcasting, and vice versa. `backward()` is the same topological walk as the scalar engine.

## A fused cross-entropy

Cross-entropy could be built from these primitives, plus `exp` and `log`. Frameworks don't do that. Exponentiating large logits overflows, so they fuse the whole thing into one operation with a numerically safe forward pass and a simple backward pass:

```python
def cross_entropy(logits, targets):
    """Mean softmax cross-entropy, computed stably, with the p - onehot gradient."""
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    rows = np.arange(len(targets))
    out = Tensor(-log_probs[rows, targets].mean(), (logits,))
    def backward():
        g = np.exp(log_probs)
        g[rows, targets] -= 1
        return (out.grad * g / len(targets),)
    out._backward = backward
    return out
```

The forward pass subtracts each row's maximum before exponentiating. That leaves the softmax unchanged and keeps every exponent at or below zero. This is the log-sum-exp trick. The backward pass is the result from [Learning as optimization](/writing/learning-as-optimization/): predicted probabilities minus the one-hot target.

## Check every operation

Every backward rule above is a place for a bug that gives slightly wrong gradients without crashing. So, as in the scalar chapter, each one is checked against central finite differences on random inputs, including the broadcasting cases and a whole two-layer network:

```python
def grad_check(f, *shapes, seed=0, h=1e-6):
    """Largest relative difference between backprop and central differences, over all inputs."""
    rng = np.random.default_rng(seed)
    inputs = [Tensor(rng.normal(size=s)) for s in shapes]
    f(*inputs).backward()
    worst = 0.0
    for t in inputs:
        for i in np.ndindex(t.data.shape):
            old = t.data[i]
            t.data[i] = old + h; up = f(*inputs).data
            t.data[i] = old - h; down = f(*inputs).data
            t.data[i] = old
            numeric = (up - down) / (2 * h)
            worst = max(worst, abs(numeric - t.grad[i]) / max(1e-8, abs(numeric) + abs(t.grad[i])))
    return worst

targets = np.array([2, 0, 1, 1])
tests = {
    "add, broadcasting a row": (lambda a, b: (a + b).sum(), (3, 4), (4,)),
    "mul, broadcasting a column": (lambda a, b: (a * b).sum(), (3, 4), (3, 1)),
    "matmul": (lambda a, b: (a @ b).sum(), (3, 4), (4, 2)),
    "relu": (lambda a: (a.relu() * a).sum(), (5, 3)),
    "sum over an axis": (lambda a: (a.sum(axis=0) * a.sum(axis=0)).sum(), (3, 4)),
    "cross-entropy": (lambda a: cross_entropy(a, targets), (4, 3)),
    "two-layer network": (lambda x, w1, w2: cross_entropy((x @ w1).relu() @ w2, targets), (4, 5), (5, 6), (6, 3)),
}
for name, (f, *shapes) in tests.items():
    print(f"{name:<28} worst relative error {grad_check(f, *shapes):.1e}")
```

```text
add, broadcasting a row      worst relative error 3.7e-10
mul, broadcasting a column   worst relative error 8.0e-11
matmul                       worst relative error 1.7e-10
relu                         worst relative error 8.4e-10
sum over an axis             worst relative error 1.7e-10
cross-entropy                worst relative error 4.5e-10
two-layer network            worst relative error 5.4e-07
```

Relative errors around $10^{-10}$ mean the analytic and numerical gradients agree to the precision finite differences allow. The two-layer network is slightly worse because ReLU has kinks, where finite differences straddle a corner. A relative error near $10^{-2}$ or larger would mean a real bug. PyTorch ships the same tool as `torch.autograd.gradcheck`. If you write a custom operation, run it.

## Train a network with it

Here is the engine training a two-layer network on three interleaved spiral arms, a problem no straight line can separate:

```python
rng = np.random.default_rng(1)
per_class, classes = 100, 3                 # three interleaved spiral arms in the plane
X = np.zeros((per_class * classes, 2))
y = np.repeat(np.arange(classes), per_class)
for c in range(classes):
    r = np.linspace(0.05, 1, per_class)
    theta = np.linspace(c * 4, (c + 1) * 4, per_class) + rng.normal(0, 0.2, per_class)
    X[y == c] = np.stack([r * np.sin(theta), r * np.cos(theta)], axis=1)

W1 = Tensor(rng.normal(0, np.sqrt(2 / 2), (2, 64)))
b1 = Tensor(np.zeros(64))
W2 = Tensor(rng.normal(0, np.sqrt(2 / 64), (64, classes)))
b2 = Tensor(np.zeros(classes))
params = [W1, b1, W2, b2]

for step in range(2001):
    logits = (Tensor(X) @ W1 + b1).relu() @ W2 + b2
    loss = cross_entropy(logits, y)
    loss.backward()
    for p in params:
        p.data -= 0.5 * p.grad
    if step % 500 == 0:
        accuracy = (logits.data.argmax(axis=1) == y).mean()
        print(f"step {step:4d}  loss {loss.data:.3f}  accuracy {accuracy:.2f}")
```

```text
step    0  loss 1.123  accuracy 0.29
step  500  loss 0.081  accuracy 1.00
step 1000  loss 0.039  accuracy 1.00
step 1500  loss 0.025  accuracy 1.00
step 2000  loss 0.018  accuracy 1.00
```

Each step builds a fresh graph by running the model, calls `backward()` and nudges every parameter against its gradient. That is the entire training loop of every framework, minus the conveniences. For contrast, here is the same loop with no hidden layer, a linear classifier:

```python
W = Tensor(np.zeros((2, classes)))
b = Tensor(np.zeros(classes))
for step in range(2001):
    logits = Tensor(X) @ W + b             # no hidden layer: a linear classifier
    loss = cross_entropy(logits, y)
    loss.backward()
    for p in (W, b):
        p.data -= 0.5 * p.grad
accuracy = (logits.data.argmax(axis=1) == y).mean()
print(f"linear model after 2,000 steps: loss {loss.data:.3f}  accuracy {accuracy:.2f}")
```

```text
linear model after 2,000 steps: loss 0.777  accuracy 0.51
```

Stuck at about half right. The hidden layer and its ReLU are what let the network bend its decision boundary around the spirals.

## What real frameworks add

This engine does what PyTorch's autograd does, at a small fraction of the engineering. The differences are worth knowing, because each one explains a habit you already have:

- **Gradients accumulate.** This engine resets every gradient on each `backward()`. PyTorch instead adds into `.grad`: leaf tensors "will have gradients accumulated into their .grad fields". That is why training loops call `optimizer.zero_grad()`, and why forgetting it silently sums gradients across steps.
- **Memory is managed.** The graph here keeps every intermediate array until the next step replaces it. Real engines save only what each backward rule needs, and free saved tensors once they have been used.
- **In-place safety.** As the previous chapter described, PyTorch's version counters catch tensors that were modified after being saved for backward.
- **Kernels and devices.** Each backward rule here is a few NumPy calls. In a framework it is a compiled kernel, possibly on a GPU, chosen by a dispatcher. Edward Yang's "PyTorch internals" walks through that machinery.
- **Higher derivatives.** Because backward rules here use plain arrays, you can't differentiate the backward pass itself. Frameworks can record the backward computation as a graph too, which is what makes gradients of gradients possible.

Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) is the classic scalar version of this engine, worth reading next to the scalar chapter. The step from it to this chapter, from numbers to arrays, is the step from a toy to the shape of a real framework.

> [!TIP] What to remember
> - A tensor autograd engine is the scalar algorithm with arrays in the nodes: record each operation and its inputs, sort the graph, and apply each backward rule in reverse.
> - Broadcasting in the forward pass becomes summation in the backward pass. Reduce every gradient back to its input's shape.
> - Fuse operations that are numerically dangerous. Softmax cross-entropy is computed with log-sum-exp, and its gradient is $p - e_y$.
> - Gradient-check every backward rule on random inputs. Wrong gradients don't crash.
> - Frameworks add gradient accumulation, memory management, in-place safety, device kernels and higher derivatives on top of the same core.

The chapters planned next build on this engine, starting with the numerics of training: floating-point formats, where they overflow, and how mixed precision works.
