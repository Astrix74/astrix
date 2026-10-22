---
title: Backpropagation is reverse-mode differentiation
description: How to get the gradient of a loss with respect to millions of parameters for about the cost of running the model once, with a working autograd engine in forty lines.
date: 2026-10-22T15:01:00Z
tags: [deep-learning, math, autodiff]
series: deep-learning-foundations
chapter: 2
references: [rumelhart1986, baydin2015]
---

The [previous chapter](/writing/learning-as-optimization/) ended with a linear model whose gradient fit on one line. A real network is dozens or hundreds of layers deep, and gradient descent needs the derivative of the loss with respect to every one of its parameters, at every step.

Doing that one parameter at a time would be hopeless. Nudging each of a billion parameters and re-running the network to see what changes would cost a billion forward passes per step. Backpropagation gets all of those derivatives for roughly the cost of *one* extra pass. This chapter shows why that is possible, works through it by hand once, and then builds a tiny automatic-differentiation engine that does it for any expression.

## Functions as graphs

Any computation a network performs can be broken into elementary operations (additions, multiplications, a `tanh`), each feeding its result into later ones. Draw each intermediate value as a node and each "is used to compute" relation as an edge, and you get a **computational graph**.

Take a small function of two inputs:

$$
f(x, y) = \tanh(xy + x) \cdot y
$$

Broken into steps, with $x = 0.5$ and $y = -1.5$:

| Step | Value |
| --- | --- |
| $a = x \cdot y$ | $-0.75$ |
| $u = a + x$ | $-0.25$ |
| $t = \tanh(u)$ | $-0.2449$ |
| $f = t \cdot y$ | $0.3674$ |

Each step is trivial to differentiate *locally*: $\partial f / \partial t = y$, $\partial t / \partial u = 1 - t^2$, and so on. The chain rule says how to combine local derivatives into the derivative we actually want. The only question is the order in which to combine them.

## Two ways to apply the chain rule

For a chain of functions $f = f_3 \circ f_2 \circ f_1$, the derivative is a product of local Jacobians:

$$
\frac{\partial f}{\partial x} = \frac{\partial f_3}{\partial f_2} \cdot \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial x}
$$

Matrix multiplication is associative, so we can evaluate this product from either end.

- **Forward mode** starts at the input, and asks: if I wiggle $x$, how does every intermediate value wiggle? One pass gives the derivative of *every output* with respect to *one input*.
- **Reverse mode** starts at the output, and asks: how much does the output care about each intermediate value? One pass gives the derivative of *one output* with respect to *every input*.

For a function from $n$ inputs to $m$ outputs, forward mode needs $n$ passes for the full Jacobian and reverse mode needs $m$. Training has a huge $n$ (all the parameters) and $m = 1$ (the loss is a single number). Reverse mode wins by a factor of $n$.

The general technique is called **automatic differentiation**, and Baydin et al.'s 2015 survey covers it as it is used in machine learning, history included. It is neither symbolic differentiation (no formulas are manipulated) nor numerical differentiation (no finite differences). It applies the chain rule to the actual numbers flowing through the program. Rumelhart, Hinton and Williams' 1986 paper showed what reverse mode makes possible for neural networks: hidden layers that learn useful internal representations on their own. Applied to networks, reverse mode goes by the name **backpropagation**.

### Doing it by hand

Reverse mode keeps one number per node, its **adjoint**: the derivative of the final output with respect to that node, written $\bar v = \partial f / \partial v$. Start with $\bar f = 1$ and walk the graph backwards. Each node passes its adjoint to its inputs, multiplied by the local derivative.

1. $f = t \cdot y$, so $\bar t = \bar f \cdot y = -1.5$, and $y$ receives $\bar f \cdot t = -0.2449$.
2. $t = \tanh(u)$, so $\bar u = \bar t \cdot (1 - t^2) = -1.5 \times 0.94 = -1.41$.
3. $u = a + x$, so $\bar a = \bar u = -1.41$, and $x$ receives $\bar u = -1.41$.
4. $a = x \cdot y$, so $x$ receives $\bar a \cdot y = 2.115$, and $y$ receives $\bar a \cdot x = -0.705$.

Notice that $x$ and $y$ each received *two* contributions, because each is used twice. The multivariable chain rule says to add them:

$$
\frac{\partial f}{\partial x} = -1.41 + 2.115 = 0.705, \qquad \frac{\partial f}{\partial y} = -0.2449 - 0.705 = -0.9499
$$

Each edge of the graph was visited exactly once, with one multiplication and one addition. That is where the efficiency comes from. The work of the backward pass is proportional to the size of the graph, which is the work of the forward pass. Getting the gradient costs a small constant multiple of evaluating the function, however many inputs there are.

## A forty-line autograd engine

Everything above is mechanical enough to automate. Here is a scalar `Value` type that records, for every operation, its inputs and its local derivatives. `backward()` sorts the graph so that each node comes after the nodes it was computed from, then walks that order in reverse, exactly as we just did by hand.

```python
import math

class Value:
    """A number that remembers how it was computed, so gradients can flow back."""

    def __init__(self, data, parents=(), local_grads=()):
        self.data = data
        self.grad = 0.0
        self.parents = parents          # the Values this one was computed from
        self.local_grads = local_grads  # d(self)/d(parent), one per parent

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    __radd__ = __add__
    __rmul__ = __mul__

    def tanh(self):
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t * t,))

    def backward(self):
        # Put every node after the nodes it was computed from ...
        order, seen = [], set()
        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for parent in v.parents:
                    visit(parent)
                order.append(v)
        visit(self)
        # ... then walk that order backwards, applying the chain rule once per edge.
        self.grad = 1.0
        for v in reversed(order):
            for parent, local in zip(v.parents, v.local_grads):
                parent.grad += local * v.grad
```

The `+=` on the last line is the "add the contributions" rule. Replace it with `=` and the engine silently gives wrong answers for any value used more than once. That bug is common enough that you should check for it in any autograd code you write.

### Always check gradients numerically

Backprop code is easy to get subtly wrong, and wrong gradients don't crash anything; training just goes badly. The standard defence is a **gradient check**: compare against a central finite difference, $\big(f(x + h) - f(x - h)\big) / 2h$, which is slow but hard to get wrong.

```python
def f(x, y):
    return (x * y + x).tanh() * y      # x appears twice, y appears twice

x, y = Value(0.5), Value(-1.5)
f(x, y).backward()

def numerical(fx, h=1e-6):             # central difference, for comparison
    return (fx(h) - fx(-h)) / (2 * h)

dx = numerical(lambda e: f(Value(0.5 + e), Value(-1.5)).data)
dy = numerical(lambda e: f(Value(0.5), Value(-1.5 + e)).data)
print(f"df/dx  backprop {x.grad:.6f}   numerical {dx:.6f}")
print(f"df/dy  backprop {y.grad:.6f}   numerical {dy:.6f}")
```

```text
df/dx  backprop 0.705011   numerical 0.705011
df/dy  backprop -0.949930   numerical -0.949930
```

The engine agrees with the hand calculation and with finite differences to six decimal places.

## From scalars to matrices

Real frameworks work on tensors, not scalars, but the rules are the same. They are just applied to whole arrays at once. The most important case is a linear layer $Y = XW$, where $X$ is a batch of inputs, one row per example. Given the adjoint $\bar Y = \partial L / \partial Y$ coming back from later layers:

$$
\bar W = X^\top \bar Y, \qquad \bar X = \bar Y \, W^\top
$$

You don't need to memorize these. The shapes force them. $\bar W$ must have the shape of $W$, and there is only one way to combine $X$ and $\bar Y$ to get it. (This is also why `X.T @ p` appeared in the previous chapter's code.) An elementwise function like $h = \tanh(a)$ simply multiplies the adjoint elementwise by its derivative: $\bar a = \bar h \odot (1 - h^2)$.

With just those rules, here is a two-layer network learning XOR, the classic function no linear model can represent. The forward pass keeps its intermediate values, and the backward pass consumes them in reverse:

```python
import numpy as np

rng = np.random.default_rng(1)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)          # XOR: not linearly separable

W1, b1 = rng.normal(size=(2, 8)), np.zeros(8)
W2, b2 = rng.normal(size=(8, 1)), np.zeros(1)

for step in range(401):
    # Forward: keep every intermediate, the backward pass needs them.
    a = X @ W1 + b1
    h = np.tanh(a)
    out = h @ W2 + b2
    loss = ((out - y) ** 2).mean()

    # Backward: the same steps in reverse, one chain-rule product each.
    d_out = 2 * (out - y) / len(y)
    dW2, db2 = h.T @ d_out, d_out.sum(axis=0)
    d_h = d_out @ W2.T
    d_a = d_h * (1 - h ** 2)                              # tanh'(a) = 1 - tanh(a)^2
    dW1, db1 = X.T @ d_a, d_a.sum(axis=0)

    for param, grad in [(W1, dW1), (b1, db1), (W2, dW2), (b2, db2)]:
        param -= 0.1 * grad
    if step % 100 == 0:
        print(f"step {step:4d}  loss {loss:.5f}  predictions {out.ravel().round(2)}")
```

```text
step    0  loss 0.95999  predictions [ 0.   -0.22 -0.43 -0.55]
step  100  loss 0.03574  predictions [0.14 0.83 0.81 0.24]
step  200  loss 0.00173  predictions [0.03 0.96 0.96 0.05]
step  300  loss 0.00004  predictions [0.   0.99 0.99 0.01]
step  400  loss 0.00000  predictions [0. 1. 1. 0.]
```

(The first prediction starts at exactly zero because the input $(0, 0)$ and the zero biases make every pre-activation zero.)

## The price: memory

Reverse mode is cheap in compute but not in memory. The backward pass needs the forward pass's intermediate values (`h` above, and every activation in a real network), so they all have to be kept until the backward pass reaches them. For large models, stored activations, not parameters, are often what fills accelerator memory during training.

The standard trade is **activation checkpointing**: keep only some activations and recompute the rest during the backward pass. You spend extra compute to save memory, the reverse of the trade backprop itself makes.

> [!TIP] What to remember
> - A network is a graph of simple operations. The chain rule combines their local derivatives, and the order of combination is a choice.
> - Reverse mode, which is backpropagation, gets the derivative of one output with respect to every input in one backward pass. A loss is one output, so the full gradient costs about as much as a forward pass.
> - When a value feeds into several places, its gradient is the *sum* of the contributions.
> - For a layer $Y = XW$: $\bar W = X^\top \bar Y$ and $\bar X = \bar Y W^\top$. Let the shapes guide you.
> - Gradient-check new backprop code against finite differences. Wrong gradients fail silently.

We can now compute gradients for networks of any depth. But having the gradient is not the same as being able to train. The [next chapter](/writing/making-depth-trainable/) is about what goes wrong when networks get deep, and the three ideas that fixed it.
