---
title: 反向传播就是反向模式自动微分
description: 如何以大约运行一次模型的代价，求出损失对数百万个参数的梯度，并附上一个只有四十行、可以实际运行的 autograd 引擎。
date: 2026-10-22T15:01:00Z
tags: [deep-learning, math, autodiff]
series: deep-learning-foundations
chapter: 2
references: [rumelhart1986, baydin2015]
---

[上一章](/zh/writing/learning-as-optimization/)结尾的那个线性模型，梯度一行就能写完。而真实的网络有几十甚至上百层，梯度下降每走一步，都需要损失对其中每一个参数的导数。

一个参数一个参数地去求，是毫无希望的。把十亿个参数逐个轻推一下，再重新运行网络看看结果怎么变，每一步就得做十亿次前向传播。反向传播只需大约多做**一次**传播的代价，就能拿到所有这些导数。本章会说明这为什么做得到，先手算一遍，然后搭一个小小的自动微分引擎，让它对任意表达式都能自动完成这件事。

## 把函数看成图

网络执行的任何计算，都可以拆成一系列基本运算（加法、乘法、一个 `tanh`），每个运算的结果再送进后面的运算。把每个中间值画成一个节点，把每一条“被用来计算”的关系画成一条边，就得到了一张**计算图**。

以一个有两个输入的小函数为例：

$$
f(x, y) = \tanh(xy + x) \cdot y
$$

取 $x = 0.5$、$y = -1.5$，把它拆成几步：

| 步骤 | 值 |
| --- | --- |
| $a = x \cdot y$ | $-0.75$ |
| $u = a + x$ | $-0.25$ |
| $t = \tanh(u)$ | $-0.2449$ |
| $f = t \cdot y$ | $0.3674$ |

每一步单独在局部求导都很简单：$\partial f / \partial t = y$，$\partial t / \partial u = 1 - t^2$，依此类推。链式法则告诉我们，如何把这些局部导数组合成我们真正想要的那个导数。唯一的问题是按什么顺序组合。

## 应用链式法则的两种方式

对于函数链 $f = f_3 \circ f_2 \circ f_1$，导数是各个局部雅可比矩阵的乘积：

$$
\frac{\partial f}{\partial x} = \frac{\partial f_3}{\partial f_2} \cdot \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial x}
$$

矩阵乘法满足结合律，所以这个乘积从哪一头开始算都可以。

- **前向模式**从输入出发，问的是：如果我轻轻晃动 $x$，每个中间值会怎样跟着晃动？一次传播得到的是所有输出对某一个输入的导数。
- **反向模式**从输出出发，问的是：输出对每个中间值有多在意？一次传播得到的是某一个输出对所有输入的导数。

对于一个从 $n$ 个输入映射到 $m$ 个输出的函数，要得到完整的雅可比矩阵，前向模式需要 $n$ 次传播，反向模式需要 $m$ 次。训练中的 $n$ 极大（所有参数），而 $m = 1$（损失只是一个数）。反向模式快了 $n$ 倍。

这一类技术统称为**自动微分**，Baydin 等人 2015 年的综述介绍了它在机器学习中的用法，连同它的历史。它既不是符号微分（不对公式做变换），也不是数值微分（不用有限差分），而是把链式法则直接用在程序里实际流动的那些数值上。Rumelhart、Hinton 和 Williams 在 1986 年的论文中展示了反向模式能为神经网络带来什么：隐藏层可以自己学到有用的内部表示。用在神经网络上时，反向模式有一个专门的名字：**反向传播**。

### 手算一遍

反向模式为每个节点保存一个数，叫作它的**伴随值**（adjoint）：最终输出对该节点的导数，记作 $\bar v = \partial f / \partial v$。从 $\bar f = 1$ 开始，沿着图倒着走。每个节点把自己的伴随值乘上局部导数，再传给它的各个输入。

1. $f = t \cdot y$，所以 $\bar t = \bar f \cdot y = -1.5$，而 $y$ 收到 $\bar f \cdot t = -0.2449$。
2. $t = \tanh(u)$，所以 $\bar u = \bar t \cdot (1 - t^2) = -1.5 \times 0.94 = -1.41$。
3. $u = a + x$，所以 $\bar a = \bar u = -1.41$，而 $x$ 收到 $\bar u = -1.41$。
4. $a = x \cdot y$，所以 $x$ 收到 $\bar a \cdot y = 2.115$，$y$ 收到 $\bar a \cdot x = -0.705$。

注意，$x$ 和 $y$ 各自收到了两份贡献，因为它们各被用了两次。多元链式法则告诉我们，把这两份加起来：

$$
\frac{\partial f}{\partial x} = -1.41 + 2.115 = 0.705, \qquad \frac{\partial f}{\partial y} = -0.2449 - 0.705 = -0.9499
$$

图中的每条边都恰好访问了一次，做一次乘法和一次加法。效率正是从这里来的。反向传播的工作量与图的规模成正比，而前向传播的工作量也正是这个规模。无论有多少个输入，求梯度的代价都只是求一次函数值的一个不大的常数倍。

## 一个四十行的 autograd 引擎

上面这些步骤足够机械，完全可以自动化。下面是一个标量类型 `Value`，它为每一次运算记录输入和局部导数。`backward()` 先对图排序，让每个节点都排在计算它所用的那些节点之后，再按倒序遍历，和我们刚才手算的过程一模一样。

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

最后一行的 `+=` 就是“把各份贡献加起来”这条规则。把它换成 `=`，对于任何被使用不止一次的值，引擎都会悄无声息地给出错误答案。这个 bug 常见到你写任何 autograd 代码时都应该专门查一查。

### 一定要用数值方法检验梯度

反向传播代码很容易出一些隐蔽的错误，而错误的梯度不会让任何东西崩溃，只会让训练效果变差。标准的防御手段是**梯度检验**：拿结果和中心有限差分 $\big(f(x + h) - f(x - h)\big) / 2h$ 做对比。这种方法很慢，但很难写错。

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

引擎算出的结果与手算一致，也与有限差分一致，精确到小数点后六位。

## 从标量到矩阵

真实的框架处理的是张量而不是标量，但规则是一样的，只不过一次作用在整个数组上。最重要的情形是线性层 $Y = XW$，其中 $X$ 是一批输入，每行一个样本。给定从后面各层传回来的伴随值 $\bar Y = \partial L / \partial Y$：

$$
\bar W = X^\top \bar Y, \qquad \bar X = \bar Y \, W^\top
$$

这两个式子不用死记，形状会逼着你写出它们。$\bar W$ 的形状必须和 $W$ 相同，而用 $X$ 和 $\bar Y$ 凑出这个形状的组合方式只有一种。（这也是上一章代码里出现 `X.T @ p` 的原因。）像 $h = \tanh(a)$ 这样的逐元素函数，只需把伴随值逐元素乘以它的导数：$\bar a = \bar h \odot (1 - h^2)$。

仅凭这些规则，就能让一个两层网络学会 XOR，这是一个经典函数，任何线性模型都无法表示它。前向传播保留中间值，反向传播再倒序消耗它们：

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

（第一个预测一开始恰好为零，因为输入 $(0, 0)$ 加上全零的偏置，使得每个预激活值都是零。）

## 代价：内存

反向模式省计算，却不省内存。反向传播需要用到前向传播的中间值（上面的 `h`，以及真实网络中的每一个激活值），所以这些值都得一直保留，直到反向传播用到它们为止。对大模型来说，训练时占满加速器内存的，往往不是参数，而是存下来的激活值。

标准的折中办法是**激活检查点**（activation checkpointing）：只保留一部分激活值，其余的在反向传播时重新计算。这是多花计算来省内存，恰好与反向传播本身所做的交换方向相反。

> [!TIP] 要点
> - 网络是由简单运算组成的图。链式法则把这些运算的局部导数组合起来，而组合的顺序是可以选择的。
> - 反向模式（也就是反向传播）只需反向走一遍，就能得到某一个输出对所有输入的导数。损失只是一个输出，所以完整梯度的代价大约相当于一次前向传播。
> - 当一个值被送往多个地方时，它的梯度是各份贡献加起来的总和。
> - 对于层 $Y = XW$：$\bar W = X^\top \bar Y$，$\bar X = \bar Y W^\top$。让形状来引导你。
> - 新写的反向传播代码，要用有限差分做梯度检验。错误的梯度会悄无声息地失败。

现在，任意深度的网络我们都能求梯度了。但有了梯度，并不等于就能训练。[下一章](/zh/writing/making-depth-trainable/)讨论网络变深之后会出什么问题，以及解决这些问题的三个想法。
