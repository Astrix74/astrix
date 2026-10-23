---
title: 为张量实现一个 autograd 引擎
description: 与标量 autograd 引擎相同的算法，只是作用在整个数组上。动手实现它，学会广播给反向传播新增的那一条规则，对每个操作做梯度检验，再用它训练一个网络。
date: 2026-10-23T08:21:00Z
tags: [under-the-hood, autograd, numpy]
series: under-the-hood
chapter: 2
references: [rumelhart1986, baydin2015, yang2019]
---

[《反向传播就是反向模式自动微分》](/zh/writing/backpropagation-is-reverse-mode-differentiation/)用四十行代码实现了一个 autograd 引擎，其中每个节点只存一个数。PyTorch、JAX 以及所有其他框架跑的都是同一个算法，区别只有两点：每个节点存的是一整个张量，每个操作的反向步骤本身也是一次张量计算。本章就来做这个改动。算法不变，新增的只是几条数组规则，其中最重要的一条来自广播。

## 一条新规则：广播反过来就是求和

在前向传播中，NumPy 的广播让形状为 $(4,)$ 的偏置可以加到 $(3, 4)$ 矩阵的每一行上。正如[上一章](/zh/writing/tensors-memory-shapes-and-strides/)所讲，偏置是通过一个为零的步长被读了三次。到了反向传播，多元链式法则告诉我们：一个值如果在多处被使用，它得到的就是来自所有这些地方的梯度之**和**。所以偏置的梯度，就是把传进来的 $(3, 4)$ 梯度的各行加在一起：

$$
y_{ij} = x_{ij} + b_j \quad\Longrightarrow\quad \frac{\partial L}{\partial b_j} = \sum_i \frac{\partial L}{\partial y_{ij}}
$$

一般的规则是：凡是被广播新增或拉伸过的维度，都要在这些维度上对梯度求和，直到它恢复成输入的形状。忘了这一步，是手写反向传播时最常见的 bug，而且它往往悄无声息地出错，因为形状不对的梯度，NumPy 照样会一声不吭地拿去广播。

## 引擎

每个 `Tensor` 保存它的数据、它的梯度、计算出它的那些张量，以及一个函数，负责把它自己的梯度转换成这些父节点的梯度。对一个神经网络来说，五个操作就够了：

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

里面的规则都是熟悉的导数的数组版。对 `a @ b`，关于 `a` 的梯度是传进来的梯度乘以 $b^\top$，关于 `b` 的梯度是 $a^\top$ 乘以传进来的梯度。和以前一样，这些都是形状逼出来的。对 `relu`，输入为正的地方梯度原样通过。对 `sum`，梯度被广播回输入的形状：求和是广播的逆过程，反之亦然。`backward()` 和标量引擎里的一样，是同一套拓扑遍历。

## 融合的交叉熵

交叉熵本可以用这些基本操作再加上 `exp` 和 `log` 搭出来，但框架不这么做。对很大的 logits 取指数会溢出，所以框架把整个计算融合成一个操作：前向传播在数值上是安全的，反向传播也很简单：

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

前向传播在取指数之前，先减去每一行的最大值。这不会改变 softmax 的结果，又能让每个指数都不大于零。这就是 log-sum-exp 技巧。反向传播用的是[《学习即优化》](/zh/writing/learning-as-optimization/)里的结论：预测概率减去 one-hot 目标。

## 检验每一个操作

上面的每条反向规则，都可能藏着一个让梯度略有偏差、却不会崩溃的 bug。所以和标量那一章一样，每条规则都要在随机输入上用中心有限差分来检验，包括广播的几种情况和一个完整的两层网络：

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

相对误差在 $10^{-10}$ 左右，说明解析梯度和数值梯度在有限差分所能达到的精度内是一致的。两层网络的误差大了大约一千倍，不过仍远不足为虑，因为 ReLU 有折点，有限差分会跨在拐角两侧。相对误差接近 $10^{-2}$ 甚至更大，才说明真有 bug。PyTorch 自带同样的工具，叫 `torch.autograd.gradcheck`。如果你写了自定义操作，就跑一跑它。

## 用它训练一个网络

下面用这个引擎在三条交错的螺旋臂上训练一个两层网络，这个问题没有哪条直线能分开：

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

每一步都通过运行模型构建一张新的计算图，调用 `backward()`，再让每个参数沿梯度的反方向挪一小步。去掉那些便利功能，每个框架的训练循环就是这么多。作为对比，下面是同样的循环，只是去掉了隐藏层，也就是一个线性分类器：

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

卡在大约一半正确。让网络能把决策边界弯过来、绕着螺旋走的，正是隐藏层和它的 ReLU。

## 真正的框架还多了什么

这个引擎做的事和 PyTorch 的 autograd 一样，工程量却只有它的一个零头。两者的差别值得了解，因为每一处差别都解释了你已经养成的某个习惯：

- **梯度会累加**。这个引擎每次调用 `backward()` 都会把所有梯度重置；PyTorch 则往 `.grad` 里累加：叶子张量“会把梯度累加到各自的 .grad 字段中”。这就是训练循环要调用 `optimizer.zero_grad()` 的原因，也是忘了调用它就会悄悄把多个步骤的梯度加在一起的原因。
- **内存会被管理**。这里的计算图会一直保留每个中间数组，直到下一步把它替换掉。真正的引擎只保存每条反向规则需要的东西，保存下来的张量一旦用过就释放。
- **原地操作的安全**。正如上一章所说，PyTorch 的版本计数器能发现那些为反向传播保存之后又被修改过的张量。
- **kernel 与设备**。这里的每条反向规则都只是几次 NumPy 调用；在框架里，它是一个编译好的 kernel，可能跑在 GPU 上，由分发器（dispatcher）挑选。Edward Yang 的“PyTorch internals”详细讲解了这套机制。
- **高阶导数**。这里的反向规则用的是普通数组，所以没法对反向传播本身再求导。框架可以把反向计算也记录成一张图，梯度的梯度正是这样才算得出来。

Andrej Karpathy 的 [micrograd](https://github.com/karpathy/micrograd) 是这个引擎的经典标量版本，值得和标量那一章对照着读。从它走到本章，也就是从数走到数组，就是从一个玩具走到真实框架的基本形态。

> [!TIP] 要点
> - 张量 autograd 引擎就是节点里存放数组的标量算法：记录每个操作及其输入，给计算图排好序，再逆序应用每条反向规则。
> - 前向传播里的广播，到了反向传播就变成求和。要把每个梯度归约回它所对应输入的形状。
> - 数值上危险的操作要融合起来。softmax 交叉熵用 log-sum-exp 来计算，它的梯度是 $p - e_y$。
> - 在随机输入上对每条反向规则做梯度检验。错误的梯度不会让程序崩溃。
> - 框架在同一个核心之上，加了梯度累加、内存管理、原地操作的安全保护、设备 kernel 和高阶导数。

[下一章](/writing/floating-point-and-mixed-precision/)转向这些数组里的数本身：浮点格式、它们会在哪里溢出，以及混合精度是怎么工作的。
