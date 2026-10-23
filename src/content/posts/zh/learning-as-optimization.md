---
title: "学习即优化：损失、似然与梯度下降"
description: 神经网络“学习”到底是什么意思，损失函数从何而来，以及为什么单单一个学习率阈值就决定了训练能不能成功。
date: 2026-10-22T15:00:00Z
tags: [deep-learning, math, optimization]
series: deep-learning-foundations
chapter: 1
references: [robbins1951, bottou2016]
---

“模型学会了识别猫”，听上去像是发生了什么神秘的事。其实没有。神经网络就是一个函数，里面有一些可以调节的数；所谓学习，就是去寻找一组数，让某个特定的量，也就是损失，变得尽可能小。深度学习里的其他一切，不过是关于选哪个函数、哪种损失、哪种搜索方法的细节。

这一章就来搭好这三块。读完之后，你会知道常用的损失函数从何而来（它们并不是随便定的），为什么沿着梯度走就能奏效，以及那个关键的不等式：它告诉你，用某个学习率训练是会收敛，还是会爆炸。

## 问题设定

我们有一个包含 $n$ 个样本 $(x_i, y_i)$ 的数据集，每个样本由一个输入和我们希望它对应的答案组成。我们还有一个模型 $f_\theta$，它是一个函数，其行为取决于参数向量 $\theta$。对大语言模型来说，$\theta$ 有数十亿个分量；本章末尾那个例子里，它只有九个。

**损失函数** $\ell(f_\theta(x), y)$ 给单个预测打分：预测完美时为零，预测越差，分数越大。把它在整个数据集上取平均，就得到**经验风险**（empirical risk）：

$$
\hat{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell\big(f_\theta(x_i),\, y_i\big)
$$

训练就是求解 $\min_\theta \hat{L}(\theta)$ 这个问题。定义就这么多。

有一点值得现在就说清楚，因为它会一直存在。我们其实并不关心手头已有的样本，我们关心的是以后才会遇到的样本。我们真正想让它变小的，是整个数据分布上的期望风险，而 $\hat L$ 只是用一份样本对它做出的估计。两者之间的差距叫作泛化差距，本系列后面会专门用一章来讲泛化。

## 损失函数从何而来

损失函数你想怎么发明都行。但实际上几乎所有人用的都是其中两种，而这两种都出自同一个原理：**最大似然**。

假设模型输出的不是单个答案，而是答案上的一个概率分布 $p_\theta(y \mid x)$。好的模型应该给实际出现的答案分配高概率。把各个样本视为相互独立，整个数据集的概率就是乘积 $\prod_i p_\theta(y_i \mid x_i)$。许多很小的数连乘会下溢，而取对数能把乘积变成求和，所以我们改为最大化对数，再翻转符号，得到一个要最小化的量：

$$
\hat{L}(\theta) = -\frac{1}{n} \sum_{i=1}^{n} \log p_\theta(y_i \mid x_i)
$$

这就是**负对数似然**。接下来给 $p_\theta$ 选一个具体形式，看看会得出什么损失。

**分类**：有 $k$ 个类别时，网络输出 $k$ 个实数 $z_1, \dots, z_k$，称为 logits，再由 softmax 函数把它们变成概率：

$$
p_j = \frac{e^{z_j}}{\sum_{m=1}^{k} e^{z_m}}
$$

正确类别 $y$ 的负对数概率是

$$
\ell = -\log p_y = -z_y + \log \sum_{m=1}^{k} e^{z_m}
$$

这就是**交叉熵损失**。每个图像分类器最小化的都是它，语言模型最小化的也是它：预测下一个 token，就是在整个词表上做分类。

**回归**：如果模型预测一个数 $\mu_\theta(x)$，并且我们假设真实值等于这个预测加上方差固定为 $\sigma^2$ 的高斯噪声，那么

$$
-\log p_\theta(y \mid x) = \frac{\big(y - \mu_\theta(x)\big)^2}{2\sigma^2} + \text{const}
$$

最小化它，恰好就是最小化**平方误差**。

所以，选择损失函数，实际上是在选择噪声模型。平方误差说的是“误差服从高斯分布”，交叉熵说的是“答案是 $k$ 个类别之一”。当某个损失表现异常时，问一问它隐含的是哪种噪声模型，往往是最快的诊断方法。

### 一个随处可见的梯度

把交叉熵对 logits 求导。对正确类别，$\partial \ell / \partial z_y = -1 + p_y$；对其他任一类别 $j$，$\partial \ell / \partial z_j = p_j$。写成一个向量就是：

$$
\frac{\partial \ell}{\partial z} = p - e_y
$$

其中 $e_y$ 是正确类别的 one-hot 向量。梯度不过就是“预测概率减去目标概率”。它有界，性质良好，而且恰恰在模型错得最自信的地方推得最狠。softmax 和交叉熵几乎总是搭配使用，原因就在这一行公式；下一章它还会再次出现，作为反向传播的起点。

## 梯度下降

我们要找的是让 $\hat L$ 最小的 $\theta$，而 $\hat L$ 可能是数十亿个变量的函数。这样的空间根本没法搜索。我们能廉价算出来的，是**梯度** $g = \nabla_\theta \hat{L}(\theta)$，也就是损失增长最快的方向。

一阶泰勒展开告诉我们，逆着梯度方向迈出大小为 $\eta$ 的一小步会发生什么：

$$
\hat{L}(\theta - \eta g) \approx \hat{L}(\theta) - \eta \, \lVert g \rVert^2
$$

只要 $\eta > 0$ 足够小，损失就会下降，除非梯度已经为零。由此得到更新规则

$$
\theta \leftarrow \theta - \eta \, \nabla_\theta \hat{L}(\theta)
$$

其中 $\eta$ 就是**学习率**。上面论证里的“足够小”几个字举足轻重，所以我们来把它说精确。

### 一步能迈多大？

取最简单的损失：一个一维的碗，$L(\theta) = \tfrac{a}{2}\theta^2$，其中 $a > 0$。它的梯度是 $a\theta$，所以梯度下降走一步得到

$$
\theta_{t+1} = \theta_t - \eta a \theta_t = (1 - \eta a)\, \theta_t
$$

走 $t$ 步之后，$\theta_t = (1 - \eta a)^t \theta_0$。它会收缩到零点处的最小值，当且仅当 $|1 - \eta a| < 1$，也就是

$$
0 < \eta < \frac{2}{a}
$$

一旦超过这个阈值，每一步冲过头的量都比它纠正的量更大，迭代值就会无界增长。下面取 $a = 4$，于是阈值是 $0.5$：

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

有两点值得注意。学习率 $0.4$ 收敛得和 $0.1$ 一样快，尽管它每一步都会越过最小值、翻转符号：$(1 - 1.6) = -0.6$ 与 $(1 - 0.4) = 0.6$ 的绝对值相同。另外，这个边界非常陡峭：取 $0.49$ 时，迭代值几乎原地不动；取 $0.6$ 时，它已经涨了将近三十倍。

真实的损失当然不是一维的碗，但在最小值附近，它看起来就像许多个碗同时叠在一起：沿黑塞矩阵（Hessian，即二阶导数构成的矩阵）的每个特征向量方向各有一个碗，曲率等于对应的特征值 $\lambda_1 \ge \dots \ge \lambda_d$。同样的分析在每个方向上都成立，因此：

- 最大的曲率决定速度上限，$\eta < 2/\lambda_1$；
- 在这个速度下，最小的曲率决定最平坦的方向收敛得有多慢。

两者之比 $\lambda_1 / \lambda_d$ 叫作条件数，它衡量的是这个问题对朴素的梯度下降来说有多难。本系列余下的大部分内容，包括归一化层、精心设计的初始化和自适应优化器，都可以看作是在设法把这个比值变小。

## 随机梯度下降

精确计算 $\nabla \hat L$，意味着每走一步都要把全部 $n$ 个样本过一遍。样本数以百万计时，这就太慢了。解决办法是改用一个随机抽取的**小批次**（minibatch）$B$ 来估计梯度：

$$
g_B = \frac{1}{|B|} \sum_{i \in B} \nabla_\theta \, \ell\big(f_\theta(x_i), y_i\big)
$$

如果小批次是均匀随机抽取的，$g_B$ 就是完整梯度的无偏估计：平均而言是对的，只是每一步都带着噪声。这样一来，每一步的代价取决于批次大小，而不再取决于数据集大小。

这个想法比神经网络还要古老。Robbins 和 Monro 在 1951 年就研究过它，称之为“随机逼近”（stochastic approximation）：为一个只能带噪声测量的函数寻找根。他们的分析给出了递减步长 $\eta_t$ 要让噪声被平均掉所需满足的经典条件：

$$
\sum_{t=1}^{\infty} \eta_t = \infty, \qquad \sum_{t=1}^{\infty} \eta_t^2 < \infty
$$

第一个条件是说，步长永远不会小到让进展停滞；第二个条件是说，步长缩小得足够快，噪声不会永无休止地把迭代值踢来踢去。$\eta_t = 1/t$ 同时满足这两个条件。现代训练很少照搬这个调度，但它描述的那种权衡，即大步长换进展、小步长换精度，正是学习率通常先预热、再衰减的原因。

为什么这样一种充满噪声的方法，到了大规模场景下反而效果这么好？Bottou、Curtis 和 Nocedal 在 2016 年的综述里对此做了细致的论证。简单来说：数据充足时，一个能走上一千次的廉价含噪步，胜过一个只能走一次的精确步。

## 合起来看

下面用大约三十行代码把整章内容串起来：在三团带噪声的 2D 点上，用小批次 SGD 训练一个 softmax 分类器。模型是线性的，logits 为 $z = xW + b$，所以共有 $2 \times 3 + 3 = 9$ 个参数；梯度用的就是上面得到的 $p - e_y$。

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

初始损失并非巧合：所有权重都为零时，每个类别的概率都是 $1/3$，而 $-\log(1/3) = \log 3 \approx 1.099$。检查初始损失是否等于 $\log k$，是一项对任何分类器都适用、几乎零成本的健全性检查。（初始准确率 $0.31$ 恰好是第一类样本所占的比例，因为 `argmax` 遇到并列时取索引 0。）

损失稳定之后，准确率仍在 $0.95$ 和 $0.96$ 之间摆动。这就是上一节说的小批次噪声：在固定的学习率下，它还在不停地轻推参数。

试着把它弄坏：把 `eta` 设成 `50`，看看损失会怎样；或者把 `batch` 设成 `1`，看看准确率跳得有多厉害。

> [!TIP] 要点
> - 学习就是最小化数据上的平均损失。常用的损失都来自最大似然：类别问题用交叉熵，高斯噪声用平方误差。
> - softmax 交叉熵对 logits 的梯度是 $p - e_y$。
> - 在曲率为 $a$ 的碗上，梯度下降只有在 $\eta < 2/a$ 时才会收敛。在多维情形下，最尖锐的那个方向决定上限。
> - 小批次梯度是无偏的，但带有噪声，而且它的代价不随数据集增大而增长。正是这笔交换让大规模训练成为可能。

这里的模型是线性的，所以梯度可以手写出来。下一章讨论模型是一摞很深的层时会怎样：如何自动求出任何东西的梯度，而代价大致只相当于把模型运行一遍。
