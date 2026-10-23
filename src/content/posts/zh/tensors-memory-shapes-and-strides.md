---
title: 张量：内存、形状与步长
description: 张量就是一块内存，外加几个说明如何遍历这块内存的整数。用三十行代码实现这种表示，拿 NumPy 验证，再弄清楚为什么转置不花任何代价、reshape 为什么有时要拷贝，以及同样的求和为什么会慢上近二十倍。
date: 2026-10-23T08:20:00Z
tags: [under-the-hood, tensors, numpy]
series: under-the-hood
chapter: 1
references: [yang2019]
---

`x.T`、`x[:, ::2]`、`x.view(4, -1)`、`a + b[None, :]`，这几种写法你大概已经写过成千上万次。每一行都对内存做了一件确切的事，而其中大多数根本不碰数据本身。张量就是一块内存，外加几个说明如何遍历这块内存的整数。张量上那些看似在干活的操作，大多只是改了这几个整数。

本章从零开始，用大约三十行代码搭出这种表示，拿 NumPy 验证它，再用它解释几件不然只像冷知识的事：

- 为什么转置一个矩阵不花任何代价
- 为什么 PyTorch 的 `.view()` 有时会拒绝 `.reshape()` 能接受的形状
- 为什么广播不占内存
- 为什么同样的求和，只因数据布局不同，就可能慢上近二十倍

## 张量是内存加一份配方

下面是一个由 32 位浮点数组成的 3 × 4 矩阵：

```python
import numpy as np

a = np.arange(12, dtype=np.float32).reshape(3, 4)
print(a)
print("shape", a.shape, " strides in bytes", a.strides, " bytes per element", a.itemsize)
```

```text
[[ 0.  1.  2.  3.]
 [ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]]
shape (3, 4)  strides in bytes (16, 4)  bytes per element 4
```

这十二个数一行接一行地放在一块 48 字节的扁平内存里。**形状**（shape）说明怎样把它们分组，**步长**（stride）说明沿每个维度走一步要在内存里跳多远：到下一行跳 16 字节，到下一列跳 4 字节。元素 $(i, j)$ 的地址为

$$
\text{address}(i, j) = \text{offset} + i \cdot \text{stride}_0 + j \cdot \text{stride}_1
$$

这种最后一个下标变化最快的布局，叫作行主序（row-major），也叫 C 顺序。本章其余的一切，都从这一个公式推出来。

## 动手实现

把整个想法写成一个类，就是下面这样。和 PyTorch 的 `.stride()` 一样，它以元素而不是字节为单位记录步长；NumPy 的 `.strides` 给出的则是字节数。

```python
class Strided:
    """A view of a flat buffer: element (i, j, ...) lives at offset + i*strides[0] + j*strides[1] + ..."""

    def __init__(self, buffer, shape, strides, offset=0):
        self.buffer, self.shape, self.strides, self.offset = buffer, tuple(shape), tuple(strides), offset

    @classmethod
    def contiguous(cls, buffer, shape):
        """Row-major layout: the last index moves fastest."""
        strides, step = [], 1
        for size in reversed(shape):
            strides.append(step)
            step *= size
        return cls(buffer, shape, reversed(strides))

    def __getitem__(self, index):
        return self.buffer[self.offset + sum(i * s for i, s in zip(index, self.strides))]

    def transpose(self):
        return Strided(self.buffer, self.shape[::-1], self.strides[::-1], self.offset)

    def slice(self, dim, start, stop, step=1):
        shape, strides = list(self.shape), list(self.strides)
        shape[dim] = len(range(start, stop, step))
        strides[dim] = self.strides[dim] * step
        return Strided(self.buffer, shape, strides, self.offset + start * self.strides[dim])

    def broadcast(self, dim, size):
        """Repeat a size-1 dimension `size` times without copying: a stride of zero."""
        assert self.shape[dim] == 1
        shape, strides = list(self.shape), list(self.strides)
        shape[dim], strides[dim] = size, 0
        return Strided(self.buffer, shape, strides, self.offset)
```

每个方法都返回一个新的 `Strided`，底下用的仍是同一个缓冲区，没有一个方法会碰数据。要检验这个类，就拿 NumPy 当参照：用两种方式构造同样的视图，每个视图随机比较 200 个元素，同时比较步长。

```python
buffer = list(range(24))
x = Strided.contiguous(buffer, (2, 3, 4))
reference = np.arange(24).reshape(2, 3, 4)
row = Strided.contiguous(list(range(4)), (1, 4))
cases = {
    "contiguous": (x, reference),
    "transpose": (x.transpose(), reference.transpose()),
    "every 2nd column": (x.slice(2, 1, 4, 2), reference[:, :, 1:4:2]),
    "rows reversed": (x.slice(1, 2, -1, -1), reference[:, ::-1, :]),
    "broadcast row": (row.broadcast(0, 3), np.broadcast_to(np.arange(4).reshape(1, 4), (3, 4))),
}
rng = np.random.default_rng(0)
for name, (mine, theirs) in cases.items():
    assert mine.shape == theirs.shape
    checks = [tuple(int(rng.integers(n)) for n in mine.shape) for _ in range(200)]
    same = all(mine[i] == theirs[i] for i in checks)
    element_strides = tuple(s // theirs.itemsize for s in theirs.strides)
    print(f"{name:<17} shape {str(mine.shape):<10} strides {str(mine.strides):<12} "
          f"numpy {str(element_strides):<12} 200 random elements match: {same}")
```

```text
contiguous        shape (2, 3, 4)  strides (12, 4, 1)   numpy (12, 4, 1)   200 random elements match: True
transpose         shape (4, 3, 2)  strides (1, 4, 12)   numpy (1, 4, 12)   200 random elements match: True
every 2nd column  shape (2, 3, 2)  strides (12, 4, 2)   numpy (12, 4, 2)   200 random elements match: True
rows reversed     shape (2, 3, 4)  strides (12, -4, 1)  numpy (12, -4, 1)  200 random elements match: True
broadcast row     shape (3, 4)     strides (0, 1)       numpy (0, 1)       200 random elements match: True
```

五种情况都与 NumPy 完全一致。除了作为起点的连续情况，其余每一种都只是一条规则：

- **转置**把形状和步长都倒过来，数据一动不动。
- **按间隔切片**把偏移量移到第一个元素，再把步长乘以间隔。
- **反转**就是间隔为 $-1$ 的切片，此时步长是负数：视图从末尾出发，倒着往回走。
- **广播**让大小为 1 的维度步长为**零**：沿这个维度每走一步，都落在同一块内存上。

在随机输入上，把自己的实现和一个可信的实现相对照，是整个系列的方法。自己动手实现一个组件时，参考库就是你的测试参照（oracle）。

## 视图共享内存

这些操作只改元数据，所以结果和原张量共享内存。**视图**（view）就是建立在已有数据之上的新张量：

```python
a = np.arange(12, dtype=np.float32).reshape(3, 4)
t = a.T
s = a[:, 1::2]
print("transpose: strides", t.strides, " shares memory with a:", np.shares_memory(a, t))
print("slice:     strides", s.strides, " shares memory with a:", np.shares_memory(a, s))
s[0, 0] = 100.0                      # write through the view...
print("a[0] after writing through the slice:", a[0])
```

```text
transpose: strides (4, 16)  shares memory with a: True
slice:     strides (16, 8)  shares memory with a: True
a[0] after writing through the slice: [  0. 100.   2.   3.]
```

经由切片写入，`a` 也跟着变了。这正是视图的意义所在，因为对大张量切片不花任何代价；这同时也是视图的隐患，后面还会讲到。PyTorch 的文档写明了同样的约定：视图“与其基张量共享同一份底层数据”，从而“避免了显式的数据拷贝”。它列出的视图操作包括基本切片、`transpose()`、`expand()`（广播）和 `view()`。高级索引（用一个下标数组去索引）返回的则是副本。

## 广播零开销

把一个行向量加到矩阵的每一行上时，这一行从来不会被复制三份，而是通过一个步长为零的视图来读取：

```python
row = np.arange(4, dtype=np.float32)
b = np.broadcast_to(row, (3, 4))
print("broadcast strides", b.strides, " writeable:", b.flags.writeable,
      " shares memory with row:", np.shares_memory(b, row))
```

```text
broadcast strides (0, 4)  writeable: False  shares memory with row: True
```

广播视图被标记为只读，这是有道理的：它的每个内存位置都由三个逻辑元素共享，写其中一个，就等于同时写了三个。计算 `a + row` 的算术 kernel 只是沿着这个零步长去读而已。

## 什么时候非拷贝不可

只有当新布局能用旧内存上的步长描述出来时，才可能得到视图。把转置后的矩阵展平就做不到：

```python
def flatten_is_a_view(shape, strides):
    """Merging all dimensions into one keeps a constant step only if each stride = next stride x next size."""
    return all(strides[k] == strides[k + 1] * shape[k + 1] for k in range(len(shape) - 1))

a = np.arange(12, dtype=np.float32).reshape(3, 4)
for name, v in [("a", a), ("a.T", a.T)]:
    flat = v.reshape(12)
    print(f"{name:<4} strides {str(v.strides):<8} C-contiguous {str(v.flags.c_contiguous):<5} "
          f"predicted view {str(flatten_is_a_view(v.shape, v.strides)):<5} "
          f"reshape(12) shares memory: {np.shares_memory(flat, a)}")
print("order of a.T.reshape(12):", a.T.reshape(12))
```

```text
a    strides (16, 4)  C-contiguous True  predicted view True  reshape(12) shares memory: True
a.T  strides (4, 16)  C-contiguous False predicted view False reshape(12) shares memory: False
order of a.T.reshape(12): [ 0.  4.  8.  1.  5.  9.  2.  6. 10.  3.  7. 11.]
```

展平后的转置矩阵必须按 0、4、8、1、5、9……的顺序访问内存，没有哪一个步长能产生这样的序列。合并维度后，只有当每个维度的步长都等于下一维的步长乘以下一维的大小时，才能保持固定的间隔。`flatten_is_a_view` 检查的正是这个条件，它对 NumPy 行为的预测分毫不差。

PyTorch 对 `.view()` 给出了同样的条件：新形状“必须与原来的大小和步长兼容”，不兼容时 `.view()` 会报错。`.reshape()` 在可能时返回“输入的一个视图”，“否则就是一份拷贝”，文档还提醒“不应依赖它究竟是拷贝还是视图”。`.contiguous()` 把拷贝摆到明面上；如果张量本来就是连续的，它直接返回张量本身。所以 `x.transpose(0, 1).contiguous().view(-1)` 才成了常见写法：转置不花代价，`contiguous()` 付一次拷贝的钱，`view` 又是免费的。

## 为什么布局决定速度

内存不是一个数一个数地搬运的，而是以**缓存行**（cache line）为单位：一段连续的内存块，在 x86 处理器上通常是 64 字节（写这篇文章用的 Apple M1 Pro 报告的是 128）。读一个浮点数，会把它所在的整行都拉进来。做一个理想化的计数，就能看出这对不同的步长意味着什么：

```python
line = 64                                          # bytes per cache line on typical CPUs
itemsize = 4                                       # float32
print("stride (bytes)   useful bytes per 64-byte line   lines fetched per 1,024 elements")
for stride in [4, 8, 64, 4096]:
    useful = itemsize * max(1, line // stride)
    lines = -(-1024 * stride // line) if stride < line else 1024
    print(f"{stride:>14} {useful:>31} {lines:>34}")
```

```text
stride (bytes)   useful bytes per 64-byte line   lines fetched per 1,024 elements
             4                              64                                 64
             8                              32                                128
            64                               4                               1024
          4096                               4                               1024
```

步长为 4 字节时，每一行的每个字节都派上了用场。步长达到 64 字节或更大时，每个元素都要占用一整行，64 字节中有 60 字节被白白浪费。这张表只是一个模型：真实的处理器会预取，数据装得进缓存时，缓存行也会被重复使用。但这个效应是测得出来的。下面对浮点数求和，先是连续存放的，再是分散开的：前两种步长各取一百万个，最宽的那种则取数组在这个间隔下所能容纳的 65,536 个：

```python
import time

x = np.ones(2**26, dtype=np.float32)               # 256 MB, far larger than any cache
for stride, n in [(1, 2**20), (16, 2**20), (1024, 2**16)]:
    view = x[::stride][:n]                          # n elements, `stride` elements apart
    best = float("inf")
    for _ in range(30):
        start = time.perf_counter()
        view.sum()
        best = min(best, time.perf_counter() - start)
    print(f"stride {stride * 4:>5} bytes: {best / n * 1e9:.2f} ns per element")
```

在 M1 Pro 上跑了几次，连续求和每个元素约 0.18 纳秒，64 字节步长约 1.2 到 1.3 纳秒，4096 字节步长则是 3.1 到 3.5 纳秒：同样的算术，慢了大约 17 到 20 倍。最后一种情况比按缓存行计数预测的还要糟，因为很大的二的幂次步长会在内存系统里招来额外的惩罚。你测出的数字会不一样，但快慢顺序不会变。

这就是各种库如此在意布局的原因。矩阵乘法之前，转置过的操作数有时会先被拷贝一份，`.contiguous()` 之所以存在，也都是这个原因。在 GPU 上，同样的道理体现得更加极端，后面的章节会讲到。

## 别名陷阱

免费视图的代价是，两个张量可能共用同一块内存：

```python
a = np.zeros(6, dtype=np.float32)
middle = a[2:4]
middle += 1                          # an in-place update to a view...
print(a)                             # ...is an update to the original
```

```text
[0. 0. 1. 1. 0. 0.]
```

通过视图做原地更新，就是在更新原张量。在训练框架里，这可能悄无声息地破坏为反向传播保存下来的值。PyTorch 对此有防护：“每个张量都有一个版本计数器”，原地操作会让它递增；如果一个为反向传播保存的张量在保存之后又被改过，反向传播就会报错，而不是算出一个错误的梯度。

## 延伸阅读

Edward Yang 的“PyTorch internals”（2019）在 PyTorch 内部把同样的思路走了一遍，从步长和存储（storage），一直讲到一个操作如何抵达它的 kernel。PyTorch 关于[张量视图](https://docs.pytorch.org/docs/stable/tensor_view.html)的文档，准确列出了哪些操作返回视图。NumPy 关于 [NumPy 数组内部组织](https://numpy.org/doc/stable/dev/internals.html)的页面，则针对 NumPy 讲了同样的内容。

> [!TIP] 要点
> - 张量是一个扁平缓冲区，加上形状、步长和偏移量。元素 $(i, j, \ldots)$ 位于偏移量 $+ \sum_k i_k \cdot \text{stride}_k$ 处。
> - 转置、切片、反转和广播都只改元数据。它们不花代价，结果与原张量共享内存。
> - 广播就是步长为零，所以它不占内存，广播视图也因此是只读的。
> - 只有步长允许时，reshape 才可能是视图，否则就要拷贝。在 PyTorch 里，`.view()` 会拒绝，`.reshape()` 会拷贝。
> - 连续的数据一次读入一整个缓存行。跨步访问会浪费每行的大部分字节，由此带来的变慢真实存在，也测得出来。
> - 通过视图写入，写的就是原张量。框架会追踪这一点，免得它破坏梯度。

[下一章](/zh/writing/autograd-for-tensors/)会搭建每个框架的另一半：一个直接作用于整个张量的 autograd 引擎。
