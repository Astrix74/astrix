---
title: "Tensors: memory, shapes and strides"
description: A tensor is a block of memory plus a few integers that say how to walk it. Build that representation in thirty lines, check it against NumPy, and see why transposes are free, why reshape sometimes copies, and why the same sum can run nearly twenty times slower.
date: 2026-10-23T08:20:00Z
tags: [under-the-hood, tensors, numpy]
series: under-the-hood
chapter: 1
references: [yang2019]
---

You have written `x.T`, `x[:, ::2]`, `x.view(4, -1)` and `a + b[None, :]` thousands of times. Each of those lines does something precise to memory, and most of them do nothing to the data at all. A tensor is a block of memory plus a few integers that say how to walk it. Most of what looks like work on a tensor is a change to those integers.

This chapter builds that representation from scratch in about thirty lines, checks it against NumPy, and uses it to explain things that otherwise look like trivia:

- why transposing a matrix costs nothing
- why PyTorch's `.view()` sometimes refuses a shape that `.reshape()` accepts
- why broadcasting uses no memory
- why the same sum can run nearly twenty times slower depending on how the data is laid out.

## A tensor is memory plus a recipe

Here is a 3 × 4 matrix of 32-bit floats:

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

The twelve numbers sit in one flat block of 48 bytes, row after row. The **shape** says how to group them, and the **strides** say how far to jump in memory to move one step along each dimension: 16 bytes to the next row, 4 bytes to the next column. Element $(i, j)$ lives at

$$
\text{address}(i, j) = \text{offset} + i \cdot \text{stride}_0 + j \cdot \text{stride}_1
$$

This layout, where the last index moves fastest, is called row-major or C order. Everything else in this chapter follows from that one formula.

## Build it

Here is the whole idea as a class. It keeps strides in *elements* rather than bytes, as PyTorch's `.stride()` does. NumPy's `.strides` reports bytes.

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

Every method returns a new `Strided` over the *same* buffer. None of them touches the data. To check the class, use NumPy as the reference: build the same views both ways and compare 200 random elements of each, along with the strides.

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

All five cases match NumPy exactly, and each one is a single rule:

- **Transpose** reverses the shape and the strides. The data doesn't move.
- **Slicing with a step** moves the offset to the first element and multiplies the stride by the step.
- **Reversing** is a slice with step $-1$, a *negative* stride. The view starts at the end and walks backwards.
- **Broadcasting** gives a size-1 dimension a stride of **zero**: every step along it lands on the same memory.

Checking your implementation against a trusted one, on random inputs, is the method of this whole series. When you build a component yourself, the reference library is your test oracle.

## Views share memory

Because these operations only change metadata, the result shares memory with the original. A **view** is a new tensor over existing data:

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

Writing through the slice changed `a`. That is the point of views, since slicing a large tensor costs nothing, and also their hazard, which we'll come back to. PyTorch documents the same contract: a view "shares the same underlying data with its base tensor", which "avoids explicit data copy". Its list of view operations includes basic slicing, `transpose()`, `expand()` (broadcasting) and `view()`. Advanced indexing, with an array of indices, returns a copy instead.

## Broadcasting costs nothing

When you add a row vector to every row of a matrix, the row is never copied three times. It is viewed with a stride of zero:

```python
row = np.arange(4, dtype=np.float32)
b = np.broadcast_to(row, (3, 4))
print("broadcast strides", b.strides, " writeable:", b.flags.writeable,
      " shares memory with row:", np.shares_memory(b, row))
```

```text
broadcast strides (0, 4)  writeable: False  shares memory with row: True
```

The broadcast view is marked read-only, and for a good reason: three of its logical elements share each memory location, so writing to one would write to all three. The arithmetic kernel that computes `a + row` simply reads through the zero stride.

## When a copy is unavoidable

A view is possible only when the new layout can be described by strides over the old memory. Flattening the transpose cannot:

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

The flattened transpose has to visit memory in the order 0, 4, 8, 1, 5, 9, …, and no single stride produces that sequence. Merging dimensions keeps a constant step only when each stride equals the next stride times the next size. That is the condition `flatten_is_a_view` checks, and it predicts NumPy's behavior exactly.

PyTorch states the same condition for `.view()`: the new shape "must be compatible with its original size and stride". When it isn't, `.view()` raises an error. `.reshape()` returns "a view of input" when possible and "otherwise, it will be a copy", and the documentation warns that "you should not depend on the copying vs. viewing behavior". `.contiguous()` makes the copy explicit and returns the tensor itself if it is already contiguous. That is why `x.transpose(0, 1).contiguous().view(-1)` is a common idiom: the transpose is free, the `contiguous()` pays for one copy, and the `view` is free again.

## Why layout decides speed

Memory does not move one number at a time. It moves in **cache lines**, contiguous blocks that are typically 64 bytes on x86 processors (the Apple M1 Pro this was written on reports 128). Reading one float pulls in its whole line. An idealized count shows what that does to different strides:

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

At a 4-byte stride every byte of every line is used. At 64 bytes or more, each element costs a whole line and 60 of its 64 bytes are wasted. The table is a model: real processors prefetch, and when data fits in cache, lines get reused. But you can measure the effect. This sums a million floats that are contiguous, then spread out:

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

Across several runs on the M1 Pro, the contiguous sum took about 0.18 nanoseconds per element, the 64-byte stride about 1.2 to 1.3, and the 4096-byte stride 3.1 to 3.5: roughly 17 to 20 times slower for the same arithmetic. The last case is worse than line counting predicts: large power-of-two strides pay extra penalties in the memory system. Your numbers will differ, but the ordering won't.

This is why libraries care so much about layout. It is why a transposed operand sometimes gets copied before a matrix multiplication, and why `.contiguous()` exists at all. On a GPU, the same logic is more extreme, which a later chapter covers.

## The aliasing trap

The cost of free views is that two tensors can share one memory:

```python
a = np.zeros(6, dtype=np.float32)
middle = a[2:4]
middle += 1                          # an in-place update to a view...
print(a)                             # ...is an update to the original
```

```text
[0. 0. 1. 1. 0. 0.]
```

An in-place update through a view is an update to the original. In a training framework, that can silently corrupt a value saved for the backward pass. PyTorch guards against it: "every tensor keeps a version counter", incremented by in-place operations, and if a tensor saved for backward has since been modified, the backward pass raises an error instead of computing a wrong gradient.

## Going further

Edward Yang's "PyTorch internals" (2019) walks through the same ideas inside PyTorch, from strides and storage to how an operation reaches its kernel. PyTorch's documentation on [tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html) lists exactly which operations return views. NumPy's page on the [internal organization of NumPy arrays](https://numpy.org/doc/stable/dev/internals.html) covers the same ground for NumPy.

> [!TIP] What to remember
> - A tensor is a flat buffer plus shape, strides and an offset. Element $(i, j, \ldots)$ lives at offset $+ \sum_k i_k \cdot \text{stride}_k$.
> - Transpose, slicing, reversing and broadcasting only change the metadata. They are free, and the results share memory.
> - Broadcasting is a stride of zero, which is why it uses no memory and why broadcast views are read-only.
> - A reshape can be a view only if the strides allow it. Otherwise it copies. In PyTorch, `.view()` refuses and `.reshape()` copies.
> - Contiguous data is read a cache line at a time. Strided access wastes most of each line, and the slowdown is real and measurable.
> - Writing through a view writes the original. Frameworks track this so that it can't corrupt gradients.

The [next chapter](/writing/autograd-for-tensors/) builds the other half of every framework: an autograd engine that works on whole tensors.
