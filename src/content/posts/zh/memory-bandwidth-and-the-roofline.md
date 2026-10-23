---
title: GPU：为什么内存带宽决定速度
description: 芯片有两种速度：算得有多快，搬字节有多快。逐一算出 Transformer 里每个操作受哪一种速度限制，在笔记本电脑上测出同样的效应，再看看批处理和融合为什么如此重要。
date: 2026-10-23T10:11:00Z
tags: [under-the-hood, gpus, numpy]
series: under-the-hood
chapter: 4
references: [williams2009, he2022brrr, dao2022]
---

加速器的规格表上，打头的是一个数字：它的算术吞吐量。NVIDIA 的 H100 SXM 写的是 bfloat16 下 1,979 teraFLOPS，旁边带着一个脚注：“稀疏情况下”。换成普通的稠密矩阵，这个数要减半，约为 989 teraFLOPS。再往下看，还有第二个数字：3.35 TB/s 的内存带宽。两者之比，决定了深度学习里几乎所有计算跑得有多快：

$$
\frac{989 \times 10^{12}\ \text{FLOP/s}}{3.35 \times 10^{12}\ \text{byte/s}} \approx 295\ \text{FLOPs per byte}
$$

要让算术单元一直有活干，芯片每从内存读一个字节，就得对它做大约 295 次运算。而 Transformer 里的大多数操作只做一两次。对这些操作来说，算术单元几乎一直闲着，决定速度的是字节到达得有多快。[上一章](/zh/writing/floating-point-and-mixed-precision/)把数变小了；本章要讲的是，为什么这件事比看上去还要重要。

## 算术强度

有用的度量是一个操作的**算术强度**（arithmetic intensity）：它执行的 FLOPs 除以它读写的字节数。两个 bfloat16 张量的逐元素加法，读两个值、写一个值，共 6 字节，只换来 1 FLOP，强度为 $1/6$。一个 $(n \times d)$ 矩阵乘以一个 $(d \times d)$ 矩阵，要在约 $2(2nd + d^2)$ 字节上做 $2nd^2$ FLOPs。每个权重对每一行都要复用一次，所以它的强度随 $n$ 增长。

**roofline 模型**（Williams 等人，2009）把这一点变成了一道限速。一个操作跑得再快，也快不过芯片的计算速度，也快不过它的字节到达的速度：

$$
\text{attainable FLOP/s} = \min\left(\text{peak FLOP/s},\ \text{bandwidth} \times \text{intensity}\right)
$$

**拐点**（ridge point）在 H100 上是每字节 295 FLOPs。强度低于它，操作就受内存限制（memory-bound）；高于它，就受计算限制（compute-bound）。下面是一个 Transformer 层里的各个操作，宽度为 4096，同时处理 4096 个 token：

```python
peak, bandwidth = 989e12, 3.35e12  # H100 SXM: dense bfloat16 FLOP/s, memory bytes/s
ridge = peak / bandwidth
B = 2                              # bytes per bfloat16 value
n, d = 4096, 4096                  # tokens in a batch, model width

# FLOPs and bytes read plus written for one call, counting each exp or division as one FLOP.
ops = {
    "add":                 (n * d, 3 * n * d * B),
    "softmax over rows":   (5 * n * d, 2 * n * d * B),
    "LayerNorm":           (8 * n * d, 2 * n * d * B),
    "matrix-vector":       (2 * d * d, (d * d + 2 * d) * B),
    "matmul, 4096 tokens": (2 * n * d * d, (2 * n * d + d * d) * B),
}
print(f"ridge point: {ridge:.0f} FLOPs per byte")
print(f"{'operation':21}{'FLOP/byte':>10}{'bound by':>10}{'time (us)':>11}{'of peak':>9}")
for name, (flops, nbytes) in ops.items():
    intensity = flops / nbytes
    seconds = max(flops / peak, nbytes / bandwidth)
    bound = "memory" if intensity < ridge else "compute"
    print(f"{name:21}{intensity:10.2f}{bound:>10}{seconds * 1e6:11.1f}{flops / seconds / peak:9.1%}")
```

```text
ridge point: 295 FLOPs per byte
operation             FLOP/byte  bound by  time (us)  of peak
add                        0.17    memory       30.0     0.1%
softmax over rows          1.25    memory       20.0     0.4%
LayerNorm                  2.00    memory       20.0     0.7%
matrix-vector              1.00    memory       10.0     0.3%
matmul, 4096 tokens     1365.33   compute      139.0   100.0%
```

矩阵乘法跑满了速度。其余操作都远远达不到芯片算术峰值的 1%，因为它们都在等内存。值得细看的是矩阵向量乘积。只有一个 token 时，每个权重从内存读出来后恰好只用一次，所以即便是矩阵乘积，也会受内存限制。语言模型每次只为一个用户生成一个 token 时，面对的正是这种情况。

## 动手测量

这些都不是 GPU 独有的。任何处理器都有同样的两道限制，所以这个模型在笔记本电脑上就能检验。下面的代码通过 NumPy，测出写这篇文章所用的 CPU（一块 Apple M1 Pro）的拷贝带宽和最快的矩阵乘法速度，再把每个操作的速度与 roofline 允许的上限相比较：

```python
import time
import numpy as np

def best_time(fn, repeats=15):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best

rng = np.random.default_rng(0)
x, y = rng.normal(size=(2, 2**25)).astype(np.float32)  # 134 MB each, far larger than the caches
out = np.empty_like(x)
cpu_bandwidth = 2 * x.nbytes / best_time(lambda: np.copyto(out, x))
mats = {k: rng.normal(size=(k, k)).astype(np.float32) for k in [32, 128, 512, 2048]}
matmul = {k: best_time(lambda M=M: M @ M) for k, M in mats.items()}
cpu_peak = max(2 * k**3 / t for k, t in matmul.items())
print(f"this CPU: {cpu_bandwidth / 1e9:.0f} GB/s, {cpu_peak / 1e12:.2f} TFLOP/s, ridge {cpu_peak / cpu_bandwidth:.0f} FLOPs per byte")

W, v = rng.normal(size=(4096, 4096)).astype(np.float32), rng.normal(size=4096).astype(np.float32)
cases = [("add", 2**25, 3 * 4 * 2**25, best_time(lambda: np.add(x, y, out=out))),
         ("matrix-vector", 2 * 4096**2, 4 * (4096**2 + 2 * 4096), best_time(lambda: W @ v))]
cases += [(f"matmul {k}", 2 * k**3, 3 * 4 * k * k, t) for k, t in matmul.items()]
print(f"{'operation':15}{'FLOP/byte':>10}{'roofline':>10}{'measured':>10}  GFLOP/s")
for name, flops, nbytes, t in cases:
    roof = min(cpu_peak, cpu_bandwidth * flops / nbytes)
    print(f"{name:15}{flops / nbytes:10.2f}{roof / 1e9:10.0f}{flops / t / 1e9:10.0f}")
```

跑了几次，拷贝带宽测得 62 到 88 GB/s，最快的矩阵乘法为 1.4 到 2.2 TFLOP/s，也就是说拐点在每字节 20 到 28 FLOPs，比 H100 的低一个数量级。roofline 只是上限，每个操作都以各自的方式达不到它：

- **加法**跑到 4 到 6 GFLOP/s，达到其带宽上限的 75% 到 90%。它是一个受内存限制的操作，以内存的速度运行；同一块芯片上的大矩阵乘法，每秒完成的 FLOPs 是它的几百倍。
- **大矩阵乘法**跑在计算上限上或接近它，正如其强度所预示的那样。其中最快的那一个，本身就定义了这个上限。
- **矩阵向量乘积**只达到其带宽上限的三分之一到一半。库里处理单个向量的那条代码路径，不像单纯的拷贝那样能把内存跑满。roofline 说的是可能达到什么，而不是某个具体实现实际达到了什么。
- **32 × 32 的矩阵乘法**只跑到其上限的十分之一，甚至更低。它只要几微秒，其中大部分时间都花在调用这个库本身上。Horace He 那篇讲 GPU 性能的文章，把这第三种情形称为**开销**（overhead）：“其余的一切”，即既没在计算、也没在搬运张量的时间。在 GPU 上，启动一个 kernel 也有类似的固定开销。

## 生成文本受内存限制

模型生成文本时，每个新 token 都要经过每个权重矩阵一次。只有一条序列时，每个权重从内存读出来，只用来做两次 FLOP：一次乘法、一次加法。所以每个 token 花的时间，至少是读完全部权重的时间：

$$
\text{tokens per second} \le \frac{\text{memory bandwidth}}{\text{bytes of weights}}
$$

以 bfloat16 存储的模型，按一块 H100 的带宽来算：

```python
print(f"{'model':>6}{'weights':>9}{'tokens/s at batch 1':>21}")
for params in [1e9, 8e9, 70e9]:
    weight_bytes = B * params
    print(f"{params / 1e9:5.0f}B{weight_bytes / 1e9:7.0f} GB{bandwidth / weight_bytes:21.0f}")
```

```text
 model  weights  tokens/s at batch 1
    1B      2 GB                 1675
    8B     16 GB                  209
   70B    140 GB                   24
```

算术单元本可以做多得多的事。一个 700 亿参数的模型连一块 H100 的 80 GB 内存都装不下，所以要拆到几块 GPU 上，而这笔账同样适用于每块 GPU 上存放的那部分字节。公式里根本没有出现模型的 FLOPs。批次大小为 1 时，生成速度只由模型大小和内存带宽决定；这也是把权重量化到 8 位或 4 位能加快生成的一个原因：每个 token 要读的字节更少了。

出路是**批处理**（batching）。如果把 64 个用户的序列放在一起处理，每个权重读一次就能用 64 次，所以强度随批次大小增长。用一个 4096 × 4096 的层，在 CPU 上就能直接看到这个效应：

```python
for batch in [1, 8, 64]:
    X = rng.normal(size=(batch, 4096)).astype(np.float32)
    separate = best_time(lambda: [W @ row for row in X], repeats=5)
    together = best_time(lambda: X @ W.T, repeats=5)
    print(f"batch {batch:2}: one at a time {separate / batch * 1e6:6.0f} us per token, "
          f"batched {together / batch * 1e6:6.0f} us per token")
```

跑了几次，逐个处理 token 时，无论批次多大，每个 token 都要花 2.1 到 2.5 毫秒，因为每个 token 都要把整个 67 MB 的矩阵重读一遍。批处理时，每一步只读一次矩阵：8 个 token 时每个花 0.30 到 0.34 毫秒，约为逐个处理时的七分之一；64 个 token 时每个花 44 到 47 微秒，约为五十分之一。这就是推理服务器要把不同用户的请求合成一批来处理的原因，也是每个 token 的成本随负载上升而急剧下降的原因。

## 融合：少搬字节

一个操作如果受内存限制，要让它变快，就得让它少搬字节。最大的杠杆是**融合**（fusion）：不再用一个 kernel 把中间结果写进内存、再用另一个 kernel 把它读回来，而是只用一个 kernel，把中间结果留在片上内存里。Horace He 称算子融合是“深度学习编译器中最重要的优化”。

注意力是最明显的例子。直接计算时，它要把 $N \times N$ 的分数矩阵写进内存，做 softmax 时再读回来，接着写出概率，然后为了乘以 $V$ 再读一遍。这些读写随 $N^2$ 增长，而输入和输出只随 $N$ 增长。FlashAttention（Dao 等人，2022）把注意力切成能放进片上 SRAM 的分块（tile）来计算，从不写出完整的矩阵。用论文的话说，过去缺少的那条原则是“让注意力算法具备 IO 感知——把 GPU 各级内存之间的读写考虑在内”。下面是一个 128 维的注意力头要搬运的字节数，以及 roofline 在 H100 上允许的时间：

```python
head_dim = 128
print(f"{'tokens':>7}{'standard MB':>13}{'fused MB':>10}{'standard us':>13}{'fused us':>10}")
for N in [1024, 4096, 16384]:
    flops = 4 * N * N * head_dim                          # QK^T and PV, one head
    standard = B * (4 * N * head_dim + 4 * N * N)         # Q, K, V, O, plus scores and probabilities written and read
    fused = B * 4 * N * head_dim                          # Q, K, V read once, O written once
    t_std = max(flops / peak, standard / bandwidth)
    t_fused = max(flops / peak, fused / bandwidth)
    print(f"{N:7}{standard / 1e6:13.1f}{fused / 1e6:10.1f}{t_std * 1e6:13.1f}{t_fused * 1e6:10.1f}")
```

```text
 tokens  standard MB  fused MB  standard us  fused us
   1024          9.4       1.0          2.8       0.5
   4096        138.4       4.2         41.3       8.7
  16384       2164.3      16.8        646.0     139.0
```

两列的算术量完全相同。在 16,384 个 token 时，直接计算每个头要搬运 2 GB 以上，受内存限制；融合版本只搬 17 MB，受计算限制，按这个估算快了约 4.6 倍。不管序列多长，直接计算的注意力强度都卡在头维度的一半附近，这里是每字节 64 FLOPs，远低于拐点。融合注意力的强度则会随序列长度增长。

## 框架是怎么做的

上面每一个想法，都已经内置在工具里：

- **编译器会融合操作**。`torch.compile` 把捕获到的计算图交给 TorchInductor，PyTorch 文档称它为“torch.compile 默认的深度学习编译器”。它会生成自己的 kernel，在 GPU 上借助 OpenAI Triton 来生成；正因如此，一串逐元素操作，比如加偏置、激活函数和 dropout，才能作为一个 kernel 运行，中间结果从不落到内存里。
- **注意力默认就是融合的**。只要输入允许，PyTorch 的 `scaled_dot_product_attention` 就会选择一个融合实现：FlashAttention，或类似的内存高效 kernel。
- **推理服务器会做批处理**。它们把许多用户的请求编进每一次前向传播，因为读取权重的代价由批次里的每一条序列分摊。
- **更小的数搬得更快**。上一章的 16 位和 8 位格式，分别把每个值的字节数减为一半和四分之一。对受内存限制的操作来说，这直接就是提速，还没算上更快的算术。

## 延伸阅读

Horace He 的“Making Deep Learning Go Brrrr From First Principles”（2022）是介绍计算、内存和开销这三种情形的最佳短文，写给训练模型的人看。roofline 论文（Williams 等人，2009）最初是针对多核 CPU 提出这个模型的，至今读来仍不费力。FlashAttention 论文给出了注意力完整的 IO 分析，也是一个围绕内存流量而不是 FLOPs 来设计算法的好例子。

> [!TIP] 要点
> - 每块芯片都有两种速度：算术速度和内存带宽。两者之比就是拐点，对 bfloat16 下的 H100 来说约为每字节 295 FLOPs。
> - 算术强度，即每搬运一个字节对应的 FLOPs，决定了是哪一种速度在限制一个操作。逐元素操作、归一化和 softmax 受内存限制；大矩阵乘法受计算限制。
> - 为一条序列生成一个 token，要把每个权重都读一遍。无论 FLOPs 多少，它的速度上限都是带宽除以权重的大小。
> - 批处理让每个权重被许多条序列复用，融合让中间结果不进内存。两者都提高了强度，也都是真实系统比朴素实现快得多的原因。
> - 小操作往往两道限制都碰不到，卡住它们的是启动它们的固定开销。
