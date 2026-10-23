---
title: "GPUs: why memory bandwidth decides speed"
description: "A chip has two speeds: how fast it computes and how fast it moves bytes. Work out which one limits each operation in a transformer, measure the same effect on a laptop, and see why batching and fusion matter so much."
date: 2026-10-23T10:25:00Z
tags: [under-the-hood, gpus, numpy]
series: under-the-hood
chapter: 4
references: [williams2009, he2022brrr, dao2022]
---

An accelerator's spec sheet leads with one number, its arithmetic throughput. For NVIDIA's H100 SXM that is 1,979 teraFLOPS of bfloat16, with a footnote saying "with sparsity". For ordinary dense matrices it is half that, about 989 teraFLOPS. Further down is a second number: 3.35 TB/s of memory bandwidth. The ratio between the two decides how fast almost everything in deep learning runs:

$$
\frac{989 \times 10^{12}\ \text{FLOP/s}}{3.35 \times 10^{12}\ \text{byte/s}} \approx 295\ \text{FLOPs per byte}
$$

To keep its arithmetic units busy, the chip has to do about 295 operations on every byte it reads from memory. Most operations in a transformer do one or two. For them, the arithmetic units are idle almost all the time, and what decides the speed is how fast bytes arrive. The [previous chapter](/writing/floating-point-and-mixed-precision/) made numbers smaller. This one is about why that matters even more than it seemed.

## Arithmetic intensity

The useful measure is an operation's **arithmetic intensity**: the FLOPs it performs divided by the bytes it reads and writes. An elementwise add of two bfloat16 tensors reads two values and writes one, 6 bytes, for 1 FLOP, an intensity of $1/6$. A matrix multiplication of an $(n \times d)$ matrix by a $(d \times d)$ one does $2nd^2$ FLOPs on about $2(2nd + d^2)$ bytes. Each weight is reused once per row, so its intensity grows with $n$.

The **roofline model** (Williams et al., 2009) turns that into a speed limit. An operation can't run faster than the chip computes, and it can't run faster than its bytes arrive:

$$
\text{attainable FLOP/s} = \min\left(\text{peak FLOP/s},\ \text{bandwidth} \times \text{intensity}\right)
$$

Below the **ridge point**, 295 FLOPs per byte on an H100, an operation is memory-bound. Above it, compute-bound. Here are the operations of a transformer layer, at a width of 4096 with 4096 tokens in flight:

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

The matrix multiplication runs at full speed. Everything else runs at well under 1% of the chip's arithmetic peak, because it is waiting for memory. The matrix-vector product is the interesting one. With a single token, each weight is read from memory and used exactly once, so even a matrix product becomes memory-bound. That is the situation every time a language model generates a token for a single user.

## Measure it

None of this is specific to GPUs. The same two limits exist on any processor, so the model can be checked on a laptop. This measures the copy bandwidth and the best matrix-multiplication speed of the CPU this was written on, an Apple M1 Pro, through NumPy, then compares each operation's speed with what the roofline allows:

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

Across several runs, the copy measured 62 to 88 GB/s and the best matrix multiplication 1.4 to 2.2 TFLOP/s: a ridge point of 20 to 28 FLOPs per byte, an order of magnitude below the H100's. The roofline is an upper bound, and each operation falls short of it in its own way:

- **The add** ran at 4 to 6 GFLOP/s, 75 to 90% of its bandwidth limit. It is a memory-bound operation running at the speed of memory, a few hundred times fewer FLOPs per second than a large matrix multiplication on the same chip.
- **The big matrix multiplications** run at or near the compute roof, as their intensity says they should. The fastest of them defines the roof.
- **The matrix-vector product** reached only a third to a half of its bandwidth limit. The library's single-vector code path doesn't saturate memory the way a plain copy does. A roofline says what is possible, not what a given implementation achieves.
- **The 32 × 32 matrix multiplication** ran at a tenth of its limit or less. It takes microseconds, and most of that is the cost of calling into the library at all. Horace He's essay on GPU performance names this third regime **overhead**: "Everything else", meaning time spent neither computing nor moving tensors. On a GPU, launching a kernel has a similar fixed cost.

## Generating text is memory-bound

When a model generates text, each new token passes through every weight matrix once. With one sequence, every weight is read from memory to do two FLOPs, one multiply and one add. So the time per token is at least the time to read all the weights:

$$
\text{tokens per second} \le \frac{\text{memory bandwidth}}{\text{bytes of weights}}
$$

For models in bfloat16 on one H100's bandwidth:

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

The arithmetic units could do far more. A 70-billion-parameter model does not even fit in an H100's 80 GB, so it is split across several GPUs, and the same arithmetic applies to the bytes each one holds. Nothing about the model's FLOPs appears in the formula. Generation speed at batch size 1 is set by model size and memory bandwidth alone, which is one reason quantizing weights to 8 or 4 bits speeds up generation: fewer bytes to read per token.

The escape is **batching**. If 64 users' sequences are processed together, each weight is read once and used 64 times, so the intensity grows with the batch size. The CPU shows the effect directly, with one 4096 × 4096 layer:

```python
for batch in [1, 8, 64]:
    X = rng.normal(size=(batch, 4096)).astype(np.float32)
    separate = best_time(lambda: [W @ row for row in X], repeats=5)
    together = best_time(lambda: X @ W.T, repeats=5)
    print(f"batch {batch:2}: one at a time {separate / batch * 1e6:6.0f} us per token, "
          f"batched {together / batch * 1e6:6.0f} us per token")
```

Across several runs, processing tokens one at a time cost 2.1 to 2.5 milliseconds each, at every batch size, because each token rereads the whole 67 MB matrix. Batched, the matrix is read once per step: with 8 tokens each one cost 0.30 to 0.34 milliseconds, about seven times less, and with 64 tokens 44 to 47 microseconds, about fifty times less. This is why inference servers batch requests from different users together, and why the cost per token falls so steeply with load.

## Fusion: moving fewer bytes

If an operation is memory-bound, the way to speed it up is to make it move fewer bytes. The biggest lever is **fusion**: instead of running one kernel that writes an intermediate result to memory and another that reads it back, run a single kernel that keeps the intermediate in on-chip memory. He calls operator fusion "the most important optimization in deep learning compilers."

Attention is the clearest case. Computed directly, it writes the $N \times N$ matrix of scores to memory, reads it back for the softmax, writes the probabilities, and reads them again to multiply by $V$. Those reads and writes grow with $N^2$, while the inputs and output only grow with $N$. FlashAttention (Dao et al., 2022) computes attention in tiles that fit in on-chip SRAM and never writes the full matrix. In the paper's words, the missing principle was "making attention algorithms IO-aware -- accounting for reads and writes between levels of GPU memory." Here is the byte count for one attention head with 128 dimensions, and the time the roofline allows on an H100:

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

The arithmetic is identical in both columns. At 16,384 tokens, the direct computation moves over 2 GB per head and is memory-bound. The fused one moves 17 MB and is compute-bound, about 4.6 times faster by this estimate. The intensity of direct attention is stuck near half the head dimension, 64 FLOPs per byte here, well below the ridge, however long the sequence. Fused attention's intensity grows with the sequence length instead.

## What frameworks do

Each of these ideas is built into the tools:

- **Compilers fuse operations.** `torch.compile` hands the captured graph to TorchInductor, which PyTorch's documentation calls "the default torch.compile deep learning compiler". It generates its own kernels, through OpenAI Triton on GPUs, which is what lets a chain of elementwise operations, such as a bias add, an activation and a dropout, run as one kernel whose intermediates never reach memory.
- **Attention is fused by default.** PyTorch's `scaled_dot_product_attention` selects a fused implementation, FlashAttention or a similar memory-efficient kernel, when the inputs allow it.
- **Inference servers batch.** They group requests from many users into each forward pass, because the cost of reading the weights is shared by every sequence in the batch.
- **Smaller numbers move faster.** The 16-bit and 8-bit formats from the previous chapter halve and quarter the bytes per value. For memory-bound operations that is a direct speedup, before any faster arithmetic.

## Going further

Horace He's "Making Deep Learning Go Brrrr From First Principles" (2022) is the best short introduction to the compute, memory and overhead regimes, written for people who train models. The roofline paper (Williams et al., 2009) introduced the model for multicore CPUs and is still readable. The FlashAttention paper shows the full IO analysis of attention, and is a good example of designing an algorithm around memory traffic rather than FLOPs.

> [!TIP] What to remember
> - Every chip has two speeds: arithmetic and memory bandwidth. Their ratio, the ridge point, is about 295 FLOPs per byte for an H100 in bfloat16.
> - Arithmetic intensity, FLOPs per byte moved, decides which speed limits an operation. Elementwise operations, normalization and softmax are memory-bound. Large matrix multiplications are compute-bound.
> - Generating one token for one sequence reads every weight once. Its speed is bounded by bandwidth divided by the size of the weights, whatever the FLOPs.
> - Batching reuses each weight for many sequences, and fusion keeps intermediates out of memory. Both raise intensity, and both are why real systems are so much faster than naive ones.
> - Small operations are often bound by neither limit, but by the fixed cost of launching them.
