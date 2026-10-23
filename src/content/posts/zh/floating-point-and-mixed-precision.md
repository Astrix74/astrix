---
title: 数值：浮点数、溢出与混合精度
description: 浮点数在比特层面怎样布局，手工解码、舍入，再拿 NumPy 验证；16 位浮点数破坏训练的四种方式；以及解决这些问题的混合精度方案，用几行代码重新搭出来。
date: 2026-10-23T09:45:00Z
tags: [under-the-hood, numerics, numpy]
series: under-the-hood
chapter: 3
references: [goldberg1991, micikevicius2017, kalamkar2019, micikevicius2022]
---

[前两章](/zh/writing/autograd-for-tensors/)里的每个数组，存的都是 32 位浮点数。大模型则用 16 位的数来训练，用 8 位的也越来越多：它们只占一半的内存和一半的内存带宽，GPU 做它们的乘法也快得多。代价是，网络算出来的有些数装不下了：softmax 会上溢，小梯度会变成零，权重更新会凭空消失，却没有任何报错。

本章从比特开始搭出浮点数，每一部分都拿 NumPy 验证。然后按低精度可能破坏训练的四种方式，逐一把训练弄坏，再像框架那样把它修好。

## 浮点数就是三个整数

一个二进制浮点数存三样东西：符号位 $s$、占 $k$ 位的指数 $e$，以及占 $m$ 位的尾数（fraction）$f$。对于普通的数（即“正规数”，normal number），它的值为

$$
x = (-1)^s \times \left(1 + \frac{f}{2^m}\right) \times 2^{\,e - \text{bias}}, \qquad \text{bias} = 2^{k-1} - 1
$$

开头那个 1 是隐含的，所以存 $m$ 位就能得到 $m + 1$ 位精度。有两个指数值是保留的。指数位全为零表示**次正规数**（subnormal）：去掉开头的 1，指数固定为 $1 - \text{bias}$，这样数值可以平滑地逐渐趋近于零，而不是在某处戛然而止。指数位全为一时，尾数为零表示无穷大，否则表示 NaN。

对深度学习来说，要紧的布局有三种：

| 格式 | 符号位 | 指数位 | 尾数位 | 字节数 |
| --- | --- | --- | --- | --- |
| float32 | 1 | 8 | 23 | 4 |
| float16 | 1 | 5 | 10 | 2 |
| bfloat16 | 1 | 8 | 7 | 2 |

bfloat16 即“Brain Floating Point”格式，它不过是 float32 的高半部分：符号位和指数位都一样，只是尾数截短到 7 位。写一个以布局为参数的解码器，三种格式就都能处理：

```python
import math
import numpy as np

def decode(bits, exp_bits, frac_bits):
    """The value of a binary floating-point number, given its bit pattern as an integer."""
    sign = -1.0 if bits >> (exp_bits + frac_bits) else 1.0
    exponent = (bits >> frac_bits) & ((1 << exp_bits) - 1)
    fraction = bits & ((1 << frac_bits) - 1)
    bias = (1 << (exp_bits - 1)) - 1
    if exponent == (1 << exp_bits) - 1:  # all ones: infinity, or NaN if any fraction bit is set
        return sign * math.inf if fraction == 0 else math.nan
    if exponent == 0:  # subnormal: no implicit leading 1
        return sign * (fraction / 2**frac_bits) * 2.0 ** (1 - bias)
    return sign * (1 + fraction / 2**frac_bits) * 2.0 ** (exponent - bias)

print(decode(0b0_01111_0000000000, 5, 10))  # float16: sign 0, exponent 15, fraction 0
print(decode(0b1_10000_1000000000, 5, 10))  # sign 1, exponent 16, fraction 0.1 in binary
print(decode(0b0_11110_1111111111, 5, 10))  # the largest finite float16
print(decode(0b0_00000_0000000001, 5, 10))  # the smallest positive float16, a subnormal
```

```text
1.0
-3.0
65504.0
5.960464477539063e-08
```

接下来是参照（oracle）。float16 和 bfloat16 各自只有 65,536 种位模式，所以每一种都可以逐一检验：前者对照 NumPy 的 float16；bfloat16 则把这 16 位放在 16 个零位之上，对照 NumPy 的 float32。至于 float32，抽查一百万个随机位模式就够了。

```python
def agree(mine, expected):
    """How many values are equal, counting NaN as equal to NaN."""
    return int(np.sum((mine == expected) | (np.isnan(mine) & np.isnan(expected))))

patterns = np.arange(2**16, dtype=np.uint16)
sample = np.random.default_rng(0).integers(0, 2**32, size=1_000_000, dtype=np.uint64).astype(np.uint32)
with np.errstate(invalid="ignore"):  # NumPy warns when it converts the NaN patterns
    float16_values = patterns.view(np.float16).astype(np.float64)
    bfloat16_values = (patterns.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    float32_values = sample.view(np.float32).astype(np.float64)

print("float16: ", agree(np.array([decode(int(b), 5, 10) for b in patterns]), float16_values), "of", 2**16)
print("bfloat16:", agree(np.array([decode(int(b), 8, 7) for b in patterns]), bfloat16_values), "of", 2**16)
print("float32: ", agree(np.array([decode(int(b), 8, 23) for b in sample]), float32_values), "of", len(sample))
```

```text
float16:  65536 of 65536
bfloat16: 65536 of 65536
float32:  1000000 of 1000000
```

## 范围和精度都由布局决定

一种格式的一切行为，都由这两个位宽推出来。指数位决定**范围**（range），从最大的有限值一直到最小的次正规数。尾数位决定**精度**（precision）：1 与下一个可表示的数之间的间隔叫作机器精度（machine epsilon），等于 $2^{-m}$，其他任何数附近的间隔都与它成比例。

```python
formats = {"float32": (8, 23), "float16": (5, 10), "bfloat16": (8, 7)}

def limits(exp_bits, frac_bits):
    bias = 2 ** (exp_bits - 1) - 1
    return {
        "largest": (2 - 2.0**-frac_bits) * 2.0**bias,
        "smallest normal": 2.0 ** (1 - bias),
        "smallest subnormal": 2.0 ** (1 - bias - frac_bits),
        "epsilon": 2.0**-frac_bits,  # the gap between 1 and the next number up
    }

print(f"{'':9}{'largest':>11}{'smallest normal':>17}{'smallest subnormal':>20}{'epsilon':>10}")
for name, (e, f) in formats.items():
    v = limits(e, f)
    print(f"{name:9}{v['largest']:11.5g}{v['smallest normal']:17.4g}{v['smallest subnormal']:20.4g}{v['epsilon']:10.3g}")

for name, dtype in [("float32", np.float32), ("float16", np.float16)]:
    info, v = np.finfo(dtype), limits(*formats[name])
    assert (info.max, info.smallest_normal, info.smallest_subnormal, info.eps) == tuple(v.values())
```

```text
             largest  smallest normal  smallest subnormal   epsilon
float32   3.4028e+38        1.175e-38           1.401e-45  1.19e-07
float16        65504        6.104e-05            5.96e-08  0.000977
bfloat16  3.3895e+38        1.175e-38           9.184e-41   0.00781
```

断言全部通过，说明这些公式与 NumPy 的 `finfo` 分毫不差。这张表可以看成一种取舍：

- **float16** 的精度尚可，大约有三位十进制有效数字，但范围很窄。超过 65,504 的数都会上溢为无穷大，小于约 $3 \times 10^{-8}$ 的数都会变成零。
- **bfloat16** 的范围和 float32 一样，但它的机器精度是 0.0078。在 1 附近，它分不清 1.000 和 1.003。

下面的每一种故障，都出在这种取舍的某一边。

## 舍入

每个算术结果都会舍入到最近的可表示值。如果结果恰好落在两个相邻值的正中间，IEEE 754 会舍入到最后一位为 0 的那一个，这叫四舍六入五成双（round half to even）；这样一来，恰在中点的情况就不会让很长的计算朝同一个方向偏。下面把它实现两遍。`round_to` 用算术来做，适用于任何布局。`to_bfloat16` 用整数位运算来做，和硬件把 float32 转成 bfloat16 的方式一样：加上 `0x7FFF`，如果保留下来的最后一位是奇数就再加 1，然后把低 16 位清零。两者没有共用任何代码，所以可以互相检验。

```python
def round_to(x, exp_bits, frac_bits):
    """Round to the nearest value the format can hold, breaking ties to even, as IEEE 754 does."""
    if x == 0 or not math.isfinite(x):
        return x
    bias = 2 ** (exp_bits - 1) - 1
    exponent = max(math.frexp(abs(x))[1] - 1, 1 - bias)  # exponent of the leading bit, floored for subnormals
    step = 2.0 ** (exponent - frac_bits)                  # the gap between neighbouring values near x
    rounded = round(x / step) * step                      # Python's round() breaks ties to even
    return math.copysign(math.inf, x) if abs(rounded) > limits(exp_bits, frac_bits)["largest"] else rounded

def to_bfloat16(x):
    """float32 to bfloat16 the way hardware does it: add a rounding bias, then clear the low 16 bits."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    bias = 0x7FFF + ((bits >> 16) & 1)  # on a tie, round toward the neighbour whose last kept bit is 0
    return ((bits + bias) & 0xFFFF0000).view(np.float32)

rng = np.random.default_rng(1)
values = rng.integers(0, 2**32, size=200_000, dtype=np.uint64).astype(np.uint32).view(np.float32)
values = values[np.isfinite(values)]
ties = ((rng.integers(0, 2**16, size=50_000, dtype=np.uint32) << 16) | 0x8000).view(np.float32)
ties = ties[np.isfinite(ties)]  # each one exactly halfway between two bfloat16 numbers
for name, xs in [("random float32", values), ("exact ties", ties)]:
    by_arithmetic = np.array([round_to(float(x), 8, 7) for x in xs])
    print(f"bfloat16, {name}: {np.sum(to_bfloat16(xs) == by_arithmetic)} of {len(xs)} agree")

xs = rng.normal(size=200_000).astype(np.float32) * np.float32(2.0) ** rng.integers(-30, 20, size=200_000).astype(np.float32)
with np.errstate(over="ignore"):  # some of these are too large for float16 on purpose
    numpy_float16 = xs.astype(np.float16).astype(np.float64)
print(f"float16 against NumPy: {np.sum(numpy_float16 == [round_to(float(x), 5, 10) for x in xs])} of {len(xs)} agree")
```

```text
bfloat16, random float32: 199254 of 199254 agree
bfloat16, exact ties: 49796 of 49796 agree
float16 against NumPy: 200000 of 200000 agree
```

两种 bfloat16 转换在每一个随机值、每一个恰在中点的值上都一致；`round_to` 在 200,000 个值上与 NumPy 的 float16 转换结果相同，其中也包括会上溢的值。从这里开始，就用 `to_bfloat16` 来代替 bfloat16 硬件。

## 低精度破坏训练的四种方式

### 上溢

注意力和损失的核心都是 softmax，而 softmax 要调用 `exp`。在 float16 中，一旦 $x > \ln 65504 \approx 11.09$，$e^{x}$ 就会超过 65,504，而这么大的 logits 再平常不过：

```python
logits = np.array([12.0, 10.0, 8.0], dtype=np.float16)
with np.errstate(over="ignore", invalid="ignore"):
    print("exp:           ", np.exp(logits))
    print("softmax:       ", np.exp(logits) / np.exp(logits).sum())
shifted = np.exp(logits - logits.max())
print("stable softmax:", shifted / shifted.sum())
```

```text
exp:            [      inf 2.203e+04 2.980e+03]
softmax:        [nan  0.  0.]
stable softmax: [0.867   0.1174  0.01587]
```

$e^{12}$ 上溢为无穷大，无穷大除以无穷大得到 NaN，而这个 NaN 会传遍整个反向传播。修复方法就是每个 softmax 实现都在用的那一招：先减去最大的 logit。这不会改变结果，因为 softmax 只取决于 logits 之间的差；但它让 `exp` 的每个参数都为零或负数，所以结果都不会超过 1。

### 下溢

梯度往往很小。一个梯度如果小于 float16 最小次正规数的一半，也就是约 $3 \times 10^{-8}$，就会被舍入为零；而在最小正规数以下，值越小，保留的精度位就越少。**损失缩放**（loss scaling）利用线性来解决这个问题：在反向传播之前把损失乘以一个常数 $S$，每个梯度也就都乘上了 $S$，从而被抬进 float16 的范围之内。在优化器更新之前，再在 float32 中除以 $S$。

下面这批梯度的量级均匀分布在 $10^{-10}$ 到 $10^{-2}$ 这八个数量级上，分别在缩放和不缩放的情况下存成 float16：

```python
rng = np.random.default_rng(0)
grads = 10.0 ** rng.uniform(-10, -2, size=100_000)  # magnitudes spread evenly over eight orders of magnitude
for scale in [1, 2**10, 2**16]:
    stored = (grads * scale).astype(np.float16)
    lost = np.mean(stored == 0)
    imprecise = np.mean((stored != 0) & (stored < np.finfo(np.float16).smallest_normal))
    print(f"loss scale {scale:>5}: {lost:6.1%} become zero, {imprecise:6.1%} subnormal, largest {float(stored.max()):g}")
```

```text
loss scale     1:  30.8% become zero,  41.6% subnormal, largest 0.0100021
loss scale  1024:   0.0% become zero,  34.5% subnormal, largest 10.2422
loss scale 65536:   0.0% become zero,  12.3% subnormal, largest 655.5
```

不缩放时，将近三分之一的梯度直接变成了零，另有 42% 成了次正规数。乘以 $2^{16}$ 之后，一个梯度也没丢，离 65,504 也还有充足的余量。提出这套方法的 Micikevicius 等人（2017）在论文中描述过一个 SSD 目标检测器：用 float16 时它怎么也训练不起来，直到对损失做了缩放才行。那个例子里，缩放因子取 8 就够了；在全部实验中，他们用过的因子从 8 到 32K 不等，同时也指出许多网络根本不需要缩放。

### 吞没

两个浮点数相加时，硬件要先对齐二者的二进制小数点，也就是把较小那个数的各位往右移。如果较小的数不到较大数附近可表示值间隔的一半，它就会被完全舍掉。放在权重更新上，就是这次更新什么也没做，而且没有任何东西报告这件事。Micikevicius 等人给出的 float16 阈值是：权重至少是其更新量的 2048 倍，也就是 $2^{11}$ 倍，这时更新量已不到权重所在量级上间隔的一半。同样的推理套到只有 7 个尾数位的 bfloat16 上，这个比值只有 256。

```python
w32, w16, wbf16 = np.float32(1.0), np.float16(1.0), to_bfloat16(1.0)
for _ in range(1000):
    w32 = w32 + np.float32(0.001)
    w16 = w16 + np.float16(0.001)
    wbf16 = to_bfloat16(wbf16 + np.float32(0.001))
print("after 1000 updates of 0.001, starting from 1:")
print("float32", w32, " float16", w16, " bfloat16", wbf16)
```

```text
after 1000 updates of 0.001, starting from 1:
float32 2.0000467  float16 1.977  bfloat16 1.0
```

bfloat16 的权重纹丝不动：每次 0.001 的更新都不到 1 上方那个 0.0078 间隔的一半，所以每次加法都直接舍入回 1。float16 的情况更隐蔽：更新保住了，但每次都被舍入成 $2^{-10} \approx 0.000977$，于是一千步之后，权重只增加了 0.977 而不是 1，少了 2.3%，既没有报错，也没有警告。

### 累加

长求和就是许多次加法：累计和不断增长，而每个新加进来的项始终很小。到最后，每一项都小于累计和所在量级上间隔的一半，和就再也不动了：

```python
values = np.full(10_000, 0.01, dtype=np.float16)
total = np.float16(0)
for v in values:
    total = total + v  # every addition rounded to float16
print("float16 running sum:", total)
print("summed in float32:  ", values.astype(np.float32).sum())
print("exact:              ", 10_000 * float(values[0]))
```

```text
float16 running sum: 32.0
summed in float32:   100.02136
exact:               100.0213623046875
```

从 32 往上，float16 的间隔是 $2^{-5} = 0.03125$。它的一半大于 0.01，所以每次加法都舍入回 32。经过 2,798 次加法后，和就到了这里，剩下的 7,202 项什么也没改变。把同样的 float16 值放在 float32 中求和，就能得到正确答案。正因如此，那篇论文规定矩阵乘法内部要把乘积累加到 float32 中，并指出“大型归约（对向量各元素求和）应当在 FP32 中进行”。GPU 的 Tensor Core 正是为此打造的：它们对 16 位输入做乘法，并且可以在 32 位中累加乘积。

## 组装混合精度

这四种故障指向同一套方案，也正是 Micikevicius 等人提出的方案：

1. **保留一份权重的 float32 主副本**（master copy），每次更新都作用在它上面。这解决了吞没问题。
2. **矩阵乘法用 16 位来算**，用的是权重的 16 位副本，乘积在 float32 中累加。速度就来自这里。
3. **归约、softmax、归一化和损失都用 float32 计算**。这解决了上溢和累加问题。
4. **使用 float16 时，要对损失做缩放**。这解决了下溢问题。bfloat16 的范围与 float32 相同，所以通常不需要这一步。

主副本比看上去更重要。下面用三种方式训练线性回归：全部用 float32；矩阵乘法用 bfloat16，权重也存成 bfloat16；以及混合精度，矩阵乘法同样用 bfloat16，但保留权重的 float32 主副本。后两种之间唯一的区别，就是权重怎么存。

```python
rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 20)).astype(np.float32)
w_true = rng.normal(size=20).astype(np.float32)
y = X @ w_true + np.float32(0.1) * rng.normal(size=1000).astype(np.float32)

def train(mode, steps=3000, lr=np.float32(0.01)):
    """Linear regression by gradient descent. Returns the loss at a few checkpoints."""
    w = np.zeros(20, dtype=np.float32)
    X_used = X if mode == "float32" else to_bfloat16(X)
    losses = []
    for step in range(1, steps + 1):
        w_used = w if mode == "float32" else to_bfloat16(w)  # 16-bit copy of the weights for the matmuls
        error = X_used @ w_used - y                          # products of bfloat16 values, summed in float32
        w = w - lr * (X_used.T @ error) / np.float32(len(y))
        if mode == "bfloat16":
            w = to_bfloat16(w)                               # no float32 master copy: store the result in bfloat16
        if step in (100, 300, 1000, 3000):
            losses.append(float(np.mean((X @ w - y) ** 2)))
    return losses

print(f"{'step':9}{100:>9}{300:>9}{1000:>9}{3000:>9}")
for mode in ["float32", "bfloat16", "mixed"]:
    print(f"{mode:9}" + "".join(f"{loss:9.4f}" for loss in train(mode)))
```

```text
step           100      300     1000     3000
float32     2.6372   0.0677   0.0092   0.0092
bfloat16    2.7304   1.2346   1.2346   1.2346
mixed       2.6371   0.0678   0.0092   0.0092
```

到第 300 步时，bfloat16 权重已经离目标值足够近，每一步的更新量都小于权重附近 bfloat16 值间隔的一半，于是每次更新都被舍掉了。损失冻结在 1.23，是噪声底（noise floor）0.01 的一百多倍。混合精度那一组用的 16 位乘法完全相同，在每个检查点上与 float32 的差距都不超过 0.0002。唯一的变化，是更新在哪里累加。

对 float16 来说，还剩最后一块：怎么选 $S$。固定值只能靠猜，而且遇到偶尔出现的大梯度仍然可能上溢。**动态损失缩放**（dynamic loss scaling）在训练过程中调整缩放系数：一开始取得很大；一旦有梯度上溢，就把缩放系数减半并跳过这一步；连续很多步都没有上溢之后，再把它翻倍。下面的默认值取自 PyTorch：

```python
class LossScaler:
    """Dynamic loss scaling, with the defaults of PyTorch's GradScaler."""

    def __init__(self, scale=2.0**16, growth_interval=2000):
        self.scale, self.growth_interval, self.clean_steps = scale, growth_interval, 0

    def unscale(self, grads):
        """The true gradients, from the float16 gradients of the scaled loss, or None if they overflowed."""
        if not all(np.isfinite(g).all() for g in grads):
            self.scale /= 2  # skip this step, and try a smaller scale on the next one
            self.clean_steps = 0
            return None
        grads = [g.astype(np.float32) / np.float32(self.scale) for g in grads]
        self.clean_steps += 1
        if self.clean_steps == self.growth_interval:
            self.scale *= 2  # no overflow for a long while: try a larger scale from the next step
            self.clean_steps = 0
        return grads

scaler = LossScaler(growth_interval=3)  # grow after 3 clean steps instead of 2000, to see it happen
for step, true_grad in enumerate([1e-8, 1e-8, 2.0, 2.0, 1e-8, 1e-8, 1e-8, 1e-8]):
    scale = scaler.scale
    with np.errstate(over="ignore"):
        backward = np.array([true_grad * scale], dtype=np.float16)  # what a float16 backward pass would hold
    grads = scaler.unscale([backward])
    print(f"step {step}: scale {scale:>6.0f}, " + ("overflow, step skipped" if grads is None else f"gradient {grads[0][0]:.3g}"))
```

```text
step 0: scale  65536, gradient 1e-08
step 1: scale  65536, gradient 1e-08
step 2: scale  65536, overflow, step skipped
step 3: scale  32768, overflow, step skipped
step 4: scale  16384, gradient 1e-08
step 5: scale  16384, gradient 1e-08
step 6: scale  16384, gradient 1e-08
step 7: scale  32768, gradient 1e-08
```

两个大梯度先在缩放系数 65,536 下上溢，又在 32,768 下再次上溢，所以这两步都被跳过，缩放系数减半了两次。又过了三个干净的步骤，它涨了回去。每一个 $10^{-8}$ 的梯度都完好无损地到达，而如果不做缩放，它在 float16 中本会是零。跳过一步只损失一个批次，而应用一个无穷大的梯度则会毁掉权重。论文还提到，在梯度裁剪或其他任何要查看梯度值的操作之前，都要先撤销缩放，这样各种阈值才能保持它们在 float32 下的含义。

## 框架是怎么做的

PyTorch 把这套方案拆给了两个工具。

- **`torch.autocast`** 在一段代码区域内，为每个操作分别选择精度。PyTorch 文档列出了以 float16 运行的 CUDA 操作，例如 `matmul`、`linear`、`bmm` 和各种卷积；也列出了以 float32 运行的操作，例如 `exp`、`log`、`softmax`、`log_softmax`、`sum`、`layer_norm`、`cross_entropy` 和 `mse_loss`。这就是上面的第 2 步和第 3 步，只是写成了两个列表。
- **`torch.amp.GradScaler`** 就是上面那个类。它的默认值是：初始缩放系数 65,536，连续 2000 个干净步骤后乘以 2，回退时乘以 0.5。用 bfloat16 时，通常就不用这个缩放器了。

Kalamkar 等人（2019）在图像、语音、语言和推荐模型上研究了 bfloat16 训练，并把它好用的原因归于它的范围：“保持与 FP32 相同的范围很重要，这样才能确保收敛无需任何超参数调优”。

同样的取舍在 16 位以下依然存在。Micikevicius 等人（2022）定义了两种 8 位格式：E5M2，有 5 个指数位和 2 个尾数位；E4M3，分别是 4 个和 3 个。E5M2 遵循 IEEE 的约定，所以前面那个布局函数就能给出它的最大值：

```python
print("E5M2 largest:", limits(5, 2)["largest"])
print("E4M3 largest, IEEE-style:", limits(4, 3)["largest"])
```

```text
E5M2 largest: 57344.0
E4M3 largest, IEEE-style: 240.0
```

E4M3 不遵循 IEEE。它放弃了无穷大，只为 NaN 保留一种尾数位模式，把腾出来的位模式拿来表示普通的数。这样一来，它的最大值就从上面的 240 提高到了 448。即便如此，一个范围也很难适合网络里的每个张量，所以 FP8 训练会在转换之前先把张量乘以缩放因子，而且往往每个张量各用一个因子。用论文的话说：“总体思路是选择一个缩放因子，使张量中的最大幅值接近相应格式所能表示的最大幅值。”这又是损失缩放，只不过作用在每个张量上，而不只是梯度。

## 延伸阅读

David Goldberg 的“What Every Computer Scientist Should Know About Floating-Point Arithmetic”（1991）至今仍是讲舍入、误差以及浮点运算为何如此表现的最佳读物。“Mixed Precision Training”（Micikevicius 等人）篇幅短、很实用，其中第 3 节涵盖了本章的每一项技术。PyTorch 的[自动混合精度文档](https://docs.pytorch.org/docs/stable/amp.html)完整列出了哪些操作以哪种精度运行。

> [!TIP] 要点
> - 浮点数由符号、指数和尾数组成。指数位换来范围，尾数位换来精度。
> - float16 有精度但范围小：超过 65,504 就会上溢，小于约 $3 \times 10^{-8}$ 的梯度会直接变成零。bfloat16 有 float32 的范围，但精度不到三位十进制有效数字。
> - 结果会舍入到最近的可表示值，恰在中点时取偶。加法会丢掉任何小于较大操作数所在量级间隔一半的项。
> - 混合精度意味着：权重的 float32 主副本，16 位矩阵乘法加 float32 累加，归约、softmax 和损失用 float32 计算，以及（对 float16 而言）动态损失缩放。
> - 低精度造成的故障大多悄无声息。要拿结果和一次 float32 运行对照，就像用有限差分检验梯度一样。

[下一章](/zh/writing/memory-bandwidth-and-the-roofline/)从数值格式转向硬件：为什么 GPU 的速度通常取决于它搬运字节有多快，而不是做乘法有多快。
