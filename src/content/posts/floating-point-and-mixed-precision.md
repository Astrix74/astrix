---
title: "Numerics: floating point, overflow and mixed precision"
description: "How a floating-point number is laid out in bits, decoded and rounded by hand and checked against NumPy; the four ways 16-bit numbers break training; and the mixed-precision recipe that fixes them, rebuilt in a few lines."
date: 2026-10-23T09:45:00Z
tags: [under-the-hood, numerics, numpy]
series: under-the-hood
chapter: 3
references: [goldberg1991, micikevicius2017, kalamkar2019, micikevicius2022]
---

Every array in the [last two chapters](/writing/autograd-for-tensors/) held 32-bit floats. Large models are trained with 16-bit numbers, and increasingly with 8-bit ones: they take half the memory and half the memory bandwidth, and GPUs multiply them much faster. The price is that some of the numbers a network computes stop fitting. A softmax overflows, small gradients turn into zeros, and weight updates vanish without any error.

This chapter builds floating point from its bits and checks every piece against NumPy. Then it breaks training in each of the four ways that low precision can, and fixes it the way frameworks do.

## A float is three integers

A binary floating-point number stores a sign bit $s$, an exponent $e$ of $k$ bits and a fraction $f$ of $m$ bits. For ordinary ("normal") numbers, the value is

$$
x = (-1)^s \times \left(1 + \frac{f}{2^m}\right) \times 2^{\,e - \text{bias}}, \qquad \text{bias} = 2^{k-1} - 1
$$

The leading 1 is implicit, so $m$ stored bits give $m + 1$ bits of precision. Two exponent values are reserved. All zeros means a **subnormal** number: the leading 1 is dropped and the exponent is fixed at $1 - \text{bias}$, which lets values fade gradually towards zero instead of stopping abruptly. All ones means infinity if the fraction is zero, and NaN otherwise.

Three layouts matter for deep learning:

| Format | Sign | Exponent | Fraction | Bytes |
| --- | --- | --- | --- | --- |
| float32 | 1 | 8 | 23 | 4 |
| float16 | 1 | 5 | 10 | 2 |
| bfloat16 | 1 | 8 | 7 | 2 |

bfloat16, the "Brain Floating Point" format, is simply the top half of a float32: the same sign and exponent, with the fraction cut to 7 bits. A decoder that takes the layout as parameters handles all three:

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

Now the oracle. float16 and bfloat16 have only 65,536 bit patterns each, so every one of them can be checked: against NumPy's float16, and against NumPy's float32 for bfloat16, by placing the 16 bits on top of 16 zero bits. For float32, a million random patterns will do.

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

## Range and precision come from the layout

Everything about a format's behaviour follows from those two widths. The exponent bits set the **range**, from the largest finite value down to the smallest subnormal. The fraction bits set the **precision**: the gap between 1 and the next representable number, called machine epsilon, is $2^{-m}$, and the gap around any other number scales with it.

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

The assertions pass, so the formulas reproduce NumPy's `finfo` exactly. Read the table as a trade:

- **float16** keeps decent precision, about three decimal digits, but its range is narrow. Anything above 65,504 overflows to infinity, and anything below about $6 \times 10^{-8}$ becomes zero.
- **bfloat16** has the range of float32, but its epsilon is 0.0078. Near 1, it cannot tell 1.000 from 1.003.

Each of the failures below comes from one side of this trade.

## Rounding

Every arithmetic result is rounded to the nearest representable value. When a result lies exactly halfway between two neighbours, IEEE 754 rounds to the one whose last bit is 0, called round half to even, so that ties don't bias long computations in one direction. Here it is twice. `round_to` does it with arithmetic, for any layout. `to_bfloat16` does it with integer bit tricks, the way hardware converts float32 to bfloat16: add `0x7FFF`, plus 1 if the last bit kept is odd, then clear the low 16 bits. The two share no code, so they can check each other.

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

Both bfloat16 conversions agree on every random value and every exact tie, and `round_to` matches NumPy's float16 conversion on 200,000 values, including ones that overflow. From here on, `to_bfloat16` stands in for bfloat16 hardware.

## Four ways low precision breaks training

### Overflow

The softmax at the heart of attention and of the loss calls `exp`. In float16, $e^{x}$ exceeds 65,504 once $x > \ln 65504 \approx 11.09$, and logits that large are ordinary:

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

$e^{12}$ overflows to infinity, infinity divided by infinity is NaN, and the NaN would spread through the whole backward pass. The fix is the one every softmax implementation uses: subtract the largest logit first. That doesn't change the result, because softmax only depends on differences between logits, but it makes every argument to `exp` zero or negative, so no result can exceed 1.

### Underflow

Gradients are often tiny. When a gradient is smaller than half of float16's smallest subnormal, about $3 \times 10^{-8}$, it rounds to zero, and below the smallest normal number each value keeps fewer bits of precision the smaller it is. **Loss scaling** fixes this with linearity: multiply the loss by a constant $S$ before the backward pass, and every gradient is multiplied by $S$ too, moving it up into float16's range. Divide by $S$ in float32 before the optimizer step.

Here are gradients spread evenly over eight orders of magnitude, from $10^{-10}$ to $10^{-2}$, stored in float16 with and without scaling:

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

Without scaling, almost a third of the gradients vanish, and another 42% are subnormal. Scaling by $2^{16}$ loses nothing and leaves plenty of headroom below 65,504. Micikevicius et al. (2017), the paper that introduced this recipe, describe an SSD object detector that failed to train in float16 until the loss was scaled. A factor of 8 was enough there, and across their experiments they used factors from 8 to 32K, noting that many networks needed none.

### Swamping

To add two floats, hardware lines up their binary points, which shifts the smaller number's bits to the right. If the smaller number is less than half the gap between representable values near the larger one, it rounds away completely. For a weight update this means nothing happens, and nothing reports it. Micikevicius et al. put the threshold for float16 at a weight 2048 times larger than its update, which is $2^{11}$, half the gap at the weight's scale. The same reasoning with bfloat16's 7 fraction bits gives a ratio of only 256 to 512.

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

The bfloat16 weight never moves: each update of 0.001 is less than half of the 0.0078 gap above 1, so every addition rounds straight back to 1. float16 is subtler. The update survives, but it is rounded to $2^{-10} \approx 0.000977$ each time, so after a thousand steps the weight is 2.3% short, with no error and no warning.

### Accumulation

A long sum is many additions, and the running total keeps growing while each new term stays small. Eventually each term falls below half the gap at the total's size, and the sum stops moving:

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

From 32 upward, float16's gap is $2^{-5} = 0.03125$. Half of that is more than 0.01, so every addition rounds back to 32. The sum gets there after 2,798 additions, and the remaining 7,202 terms change nothing. Summing the same float16 values in float32 gives the right answer. This is why the paper prescribes accumulating products into float32 inside matrix multiplications, and says that "Large reductions (sums across elements of a vector) should be carried out in FP32". GPU tensor cores were built for this: they multiply 16-bit inputs and can accumulate the products in 32 bits.

## Mixed precision, assembled

The four failures point to one recipe, which is what Micikevicius et al. proposed:

1. **Keep a float32 master copy of the weights**, and apply every update to it. This fixes swamping.
2. **Run the matrix multiplications in 16 bits**, using a 16-bit copy of the weights, and accumulate their products in float32. This is where the speed comes from.
3. **Do reductions, softmax, normalization and the loss in float32**. This fixes overflow and accumulation.
4. **With float16, scale the loss**. This fixes underflow. bfloat16 has float32's range, so it usually doesn't need this step.

The master copy matters more than it looks. Here is linear regression trained three ways: entirely in float32; with bfloat16 matrix multiplications and bfloat16 weights; and mixed, with the same bfloat16 multiplications but a float32 master copy of the weights. Only the storage of the weights differs between the last two.

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

By step 300, the bfloat16 weights are close enough to their targets that each step's update is smaller than half the gap between bfloat16 values near the weight, so every update rounds away. The loss freezes at 1.23, more than a hundred times the noise floor of 0.01. The mixed run uses exactly the same 16-bit multiplications and stays within 0.0002 of float32 at every checkpoint. The only change is where the updates accumulate.

For float16, the remaining piece is choosing $S$. A fixed value has to be guessed, and it can still overflow on a rare large gradient. **Dynamic loss scaling** adjusts the scale while training: start large, halve the scale and skip the step whenever a gradient overflows, and double it again after a long run of steps without overflow. The defaults below are PyTorch's:

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

The two large gradients overflow at scale 65,536 and again at 32,768, so both steps are skipped and the scale halves twice. Three clean steps later it grows back. Every gradient of $10^{-8}$ arrives intact, though unscaled it would have been zero in float16. Skipping a step costs one batch, while applying an infinite gradient would ruin the weights. The paper also says to unscale before gradient clipping or anything else that looks at gradient values, so that thresholds keep their float32 meaning.

## What frameworks do

PyTorch splits the recipe across two tools.

- **`torch.autocast`** picks a precision per operation inside a region of code. The PyTorch documentation lists the CUDA operations that run in float16, such as `matmul`, `linear`, `bmm` and the convolutions, and the ones that run in float32, such as `exp`, `log`, `softmax`, `log_softmax`, `sum`, `layer_norm`, `cross_entropy` and `mse_loss`. That is step 3 above, written as two lists.
- **`torch.amp.GradScaler`** is the class above. Its defaults are an initial scale of 65,536, growth by 2 after 2000 clean steps, and back-off by 0.5. With bfloat16, the scaler is usually left out.

Kalamkar et al. (2019) studied bfloat16 training across image, speech, language and recommendation models, and give its range as the reason it is easy to use: "Maintaining the same range as FP32 is important to ensure that no hyper-parameter tuning is required for convergence".

The same trade continues below 16 bits. Micikevicius et al. (2022) define two 8-bit formats: E5M2, with 5 exponent bits and 2 fraction bits, and E4M3, with 4 and 3. E5M2 follows IEEE conventions, so the layout function from earlier gives its largest value:

```python
print("E5M2 largest:", limits(5, 2)["largest"])
print("E4M3 largest, IEEE-style:", limits(4, 3)["largest"])
```

```text
E5M2 largest: 57344.0
E4M3 largest, IEEE-style: 240.0
```

E4M3 doesn't follow IEEE. It gives up infinities and keeps only one NaN pattern, and reuses the freed bit patterns as ordinary numbers. That raises its largest value from the 240 above to 448. Even so, one range rarely fits every tensor in a network, so FP8 training multiplies tensors by scaling factors before converting them, often a separate factor for each tensor. In the paper's words, "the general idea is to choose a scaling factor such that the maximum magnitude in the tensor becomes close to the maximum representable magnitude in the corresponding format." It is loss scaling again, applied to every tensor rather than just the gradients.

## Going further

David Goldberg's "What Every Computer Scientist Should Know About Floating-Point Arithmetic" (1991) is still the best account of rounding, error and why floating-point arithmetic behaves as it does. "Mixed Precision Training" (Micikevicius et al.) is short and practical, and Section 3 covers each technique in this chapter. PyTorch's [automatic mixed precision documentation](https://docs.pytorch.org/docs/stable/amp.html) has the full lists of which operations run in which precision.

> [!TIP] What to remember
> - A float is a sign, an exponent and a fraction. Exponent bits buy range, fraction bits buy precision.
> - float16 has precision but little range: it overflows above 65,504 and flushes gradients below about $3 \times 10^{-8}$ to zero. bfloat16 has float32's range but less than three decimal digits of precision.
> - Results are rounded to the nearest representable value, ties to even. Additions lose any term smaller than half the gap at the larger operand's size.
> - Mixed precision means float32 master weights, 16-bit matrix multiplications with float32 accumulation, float32 reductions, softmax and losses, and, for float16, dynamic loss scaling.
> - Low-precision failures are mostly silent. Check results against a float32 run, the same way gradients are checked against finite differences.

The next chapter moves from number formats to hardware: why a GPU's speed is usually set by how fast it can move bytes, not how fast it can multiply.
