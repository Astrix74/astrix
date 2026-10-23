---
title: 缩放定律：模型该多大，数据该多少
description: 损失随参数量、数据量和计算量按幂律下降。Kaplan 和 Chinchilla 如何把这一点变成分配训练预算的规则，为什么 Chinchilla 的其中一个拟合会给出错误的答案，以及为什么如今的模型都训练得远远越过了那个“最优点”。
date: 2026-10-22T15:13:00Z
tags: [transformers, scaling, math]
series: transformers
chapter: 4
references: [kaplan2020, hoffmann2022, besiroglu2024, touvron2023, sardana2023]
---

[第 2 章](/zh/writing/inside-the-transformer-block/)结尾给出了训练开销的公式：一个有 $N$ 个参数的模型在 $D$ 个 token 上训练，大约需要 $6ND$ 次浮点运算。这个公式把预算变成了一个约束，却没有变成一个决策。手上有 $10^{24}$ FLOPs 可花，你该用 1.7 万亿个 token 训练一个 1000 亿参数的模型，还是用 8 万亿个 token 训练一个 200 亿参数的模型？两者的开销大致相同。

在 2020 年之前，这个问题的答案大多是口口相传的经验。后来，两篇相隔两年的论文把它变成了算术，却得出了相反的结论。本章依次讲解这两篇论文、一次平息了分歧的复现，以及为什么人们实际训练的模型反正都不理会那个“最优”答案。

## 损失是幂律

Kaplan 等人（2020）训练了一大批语言模型，发现测试损失随每一种资源都按幂律平滑下降，“其中一些趋势跨越了超过七个数量级”。就模型规模而言，在数据和训练都充足时：

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad \alpha_N \approx 0.076, \quad N_c \approx 8.8 \times 10^{13}
$$

其中 $N$ 统计的是非嵌入参数，损失以每个 token 的 nat 数计。他们还报告了数据和计算量上与之对应的定律，指数分别为 $\alpha_D \approx 0.095$ 和 $\alpha_C^{\min} \approx 0.050$。幂律意味着，资源每乘上一个固定的倍数，损失就按一个固定的比例下降：

```python
alpha_N, N_c = 0.076, 8.8e13            # Kaplan et al. (2020), eq. 1.1: L(N) = (N_c / N) ** alpha_N

def kaplan_loss(N):
    return (N_c / N) ** alpha_N

for N in [1e8, 1e9, 1e10, 1e11]:
    print(f"N = {N:.0e} parameters: loss {kaplan_loss(N):.3f}")
print(f"each doubling of N multiplies the loss by {2 ** -alpha_N:.3f}")
print(f"each 10x multiplies it by {10 ** -alpha_N:.3f}")
```

```text
N = 1e+08 parameters: loss 2.830
N = 1e+09 parameters: loss 2.376
N = 1e+10 parameters: loss 1.994
N = 1e+11 parameters: loss 1.674
each doubling of N multiplies the loss by 0.949
each 10x multiplies it by 0.839
```

由此可以得出两点。进展很慢：模型规模每翻一倍，损失只减少约 5%。进展又是可预测的：这条直线可以外推，这正是值得去拟合它的原因。你可以先在小模型上试验一套方案，再预测它在一个你只负担得起一次的规模上会有什么表现。

### Kaplan 的处方：模型更大，token 更少

接着，Kaplan 等人研究了如何分配一笔固定的计算预算。他们的答案是把大部分预算投入模型规模：计算最优的模型规模按 $N \propto C^{0.73}$ 增长，而“数据需求”的增长则“非常缓慢，$D \sim C^{0.27}$”。他们的结论是，计算量的最佳用法是“训练非常大的模型，并在远未收敛时就停下来”。按照 Hoffmann 等人后来的概括，预算扩大到 10 倍时，模型应该扩大到 5.5 倍，而 token 数只需增加到 1.8 倍。

整个领域都照此行事。Hoffmann 等人列举了当时的一批大模型，其中 GPT-3（175B 参数）和 Gopher（280B）都各自在约 3000 亿个 token 上训练，“这与增加计算量时主要扩大模型规模的做法一致”。

## Chinchilla：两者等比例扩展

Hoffmann 等人（2022）用 400 多个模型重新研究了这个问题，模型规模从 7000 万到超过 160 亿参数不等，并用三种相互独立的方法来寻找计算最优的分配。记 $N_{\text{opt}} \propto C^{a}$、$D_{\text{opt}} \propto C^{b}$：

| 方法 | $a$（参数） | $b$（token） |
| --- | --- | --- |
| 1. 取各条训练曲线的最小值 | 0.50 | 0.50 |
| 2. IsoFLOP 曲线 | 0.49 | 0.51 |
| 3. 对损失做参数化拟合 | 0.46 | 0.54 |

他们的结论是：“模型规模和训练 token 数应当等比例扩展：模型规模每翻一倍，训练 token 数也应当翻一倍。”他们的表 3 把这一结论换算成具体数字，算下来大约是每个参数 20 个 token。例如，一个 670 亿参数的模型应该看 1.5 万亿个 token。

按这个标准，那些在 3000 亿个 token 上训练的模型都严重训练不足。Hoffmann 等人直接做了检验。他们用与 Gopher 相同的计算预算训练了 **Chinchilla**：700 亿参数，1.4 万亿个 token，规模只有 Gopher 的四分之一，数据量却接近它的五倍。Chinchilla“全面且显著地优于”Gopher、GPT-3 以及其他大得多的模型。

为什么 Kaplan 会得出不同的答案？Hoffmann 等人认为主要原因在于训练设置。Kaplan 等人“对所有模型都使用固定的训练 token 数和学习率调度”，于是，从一次长训练中途取出来检查的模型，评判它所用的损失，并不是它的学习率调度设计时要达到的那个目标。用 Hoffmann 等人的话说，使用这些中间损失“导致低估了用少于 130B token 的数据训练模型的效果，并最终促成了这样一个结论：随着计算预算增加，模型规模应当比训练数据规模增长得更快。”

## 最优分配的推导

方法 3 拟合了一个公式，描述有 $N$ 个参数、在 $D$ 个 token 上训练的模型的损失：

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

每一项都有含义。$E$ 是任何模型都无法消除的损失：文本本身的熵。$A / N^{\alpha}$ 是模型有限带来的代价，$B / D^{\beta}$ 则是数据有限带来的代价。Hoffmann 等人报告的值为 $E = 1.69$、$A = 406.4$、$B = 410.7$、$\alpha = 0.34$、$\beta = 0.28$。

要以最优方式花掉预算 $C = 6ND$，就代入 $D = C / 6N$，并令对 $N$ 的导数为零：

$$
\frac{\partial L}{\partial N} = -\alpha A N^{-\alpha - 1} + \beta B \left(\frac{C}{6}\right)^{-\beta} N^{\beta - 1} = 0
$$

解得

$$
N_{\text{opt}} = G \left(\frac{C}{6}\right)^{\frac{\beta}{\alpha + \beta}}, \qquad
D_{\text{opt}} = G^{-1} \left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha + \beta}}, \qquad
G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha + \beta}}
$$

指数给出 $a = \beta / (\alpha + \beta)$，而模型规模和数据谁该增长得更快，只取决于 $\alpha$ 和 $\beta$ 哪个更大。下面在 Chinchilla 的预算上应用这个公式，并对模型规模做一次暴力搜索作为检验：

```python
import numpy as np

fits = {   # L(N, D) = E + A / N**alpha + B / D**beta
    "printed": dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28),           # Hoffmann et al., eq. 10
    "source": dict(E=1.6934, A=406.4, B=410.7, alpha=0.3392, beta=0.2849),      # the same fit, unrounded
    "refit": dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658),    # Besiroglu et al.
}

def loss(fit, N, D):
    return fit["E"] + fit["A"] / N ** fit["alpha"] + fit["B"] / D ** fit["beta"]

def compute_optimal(fit, C):
    """Closed form: minimize loss(N, D) subject to C = 6 N D."""
    a, b = fit["alpha"], fit["beta"]
    G = (a * fit["A"] / (b * fit["B"])) ** (1 / (a + b))
    N = G * (C / 6) ** (b / (a + b))
    return N, C / (6 * N)

def brute_force(fit, C):
    """The same optimum by searching a fine grid of model sizes."""
    N = np.logspace(7, 14, 200_001)
    best = N[np.argmin(loss(fit, N, C / (6 * N)))]
    return best, C / (6 * best)

C = 5.76e23   # Gopher's training budget, the budget Chinchilla was trained with (Hoffmann et al., Table 3)
print("fit       N        D          tokens/param   grid search")
for name, fit in fits.items():
    N, D = compute_optimal(fit, C)
    N_grid, _ = brute_force(fit, C)
    print(f"{name:<8} {N / 1e9:5.1f}B   {D / 1e12:5.2f}T   {D / N:12.1f}   N = {N_grid / 1e9:.1f}B")
print(f"Chinchilla as trained: 70B on 1.4T tokens, {1.4e12 / 70e9:.0f} tokens per parameter")
```

```text
fit       N        D          tokens/param   grid search
printed   32.2B    2.98T           92.6   N = 32.2B
source    40.3B    2.38T           59.1   N = 40.3B
refit     72.2B    1.33T           18.4   N = 72.2B
Chinchilla as trained: 70B on 1.4T tokens, 20 tokens per parameter
```

闭式解与网格搜索的结果一致，说明代数推导没有问题。有问题的是答案。论文里印出来的那个拟合，推荐的模型规模不到 Chinchilla 的一半，训练数据却是它的两倍多：每个参数 93 个 token，而 Chinchilla 用的是 20 个。论文自己的拟合，反倒与它被用来论证的那个模型相矛盾。

## 一次复现

Besiroglu 等人（2024）注意到了这一点。他们从论文的图中重建出方法 3 背后的数据，并用同一个公式重新拟合。他们发现，已发表的估计值“与其前两种估计方法不一致，无法拟合提取出的数据，并且给出了窄得令人难以置信的置信区间”。

他们把问题追溯到两个原因，在上面的表里都能看到。

**四舍五入**。论文的 TeX 源文件里保存着更精确的值，例如 $\beta \approx 0.2849$，而不是 0.28。由于 $\beta$ 是作用在 $10^{11}$ 量级数字上的指数，这点差别很要紧：Besiroglu 等人估计，在这个量级上，它会让数据项产生约 13% 的偏差。在上表中，未经四舍五入的“source”取值，把最优点从每个参数 93 个 token 挪到了 59 个。

**提前停下的优化器**。即使不做四舍五入，这个拟合也比同一公式的最优拟合要差。正如原作者之一后来在一份说明中澄清的（这次复现引用了那份说明），拟合程序对各个样本的损失取了平均，而不是求和，而且优化器停得太早。这也解释了那些置信区间：它们之所以过窄，是因为参数几乎没有离开初始值。

重新拟合得到 $a = 0.5126$，在 Chinchilla 的预算下落在 720 亿参数和 1.3 万亿个 token，“大约每个参数 20 个 token”，与方法 1 和方法 2 一致，也与 Chinchilla 实际的训练方式一致。规模越大，差别越大：

```python
print("optimal tokens per parameter")
print("compute     " + "".join(f"{name:>10}" for name in fits))
for C in [1e21, 1e23, 1e25, 1e27]:
    ratios = [D / N for N, D in (compute_optimal(fit, C) for fit in fits.values())]
    print(f"{C:<10.0e}  " + "".join(f"{r:>10.1f}" for r in ratios))
```

```text
optimal tokens per parameter
compute        printed    source     refit
1e+21             50.1      34.0      21.6
1e+23             78.2      50.7      19.2
1e+25            122.1      75.7      17.1
1e+27            190.7     113.1      15.2
```

按论文印出的拟合，理想比例随计算量稳步攀升。而按重新拟合的结果，它在六个数量级的跨度上始终保持在 15 到 22 之间，这才是“两者等比例扩展”的含义。

这个教训与 Chinchilla 本身无关，它的主要结论站得住。教训在于怎样解读拟合出来的常数。拟合是对实验的总结，它可能错在实验本身并没有错的地方。当一篇论文对同一个量给出好几种估计时，先检查它们是否一致，再去相信其中任何一个。

## 训练越过最优点

“计算最优”回答的只是一个问题：在训练预算固定时，怎样得到最低的损失。但模型训练一次之后就要投入运行，可能要处理数万亿个 token。推理时每个 token 大约花费 $2N$ FLOPs，也就是第 2 章讲的前向传播，所以更小的模型在整个服役期内都更便宜。

这就是为什么 [LLaMA](https://arxiv.org/abs/2302.13971) 模型都训练得远远越过了 Chinchilla 比例，7B 模型用了 1 万亿个 token。论文把理由说得很直白：“尽管 Hoffmann 等人（2022）建议用 200B token 训练一个 10B 模型，我们发现 7B 模型的性能即使在 1T token 之后仍在继续提升”，而一个训练得更久的小模型“最终在推理时会更便宜”。Sardana 等人（2023）把这种权衡量化了，发现在预期需求很大时，模型应当“比 Chinchilla 最优更小、训练更久”。

借助拟合，这笔账很容易算。先定下想要的质量（这里取 Chinchilla 预算下计算最优模型的损失），再找出以最低总成本达到这一质量的模型，总成本即训练加上推理服务：

```python
fit = fits["refit"]
N0, D0 = compute_optimal(fit, 5.76e23)
target = loss(fit, N0, D0)              # the quality of the compute-optimal model

def cheapest(served_tokens):
    """Model size and training tokens that reach `target` loss at the lowest total cost."""
    N = np.logspace(9.5, 12, 200_001)
    gap = target - fit["E"] - fit["A"] / N ** fit["alpha"]          # what data must still buy
    N = N[gap > 0]
    D = (fit["B"] / gap[gap > 0]) ** (1 / fit["beta"])                # tokens that close the gap
    cost = 6 * N * D + 2 * N * served_tokens                         # training + inference FLOPs
    i = np.argmin(cost)
    return N[i], D[i], cost[i]

print(f"target loss {target:.4f}")
for served in [0, 1e12, 1e13, 1e14]:
    N, D, cost = cheapest(served)
    print(f"serving {served:.0e} tokens: N = {N / 1e9:5.1f}B, D = {D / 1e12:5.2f}T, "
          f"{D / N:6.1f} tokens/param, total {cost:.2e} FLOPs")
```

```text
target loss 1.9744
serving 0e+00 tokens: N =  72.2B, D =  1.33T,   18.4 tokens/param, total 5.76e+23 FLOPs
serving 1e+12 tokens: N =  57.0B, D =  1.72T,   30.2 tokens/param, total 7.03e+23 FLOPs
serving 1e+13 tokens: N =  34.1B, D =  3.73T,  109.4 tokens/param, total 1.44e+24 FLOPs
serving 1e+14 tokens: N =  20.9B, D = 13.07T,  624.5 tokens/param, total 5.83e+24 FLOPs
```

不考虑推理时，答案就是 Chinchilla 式的最优点。如果模型在整个生命周期内要服务十万亿个 token，那么同等质量下最便宜的模型只有一半大，训练数据却接近三倍，达到每个参数 109 个 token。到了一百万亿个 token，最佳比例超过 600。在这些情况下，训练成本无一例外都更高，但总成本更低。

这些数字还需要附带一句提醒。它们把拟合外推到了远远超出其拟合范围的比例上，而 Sardana 等人发现，用常规比例构建的拟合，在极端比例下会高估额外 token 带来的收益。方向可以视为可靠，具体数字只当粗略参考。

> [!TIP] 要点
> - 损失随参数量、数据量和计算量按幂律下降：资源每乘上固定的倍数，损失就按固定的比例下降。这让规模的效果变得可预测。
> - Kaplan 等人把预算的大头投入模型规模。Chinchilla 表明参数和 token 应当一起增长，大约每个参数 20 个 token。
> - 由 $L = E + A/N^{\alpha} + B/D^{\beta}$ 和 $C = 6ND$，几行推导就能得出计算最优的分配。指数决定了谁该增长得更快。
> - Chinchilla 论文中印出的参数化拟合与它自己的实验相矛盾。一次复现把原因追溯到四舍五入和一个过早停下的优化器。要拿论文中的其他证据来核对拟合出的常数。
> - 计算最优忽略了推理。当一个模型要服务大量 token 时，就该把它训练得更小、更久。

下一章转向推理本身：逐个 token 地生成文本，让推理成本变得可以承受的 KV cache，以及让推理变快的各种技巧。想提前了解它的内存开销怎么算，可以看笔记[《KV cache 的内存账》](/notes/kv-cache-arithmetic/)。
