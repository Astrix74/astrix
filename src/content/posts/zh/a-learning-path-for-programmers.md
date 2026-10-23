---
title: 如何搞懂深度学习的原理：写给程序员的学习路线
description: 从会用深度学习工具，到理解它们底层如何运作的一条路线：先学什么、后学什么，每个阶段最好的课程、书籍和代码库，以及怎样用 AI 编程智能体学习，又不跳过理解这一步。
date: 2026-10-23T08:25:00Z
tags: [guide, learning, deep-learning]
---

> [!NOTE] 核对于 2026 年 10 月
> 下面的每一项外部资源，都在 2026 年 10 月逐一到它自己的页面上核对过。课程和代码库会变，这条路线的顺序不会。

会用深度学习和理解深度学习是两种不同的能力。你可以微调模型、通过 API 调用模型，让 Claude Code 或 Codex 写一半代码来交付产品，却说不清 `loss.backward()` 做了什么、一次训练为什么发散，或者一次转置要付出什么代价。大多数时候这没什么问题。可一旦出了故障、某样东西必须跑得快，或者你读到一篇论文、需要判断该不该相信它，问题就来了。

这份指南是一条跨越这道鸿沟的路线，写给程序员。它假定你会编程，也用过深度学习工具；不假定你懂任何机器学习理论，数学也只需要一点点，而且可以边学边补。

## “理解实现”指的是什么

先把目标说清楚会有帮助。走完这条路线，你应该能够：

1. **亲手写出核心部分**：训练循环、小型网络的反向传播、注意力，以及一个分词器（tokenizer），每一样都对照真正的库检验过。
2. **预判框架会做什么**：一个操作是复制数据还是生成视图，一层要花多少次浮点运算、读写多少字节，内存都去了哪里。
3. **调试训练**：分辨出问题出在初始化、学习率，还是数据里有 bug。
4. **批判性地阅读研究**：在小规模上复现一篇论文的核心结论，并且能看出它的证据什么时候撑不起结论。

下面每个阶段都以一个检查点结束，各自对应其中一项能力。

## 方法：搭建、检验、破坏、对照

我最信任的学习方法分四步，[《深入底层》](/zh/series/under-the-hood/)的每一章走的都是这个循环。

1. **搭建**：亲手实现这一部分，代码短到坐下来一次就能读完。在 GPU 登场之前，Python 和 NumPy 就够用了。
2. **用参照（oracle）检验**：在随机输入上，把你的输出和一个可信的实现相比较：NumPy、PyTorch，或者用有限差分来检验梯度。正是这一步把“能跑”变成“是对的”。
3. **故意把它弄坏**：删掉一行、翻转一个符号，或者改一个超参数，但动手之前先预测会出现什么症状。看着少了某个部分之后哪里出错，你就明白了这个部分是做什么用的。
4. **和真东西对照**：读一读生产级的库是怎么做同一件事的，列出它多做了哪些：内存管理、kernel、各种边界情况。

光靠读和看，只能培养出熟悉感。熟悉感摸起来像理解，可一面对空白的编辑器就站不住了。把它变成知识的，是动手搭建和检验。

## 路线

### 第 1 阶段：损失、梯度与反向传播

一切的核心：一个要最小化的损失，用链式法则求出的梯度，以及一个自动计算梯度的引擎。

- **本站**：[《学习即优化》](/zh/writing/learning-as-optimization/)、[《反向传播就是反向模式自动微分》](/zh/writing/backpropagation-is-reverse-mode-differentiation/)，然后是[《张量：内存、形状与步长》](/zh/writing/tensors-memory-shapes-and-strides/)和[《为张量实现一个 autograd 引擎》](/zh/writing/autograd-for-tensors/)。
- **其他地方**：Andrej Karpathy 的 [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) 第一讲，它从零写出 [micrograd](https://github.com/karpathy/micrograd)，一个大约一百行的标量 autograd 引擎。斯坦福 [CS231n 的讲义](https://cs231n.github.io/)把反向传播讲成梯度在电路中的流动。
- **检查点**：凭记忆写出一个两层网络的前向和反向传播，并做梯度检验。

### 第 2 阶段：让训练跑起来

深层网络为什么难训练，以及大家都在用的那些解决办法：初始化、归一化、残差连接和更好的优化器。

- **本站**：[《让深度可训练》](/zh/writing/making-depth-trainable/)，关于 [Adam 与 AdamW](/zh/notes/adam-and-adamw/) 和 [LayerNorm 与 RMSNorm](/zh/notes/layernorm-vs-rmsnorm/) 的笔记，以及讲训练改用 16 位数字后会发生什么的[《数值》](/zh/writing/floating-point-and-mixed-precision/)。
- **其他地方**：Zero to Hero 里的 makemore 系列讲座，尤其是“Activations & Gradients, BatchNorm”和“Becoming a Backprop Ninja”，后者让你亲手写出整个网络的反向传播。fast.ai 的 [Practical Deep Learning for Coders](https://course.fast.ai/) 第 2 部分从零实现 Stable Diffusion，一路上还搭起了自己的训练框架，其中有专门讲初始化、归一化和混合精度的课。
- **检查点**：给你一个训练不起来的网络，判断原因是初始化、学习率还是 bug，并拿出证据。

### 第 3 阶段：Transformer

语言模型背后的架构，一块一块来。

- **本站**：[《从第一性原理理解 Transformer》](/zh/series/transformers/)的第 1 到第 3 章，以及论文[《注意力何时冻结》](/zh/papers/when-attention-freezes/)。
- **其他地方**：先看 3Blue1Brown 的[神经网络系列课](https://www.3blue1brown.com/topics/neural-networks)，包括“Attention in transformers, step-by-step”，建立直观印象。然后是 Karpathy 的“Let's build GPT”，以及 [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)，它逐行实现了原始论文。
- **检查点**：实现带因果掩码的多头注意力，并测试改动未来的 token 永远不会改变过去位置的输出。

### 第 4 阶段：端到端的语言模型

分词、小规模训练、采样，以及给定预算时模型应该多大。

- **本站**：[《缩放定律》](/zh/writing/scaling-laws/)。《深入底层》计划写一章端到端的内容。
- **其他地方**：Karpathy 的“Let's build the GPT Tokenizer”。然后是 [nanochat](https://github.com/karpathy/nanochat)，他目前的最小化代码库，涵盖“分词、预训练、微调、评估和推理”。它的前身 [nanoGPT](https://github.com/karpathy/nanoGPT) 如今自称“非常老旧，已被弃用”，但其中各约 300 行的模型文件和训练文件仍然清晰易读。Sebastian Raschka 的书《Build a Large Language Model (From Scratch)》（Manning 出版）一步步搭出一个 GPT，书中的全部代码都免费放在[一个公开仓库](https://github.com/rasbt/LLMs-from-scratch)里。
- **检查点**：用你自选的一段文本训练一个小型 Transformer，从中采样，并解释它的损失曲线。

### 第 5 阶段：内存、GPU 与 kernel

时间和内存究竟花在了哪里，以及为什么同样的数学既可以跑得快，也可以跑得慢。

- **本站**：[张量那一章](/zh/writing/tensors-memory-shapes-and-strides/)里讲内存布局的一节，[《Transformer 块的内部》](/zh/writing/inside-the-transformer-block/)里的 FLOP 计数，[关于 KV cache 的笔记](/zh/notes/kv-cache-arithmetic/)，以及[《GPU：为什么内存带宽决定速度》](/zh/writing/memory-bandwidth-and-the-roofline/)。
- **其他地方**：Edward Yang 的 [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/)，讲一个张量操作如何一路抵达它的 kernel。Karpathy 的 [llm.c](https://github.com/karpathy/llm.c) 用纯 C 和 CUDA 训练 GPT-2，其中 CPU 参考实现只有一个文件，大约一千行。[tinygrad](https://github.com/tinygrad/tinygrad) 用它自己的话说，是“介于 PyTorch 和 karpathy/micrograd 之间的东西”：一个能用的框架，比 PyTorch 小得多。
- **检查点**：在运行之前估算一步训练的内存和计算量，然后实际测量。

### 第 6 阶段：阅读研究

- **本站**：[论文](/zh/papers/)、[阅读清单](/zh/reading/)，以及[缩放定律那一章](/zh/writing/scaling-laws/)里的复现部分，它示范了怎样拿一篇论文自己的证据去检验它。
- **检查点**：在小规模上复现某篇论文里的一张图或一张表。

### 数学：用到再学

你需要导数和链式法则、矩阵乘法，以及基础的概率论。每一样都等到某一章需要时再学，不必一开始全部学完。3Blue1Brown 的课程提供直观理解。Terence Parr 和 Jeremy Howard 的 [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528) 讲矩阵求导，只假定你学过大一的微积分。Goodfellow、Bengio 和 Courville 合著的 [Deep Learning](https://www.deeplearningbook.org/)（中译本《深度学习》）有免费的在线版，其余内容可以拿它当参考书。

## 如果只上一门课

有两门课单独就能覆盖这条路线的大部分内容，而且都是免费的。

**[Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)** 由 Tatsunori Hashimoto 和 Percy Liang 讲授，自称是“一门非常注重实现的课”。第一次作业里，你要实现一个 Transformer 语言模型的“各个组件（分词器、模型架构、优化器）”，并训练出一个模型。第二次作业里，你要对它做性能剖析，并用 Triton 写出你自己的 FlashAttention 2 kernel。课程录像是公开的。

**Neural Networks: Zero to Hero** 是更平缓的起点：“从零开始，用代码构建神经网络”，从 micrograd 一直讲到 GPT 和它的分词器。

## 书籍与参考资料一览

| 资源 | 是什么 | 最适合 |
| --- | --- | --- |
| [Dive into Deep Learning](https://d2l.ai/) | 免费的交互式图书（中文版《动手学深度学习》），代码可运行，有 PyTorch、JAX 等版本 | 把每个想法都跑一遍来学习 |
| [Understanding Deep Learning](https://udlbook.github.io/udlbook/) | Simon Prince 著，MIT Press 2023 年出版，PDF 免费 | 讲解清楚，插图出色 |
| [The Little Book of Deep Learning](https://fleuret.org/francois/lbdl.html) | François Fleuret 著，一本为手机排版的免费小书 | 随身携带的简明概览 |
| [Deep Learning](https://www.deeplearningbook.org/) | Goodfellow、Bengio 和 Courville 合著，可在线免费阅读 | 经典参考书 |
| [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) | 逐行实现的原始 Transformer 论文 | 看一篇论文如何变成代码 |
| [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/) | Edward Yang 关于 PyTorch 设计的长文 | 了解 `torch.Tensor` 底下发生了什么 |

## 用 Claude Code 和 Codex 学习

这一节是建议，依据的是通常行之有效的做法。AI 编程智能体既能让这条路线快得多，也可能悄悄把它掏空。区别在于你把什么交给它们。和智能体协作的一般习惯，见[这份实战指南](/zh/writing/working-with-a-coding-agent/)。专就学习而言：

- **核心自己写，脚手架交出去**：让智能体去写数据加载、绘图和测试框架，前向和反向传播则自己动手敲。把它们交出去，你省掉的那份理解，恰恰是你来这里要得到的东西。
- **要参照，不要答案**：“写一个测试，在随机输入上把我的 softmax 和 NumPy 的结果相比较，包括非常大的 logits。”智能体擅长写测试框架，而一个失败的测试比一个能用的解答教给你的更多。
- **让它出练习**：“给我三个可以埋进这段反向传播的隐蔽 bug，以及每个 bug 会引起的症状。”把它们埋进去，然后不看答案把它们找出来。
- **让它追踪真实的代码，然后自己去读**：问它某件事发生在一个库源码的什么地方，然后打开它点名的那些文件。模型谈起库的内部实现和版本时，可能错得理直气壮。源码才是权威。
- **把它当审稿人**：写完一段代码后，请它按某个具体标准来审查，比如数值稳定性、形状处理或边界情况，再把每一条确有其事的发现变成一个测试。
- **让它考你**：每次开始学习时，让它就你上次学的内容出五道题。主动回忆比重读更有效。

## 节奏

这里没有时间表，只有检查点。当你不查任何资料就能通过某个阶段的检查点时，就进入下一阶段。不要更早，也不必拖得太久：后面的阶段还会再次用到前面的内容。

> [!TIP] 要点
> - 瞄准四项能力：亲手写出核心部分、预判框架会做什么、调试训练、批判性地阅读研究。
> - 通过搭建、用参照检验、故意破坏、与真实实现对照来学习。
> - 按顺序走这条路线：反向传播、训练、Transformer、端到端模型、系统、研究。用检查点来决定何时继续前进。
> - 如果只上一门课，就选 CS336 或 Zero to Hero，并把它们要求的东西全部亲手做出来。
> - 把编程智能体用在脚手架、测试和练习上，而不是用在你正想弄懂的那些部分。
