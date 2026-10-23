---
title: "How to learn how deep learning works: a path for programmers"
description: "A route from using deep learning tools to understanding how they work underneath: what to learn in what order, the best courses, books and codebases for each stage, and how to use AI coding agents without skipping the understanding."
date: 2026-10-23T08:25:00Z
tags: [guide, learning, deep-learning]
---

> [!NOTE] Checked in October 2026
> Every outside resource below was checked on its own page in October 2026. Courses and repositories change. The order of the route won't.

Using deep learning and understanding it are different skills. You can fine-tune models, call them through APIs and ship products with Claude Code or Codex writing half the code, without being able to say what `loss.backward()` does, why a training run diverged, or what a transpose costs. Most of the time that's fine. It stops being fine when something breaks, when something has to be fast, or when you read a paper and need to decide whether to believe it.

This guide is a route across that gap, written for programmers. It assumes you can program and have used deep learning tools. It assumes no machine learning theory, and only a little math, which you can pick up as you go.

## What "understanding the implementation" means

It helps to name the target. By the end of this route, you should be able to:

1. **Write the core yourself.** A training loop, backpropagation for a small network, attention, and a tokenizer, each checked against a real library.
2. **Predict what the framework does.** Whether an operation copies or makes a view, how many floating-point operations and bytes a layer costs, where the memory goes.
3. **Debug training.** Tell an initialization problem from a learning-rate problem from a data bug.
4. **Read research critically.** Reproduce a paper's central claim at small scale, and notice when its evidence doesn't support it.

Each stage below ends with a checkpoint for one of these.

## The method: build, check, break, compare

The way to learn this material that I trust most has four steps, and it is the loop every chapter of [Under the hood](/series/under-the-hood/) follows.

1. **Build** the piece yourself, small enough to read in one sitting. Python and NumPy are enough until GPUs enter the picture.
2. **Check it against an oracle.** Compare your output with a trusted implementation on random inputs: NumPy, PyTorch, or finite differences for gradients. This is what turns "it runs" into "it's right".
3. **Break it on purpose.** Delete a line, flip a sign or change a hyperparameter, but first predict the symptom. You learn what each part is for by watching what fails without it.
4. **Compare with the real thing.** Read how a production library does the same job, and list what it adds: memory management, kernels, edge cases.

Reading and watching alone builds familiarity, which feels like understanding but doesn't survive a blank editor. The building and checking is what turns it into knowledge.

## The route

### Stage 1: losses, gradients and backpropagation

The core of everything: a loss to minimize, gradients from the chain rule, and an engine that computes them automatically.

- **On this site:** [Learning as optimization](/writing/learning-as-optimization/), [Backpropagation is reverse-mode differentiation](/writing/backpropagation-is-reverse-mode-differentiation/), then [Tensors: memory, shapes and strides](/writing/tensors-memory-shapes-and-strides/) and [An autograd engine for tensors](/writing/autograd-for-tensors/).
- **Elsewhere:** the first lecture of Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), which builds [micrograd](https://github.com/karpathy/micrograd), a scalar autograd engine of about a hundred lines. Stanford's [CS231n notes](https://cs231n.github.io/) explain backpropagation as gradients flowing through a circuit.
- **Checkpoint:** write the forward and backward pass of a two-layer network from memory, and gradient-check it.

### Stage 2: making training work

Why deep networks are hard to train, and the fixes everyone uses: initialization, normalization, residual connections and better optimizers.

- **On this site:** [Making depth trainable](/writing/making-depth-trainable/), and the notes on [Adam and AdamW](/notes/adam-and-adamw/) and [LayerNorm vs RMSNorm](/notes/layernorm-vs-rmsnorm/).
- **Elsewhere:** the makemore lectures in Zero to Hero, especially "Activations & Gradients, BatchNorm" and "Becoming a Backprop Ninja", which has you write the backward pass of a whole network by hand. Part 2 of fast.ai's [Practical Deep Learning for Coders](https://course.fast.ai/) implements Stable Diffusion from scratch, and along the way builds its own training framework, with lessons on initialization, normalization and mixed precision.
- **Checkpoint:** given a network that won't train, work out whether the cause is initialization, the learning rate or a bug, and show the evidence.

### Stage 3: transformers

The architecture behind language models, piece by piece.

- **On this site:** [Transformers from first principles](/series/transformers/), chapters 1 to 3, and the paper [When attention freezes](/papers/when-attention-freezes/).
- **Elsewhere:** 3Blue1Brown's [neural network lessons](https://www.3blue1brown.com/topics/neural-networks), including "Attention in transformers, step-by-step", for visual intuition first. Then Karpathy's "Let's build GPT", and [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/), a line-by-line implementation of the original paper.
- **Checkpoint:** implement causal multi-head attention, and test that changing future tokens never changes past outputs.

### Stage 4: a language model, end to end

Tokenization, training at small scale, sampling, and how big a model should be for a budget.

- **On this site:** [Scaling laws](/writing/scaling-laws/). An end-to-end chapter is planned for Under the hood.
- **Elsewhere:** Karpathy's "Let's build the GPT Tokenizer". Then [nanochat](https://github.com/karpathy/nanochat), his current minimal repository, which covers "tokenization, pretraining, finetuning, evaluation, and inference". Its predecessor [nanoGPT](https://github.com/karpathy/nanoGPT) now calls itself "very old and deprecated", but its roughly 300-line model and training files are still a clear read. Sebastian Raschka's book *Build a Large Language Model (From Scratch)* (Manning) builds a GPT step by step, with all its code free in [a public repository](https://github.com/rasbt/LLMs-from-scratch).
- **Checkpoint:** train a small transformer on a text of your choice, sample from it, and explain its loss curve.

### Stage 5: memory, GPUs and kernels

Where the time and memory actually go, and why the same math can run fast or slow.

- **On this site:** the memory layout section of the [tensors chapter](/writing/tensors-memory-shapes-and-strides/), the FLOP counts in [Inside the transformer block](/writing/inside-the-transformer-block/), and the [KV-cache note](/notes/kv-cache-arithmetic/). A chapter on GPUs is planned for Under the hood.
- **Elsewhere:** Edward Yang's [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/), on how a tensor operation reaches its kernel. Karpathy's [llm.c](https://github.com/karpathy/llm.c) trains GPT-2 in plain C and CUDA, with a CPU reference of about a thousand lines in one file. [tinygrad](https://github.com/tinygrad/tinygrad), in its own words, is "something between PyTorch and karpathy/micrograd": a working framework, much smaller than PyTorch.
- **Checkpoint:** estimate the memory and compute of one training step before running it, then measure it.

### Stage 6: reading research

- **On this site:** the [papers](/papers/), the [reading list](/reading/), and the replication section of the [scaling laws chapter](/writing/scaling-laws/), as an example of checking a paper against its own evidence.
- **Checkpoint:** reproduce one figure or table from a paper at small scale.

### The math, as needed

You need derivatives and the chain rule, matrix multiplication, and basic probability. Learn each one when a chapter needs it, not all up front. 3Blue1Brown's lessons give the intuition. Terence Parr and Jeremy Howard's [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528) covers matrix derivatives, assuming only first-year calculus. The free online edition of [Deep Learning](https://www.deeplearningbook.org/) by Goodfellow, Bengio and Courville is a reference for the rest.

## If you want one course

Two courses cover most of the route on their own, and both are free.

**[Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)**, taught by Tatsunori Hashimoto and Percy Liang, describes itself as "a very implementation-heavy class". In its first assignment you implement "the components (tokenizer, model architecture, optimizer)" of a transformer language model and train one. In the second you profile it and write your own FlashAttention 2 kernel in Triton. Lecture recordings are public.

**Neural Networks: Zero to Hero** is the gentler start: "building neural networks, from scratch, in code", from micrograd to GPT and its tokenizer.

## Books and references at a glance

| Resource | What it is | Best for |
| --- | --- | --- |
| [Dive into Deep Learning](https://d2l.ai/) | Free interactive book with runnable code in PyTorch, JAX and others | Learning by running every idea |
| [Understanding Deep Learning](https://udlbook.github.io/udlbook/) | Simon Prince, MIT Press 2023, free PDF | Clear explanations with excellent figures |
| [The Little Book of Deep Learning](https://fleuret.org/francois/lbdl.html) | François Fleuret, a short free book formatted for phones | A compact overview to carry around |
| [Deep Learning](https://www.deeplearningbook.org/) | Goodfellow, Bengio and Courville, free online | The classic reference |
| [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) | The original transformer paper, implemented line by line | Seeing a paper become code |
| [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/) | Edward Yang's long essay on PyTorch's design | What happens below `torch.Tensor` |

## Learning with Claude Code and Codex

*This section is advice, from what tends to work.* AI coding agents can make this route much faster or quietly hollow it out. The difference is what you hand them. For general working habits with agents, see [the field guide](/writing/working-with-a-coding-agent/). For learning specifically:

- **Write the core yourself; delegate the scaffolding.** Let the agent write data loading, plotting and test harnesses. Type the forward and backward passes yourself. The understanding you'd skip by delegating them is the thing you're here for.
- **Ask for the oracle, not the answer.** "Write a test that compares my softmax with NumPy's on random inputs, including very large logits." Agents are good at test harnesses, and a test that fails teaches you more than a solution that works.
- **Ask for exercises.** "Give me three subtle bugs to plant in this backward pass, and the symptom each one causes." Plant them, then find them without looking.
- **Have it trace real code, then read the code.** Ask where in a library's source something happens, then open the files it names. Models can be confidently wrong about library internals and versions. The source is the authority.
- **Use it as a reviewer.** After you write something, ask for a review against a specific standard, such as numerical stability, shape handling or edge cases, and turn each real finding into a test.
- **Make it quiz you.** At the start of a session, ask for five questions on what you studied last time. Retrieval works better than rereading.

## Pace

There is no timetable here, only checkpoints. Move on from a stage when you can pass its checkpoint without looking anything up. Not before, and not much after: the later stages will exercise the earlier ones again.

> [!TIP] What to remember
> - Aim for four abilities: write the core yourself, predict what the framework does, debug training, and read research critically.
> - Learn by building, checking against an oracle, breaking things on purpose, and comparing with real implementations.
> - Follow the route in order: backpropagation, training, transformers, an end-to-end model, systems, research. Use the checkpoints to decide when to move on.
> - If you take one course, take CS336 or Zero to Hero, and build everything they ask you to.
> - Use coding agents for scaffolding, tests and exercises, not for the parts you are trying to understand.
