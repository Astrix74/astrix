---
title: Transformers from first principles
description: The architecture behind modern language models, rebuilt one piece at a time, with the math worked out and the code small enough to read.
order: 3
prerequisites: [deep-learning-foundations]
planned:
  - "Inference: the KV cache, sampling and speculative decoding"
  - "From pretraining to assistants: instruction tuning, RLHF and DPO"
---

The transformer was introduced in 2017 as a better machine-translation model. Within a few years it had become the architecture behind nearly every large language model. The core of it is surprisingly small: a way for tokens to look at each other (attention), a per-token computation (the MLP), and a shared channel that carries information between layers (the residual stream).

This series rebuilds that core from first principles. Each chapter answers one question the original paper takes for granted. Why is attention scaled by $1/\sqrt{d_k}$? Where do a model's parameters and compute actually go? Why does a model that sees every token at once need to be told what order they came in?

**Who it is for.** Anyone who has finished *Deep learning foundations*, or already knows how gradient descent and backpropagation work. You will need matrix multiplication and a little probability.

**How to read it.** In order. The chapters lean on each other: chapter 2 uses the attention from chapter 1, and chapter 3 changes it.

When you finish, you should be able to implement a transformer block from memory, estimate a model's parameter count and training compute from its shape, and read new architecture papers without getting lost in the notation.
