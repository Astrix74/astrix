---
title: Deep learning foundations
description: How a neural network actually learns, from the loss function to backpropagation to the tricks that make deep networks trainable at all.
order: 1
planned:
  - "Optimizers: momentum, Adam and AdamW"
  - Generalization, overfitting and regularization
  - Convolutions and the idea of inductive bias
---

Almost everything in modern AI, including the largest language models, comes down to one loop: make a prediction, measure how wrong it is, and nudge millions of numbers so the next prediction is a little less wrong. This series takes that loop apart.

It starts with what "learning" means as an optimization problem, works through backpropagation until it stops being magic, and ends with the three ideas (careful initialization, residual connections and normalization) that took networks from a handful of layers to hundreds.

**Who it is for.** You can write code and remember some calculus: derivatives and the chain rule. Linear algebra helps but is introduced where it is needed. No machine-learning background is assumed.

**How to read it.** In order. Each chapter has a short NumPy program you can run and change; the fastest way to understand a formula is to break the code that implements it.

When you finish, you should be able to derive the gradient of a small network by hand, write a working autograd engine, and explain why a 100-layer network needs more than a good learning rate. The next series, *Transformers from first principles*, builds on all three.
