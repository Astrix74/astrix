---
title: Under the hood
description: Build the machinery of deep learning yourself, from tensors and autograd up to a small language model, checking every piece against a real library.
order: 2
prerequisites: [deep-learning-foundations]
planned:
  - A small language model, end to end
---

You can use deep learning frameworks for years without knowing what a tensor looks like in memory or what `loss.backward()` actually does. That works until something breaks: a shape error that makes no sense, a gradient that is silently wrong, a training run that is ten times slower than it should be. This series builds the machinery yourself, so the frameworks stop being black boxes.

**How each chapter works.** Build one component from scratch in plain Python and NumPy, small enough to read in one sitting. Check it against a trusted implementation on random inputs, using NumPy, finite differences or PyTorch as the oracle. Then look at what real frameworks do differently, and why. Every chapter also points to the best outside resource on its topic.

**Who it is for.** Programmers who already use deep learning tools, including AI coding assistants, and want to see what is underneath. It pairs with *Deep learning foundations*, which explains the ideas this series implements, so read its first two chapters before starting here.

**How to read it.** In order. Each chapter's code is self-contained and runs on a laptop CPU in seconds, with no GPU and no framework to install. If you are deciding where to start overall, [the learning path](/writing/a-learning-path-for-programmers/) maps this site together with the best courses, books and codebases elsewhere.
