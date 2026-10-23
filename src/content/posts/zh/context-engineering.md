---
title: "上下文工程：窗口里该放什么"
description: 智能体的上下文窗口就是它的整个工作世界，而且每一步都要重新发送一遍。为什么更多的上下文并非免费，哪些东西该预先放入、哪些该按需获取，以及长时间运行的智能体怎样让上下文保持精简。
date: 2026-10-22T15:22:00Z
tags: [agents, harness, context]
series: agent-harnesses
chapter: 3
references: [anthropic2025context, liu2023, lewis2020, shinn2023]
---

在[第 1 章](/zh/writing/what-a-harness-is/)的循环里，每次调用模型都会发送整个 `messages` 列表：任务，以及到目前为止的每一次工具调用和每一个工具结果。真实的 harness 每次调用时还会一并发送指令和工具定义。模型没有别的记忆。列表里有的，就是它对这个任务所知道的全部；列表里没有的，它就不知道。这个列表就是模型的上下文窗口。在长任务中，管理这个窗口会成为 harness 的核心设计问题。

Anthropic 的工程团队（2025 年 9 月）把这叫作**上下文工程**（context engineering）：在模型工作期间，“用来筛选和维护最优 token 集合的一系列策略”。提示词工程关心的是写出好的指令；上下文工程关心的则是在任务的整个生命周期里，最终进入窗口的一切。

## 更多上下文并非免费

上下文窗口很大，那为什么不把所有东西都留着？原因有两个。

**成本**。每一步都要处理整个上下文，所以不断变长的对话记录要一次又一次地付费。下面的模拟能让你对这些数字有个直观的感受。而且正如 [Transformer 系列](/zh/writing/attention-as-a-differentiable-lookup/)所讲的，注意力的开销随序列长度的平方增长。许多 API 都提供提示词缓存（prompt caching），能让重新发送一段不变的前缀便宜一些，但模型仍然要对全部内容做注意力计算。

**质量**。长上下文里的内容，模型并不能一视同仁地用好。“Lost in the Middle”（Liu 等人，2023）用多文档问答和键值检索检验了这一点。当“相关信息出现在输入上下文的开头或结尾时”，模型的表现“往往最好”；而当“模型必须获取长上下文中间的相关信息时”，表现会“显著下降”，“即使是明确面向长上下文的模型也是如此”。2023 年以来，模型已经有所进步，但 2025 年的这份指南仍然把上下文当作一种有限的资源。指南指出，随着“上下文窗口中 token 数量的增加，模型准确回忆”窗口内容的能力会下降，并把这种效应称为**上下文腐化**（context rot）。它还把模型描述为拥有一份“注意力预算”，每个 token 都要从中支取一部分。（在一个玩具模型里，有一个朝着这个方向的简单机制：锐度固定时，任何单个键所能得到的 softmax 权重，都会随上下文变长而缩小。见[《注意力何时冻结》](/zh/papers/when-attention-freezes/)。）

所以，指导原则恰好与“什么都留着”相反：找到“尽可能小的一组高信号 token，使某个期望结果出现的可能性最大”。

## 预先放入，还是即时获取

第一个决定是：模型开始工作之前能看到什么。经典的答案是**检索增强生成**（retrieval-augmented generation，RAG；Lewis 等人，2020）。它的动机在于，模型“获取和精确操纵知识的能力仍然有限”，而且“为自己的决策给出出处、更新自身的世界知识，仍是悬而未决的研究问题”。RAG 给模型配上“一个维基百科的稠密向量索引”，在其中搜索与输入相关的段落，再把它们摆到模型面前。检索早在模型开始工作之前就完成了，依据的是与问题的相似度。

智能体让第二种做法成为可能：让模型在需要的时候，自己去取需要的东西。这样构建的智能体“维护轻量的标识符”（文件路径、查询、链接），只在内容变得相关时才去加载。编程智能体不需要把整个仓库都放进上下文。它只需要知道仓库在那里，再配上 `grep` 这样的工具，用来找出真正要紧的那三个文件。

大多数实用的 harness 会把两种做法结合起来。指南在描述 Claude Code 时说，项目说明文件（`CLAUDE.md`）“会被原样预先放进上下文”，而“glob 和 grep 这样的基本操作让它能够在环境中浏览，并即时检索文件”。一个合理的默认做法是：体积小、几乎每个任务都要用到的东西，比如约定和指令，预先放进去；其余的一切都按需加载。

## 让长任务保持精简

一个要跑几百步的任务，迟早会撑破任何窗口。指南为此介绍了三种技术。

**压缩**（compaction）。上下文快到上限时，把它总结成一份摘要，再从摘要重新开始。总结好并不容易：典型的失败是丢掉了某个细节，而它恰恰在三百步之后变得要紧。指南特别提到了一种安全得多、也轻量得多的做法：**工具结果清除**（tool result clearing）。模型一旦根据某个工具结果采取了行动，原始输出就很少还需要留着。用一个说明它原本是什么的占位符替换掉它就行。模型随时可以再调用一次这个工具。

下面看看在一个 30 步的会话里这样做的效果，其中每一步都返回一个 3,000 token 的工具结果。（token 数按每个 token 约四个字符的粗略规则估算。）

```python
def tokens(text):
    return len(text) // 4  # rough rule of thumb for English text and code

def clear_old_results(messages, keep_last=3):
    """Replace all but the most recent tool results with a one-line placeholder."""
    tool_turns = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    old = set(tool_turns[:max(0, len(tool_turns) - keep_last)])
    return [{**m, "content": f"[cleared: {m['about']}]"} if i in old else m
            for i, m in enumerate(messages)]

def simulate(steps=30, clear=False):
    messages = [{"role": "system", "content": "x" * 8_000}]  # instructions and tools
    sent = []
    for step in range(1, steps + 1):
        messages.append({"role": "assistant", "content": "x" * 400})  # a tool call
        messages.append({"role": "tool", "content": "x" * 12_000,  # 3,000 tokens
                         "about": f"output of step {step}"})
        if clear:
            messages = clear_old_results(messages)
        # The next model call sends all of it.
        sent.append(sum(tokens(m["content"]) for m in messages))
    return sent

for clear in (False, True):
    sent = simulate(clear=clear)
    label = "clear old results" if clear else "keep everything"
    print(f"{label:<18} context at step 10: {sent[9]:>6,}  at step 30: {sent[29]:>6,}"
          f"  total sent: {sum(sent):>9,}")
```

```text
keep everything    context at step 10: 33,000  at step 30: 95,000  total sent: 1,501,500
clear old results  context at step 10: 12,042  at step 30: 14,180  total sent:   369,939
```

不清除时，上下文每一步增长约 3,100 个 token，处理的总量则随步数的**平方**增长。清除之后，上下文几乎保持不变，整个会话发送的 token 数只有原来的四分之一左右。有一点需要权衡：改写之前的消息会改变前缀，这可能削弱提示词缓存带来的好处。成批地清除，而不是每一步都清除，可以让前缀在更长时间里保持稳定。

**结构化笔记**。智能体“定期写下笔记，并把它们持久保存在上下文窗口之外的记忆中”，比如进度文件、待办清单或决策日志，需要时再读回来。这些笔记在压缩之后依然保留，而且比它们所概括的历史小得多。这种做法在研究上的前身是 Reflexion（Shinn 等人，2023）。它的智能体不更新任何权重，而是用文字反思任务反馈，并“把自己的反思文本保存在一个情景记忆缓冲区里”，供之后的尝试使用。这在 HumanEval 编程基准测试上达到了 91% 的 pass@1，而当时最先进的 GPT-4 是 80%。

**子智能体**。把一项聚焦的工作交给另一个拥有“干净上下文窗口”的智能体，让它返回一份精炼的结果。主智能体看到的是一页篇幅的搜索摘要，而不是子智能体为了写出这份摘要读过的五十个文件。代价是协调。子智能体看不到主智能体的任何上下文，所以交给它的任务描述必须包含它需要的一切。

## 一份检查清单

智能体表现不对劲时，先别怪模型，先检查它能看到什么：

- 它需要的信息到底在不在窗口里？是从来没有加载过，还是已经被清除了？
- 信息是不是埋在长长的对话记录中间，压在好几页过时的工具输出后面？
- 是不是某个工具返回的内容远远超出了需要？那是工具的问题，见[第 2 章](/zh/writing/tools-and-the-agent-computer-interface/)。
- 窗口里有没有什么内容和指令相矛盾？比如一条旧的错误信息，或者一份过时的计划？

> [!TIP] 要点
> - 上下文工程是在整个任务过程中管理窗口里的一切，而不只是写提示词。
> - 每个 token 在每一步都有成本，还要争夺模型的注意力。目标是最小的一组高信号 token。
> - 体积小、总要用到的材料预先放入，其余的一切都通过工具按需获取。
> - 对于长任务：清除旧的工具结果，把笔记记在窗口之外，把聚焦的工作交给拥有干净上下文的子智能体。
> - 智能体失败时，先检查它当时能看到什么。

这三章覆盖了任何 harness 的核心：循环、工具和上下文。[系列页面](/zh/series/agent-harnesses/)列出了接下来的内容：评估智能体、编排子智能体，以及权限与沙箱。
