---
title: "Context engineering: what goes in the window"
description: An agent's context window is its entire working world, and it is resent on every step. Why more context is not free, what to load up front versus on demand, and how long-running agents keep their context small.
date: 2026-10-22T15:22:00Z
tags: [agents, harness, context]
series: agent-harnesses
chapter: 3
references: [anthropic2025context, liu2023, lewis2020, shinn2023]
---

In the loop from [chapter 1](/writing/what-a-harness-is/), every model call sends the entire `messages` list: the task, every tool call and every tool result so far. Real harnesses also send their instructions and tool definitions with every call. The model has no other memory. Whatever is in that list is everything it knows about the task, and whatever isn't, it doesn't know. The list is the model's context window, and over a long task, managing it becomes the central design problem of a harness.

Anthropic's engineering team calls this **context engineering** (September 2025): "the set of strategies for curating and maintaining the optimal set of tokens" during a model's work. Prompt engineering is about writing good instructions. Context engineering is about everything that ends up in the window, over the whole life of a task.

## More context is not free

Context windows are large, so why not keep everything? There are two reasons.

**Cost.** The whole context is processed on every step, so a growing transcript is paid for again and again. The simulation below gives a feel for the numbers. And as the [transformer series](/writing/attention-as-a-differentiable-lookup/) showed, attention's cost grows with the square of the sequence length. Prompt caching, offered by many APIs, makes resending an unchanged prefix cheaper, but the model still attends over all of it.

**Quality.** Models don't use everything in a long context equally well. "Lost in the Middle" (Liu et al., 2023) tested this with multi-document question answering and key-value retrieval. Performance was "often highest when relevant information occurs at the beginning or end of the input context", and it "significantly degrades when models must access relevant information in the middle of long contexts", "even for explicitly long-context models". Models have improved since 2023, but the 2025 guidance still treats context as a finite resource. As "the number of tokens in the context window increases, the model's ability to accurately recall" what is in it declines, an effect it calls *context rot*. And it describes models as having an "attention budget" that every token draws on. (For one simple mechanism in this direction, in a toy model: at a fixed sharpness, the softmax weight any single key can get shrinks as the context grows. See [When attention freezes](/papers/when-attention-freezes/).)

So the guiding principle is the opposite of "keep everything": find "the smallest possible set of high-signal tokens that maximize the likelihood of some desired outcome."

## Up front or just in time

The first decision is what the model sees before it starts. The classic answer is **retrieval-augmented generation** (Lewis et al., 2020). The motivation was that a model's "ability to access and precisely manipulate knowledge is still limited", and that "providing provenance for their decisions and updating their world knowledge remain open research problems". RAG pairs the model with "a dense vector index of Wikipedia", searches it for passages relevant to the input, and puts them in front of the model. The retrieval happens *before* the model starts working, chosen by similarity to the question.

Agents make a second option possible: let the model fetch what it needs, when it needs it. Agents built this way "maintain lightweight identifiers" (file paths, queries, links) and load the content only when it becomes relevant. A coding agent doesn't need the whole repository in context. It needs to know the repository exists, plus tools like `grep` to find the three files that matter.

Most practical harnesses mix the two. In the guide's description of Claude Code, project instruction files (`CLAUDE.md`) "are naively dropped into context up front", while "primitives like glob and grep allow it to navigate its environment and retrieve files just-in-time". A reasonable default is to put things up front when they are small and needed on almost every task, like conventions and instructions, and to load everything else on demand.

## Keeping long tasks small

A task that runs for hundreds of steps will outgrow any window. The guide describes three techniques for this.

**Compaction.** When the context nears its limit, summarize it and start over from the summary. Summarizing well is hard: dropping the one detail that matters three hundred steps later is the typical failure. There is a much safer, lighter-touch version that the guide singles out, *tool result clearing*. Once the model has acted on a tool result, the raw output rarely needs to stay. Replace it with a placeholder that says what it was. The model can always call the tool again.

Here is what that does over a 30-step session, where each step returns a 3,000-token tool result. (Token counts use the rough rule of four characters per token.)

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

Without clearing, the context grows by about 3,100 tokens a step, and the total processed grows with the *square* of the number of steps. With clearing, the context stays almost flat and the session sends about a quarter as many tokens. One trade-off: rewriting earlier messages changes the prefix, which can reduce the benefit of prompt caching. Clearing in batches rather than every step keeps the prefix stable for longer.

**Structured note-taking.** The agent "regularly writes notes persisted to memory outside of the context window", such as a progress file, a to-do list or a decisions log, and reads them back when needed. The notes survive compaction, and they are much smaller than the history they summarize. The research ancestor here is Reflexion (Shinn et al., 2023). Its agents don't update any weights; they reflect on task feedback in words and "maintain their own reflective text in an episodic memory buffer" for later attempts. That reached 91% pass@1 on the HumanEval coding benchmark, against 80% for GPT-4, then the state of the art.

**Sub-agents.** Hand a focused piece of work to a separate agent with "clean context windows", and have it return a condensed result. The main agent sees a page-long summary of a search, not the fifty files the sub-agent read to produce it. The cost is coordination. The sub-agent sees none of the main agent's context, so its task description has to carry everything it needs.

## A checklist

When an agent misbehaves, check what it can see before blaming the model:

- Is the information it needs in the window at all? Or was it never loaded, or already cleared?
- Is it buried in the middle of a long transcript, behind pages of stale tool output?
- Did a tool return far more than needed? That is a tool problem; see [chapter 2](/writing/tools-and-the-agent-computer-interface/).
- Is anything in the window contradicting the instructions? An old error message or an outdated plan?

> [!TIP] What to remember
> - Context engineering is managing everything in the window over the whole task, not just writing the prompt.
> - Every token costs on every step and competes for the model's attention. Aim for the smallest set of high-signal tokens.
> - Load small, always-needed material up front. Fetch everything else on demand through tools.
> - For long tasks: clear old tool results, keep notes outside the window, and give focused work to sub-agents with clean contexts.
> - When an agent fails, first check what it could see.

These three chapters cover the core of any harness: the loop, the tools and the context. The [series page](/series/agent-harnesses/) lists what comes next: evaluating agents, orchestrating sub-agents, and permissions and sandboxing.
