---
title: "What a harness is: the agent loop"
description: An agent is a language model calling tools in a loop, and the harness is the program that runs that loop. A complete one fits in thirty lines and runs here without an API key.
date: 2026-10-22T15:20:00Z
tags: [agents, harness, llm]
series: agent-harnesses
chapter: 1
references: [yao2022, schick2023, anthropic2024agents]
---

A language model, on its own, does one thing: given text, it produces more text. It can't open a file, run a test or look anything up. Yet "agents" built on these models now read codebases, run commands, notice their own mistakes and fix them. None of that comes from the model alone. It comes from a program wrapped around the model that lets its text *do* things and shows it what happened. That program is the **harness**.

This chapter builds one. The core turns out to be small enough to read in a minute, which makes it easier to see what actually matters: the decisions around the loop.

## Agents and workflows

There is a useful distinction in Anthropic's "Building effective agents" (December 2024). **Workflows** "are systems where LLMs and tools are orchestrated through predefined code paths". The programmer decides the sequence of steps, and the model fills in each one. **Agents** are systems where "LLMs dynamically direct their own processes and tool usage". The model decides what to do next. In practice, the post says, agents "are typically just LLMs using tools based on environmental feedback in a loop."

That one sentence is the whole architecture. The rest of this series is about getting the details right.

The idea of interleaving thinking with acting comes from research a few years older. ReAct (Yao et al., 2022) had a model "generate both reasoning traces and task-specific actions in an interleaved manner". The reasoning helps the model "induce, track, and update action plans as well as handle exceptions", and the actions let it "interface with external sources, such as knowledge bases or environments". On two interactive benchmarks, ALFWorld and WebShop, this beat imitation- and reinforcement-learning baselines by 34 and 10 percentage points of absolute success rate. Toolformer (Schick et al., 2023) showed a model could learn on its own "which APIs to call, when to call them, what arguments to pass", from "nothing more than a handful of demonstrations for each API", for tools like a calculator, a search engine and a calendar.

Today, tool use is built into model APIs. You describe each tool, with its name, what it does and its arguments, and the model can reply with a structured *tool call* instead of plain text. Running the tool is not the model's job. It is the harness's.

## The loop

Here is a complete harness. The environment is a tiny in-memory file system, the tools are two ordinary Python functions, and `run_agent` is the loop. The model is a stand-in, a scripted function that reads the transcript and decides what to do next, so everything runs without an API key. Swapping in a real model changes only the line that calls `model(messages)`.

```python
import json

# The environment: a tiny in-memory file system.
FILES = {
    "notes/todo.md": "- [x] buy a domain\n- [ ] write chapter 2\n"
                     "- [ ] fix the RSS feed\n",
    "notes/reading.md": "- [ ] Attention Is All You Need\n- [x] The Bitter Lesson\n",
    "src/app.py": "print('hello')\n",
}

# Tools are ordinary functions.
def list_files(prefix=""):
    return sorted(path for path in FILES if path.startswith(prefix))

def read_file(path):
    if path not in FILES:
        raise FileNotFoundError(f"no such file: {path}")
    return FILES[path]

TOOLS = {"list_files": list_files, "read_file": read_file}

# The harness: call the model, run the tool it asks for, append the result, repeat.
def run_agent(model, task, max_steps=8):
    messages = [{"role": "user", "content": task}]
    for _ in range(max_steps):
        reply = model(messages)
        messages.append({"role": "assistant", **reply})
        if "tool" not in reply:                      # no tool call: the model is done
            return reply["content"], messages
        try:
            result = TOOLS[reply["tool"]](**reply["args"])
        except Exception as error:                   # errors go back to the model
            result = f"error: {error}"
        messages.append({"role": "tool", "content": json.dumps(result)})
    return "stopped: step budget used up", messages

# A stand-in for a language model: it reads the transcript and decides what to do next.
def stub_model(messages):
    results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
    calls = [m for m in messages if m["role"] == "assistant"]
    if not calls:
        return {"content": "The to-do list is probably notes/todo.txt.",
                "tool": "read_file", "args": {"path": "notes/todo.txt"}}
    listing = next((r for r in results if isinstance(r, list)), None)
    if listing is None:
        return {"content": "That file doesn't exist. I'll list the notes folder.",
                "tool": "list_files", "args": {"prefix": "notes/"}}
    read = {c["args"]["path"] for c in calls if c.get("tool") == "read_file"}
    unread = [path for path in listing if path not in read]
    if unread:
        return {"content": f"Reading {unread[0]}.",
                "tool": "read_file", "args": {"path": unread[0]}}
    texts = [r for r in results if isinstance(r, str) and not r.startswith("error")]
    open_items = sum(text.count("- [ ]") for text in texts)
    return {"content": f"You have {open_items} open items "
                       f"across {len(texts)} note files."}

answer, transcript = run_agent(stub_model, "How many open to-do items are in my notes?")
for m in transcript:
    args = ", ".join(map(repr, m.get("args", {}).values()))
    detail = f"  -> {m['tool']}({args})" if "tool" in m else ""
    print(f"{m['role']:>9}: {m['content'][:60]}{detail}")
```

```text
     user: How many open to-do items are in my notes?
assistant: The to-do list is probably notes/todo.txt.  -> read_file('notes/todo.txt')
     tool: "error: no such file: notes/todo.txt"
assistant: That file doesn't exist. I'll list the notes folder.  -> list_files('notes/')
     tool: ["notes/reading.md", "notes/todo.md"]
assistant: Reading notes/reading.md.  -> read_file('notes/reading.md')
     tool: "- [ ] Attention Is All You Need\n- [x] The Bitter Lesson\n"
assistant: Reading notes/todo.md.  -> read_file('notes/todo.md')
     tool: "- [x] buy a domain\n- [ ] write chapter 2\n- [ ] fix the RS
assistant: You have 3 open items across 2 note files.
```

Read the transcript as the model sees it. It guesses a file name, gets an error, changes approach, gathers what it needs and answers. That is "environmental feedback in a loop": every tool result becomes part of the input for the next decision. Of course, a scripted stub makes these choices from fixed rules, where a real model makes them from the whole transcript. The loop around it is identical.

## What the harness decides

The model chooses each next action. Everything else is the harness's job, and these choices shape an agent's behaviour as much as the model does.

**Errors are observations, not crashes.** The `try`/`except` is the most important line in the loop. A missing file, a failing command or a bad argument goes back to the model as a tool result, so it can recover, as it did above. A harness that raises instead ends the run at the first surprise.

**When to stop.** The loop ends when the model replies without a tool call. That is the model's judgment that it's done. The harness adds a hard limit as well, because a model can get stuck:

```python
# A model that never decides it is done.
def stuck_model(messages):
    return {"content": "Let me look again.", "tool": "list_files", "args": {}}

answer, transcript = run_agent(stuck_model, "Find the bug.")
print(answer, f"({len(transcript)} messages)")
```

```text
stopped: step budget used up (17 messages)
```

Real harnesses budget more than steps: time, tokens and cost.

**What the model sees.** Every tool result is appended to `messages`, and the whole list is sent to the model on every step. A tool that returns a 10,000-line file puts all of it into the next call, and the one after that. What goes into that list, and in what form, is the subject of chapter 3.

**Which tools exist, and what they look like.** The model can only act through the tools it is given, and it learns what each one does only from its name, description and arguments. The post that drew the workflow/agent line also argues for crafting this "agent-computer interface" as carefully as a human interface. That is chapter 2.

**What is allowed.** `read_file` is harmless; a tool that deletes files or sends email is not. A harness decides which actions need a person's approval, and it often runs tools in a sandbox. A later chapter covers permissions.

## Start simpler than an agent

An agent is not always the right design. The same post recommends "finding the simplest solution possible, and only increasing complexity when needed", which "might mean not building agentic systems at all." If the steps of a task are known in advance, a workflow is easier to test and to predict. Prompt chaining, routing, parallelization, orchestrator-workers and evaluator-optimizer are the patterns it lists. An agent earns its cost when the path really can't be known ahead of time, for example fixing a bug in an unfamiliar codebase.

The post now carries a note that "much of the tooling landscape described in this post has changed since December 2024". The specific frameworks have moved on. Its three principles have not: keep the design simple, make the agent's planning visible, and carefully craft and test the tool interface.

> [!TIP] What to remember
> - An agent is a model choosing tool calls in a loop. The harness is the program that runs the loop, executes the tools and feeds results back.
> - The core loop is small. The design decisions are around it: how errors are returned, when to stop, what goes into context, which tools exist, what needs permission.
> - Return tool errors to the model as results. Recovering from them is a large part of what makes an agent useful.
> - Always set a budget. Models can loop.
> - If you can write down the steps in advance, build a workflow, not an agent.

The stand-in model was handed two tools with obvious names. Real tools are messier, and the difference between a tool a model uses well and one it fumbles is mostly design. The [next chapter](/writing/tools-and-the-agent-computer-interface/) is about that interface.
