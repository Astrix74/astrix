---
title: "harness 是什么：智能体循环"
description: 智能体就是一个在循环中调用工具的语言模型，harness 则是运行这个循环的程序。一个完整的 harness 只要三十行代码，在这里不需要 API 密钥就能运行。
date: 2026-10-22T15:20:00Z
tags: [agents, harness, llm]
series: agent-harnesses
chapter: 1
references: [yao2022, schick2023, anthropic2024agents]
---

语言模型本身只做一件事：给它一段文本，它接着生成更多文本。它打不开文件，跑不了测试，也查不了任何资料。可是建立在这些模型之上的“智能体”，如今却能读代码库、执行命令、发现自己的错误并加以改正。这些能力没有一样是单靠模型得来的。它们来自包在模型外面的一个程序：这个程序让模型输出的文本真正去做事，再把做事的结果拿给模型看。这个程序就是 **harness**。

本章就来实现一个。它的核心小到一分钟就能读完，这反而让真正要紧的东西更容易看清：围绕这个循环所做的那些决定。

## 智能体与工作流

Anthropic 的“Building effective agents”（2024 年 12 月）做了一个很有用的区分。**工作流**是“通过预定义的代码路径来编排 LLM 和工具的系统”：步骤的先后由程序员决定，模型负责完成其中的每一步。**智能体**则是“由 LLM 动态主导自身流程和工具使用的系统”：下一步做什么，由模型自己决定。文章还说，在实践中，智能体“通常不过是在循环中根据环境反馈来使用工具的 LLM”。

这一句话就是全部的架构。本系列余下的内容，讲的都是怎样把细节做对。

把思考和行动交替进行，这个想法来自早几年的研究。ReAct（Yao 等人，2022）让模型“以交替的方式，既生成推理轨迹，也生成针对具体任务的动作”。推理帮助模型“归纳、跟踪和更新行动计划，并处理异常情况”，动作则让模型能够“与知识库或环境等外部来源交互”。在 ALFWorld 和 WebShop 这两个交互式基准测试上，这种做法胜过了模仿学习和强化学习的基线，绝对成功率分别高出 34 和 10 个百分点。Toolformer（Schick 等人，2023）则表明，模型可以自己学会“调用哪些 API、何时调用、传入什么参数”，所需的“不过是每个 API 寥寥几个示范”，涉及的工具包括计算器、搜索引擎和日历。

如今，工具使用已经内置在模型 API 中。你描述每一个工具：它叫什么、做什么、有哪些参数，模型就可以不回复纯文本，而是回复一个结构化的**工具调用**（tool call）。执行工具不是模型的事，而是 harness 的事。

## 循环

下面是一个完整的 harness。环境是一个放在内存里的小型文件系统，工具是两个普通的 Python 函数，`run_agent` 就是那个循环。模型用的是一个替身：一个按脚本行事的函数，它读取对话记录，决定下一步做什么，所以整个程序不需要 API 密钥就能运行。换成真正的模型，只需要改动调用 `model(messages)` 的那一行。

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

站在模型的角度读一遍这份对话记录：它猜了一个文件名，碰了一个错误，换了思路，收集到所需的信息，然后作答。这就是“在循环中根据环境反馈”：每一个工具结果，都会成为下一次决策的输入的一部分。当然，脚本化的替身是按固定规则做出这些选择的，而真正的模型依据的是整份对话记录。包在外面的循环则一模一样。

## 由 harness 决定的事

每一步做什么，由模型选择；其余的一切都是 harness 的工作。这些选择对智能体行为的塑造，不亚于模型本身。

**错误是观察，不是崩溃**。`try`/`except` 是整个循环里最重要的一行。文件不存在、命令执行失败、参数不对，都会作为工具结果返回给模型，让它有机会恢复过来，就像上面那样。如果 harness 直接抛出异常，第一次遇到意外，整个运行就结束了。

**何时停止**。模型的回复里不再带有工具调用时，循环就结束，这代表模型判断任务已经完成。harness 还会另加一个硬性上限，因为模型可能会卡住：

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

真实的 harness 要控制的预算不只是步数，还有时间、token 和费用。

**模型看到什么**。每个工具结果都会追加到 `messages` 里，而每一步都会把整个列表发给模型。一个工具如果返回一个 10,000 行的文件，这些内容会全部进入下一次调用，以及再下一次调用。这个列表里放什么、以什么形式放，是第 3 章的主题。

**有哪些工具，长什么样**。模型只能通过交给它的工具来行动，而每个工具是做什么的，它只能从工具的名字、描述和参数中得知。划出工作流与智能体界线的那篇文章还主张，要像打磨面向人的界面一样，用心打磨这个“智能体–计算机接口”。这是第 2 章的内容。

**允许做什么**。`read_file` 无害，删除文件或发送邮件的工具就不是了。harness 决定哪些操作需要人来批准，并且常常把工具放在沙箱里运行。权限问题留到后面的章节讨论。

## 先从比智能体更简单的方案做起

智能体并不总是正确的设计。同一篇文章建议“找到尽可能简单的方案，只在需要时才增加复杂度”，而这“可能意味着根本不去构建智能体系统”。如果任务的步骤事先就知道，工作流更容易测试，行为也更容易预测。文章列出的模式有提示词链、路由、并行化、编排者–工作者和评估者–优化者。只有当路径确实无法提前知道时，比如在一个不熟悉的代码库里修 bug，智能体才值得它的成本。

这篇文章现在附了一条说明：“文中描述的工具生态，很大一部分自 2024 年 12 月以来已经变了。”具体的框架已经更新换代，它的三条原则却没有过时：保持设计简单，让智能体的规划过程清晰可见，用心打磨并测试工具接口。

> [!TIP] 要点
> - 智能体就是一个在循环中选择工具调用的模型。harness 是运行这个循环的程序：它执行工具，再把结果反馈给模型。
> - 核心循环很小，设计上的决定都在它周围：错误怎样返回，何时停止，什么内容进入上下文，有哪些工具，哪些操作需要授权。
> - 把工具错误作为结果返回给模型。能从错误中恢复过来，是智能体之所以有用的一大原因。
> - 一定要设预算。模型可能会陷入死循环。
> - 如果能事先写下步骤，就构建工作流，而不是智能体。

替身模型拿到的是两个名字一目了然的工具。真实的工具要乱得多；一个工具模型用得顺手还是磕磕绊绊，差别主要在于设计。[下一章](/zh/writing/tools-and-the-agent-computer-interface/)讲的就是这个接口。
