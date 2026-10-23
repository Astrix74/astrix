---
title: 工具与智能体–计算机接口
description: 智能体只能通过名字、描述和返回结果来认识它的工具。为什么工具设计对智能体表现的影响不亚于模型本身，设计良好的工具又是什么样子，附可以运行的示例。
date: 2026-10-22T15:21:00Z
tags: [agents, harness, tools]
series: agent-harnesses
chapter: 2
references: [jimenez2023, yang2024, anthropic2024agents, anthropic2025tools, mcp2024]
---

在[第 1 章](/zh/writing/what-a-harness-is/)里，替身模型有两个工具，`list_files` 和 `read_file`，用起来毫不费力。智能体的成败，取决于真实的工具。模型看不到你的代码、你的 API，也看不到你的文件系统。它对一个工具的全部了解，只有工具的名字、描述、参数，以及调用之后返回的那段文本。对智能体来说，工具本身就是它与世界之间的接口。

本章讲怎样设计这个接口：它为什么如此重要，以及让模型更容易正确使用工具的几条原则。

## 为什么接口如此重要

SWE-bench（Jimenez 等人，2023）让其中的利害变得具体可见。它收录了 2,294 个软件工程问题，取自 12 个流行 Python 仓库中“真实的 GitHub issue 及其对应的 pull request”。模型拿到代码库和 issue，“任务是修改代码库以解决这个 issue”。只有当补丁能够成功应用，并且用来检验修复的那些测试全部通过时，才算解决。论文发表时，它评测过的最好的模型 Claude 2 只解决了 1.96% 的 issue。

第二年，SWE-agent（Yang 等人，2024）提出了一个问题：如果不把人用的 shell 直接交给智能体，而是专门为它设计工具，会怎样？它的出发点是：“语言模型智能体是一类新的终端用户，有自己的需求和能力”，它们“会受益于为其所用软件专门打造的接口”。作者们构建了一个**智能体–计算机接口**（ACI）：
- 一个文件查看器，每次最多显示 100 行带行号的内容
- 几个搜索命令，返回结果摘要，而不是原始输出
- 一个编辑命令，替换指定范围内的行并运行 linter；被 linter 判为无效的编辑会被丢弃，并要求智能体重试。

有了它，他们的智能体在 SWE-bench 上达到了 12.5%，远远高于此前那些非交互式方法的结果。能把接口的作用单独剥离出来的，是他们在 SWE-bench Lite（一个包含 300 个任务的子集）上做的消融实验：用的是同一个模型 GPT-4 Turbo，专门设计的接口解决的 issue 数比普通 shell 相对提高了 64%。

Anthropic 自己在 SWE-bench 上的工作也指向同一个方向。他们在“Building effective agents”中写道，他们“花在优化工具上的时间，实际上比花在整体提示词上的还多”，并给出了一条经验法则：想一想“人们在人机接口（HCI）上投入了多少精力”，然后“在打造好的智能体–计算机接口（ACI）上投入同样多的精力”。

## 模型看到的是什么

下面是从模型那一侧看到的工具。大多数模型 API 接受的定义大致是这个样子：一个名字，一段描述，再加一份描述参数的 JSON Schema：

```python
read_file_tool = {
    "name": "read_file",
    "description": (
        "Read part of a text file, with line numbers. Returns at most `limit` "
        "lines starting at line `offset` (1-based). If the file is longer, the "
        "result ends with a note giving the offset to request next. Use this "
        "before editing a file, so your edit matches its current contents."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "Path from the repository root, e.g. src/app.py"},
            "offset": {"type": "integer",
                       "description": "First line to return. Default 1."},
            "limit": {"type": "integer",
                      "description": "Maximum lines to return. Default 100."},
        },
        "required": ["path"],
    },
}
```

对人类程序员来说，文档、自动补全和经验所做的事，在这里全靠这段描述来完成。Anthropic 关于编写工具的指南（Aizawa，2025）建议，写描述时就像在向“团队里的新同事”介绍这个工具。讲清楚工具返回什么、什么时候该用、各个参数是什么意思。名字要尽量没有歧义：用 `user_id`，而不是 `user`。

## 原则与代码

这几份资料给出的建议有大量重合。下面挑出最要紧的几条，并用一个很小的实现演示两个遵循这些原则的工具。示例文件是 303 行配置，结尾的几项设置马上就会派上用场：

```python
config_lines = [f"SETTING_{i} = {i}" for i in range(1, 300)]
config_lines += ["DEBUG = True", "TIMEOUT = 30", "RETRIES = 3", "TIMEOUT_MS = 30000"]
FILES = {"src/config.py": "\n".join(config_lines) + "\n"}

def read_file(path, offset=1, limit=100):
    """Return lines offset..offset+limit-1, numbered, with a note if more remain."""
    if path not in FILES:
        known = ", ".join(sorted(FILES))
        raise FileNotFoundError(f"{path} does not exist. Known files: {known}")
    lines = FILES[path].splitlines()
    window = lines[offset - 1 : offset - 1 + limit]
    body = "\n".join(f"{n:>5}  {line}" for n, line in enumerate(window, start=offset))
    end = offset + len(window) - 1
    if end < len(lines):
        body += (f"\n[lines {offset}-{end} of {len(lines)}; "
                 f"call again with offset={end + 1} for more]")
    return body

def edit_file(path, old, new):
    """Replace one exact occurrence of `old`. Fails loudly instead of guessing."""
    if path not in FILES:
        known = ", ".join(sorted(FILES))
        raise FileNotFoundError(f"{path} does not exist. Known files: {known}")
    text = FILES[path]
    count = text.count(old)
    if count == 0:
        raise ValueError(
            f"The text to replace was not found in {path}.\n"
            "Read the file again; it may have changed since you last saw it.")
    if count > 1:
        raise ValueError(
            f"The text to replace appears {count} times in {path}.\n"
            "Include more of the surrounding lines so it matches exactly once.")
    FILES[path] = text.replace(old, new, 1)
    return f"Edited {path}."

def call(tool, **args):
    """What the harness sends back: the result, or the error message as text."""
    try:
        return tool(**args)
    except Exception as error:
        return f"error: {error}"
```

### 1. 给每一次输出设上限

上下文是智能体的工作记忆，工具返回的一切都会进入其中，不只是这一步，之后的每一步也都带着它。一个能返回无限多文本的工具，迟早会真的这么做。指南建议“采用分页、范围选择、过滤和/或截断，并配上合理的默认参数值”，还提到“在 Claude Code 中，我们默认把工具响应限制在 25,000 个 token 以内”（截至 2025 年 9 月）。

输出被截断时，一定要明说，并告诉模型怎样获取剩下的部分。悄无声息的截断比不截断还糟，因为模型会把不完整的文件当成完整的来推理：

```python
print(call(read_file, path="src/config.py", offset=297, limit=4))
```

```text
  297  SETTING_297 = 297
  298  SETTING_298 = 298
  299  SETTING_299 = 299
  300  DEBUG = True
[lines 297-300 of 303; call again with offset=301 for more]
```

行号同样是为模型准备的。有了行号，模型就能精确地指称位置；SWE-agent 的文件查看器，正是围绕这类便利设计的。

### 2. 让错误信息教会模型

错误信息是模型了解哪里出了问题的唯一线索。工具指南要求错误信息“清楚地传达具体、可操作的改进方向，而不是晦涩的错误码或调用栈”。把一句光秃秃的 `KeyError: 'src/settings.py'` 和下面的输出比较一下：

```python
print(call(read_file, path="src/settings.py"))
```

```text
error: src/settings.py does not exist. Known files: src/config.py
```

模型下一步该做什么，一目了然。

### 3. 让模型难以犯错

“Building effective agents”把这叫作**防呆**（poka-yoke），这个词来自日本制造业，指的是防止出错的设计：“修改参数，让犯错变得更难。”他们举的例子是：在他们的 SWE-bench 智能体中，模型切换目录之后，“在使用相对文件路径的工具上会出错”，于是“我们把工具改成了始终要求使用绝对文件路径”。

`edit_file` 遵循的是同样的思路。它替换的是一段精确的文本，而当这段文本有歧义时，它拒绝猜测：

```python
print(call(edit_file, path="src/config.py", old="TIMEOUT", new="TIMEOUT_S"))
print(call(edit_file, path="src/config.py",
           old="TIMEOUT = 30\n", new="TIMEOUT_S = 30\n"))
print(call(read_file, path="src/config.py", offset=300, limit=10))
```

```text
error: The text to replace appears 2 times in src/config.py.
Include more of the surrounding lines so it matches exactly once.
Edited src/config.py.
  300  DEBUG = True
  301  TIMEOUT_S = 30
  302  RETRIES = 3
  303  TIMEOUT_MS = 30000
```

如果天真地全部替换，`TIMEOUT_MS` 也会被悄悄改坏。而在这里，第一次调用失败了，同时说明了该怎么做；第二次调用成功，并且恰好只改了一处。“找不到”这个错误还有第二个用途：基于文件旧副本所做的编辑会失败，而不会覆盖更新的内容。

### 4. 工具少而精

把 API 的每个端点都单独暴露成一个工具，是很诱人的做法。指南提醒说，“工具越多，结果不一定越好”，而且“工具可以整合功能，在底层处理可能多个独立的操作（或 API 调用）”。每个工具定义都要占用上下文，也给模型多添了一个需要权衡的选项。一个既能找空闲时段、又能完成预订的 `schedule_meeting` 工具，比按正确顺序依次使用 `list_users`、`list_events` 和 `create_event` 更容易用好。工具很多时，指南还建议使用**命名空间**（namespacing），也就是给相关的工具加上共同的前缀，这样来自不同服务的相似工具仍然区分得开。

### 5. 返回有意义的内容

返回模型推理所需的内容，并且用它能理解的说法。指南指出，智能体处理“自然语言的名称、术语或标识符时，比处理晦涩的标识符要成功得多”，所以应该返回人名和文件路径，而不是 UUID 和 MIME 类型，除非下一次调用需要用到它们。

## 连接工具：MCP

到目前为止，工具都是 harness 自身进程里的函数。**模型上下文协议**（Model Context Protocol）把这种连接标准化了。Anthropic 于 2024 年 11 月发布它时，称之为“将 AI 助手连接到数据所在系统的新标准”。开发者“通过 MCP 服务器暴露自己的数据”，AI 应用则作为 MCP 客户端，可以使用任何服务器。这样，一个工具只要写一次，就能在任何支持这个协议的 harness 中使用。

MCP 改变的是工具的交付方式，而不是工具应该怎样设计。一个描述糟糕、输出不设上限的工具，放在协议的另一端，和放在进程内一样糟糕。

## 测试工具：看模型怎么用

工具设计是一门实证的功夫。一段在作者看来再清楚不过的描述，可能会被模型以某种固定的方式误读，而你只有运行智能体、阅读对话记录才能发现：它会去用哪些工具，在哪里传错了参数，哪些错误它没能恢复过来。“Building effective agents”把这一点列为三条原则之一：“通过详尽的工具文档和测试”来打磨接口。

> [!TIP] 要点
> - 模型对一个工具的了解，只来自它的定义和输出。描述要写给一位能干的新人看，参数名要没有歧义。
> - SWE-agent 表明，专为智能体设计的接口，其重要性可以不亚于背后的模型。
> - 给每一次输出设上限；被截断时要说明，并告诉模型怎样获取更多。
> - 错误信息应当告诉模型下一步该做什么。
> - 把工具设计得让模型难以犯错：精确匹配、绝对路径、拒绝猜测。
> - 宁要少数几个对应真实任务的工具，也不要一大堆薄薄的封装。通过阅读智能体的对话记录来测试它们。

本章的每一个工具结果，都直接流进了模型的上下文。在一个长任务里，上下文会被填满，于是放进什么、留下什么、丢掉什么，就成了主要的设计问题。这是[下一章](/zh/writing/context-engineering/)的内容。
