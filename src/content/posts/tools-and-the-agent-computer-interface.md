---
title: Tools and the agent-computer interface
description: An agent knows its tools only by their names, descriptions and results. Why tool design moves agent performance as much as the model does, and what well-designed tools look like, with examples you can run.
date: 2026-10-22T15:21:00Z
tags: [agents, harness, tools]
series: agent-harnesses
chapter: 2
references: [jimenez2023, yang2024, anthropic2024agents, anthropic2025tools, mcp2024]
---

In [chapter 1](/writing/what-a-harness-is/) the stand-in model had two tools, `list_files` and `read_file`, and used them without trouble. Real tools are where agents succeed or fail. The model can't see your code, your API or your file system. It knows a tool only by its name, its description, its arguments and whatever text comes back when it calls it. For an agent, the tools *are* the interface to the world.

This chapter is about designing that interface: why it matters as much as it does, and the handful of principles that make a tool easy for a model to use correctly.

## Why the interface matters

SWE-bench (Jimenez et al., 2023) made the stakes concrete. It is a set of 2,294 software-engineering problems built from "real GitHub issues and corresponding pull requests" across 12 popular Python repositories. A model gets the codebase and the issue, and it is "tasked with editing the codebase to address the issue". A patch counts only if it applies and the tests chosen to check the fix all pass. When the paper came out, the best model it evaluated, Claude 2, resolved 1.96% of the issues.

A year later, SWE-agent (Yang et al., 2024) asked what happens if you design the tools for the agent rather than handing it a human's shell. Its starting point: "LM agents represent a new category of end users with their own needs and abilities", and they "would benefit from specially-built interfaces to the software they use". The authors built an **agent-computer interface** (ACI):
- a file viewer that shows at most 100 numbered lines at a time
- search commands that return a summary of results, not raw output
- an edit command that replaces a range of lines and runs a linter, discarding edits the linter flags as invalid and asking the agent to try again.

With it, their agent reached 12.5% on SWE-bench, far above the earlier non-interactive results. The comparison that isolates the interface is their ablation on SWE-bench Lite, a 300-task subset: with the same model, GPT-4 Turbo, the purpose-built interface gave a 64% relative increase in solved issues over a plain shell.

Anthropic's own SWE-bench work points the same way. In "Building effective agents" they write that they "actually spent more time optimizing our tools than the overall prompt", and they offer a rule of thumb: think about "how much effort goes into human-computer interfaces (HCI)", and invest "just as much effort in creating good agent-computer interfaces (ACI)."

## What the model sees

Here is what a tool looks like from the model's side. Most model APIs take a definition shaped roughly like this, a name, a description and a JSON Schema for the arguments:

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

The description does the work that documentation, autocomplete and experience do for a human programmer. Anthropic's guide to writing tools (Aizawa, 2025) suggests writing it the way you would explain the tool "to a new hire on your team". Spell out what the tool returns, when to use it, and what its arguments mean. Prefer unambiguous names: `user_id`, not `user`.

## Principles, with code

The guidance in these sources overlaps a great deal. Here are the parts that matter most, with a tiny implementation of two tools that follow them. The file is 303 lines of configuration, ending with a few settings that will matter in a moment:

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

### 1. Bound every output

Context is the agent's working memory, and everything a tool returns goes into it, on this step and every step after. A tool that can return an unbounded amount of text will, sooner or later. The guide recommends "pagination, range selection, filtering, and/or truncation with sensible default parameter values", and notes that "for Claude Code, we restrict tool responses to 25,000 tokens by default" (as of September 2025).

When output is cut off, *say so and say how to get more*. Silent truncation is worse than none, because the model will reason from a partial file as if it were whole:

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

Line numbers are there for the model's benefit, too. They give it a way to refer to locations precisely, the kind of affordance SWE-agent's file viewer was built around.

### 2. Make errors teach

An error message is the model's only clue about what went wrong. The tools guide asks for errors that "clearly communicate specific and actionable improvements, rather than opaque error codes or tracebacks". Compare a bare `KeyError: 'src/settings.py'` with:

```python
print(call(read_file, path="src/settings.py"))
```

```text
error: src/settings.py does not exist. Known files: src/config.py
```

The model's next step is obvious.

### 3. Make mistakes hard to make

"Building effective agents" calls this *poka-yoke*, from the Japanese manufacturing term for mistake-proofing: "Change the arguments so that it is harder to make mistakes." Their example: in their SWE-bench agent, the model "would make mistakes with tools using relative filepaths" after changing directories, so "we changed the tool to always require absolute filepaths".

`edit_file` follows the same idea. It replaces an exact piece of text, and it refuses to guess when that text is ambiguous:

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

A naive replace-all would have silently corrupted `TIMEOUT_MS` as well. Here the first call fails with instructions, and the second succeeds with exactly one change. The "not found" error has a second purpose: an edit based on a stale copy of the file fails instead of overwriting newer content.

### 4. Fewer, better tools

It's tempting to expose every endpoint of an API as its own tool. The guide warns that "more tools don't always lead to better outcomes", and that "tools can consolidate functionality, handling potentially multiple discrete operations (or API calls) under the hood". Every tool definition takes up context and adds another option for the model to weigh. One `schedule_meeting` tool that finds a free slot and books it is easier to use well than `list_users`, `list_events` and `create_event` used in the right order. When there are many tools, the guide also suggests *namespacing*, grouping related tools under common prefixes, so similar tools from different services stay distinguishable.

### 5. Return what is meaningful

Return what the model needs to reason about, in terms it understands. The guide reports that agents handle "natural language names, terms, or identifiers significantly more successfully than they do with cryptic identifiers", so return a person's name and a file's path, not a UUID and a MIME type, unless the next call needs them.

## Connecting tools: MCP

So far the tools have been functions in the harness's own process. The **Model Context Protocol**, announced by Anthropic in November 2024 as "a new standard for connecting AI assistants to the systems where data lives", standardizes the connection. Developers "expose their data through MCP servers", and AI applications act as MCP clients that can use any server. A tool written once can then be used from any harness that speaks the protocol.

MCP changes how tools are *delivered*, not how they should be *designed*. A badly described, unbounded tool is just as bad over a protocol as it is in-process.

## Test tools by watching them used

Tool design is empirical. A description that is perfectly clear to its author can be misread by a model in a consistent way, and you only find out by running the agent and reading transcripts: which tools it reaches for, where it passes wrong arguments, which errors it fails to recover from. "Building effective agents" puts it as one of its three principles: craft the interface "through thorough tool documentation and testing".

> [!TIP] What to remember
> - The model knows a tool only by its definition and its outputs. Write descriptions for a capable newcomer, and use unambiguous argument names.
> - SWE-agent showed that an interface designed for agents can matter as much as the model behind it.
> - Bound every output and say when it was truncated and how to get more.
> - Errors should tell the model what to do next.
> - Design tools so mistakes are hard: exact matches, absolute paths, refusing to guess.
> - Prefer a few tools that match real tasks over many thin wrappers. Test them by reading agent transcripts.

Every tool result in this chapter flowed straight into the model's context. Over a long task that context fills up, and what goes into it, what stays and what gets dropped becomes the main design problem. That is the [next chapter](/writing/context-engineering/).
