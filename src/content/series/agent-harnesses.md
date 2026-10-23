---
title: Agent harnesses
description: "The engineering around a language model that turns it into an agent: the loop, the tools, and the context it works from."
order: 4
prerequisites: [transformers]
planned:
  - Evaluating agents
  - Sub-agents and orchestration
  - Permissions, sandboxes and safety
---

A language model on its own maps text to more text. An *agent* reads a task, decides what to do, does it through tools, looks at what happened, and keeps going until the job is done. The difference is not in the model's weights; it is in the program wrapped around it. That program is the **harness**: the loop that calls the model, the tools it is allowed to use, and the decisions about what goes into its context window at every step.

Coding agents are the clearest example. The same model can be a chat assistant or an agent that fixes a failing test across a dozen files; what changes is the harness. As models improve, more of the difference between a useful agent and a frustrating one comes from harness design.

This series covers the parts every harness has: the agent loop, the interface between the agent and its tools, and context engineering, meaning what the model sees and what it doesn't.

**Who it is for.** Developers who build with language models or want to understand the agents they use. *Transformers from first principles* explains why context length and token order matter so much here, but you can start with this series if you already know what a context window is.

**How to read it.** In order. The examples are small Python programs that run without an API key, using a stand-in model, so you can see every step of the loop.
