---
title: "Working with a coding agent: a field guide"
description: "Practical habits for getting reliable work out of a coding agent, each tied to how agents actually work: context, tools, feedback loops and review."
date: 2026-10-22T15:31:00Z
tags: [agents, guide, practice]
references: [anthropic2024agents, anthropic2025context]
---

The [agent harnesses series](/series/agent-harnesses/) looks at coding agents from the builder's side: the loop, the tools, the context window. This post looks from the other side, the developer handing an agent a task. Most of what goes wrong in that hand-off is predictable once you know what the agent can and can't see, so each habit below comes with the reason behind it.

These are working habits, not laws. Agents and their tools change quickly, so treat this as a snapshot from late 2026 and check it against your own results.

## Before the task

**Give it what a new teammate would need.** An agent starts every session knowing nothing about your project beyond what it can read. Most coding agents load a project instructions file into context up front. Claude Code, for example, puts `CLAUDE.md` files in the context at the start of a session ([context engineering](/writing/context-engineering/)). Use that file for what is true on every task and hard to discover: how to build and test, the conventions that aren't obvious from the code, the directories not to touch. Keep it short. Everything in it is resent on every step, and a long file of rarely relevant rules competes with the task for the model's attention.

**Say what done looks like.** "Fix the date bug" can be satisfied in many ways, some of them wrong. "Dates in the RSS feed should be RFC 822; add a test that fails before the fix and passes after" can be checked. An agent works by acting and reading feedback ([the agent loop](/writing/what-a-harness-is/)), so give it a check it can run itself. A command that proves success is worth more than a paragraph describing it.

A short brief covers most of it:

```text
Goal:        RSS feed dates are RFC 822, e.g. "Wed, 21 Oct 2026 00:00:00 GMT".
Where:       src/pages/rss.xml.ts; date helpers are in src/lib/posts.ts.
Done when:   `pnpm build` passes and dist/rss.xml has RFC 822 dates.
Constraints: no new dependencies; don't change the post URLs.
```

**Scope the task to fit one context.** A task that needs the agent to hold a whole subsystem in mind will push it toward the limits of its window, where recall gets worse. Split large changes into steps that each have their own check. Start a fresh session for unrelated work, rather than letting an old transcript fill up with stale output.

## During the task

**Ask for a plan first on anything non-trivial.** "Building effective agents" lists *transparency*, "explicitly showing the agent's planning steps", as one of its three principles. Reading a plan takes a minute. Reading a wrong 400-line diff takes much longer, and so does untangling it. A plan is also where you catch a misunderstood requirement before it becomes code.

**Let it run the checks, and read what it ran.** The point of an agent over a chat window is that it can run tests, read the failures and try again. Let it. But read the transcript, not just the final message. "All tests pass" is a claim. The command and its output are the evidence. Check that the right tests ran and that none were skipped or deleted to get there.

**Interrupt early.** If the agent heads somewhere wrong, stop it and redirect it then. Every step it takes in the wrong direction adds output to its context that the rest of the session has to work around.

**Reset rather than argue.** When a session has gone in circles, the context is full of failed attempts and contradicting error messages. Often the fastest fix is a new session with a better brief that includes what you learned: "the obvious fix in X doesn't work because Y".

## After the task

**Review the diff like a colleague's pull request.** Agents can be confidently wrong, in ways that look plausible. Common failure modes are worth scanning for:

- a test changed to match the code instead of the other way round
- an error swallowed instead of handled
- a new helper that duplicates one that already exists
- an edit outside the files the task needed
- a dependency added for something the standard library already does

**Verify claims about the outside world.** An agent's knowledge of library APIs, versions and documentation has a cutoff, and it can be wrong even before that. When a change depends on how an external API behaves, check the current documentation, or ask the agent to fetch it and cite what it found.

**Keep the decision with you.** The agent can find the bug, write the fix and run the tests. Whether the change is right for the project is still your call. So are pushing, deploying and anything that touches other people.

## Safety habits

**Decide what it may do without asking.** Reading files and running tests are low risk. Deleting files, rewriting git history, running migrations or sending anything over the network are not. Most coding agents let you set which commands need approval. Use that, and loosen it deliberately, not by clicking "allow" until the prompts stop.

**Keep secrets out of reach.** Anything the agent reads can end up in its context, and from there in a log, a commit message or an API request. Don't paste credentials into a session, and prefer environment setups where the agent doesn't need them.

**Treat fetched content as untrusted.** A web page or a file can contain text written to look like instructions. A well-built harness treats tool output as data, not commands. Make sure yours does, and be careful when you point an agent at content you don't control.

> [!TIP] What to remember
> - Brief the agent like a new teammate: where things are, what done looks like, and a command that proves it.
> - Keep tasks small enough to fit one clean context, and start fresh sessions for unrelated work.
> - Ask for a plan on non-trivial work. Interrupt early. Reset instead of arguing.
> - Read the commands and outputs, not just the summary. Review the diff as you would a colleague's.
> - Set permissions on purpose, keep secrets out, and treat fetched content as data.
