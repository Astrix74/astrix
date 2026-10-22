---
title: What would count as AGI?
description: Three serious attempts to define general intelligence (reward across all environments, efficiency at acquiring new skills, and levels of capability), what each one measures, and why I think the useful question is never a yes or no.
date: 2026-10-22T15:30:00Z
tags: [agi, essay, evaluation]
references: [legg2007, chollet2019, morris2023, chollet2024arc, yang2024]
---

> [!NOTE] An essay
> The definitions, quotes and numbers below come from the cited papers. The conclusions in the last part are my opinion, and they are labelled as such.

Arguments about whether some system "is AGI" tend to go in circles, and usually for the same reason: the two sides mean different things by the word. One person means "can do most jobs a person can do at a computer". Another means "can learn anything new as quickly as a person can". A third means something closer to "is smarter than us". Each of these is a reasonable thing to care about, and they are not the same thing.

So before asking whether we are close, it helps to look at what careful people have proposed the word should mean. Here are three proposals that have shaped the debate. Each one measures something different.

## 1. Reward across every environment

Shane Legg and Marcus Hutter surveyed the many informal definitions of intelligence from psychology and AI and distilled them into one sentence: "Intelligence measures an agent's ability to achieve goals in a wide range of environments" (2007). Then they made it precise. For an agent $\pi$, its **universal intelligence** is

$$
\Upsilon(\pi) = \sum_{\mu \in E} 2^{-K(\mu)}\, V_\mu^\pi
$$

The sum runs over every computable environment $\mu$. $V_\mu^\pi$ is how much reward the agent expects to earn in that environment. $K(\mu)$ is the environment's Kolmogorov complexity, the length of the shortest program that describes it. The weight $2^{-K(\mu)}$ is Occam's razor written as a formula: simple environments count for more than complicated ones, but every environment counts for something.

The definition is elegant, and it says something true. Intelligence is about *breadth*. An agent that is superb in one environment and helpless in all the others scores low. But it is a definition, not a test. The authors note that "the Kolmogorov complexity function K is not computable and can only be approximated", so $\Upsilon$ can't be computed for any real system. And it measures what an agent *achieves*, regardless of how much it had to be told or trained in advance.

## 2. Efficiency at acquiring new skills

François Chollet's "On the Measure of Intelligence" (2019) starts from exactly that last gap. The AI field, he argues, measures intelligence by comparing skill at specific tasks, and that "falls short of measuring intelligence, because skill is heavily modulated by prior knowledge and experience: unlimited priors or unlimited training data allow experimenters to 'buy' arbitrary levels of skills for a system, in a way that masks the system's own generalization power."

His alternative:

> The intelligence of a system is a measure of its skill-acquisition efficiency over a scope of tasks, with respect to priors, experience, and generalization difficulty.

On this view, a system that reaches expert skill after seeing a million examples is less intelligent than one that reaches the same skill from ten, even though both end up equally capable. What matters is how far a system can generalize from how little.

To measure it, Chollet built the **Abstraction and Reasoning Corpus** (ARC), later renamed ARC-AGI: small grid puzzles, each defined by only a few examples and each built to be new. The tasks rest on "an explicit set of priors designed to be as close as possible to innate human priors", like objects, counting and basic geometry, so that memorized knowledge doesn't help much. The ARC Prize 2024 technical report (Chollet et al.) describes the benchmark's state at the end of 2024: "As of December 5, 2024, the ARC-AGI benchmark is five years old and remains unbeaten." The prize target was 85%. During the 2024 competition the best score on the private evaluation set rose from 33% to 55.5%, with "deep learning-guided program synthesis and test-time training" among the techniques behind the gain.

## 3. Levels, not a line

A group at Google DeepMind (Morris et al., 2023) took a more practical approach. Instead of a single definition, they proposed six principles for any useful definition:

1. Focus on capabilities, not processes.
2. Focus on generality and performance.
3. Focus on cognitive and metacognitive, but not physical, tasks.
4. Focus on potential, not deployment.
5. Focus on ecological validity.
6. Focus on the path to AGI, not a single endpoint.

The last principle produces their main contribution, a grid. One axis is *generality*: narrow, meaning a specific task, or general, meaning a wide range of non-physical tasks. The other is *performance*, measured against people:

| Level | Performance | General version |
| --- | --- | --- |
| 1 Emerging | equal to or somewhat better than an unskilled human | Emerging AGI |
| 2 Competent | at least 50th percentile of skilled adults | Competent AGI |
| 3 Expert | at least 90th percentile of skilled adults | Expert AGI |
| 4 Exceptional | at least 99th percentile of skilled adults | Exceptional AGI |
| 5 Superhuman | outperforms 100% of humans | Artificial superintelligence |

In this scheme, AlphaFold is *Superhuman Narrow AI*. The paper placed 2023's frontier chatbots (ChatGPT, Bard, Llama 2, Gemini) at *Emerging AGI*, and listed Competent AGI and every level above it as "not yet achieved". Later versions of the paper renamed Level 4, previously "Virtuoso", to "Exceptional". The paper also separates capability from *autonomy*, meaning how much a system acts on its own versus as a tool. The two are different questions with different risks.

## What each one measures

Put side by side, the three definitions are not competitors so much as measurements of different things:

- Legg and Hutter measure **breadth of achievement**: how well you do, across how many kinds of situation.
- Chollet measures **efficiency of learning**: how much new skill you get from how little information.
- Morris et al. measure **level of capability against people**, across a range of tasks.

A system can score very differently on each. One trained on most of the text humans have written could have enormous breadth and reach high percentiles on many tasks while being unremarkable by Chollet's measure, because its experience is so large. That is precisely the case his paper warns about.

## Where I land

*This section is opinion.*

**"Is it AGI?" is the wrong question.** It asks for a yes or no about something that is several continuous quantities. I find the Levels framing the most useful for everyday conversation, because it forces the follow-ups: general at what level, on which tasks, compared with whom?

**Chollet's measure is the hardest to argue with, and the most neglected.** Breadth bought with vast amounts of data is real and useful. But "how much did it need to see?" is the question that separates a system that learns from one that remembers, and most benchmark headlines don't report it.

**Capability belongs to a system, not a model.** SWE-agent (Yang et al., 2024) reported that, with the same model, a purpose-built tool interface solved 64% more real software issues than a plain shell (a relative increase, on a 300-task subset of SWE-bench). That is the subject of the [agent harnesses series](/series/agent-harnesses/). Any claim about how general a system is should say what tools, scaffolding and attempts it was given. Morris et al.'s principle of "potential, not deployment" points the same way: what you measure depends on the setup you allow.

**Ecological validity matters more as systems get better.** Puzzles and exams were good proxies when models couldn't do much. For systems that can work on real tasks for hours, the more informative test is the real task itself: does the work get done, reliably, without someone checking every step?

None of this settles when, or whether, any particular threshold will be crossed. But it makes the argument productive. Say which definition you are using, and the disagreement usually becomes something you can measure.

> [!TIP] What to remember
> - Legg and Hutter define intelligence as achieving goals across a wide range of environments, weighted toward simple ones. Precise and elegant, but not computable.
> - Chollet defines it as skill-acquisition efficiency, given priors, experience and generalization difficulty. ARC tests that with few-shot puzzles; as of December 2024 it remained unbeaten, with a best private-set score of 55.5% against an 85% target.
> - Morris et al. replace the line with levels of performance (Emerging to Superhuman) crossed with generality, and treat autonomy separately.
> - They measure different things: breadth, learning efficiency and level against people. Ask which one a claim is about.
