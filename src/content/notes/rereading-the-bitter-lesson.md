---
title: Rereading the Bitter Lesson
date: 2026-10-22T15:44:00Z
references: [sutton2019]
---

Rich Sutton's [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) is a one-page essay from March 2019. It opens: "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin."

The evidence is a pattern Sutton traces through computer chess, computer Go, speech recognition and computer vision. Researchers build human knowledge into their systems. That "always helps in the short term, and is personally satisfying to the researcher", but "in the long run it plateaus and even inhibits further progress". The breakthrough then "eventually arrives by an opposing approach based on scaling computation by search and learning". The two methods "that seem to scale arbitrarily in this way are search and learning."

Two lines are worth rereading slowly. One is the lesson itself: "building in how we think we think does not work in the long run." The other is why: "the actual contents of minds are tremendously, irredeemably complex". So instead "we should build in only the meta-methods that can find and capture this arbitrary complexity." He sums it up as "We want AI agents that can discover like we can, not which contain what we have discovered."

*The rest is my reading, not Sutton's.* The essay is often quoted as "just scale it". Read whole, it is more specific than that. It is an argument about *where to put human effort*: into methods that improve as computation grows, not into knowledge that a method will eventually learn better on its own. Seen that way, it doesn't say architecture is irrelevant. Residual connections and attention, from the [foundations](/series/deep-learning-foundations/) and [transformers](/series/transformers/) series, are exactly the kind of thing it endorses. They hard-code nothing about language. They make learning work at a larger scale.

It also cuts both ways for [agent harnesses](/series/agent-harnesses/). A harness full of hand-written rules for what the model should do in each situation is the pattern the essay warns about. A harness that gives a capable model good tools and clean context, and gets out of its way, is closer to building in meta-methods. I find that a useful test for any piece of scaffolding: will this still help when the model is much better, or will it be in the way?
