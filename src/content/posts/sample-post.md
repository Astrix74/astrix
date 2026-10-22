---
title: Sample post
description: A draft that shows the frontmatter and the Markdown styles. It only appears in development.
date: 2026-10-21
tags: [meta]
draft: true
cover: ./sample-image.png
references: [vaswani2017]   # citation keys from src/content/references.yaml, listed under the post
# series: transformers   # a file name from src/content/series; needs a chapter number too
# chapter: 4
# project: my-repo   # a repo name from the projects list, or "owner/name"; adds a project link under the post
# canonical: https://example.com/original-post/   # set when the post was first published elsewhere
# updated: 2026-11-02   # shown next to the date
---

This post is a draft, so it shows in `pnpm dev` and is left out of the built site. Copy it as a starting point for a real post, then delete it.

## Headings, lists and links

Posts are plain Markdown files in `src/content/posts`. The file name becomes the address, so `my-first-post.md` is published at `/writing/my-first-post/`.

- Dates use the `date` field, newest first.
- `tags` are shown next to the date and get their own pages.
- Set `draft: false`, or remove the line, to publish.

A contents list appears above any post with three or more headings.

## Code

Code blocks are highlighted at build time and follow the light and dark theme.

```ts
export function greet(name: string) {
  return `Hello, ${name}`;
}
```

Inline `code` uses the monospace face too.

> A quote is set in italics and indented.

## Math

Inline math sits in single dollar signs, so $e^{i\pi} + 1 = 0$ stays in the sentence. Write `\$` for a literal dollar sign, as in \$100. Display math goes between double dollar signs on their own lines:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

Formulas are rendered when the site is built, and a formula KaTeX cannot read stops the build with the file name. Keep math out of headings: the contents list shows heading text as typed.

## Callouts and footnotes

> [!TIP] What to remember
> A quote that starts with `[!NOTE]`, `[!TIP]`, `[!IMPORTANT]`, `[!WARNING]` or `[!CAUTION]` becomes a callout. Text after the marker replaces the default label.

> [!WARNING]
> Warnings and cautions get a warm rule instead of the accent colour.

Footnotes work the GitHub way.[^aside] They are collected at the end of the post, with links back.

[^aside]: Use them for asides. Sources go in the `references` field instead: mention the authors and year in the text, and the full reference is listed under the post.

## Images

Put an image next to the post and refer to it with a relative path. It is resized and converted when the site builds, so always write alt text.

![A checkerboard of two warm greys, used as a sample image](./sample-image.png)

The `cover` field in the frontmatter picks the image used when the post is shared.
