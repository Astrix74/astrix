---
title: Sample post
description: A draft that shows the frontmatter and the Markdown styles. It only appears in development.
date: 2026-10-21
tags: [meta]
draft: true
cover: ./sample-image.png
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

## Images

Put an image next to the post and refer to it with a relative path. It is resized and converted when the site builds, so always write alt text.

![A checkerboard of two warm greys, used as a sample image](./sample-image.png)

The `cover` field in the frontmatter picks the image used when the post is shared.
