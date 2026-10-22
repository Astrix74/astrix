# astrix

A developer's notebook on AI: step-by-step series on deep learning, transformers and agent harnesses, plus notes, a reading list of the papers behind them, and projects. Built with [Astro](https://astro.build) as a static site and served from Cloudflare.

```sh
pnpm install
pnpm dev                    # http://localhost:4321, drafts included
pnpm build                  # static site in dist/
pnpm check                  # type-check
pnpm new post "Title"       # new draft post (or: note, paper, series)
pnpm deploy                 # build, then deploy with Wrangler
```

Needs Node 22.22 or newer (even-numbered releases). `package.json` pins pnpm. pnpm 11 refuses to run dependency install scripts unless they are approved, so `pnpm-workspace.yaml` allows `esbuild` and `workerd`, which Wrangler needs.

## Configure

Everything you are likely to change is in [`src/config/site.ts`](src/config/site.ts):

- **Social links.** Add `{ label, href }` to `socials`. An empty `href` hides the link everywhere, so the X entry appears once you fill it in.
- **Navigation.** Edit `nav`. Use trailing slashes (`/writing/`).
- **GitHub account.** `github` is the username projects are read from.
- **Analytics.** Paste the Cloudflare Web Analytics site token into `analytics.cloudflareToken`. Empty means off; the snippet is only added to production builds.
- **Comments.** Fill in `comments` to turn on giscus (see below). An empty `repoId` or `categoryId` means off.

Environment variables (see [`.env.example`](.env.example)):

| Variable | Purpose |
| --- | --- |
| `SITE_URL` | Production origin. Used for canonical links, the RSS feeds, share images and the sitemap. |
| `GITHUB_TOKEN` | Raises GitHub's API rate limit while the project list is built. Set it on the build service, because builds there share IP addresses. |

## Writing

Content lives in `src/content/`. Posts with `draft: true` show in `pnpm dev` only. Drafts are still checked during `pnpm build`: a formula KaTeX cannot read, an unknown citation key or series name, or a repeated chapter number fails the build even in a draft.

- **Posts** (`posts/*.md`): the file name is the address, so `my-post.md` is published at `/writing/my-post/`. `sample-post.md` shows every frontmatter field. Images go next to the post and are optimised at build time; always write alt text. Tags get their own pages, a table of contents appears on posts with three or more headings, and related posts are chosen by shared tags.
  - `project: repo-name` links the post to a project.
  - `cover: ./image.png` is used as the share preview; without it a card is generated. A draft's cover file is still copied into the build, at an unlinked address named by a hash of its contents, so don't give a draft a `cover` if that image must stay private.
  - `canonical: https://...` for a post first published elsewhere.
  - `updated: 2026-11-02` is shown next to the date.
  - `series: file-name` and `chapter: 2` make the post a chapter of a series (see below).
  - `references: [vaswani2017]` lists works from `references.yaml` under the post (see Reading list below).
- **Notes** (`notes/*.md`): short entries with a title and date, and optionally `references:`. `/notes/` shows each one in full, and `/notes/rss.xml` is their feed.
- **Papers** (`papers/*.md`, `pnpm new paper "Title"`): long-form analysis at `/papers/<name>/`, with a one-sentence `description`, a plain-text `abstract` shown above the paper, and `references:`. Figures go next to the paper; an image alone in its paragraph with a title, `![alt](./chart.svg "Figure 1. What it shows.")`, becomes a figure with that caption. Papers are in the main RSS feed with the posts.
- **Who wrote it.** `writtenBy` in `src/config/site.ts` is shown under the title of every post, note and paper, linking to the About page; set `writtenBy:` in an entry's frontmatter to override it for that entry, or to an empty string to hide it.
- **Pages** (`pages/*.md`): `about.md` is the About page. Add `now.md` or `uses.md` and they exist at `/now/` and `/uses/`; set `footer: true` in a page's frontmatter to link it from the footer. Don't name a page after a built-in route (`writing`, `series`, `papers`, `notes`, `reading`, `projects`, `og`).

### Math, callouts and footnotes

- **Math.** `$…$` inline and `$$…$$` on their own lines for display, written in LaTeX and rendered with [KaTeX](https://katex.org) at build time, so pages load no script. A formula KaTeX cannot read stops the build and names the file. Write `\$` for a literal dollar sign. Keep math out of headings, since the contents list shows heading text as typed.
- **Callouts.** A quote whose first line is `[!NOTE]`, `[!TIP]`, `[!IMPORTANT]`, `[!WARNING]` or `[!CAUTION]` becomes a callout, the same syntax GitHub uses. Text after the marker replaces the default label: `> [!TIP] What to remember`. An unknown marker stops the build.
- **Footnotes.** `text[^1]` and `[^1]: the note`, collected at the end of the post. Keep them for asides: cite sources through `references:` and name the authors and year in the text, so each reference appears once, in the list under the post.

`posts/sample-post.md` shows all three.

### Series

A series is an ordered path through one topic. Tags group posts automatically; a series adds a chapter order, an introduction and a list of what comes next.

- Create one with `pnpm new series "Title"`, which writes `series/<name>.md`. The body is the introduction shown on `/series/<name>/`.
  - `order` sets its place on `/series/`, where series are listed in the order they are meant to be read.
  - `prerequisites: [other-series]` adds a "read first" link.
  - `planned: ["Title", …]` lists chapters not written yet, shown as coming.
- Add `series: <name>` and `chapter: <n>` to a post to make it a chapter. Two chapters with the same number, or a chapter number without a series, stop the build.
- A chapter shows "Chapter n · Series" under its title, and its pager moves between chapters instead of by date.
- A series with no published chapter is left out everywhere, so nothing links to an empty page. Only published chapters are listed.

### Reading list

`references.yaml` holds every work the site cites: one entry per paper or article with an `id` (the citation key), `title`, `authors`, `year`, an optional `venue`, and `url`. A post or note lists the keys it cites in `references:`; an unknown key stops the build, even in a draft. `/reading/` shows every work that something published cites, oldest first, with links to where it is discussed, and the footer links to it once there is one.

## Projects

The Projects section is read from GitHub whenever the site is built, so a new project appears on the next build. A repo is shown when it is:

1. listed in `projects.include` in `site.ts` (always shown, in that order, forks allowed; use `owner/name` for someone else's repo), or
2. one of your public repos with the GitHub topic set in `projects.topic` (default `showcase`). Forks and archived repos are skipped unless `includeForks` is set, and `exclude` hides a repo that has the topic.

If GitHub cannot be reached, the build keeps the previous project data and logs a warning.

## Comments (giscus)

Comments are stored in GitHub Discussions and shown on posts. One-time setup:

1. The repository named in `comments.repo` must be public, with Discussions turned on.
2. Install the [giscus app](https://github.com/apps/giscus) on it, and create a Discussions category (an Announcements-type one means only you start threads).
3. On [giscus.app](https://giscus.app), enter the repository and category, then copy `data-repo-id` and `data-category-id` into `comments` in `site.ts`.

Comments are matched by page address, so renaming a post detaches its comments.

## Share previews

`src/pages/og/[...route].ts` generates a 1200×630 card at build time for every post (title, then the date or the chapter and series, and the site name) and every series, plus `/og/default.png` for all other pages. The font is Newsreader, under the SIL Open Font License; the licence text is in `src/assets/fonts/`.

## Deploy

The site builds to `dist/` and is served by Cloudflare Workers static assets; no adapter is needed. [`wrangler.jsonc`](wrangler.jsonc) holds the config, and [`public/_headers`](public/_headers) and [`public/_redirects`](public/_redirects) set the security headers, long-lived caching for `/_astro/*`, and redirects for renamed pages.

In Cloudflare, connect this repository under Workers & Pages. Build command `pnpm build`, deploy command `npx wrangler deploy`, and build variables `SITE_URL` and `GITHUB_TOKEN`. Attach the domain under the Worker's Domains & Routes. The Content-Security-Policy in `_headers` allows only this site, giscus and Cloudflare's analytics script; if you add another third-party script, add its host there.
