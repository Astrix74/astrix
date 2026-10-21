# astrix

Personal technical blog and project hub, built with [Astro](https://astro.build) as a static site and served from Cloudflare.

```sh
pnpm install
pnpm dev                    # http://localhost:4321, drafts included
pnpm build                  # static site in dist/
pnpm check                  # type-check
pnpm new post "Title"       # new draft post (or: pnpm new note "Title")
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

Content lives in `src/content/`. Posts with `draft: true` show in `pnpm dev` only.

- **Posts** (`posts/*.md`): the file name is the address, so `my-post.md` is published at `/writing/my-post/`. `sample-post.md` shows every frontmatter field. Images go next to the post and are optimised at build time; always write alt text. Tags get their own pages, a table of contents appears on posts with three or more headings, and related posts are chosen by shared tags.
  - `project: repo-name` links the post to a project.
  - `cover: ./image.png` is used as the share preview; without it a card is generated. A draft's cover file is still copied into the build, at an unlinked address named by a hash of its contents, so don't give a draft a `cover` if that image must stay private.
  - `canonical: https://...` for a post first published elsewhere.
  - `updated: 2026-11-02` is shown next to the date.
- **Notes** (`notes/*.md`): short entries with a title and date. `/notes/` shows each one in full, and `/notes/rss.xml` is their feed.
- **Pages** (`pages/*.md`): `about.md` is the About page. Add `now.md` or `uses.md` and they exist at `/now/` and `/uses/`; set `footer: true` in a page's frontmatter to link it from the footer. Don't name a page after a built-in route (`writing`, `notes`, `projects`, `og`).

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

`src/pages/og/[...route].ts` generates a 1200×630 card for every post at build time (title, date, site name), plus `/og/default.png` for all other pages. The font is Newsreader, under the SIL Open Font License; the licence text is in `src/assets/fonts/`.

## Deploy

The site builds to `dist/` and is served by Cloudflare Workers static assets; no adapter is needed. [`wrangler.jsonc`](wrangler.jsonc) holds the config, and [`public/_headers`](public/_headers) and [`public/_redirects`](public/_redirects) set the security headers, long-lived caching for `/_astro/*`, and redirects for renamed pages.

In Cloudflare, connect this repository under Workers & Pages. Build command `pnpm build`, deploy command `npx wrangler deploy`, and build variables `SITE_URL` and `GITHUB_TOKEN`. Attach the domain under the Worker's Domains & Routes. The Content-Security-Policy in `_headers` allows only this site, giscus and Cloudflare's analytics script; if you add another third-party script, add its host there.
