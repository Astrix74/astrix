import { localize, type Locale } from '../i18n';
import type { Post } from './posts';

export function slugifyTag(tag: string): string {
  return tag
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, '-')
    .replace(/^-+|-+$/g, '');
}

export function tagUrl(locale: Locale, tag: string): string {
  return localize(locale, `/writing/tags/${slugifyTag(tag)}/`);
}

export interface TagCount {
  tag: string;
  slug: string;
  count: number;
}

/** Tags used by the given posts, most used first. The first spelling seen is the display name. */
export function collectTags(posts: Post[]): TagCount[] {
  const tags = new Map<string, TagCount>();
  for (const post of posts) {
    for (const tag of post.data.tags) {
      const slug = slugifyTag(tag);
      if (!slug) continue;
      const entry = tags.get(slug) ?? { tag, slug, count: 0 };
      entry.count += 1;
      tags.set(slug, entry);
    }
  }
  return [...tags.values()].sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag));
}

/** Other posts that share a tag with this one: most shared tags first, then newest. */
export function relatedPosts(post: Post, posts: Post[], limit = 3): Post[] {
  const own = new Set(post.data.tags.map(slugifyTag));
  return posts
    .filter((other) => other.id !== post.id)
    .map((other) => ({
      other,
      shared: other.data.tags.filter((tag) => own.has(slugifyTag(tag))).length,
    }))
    .filter(({ shared }) => shared > 0)
    .sort((a, b) => b.shared - a.shared || b.other.data.date.valueOf() - a.other.data.date.valueOf())
    .slice(0, limit)
    .map(({ other }) => other);
}
