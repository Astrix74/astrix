import { getCollection, type CollectionEntry } from 'astro:content';
import { defaultLocale, inLocale, type Locale } from '../i18n';

export type Post = CollectionEntry<'posts'>;

/** Newest first, in one language. Drafts are only included in development. */
export async function getPosts(locale: Locale = defaultLocale): Promise<Post[]> {
  const posts = await getCollection('posts', ({ data }) => import.meta.env.DEV || !data.draft);
  return inLocale(posts, locale).sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}

export function formatDate(date: Date): string {
  return date.toISOString().slice(0, 10);
}

const cjk = /[぀-ヿ㐀-䶿一-鿿豈-﫿]/g;

/** About 220 words a minute for English, and 400 characters a minute for Chinese or Japanese. */
export function readingMinutes(body = ''): number {
  const characters = (body.match(cjk) ?? []).length;
  const words = body.replace(cjk, ' ').trim().split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.round(words / 220 + characters / 400));
}
