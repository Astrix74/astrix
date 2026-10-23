import { getCollection, type CollectionEntry } from 'astro:content';
import { defaultLocale, idFor, inLocale, localize, type Locale } from '../i18n';
import { getPosts, type Post } from './posts';

export type Series = CollectionEntry<'series'>;

export interface SeriesWithChapters {
  /** The English entry, which owns the series' id, order and prerequisites. */
  series: Series;
  /** What to show: the translation of the introduction if there is one, otherwise the English. */
  display: Series;
  /** Visible chapters in this language, in chapter order. */
  chapters: Post[];
}

/**
 * Series in reading order, in one language. A series with no visible chapter in that language is left out,
 * so nothing links to an empty page.
 */
export async function getSeries(locale: Locale = defaultLocale): Promise<SeriesWithChapters[]> {
  const all = await getCollection('series');
  const byId = new Map(all.map((entry) => [entry.id, entry]));
  const posts = await getPosts(locale);
  return inLocale(all, defaultLocale)
    .sort((a, b) => a.data.order - b.data.order)
    .map((entry) => ({
      series: entry,
      display: byId.get(idFor(locale, entry.id)) ?? entry,
      chapters: posts
        .filter((post) => post.data.series?.id === entry.id)
        .sort((a, b) => a.data.chapter! - b.data.chapter!),
    }))
    .filter(({ chapters }) => chapters.length > 0);
}

export function seriesUrl(locale: Locale, id: string): string {
  return localize(locale, `/series/${id}/`);
}
