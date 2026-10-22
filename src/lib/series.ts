import { getCollection, type CollectionEntry } from 'astro:content';
import { getPosts, type Post } from './posts';

export type Series = CollectionEntry<'series'>;

export interface SeriesWithChapters {
  series: Series;
  /** Visible chapters, in chapter order. */
  chapters: Post[];
}

/**
 * Every post and series, drafts included, is checked so a mistake shows up before it is published.
 * Astro only logs an unknown reference, so unknown series names are made to fail the build here.
 */
async function checkSeries(all: Series[]): Promise<void> {
  const ids = new Set(all.map((entry) => entry.id));
  for (const entry of all) {
    const missing = entry.data.prerequisites.find(({ id }) => !ids.has(id));
    if (missing) throw new Error(`Series "${entry.id}" lists "${missing.id}" as a prerequisite, but there is no such series.`);
  }

  const seen = new Map<string, string>();
  for (const post of await getCollection('posts')) {
    const { series, chapter } = post.data;
    if (!series && chapter === undefined) continue;
    if (!series) throw new Error(`Post "${post.id}" has a chapter number but no series.`);
    if (!ids.has(series.id)) throw new Error(`Post "${post.id}" is in the series "${series.id}", but there is no such series.`);
    if (chapter === undefined) throw new Error(`Post "${post.id}" is in the series "${series.id}" but has no chapter number.`);
    const key = `${series.id}#${chapter}`;
    const other = seen.get(key);
    if (other) throw new Error(`Posts "${other}" and "${post.id}" are both chapter ${chapter} of "${series.id}".`);
    seen.set(key, post.id);
  }
}

/** Series in reading order. A series with no visible chapter is left out, so nothing links to an empty page. */
export async function getSeries(): Promise<SeriesWithChapters[]> {
  const series = await getCollection('series');
  await checkSeries(series);
  const posts = await getPosts();
  return series
    .sort((a, b) => a.data.order - b.data.order)
    .map((entry) => ({
      series: entry,
      chapters: posts
        .filter((post) => post.data.series?.id === entry.id)
        .sort((a, b) => a.data.chapter! - b.data.chapter!),
    }))
    .filter(({ chapters }) => chapters.length > 0);
}

export function seriesUrl(id: string): string {
  return `/series/${id}/`;
}
