import { getCollection, type CollectionEntry } from 'astro:content';

export type Paper = CollectionEntry<'papers'>;

/** Newest first. Drafts are only included in development. */
export async function getPapers(): Promise<Paper[]> {
  const papers = await getCollection('papers', ({ data }) => import.meta.env.DEV || !data.draft);
  return papers.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}
