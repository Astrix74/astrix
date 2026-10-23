import { getCollection, type CollectionEntry } from 'astro:content';
import { defaultLocale, inLocale, type Locale } from '../i18n';

export type Paper = CollectionEntry<'papers'>;

/** Newest first, in one language. Drafts are only included in development. */
export async function getPapers(locale: Locale = defaultLocale): Promise<Paper[]> {
  const papers = await getCollection('papers', ({ data }) => import.meta.env.DEV || !data.draft);
  return inLocale(papers, locale).sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}
