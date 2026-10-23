import { getCollection, type CollectionEntry } from 'astro:content';
import { defaultLocale, inLocale, type Locale } from '../i18n';

export type Note = CollectionEntry<'notes'>;

/** Newest first, in one language. Drafts are only included in development. */
export async function getNotes(locale: Locale = defaultLocale): Promise<Note[]> {
  const notes = await getCollection('notes', ({ data }) => import.meta.env.DEV || !data.draft);
  return inLocale(notes, locale).sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}
