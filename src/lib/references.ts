import { getCollection, type CollectionEntry } from 'astro:content';
import { defaultLocale, type Locale } from '../i18n';
import { getNotes } from './notes';
import { getPapers } from './papers';
import { getPosts } from './posts';

export type Reference = CollectionEntry<'references'>;
/** An entry that can cite references. */
export type CitingEntry = CollectionEntry<'posts'> | CollectionEntry<'notes'> | CollectionEntry<'papers'>;

export interface CitedReference {
  reference: Reference;
  /** Visible entries in the same language that cite this reference. */
  citedBy: CitingEntry[];
}

/** The references an entry lists, in its order. Unknown keys are caught by checkContent. */
export async function referencesOf(entry: CitingEntry): Promise<Reference[]> {
  const references = new Map((await getCollection('references')).map((reference) => [reference.id, reference]));
  return entry.data.references.flatMap(({ id }) => references.get(id) ?? []);
}

/** References cited by at least one visible entry in this language, oldest first. */
export async function getCitedReferences(locale: Locale = defaultLocale): Promise<CitedReference[]> {
  const references = await getCollection('references');
  const visible: CitingEntry[] = [...(await getPapers(locale)), ...(await getPosts(locale)), ...(await getNotes(locale))];
  return references
    .map((reference) => ({
      reference,
      citedBy: visible.filter((entry) => entry.data.references.some(({ id }) => id === reference.id)),
    }))
    .filter(({ citedBy }) => citedBy.length > 0)
    .sort((a, b) => a.reference.data.year - b.reference.data.year || a.reference.data.title.localeCompare(b.reference.data.title));
}
