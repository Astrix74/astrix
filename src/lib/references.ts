import { getCollection, type CollectionEntry } from 'astro:content';
import { getNotes } from './notes';
import { getPapers } from './papers';
import { getPosts } from './posts';

export type Reference = CollectionEntry<'references'>;
/** An entry that can cite references. */
export type CitingEntry = CollectionEntry<'posts'> | CollectionEntry<'notes'> | CollectionEntry<'papers'>;

export interface CitedReference {
  reference: Reference;
  /** Visible entries that cite this reference. */
  citedBy: CitingEntry[];
}

/** The address of an entry that cites references. */
export function entryHref(entry: CitingEntry): string {
  const section = { posts: 'writing', notes: 'notes', papers: 'papers' }[entry.collection];
  return `/${section}/${entry.id}/`;
}

async function getReferenceMap(): Promise<Map<string, Reference>> {
  return new Map((await getCollection('references')).map((reference) => [reference.id, reference]));
}

/** Astro only logs an unknown reference, so an unknown citation key is made to fail the build here. */
function cited(entry: CitingEntry, references: Map<string, Reference>): Reference[] {
  return entry.data.references.map(({ id }) => {
    const reference = references.get(id);
    if (!reference) {
      throw new Error(`"${entry.id}" in ${entry.collection} cites "${id}", which is not in src/content/references.yaml.`);
    }
    return reference;
  });
}

/** The references an entry lists, in its order. */
export async function referencesOf(entry: CitingEntry): Promise<Reference[]> {
  return cited(entry, await getReferenceMap());
}

/** References cited by at least one visible entry, oldest first. */
export async function getCitedReferences(): Promise<CitedReference[]> {
  const references = await getReferenceMap();
  // Every entry, drafts included, is checked so a bad key shows up before it is published.
  const all = [...(await getCollection('posts')), ...(await getCollection('notes')), ...(await getCollection('papers'))];
  for (const entry of all) cited(entry, references);

  const visible: CitingEntry[] = [...(await getPapers()), ...(await getPosts()), ...(await getNotes())];
  return [...references.values()]
    .map((reference) => ({
      reference,
      citedBy: visible.filter((entry) => entry.data.references.some(({ id }) => id === reference.id)),
    }))
    .filter(({ citedBy }) => citedBy.length > 0)
    .sort((a, b) => a.reference.data.year - b.reference.data.year || a.reference.data.title.localeCompare(b.reference.data.title));
}
