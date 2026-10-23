import { localeOf, localize, slugOf } from '../i18n';

const sections = { posts: 'writing', notes: 'notes', papers: 'papers', series: 'series' } as const;

/** The address of a post, note, paper or series, in the language it is written in. */
export function entryHref(entry: { collection: keyof typeof sections; id: string }): string {
  return localize(localeOf(entry.id), `/${sections[entry.collection]}/${slugOf(entry.id)}/`);
}
