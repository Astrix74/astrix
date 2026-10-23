import { getCollection } from 'astro:content';
import { defaultLocale, idFor, localeOf, slugOf } from '../i18n';

// Build-time checks over every entry, drafts and translations included, so a mistake fails the build
// before it is published. Astro only logs unknown references, so they are made fatal here.

let checked: Promise<void> | undefined;

/** Runs the checks once per build; every page awaits the same result. */
export function checkContent(): Promise<void> {
  checked ??= run();
  return checked;
}

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const ids = (refs: { id: string }[]) => refs.map(({ id }) => id);

async function run(): Promise<void> {
  const [posts, notes, papers, series, pages, references] = await Promise.all([
    getCollection('posts'),
    getCollection('notes'),
    getCollection('papers'),
    getCollection('series'),
    getCollection('pages'),
    getCollection('references'),
  ]);

  // Citations point at works in references.yaml.
  const known = new Set(references.map(({ id }) => id));
  for (const entry of [...posts, ...notes, ...papers]) {
    const missing = entry.data.references.find(({ id }) => !known.has(id));
    if (missing) {
      throw new Error(`"${entry.id}" in ${entry.collection} cites "${missing.id}", which is not in src/content/references.yaml.`);
    }
  }

  // Series are defined by their English entries; prerequisites and chapters refer to those.
  const seriesIds = new Set(series.filter(({ id }) => localeOf(id) === defaultLocale).map(({ id }) => id));
  for (const entry of series) {
    const missing = entry.data.prerequisites.find(({ id }) => !seriesIds.has(id));
    if (missing) throw new Error(`Series "${entry.id}" lists "${missing.id}" as a prerequisite, but there is no such series.`);
  }
  const seen = new Map<string, string>();
  for (const post of posts) {
    const { series: inSeries, chapter } = post.data;
    if (!inSeries && chapter === undefined) continue;
    if (!inSeries) throw new Error(`Post "${post.id}" has a chapter number but no series.`);
    if (!seriesIds.has(inSeries.id)) throw new Error(`Post "${post.id}" is in the series "${inSeries.id}", but there is no such series.`);
    if (chapter === undefined) throw new Error(`Post "${post.id}" is in the series "${inSeries.id}" but has no chapter number.`);
    const key = `${localeOf(post.id)}:${inSeries.id}#${chapter}`;
    const other = seen.get(key);
    if (other) throw new Error(`Posts "${other}" and "${post.id}" are both chapter ${chapter} of "${inSeries.id}".`);
    seen.set(key, post.id);
  }

  // A translation needs an original, and must agree with it on everything except its words.
  const collections = { posts, notes, papers, series, pages };
  for (const [name, entries] of Object.entries(collections)) {
    const originals = new Map<string, (typeof entries)[number]>(
      entries.filter(({ id }) => localeOf(id) === defaultLocale).map((entry) => [entry.id, entry]),
    );
    for (const entry of entries) {
      if (localeOf(entry.id) === defaultLocale) continue;
      const original = originals.get(slugOf(entry.id));
      if (!original) {
        throw new Error(`"${entry.id}" in ${name} is a translation, but there is no "${idFor(defaultLocale, slugOf(entry.id))}" to translate.`);
      }
      const a = original.data as Record<string, unknown>;
      const b = entry.data as Record<string, unknown>;
      for (const field of ['date', 'series', 'chapter', 'tags', 'order', 'prerequisites']) {
        if (field in a && !same(a[field], b[field])) {
          throw new Error(`"${entry.id}" has a different ${field} from its original "${original.id}".`);
        }
      }
      if ('references' in a && !same(ids(a.references as { id: string }[]), ids(b.references as { id: string }[]))) {
        throw new Error(`"${entry.id}" cites different works from its original "${original.id}".`);
      }
    }
  }
}
