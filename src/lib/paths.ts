import { getCollection } from 'astro:content';
import { inLocale, localize, locales, slugOf, type Alternates, type Locale } from '../i18n';
import { getNotes } from './notes';
import { getPapers } from './papers';
import { getPosts } from './posts';
import { getSeries } from './series';
import { collectTags, relatedPosts, slugifyTag } from './tags';

// Static paths for every page that has one per entry, in one language. Each page also gets
// `alternates`: its own address in every language it is published in.

const getters = { posts: getPosts, notes: getNotes, papers: getPapers };
const sections = { posts: 'writing', notes: 'notes', papers: 'papers' } as const;

async function alternatesFor(collection: keyof typeof getters, slug: string): Promise<Alternates> {
  const alternates: Alternates = {};
  for (const locale of locales) {
    const entries: { id: string }[] = await getters[collection](locale);
    if (entries.some((entry) => slugOf(entry.id) === slug)) {
      alternates[locale] = localize(locale, `/${sections[collection]}/${slug}/`);
    }
  }
  return alternates;
}

export async function postPaths(locale: Locale) {
  const posts = await getPosts(locale);
  const allSeries = await getSeries(locale);
  return Promise.all(
    posts.map(async (post, index) => {
      const series = allSeries.find((item) => item.series.id === post.data.series?.id);
      const position = series ? series.chapters.findIndex((chapter) => chapter.id === post.id) : -1;
      return {
        params: { id: slugOf(post.id) },
        props: {
          post,
          locale,
          // Posts are newest first, so the next index is the older post.
          newer: posts[index - 1],
          older: posts[index + 1],
          series,
          previousChapter: series?.chapters[position - 1],
          nextChapter: series?.chapters[position + 1],
          // Other chapters of the same series are already linked from the chapter pager.
          related: relatedPosts(post, series ? posts.filter((other) => other.data.series?.id !== series.series.id) : posts),
          alternates: await alternatesFor('posts', slugOf(post.id)),
        },
      };
    }),
  );
}

export async function notePaths(locale: Locale) {
  return Promise.all(
    (await getNotes(locale)).map(async (note) => ({
      params: { id: slugOf(note.id) },
      props: { note, locale, alternates: await alternatesFor('notes', slugOf(note.id)) },
    })),
  );
}

export async function paperPaths(locale: Locale) {
  return Promise.all(
    (await getPapers(locale)).map(async (paper) => ({
      params: { id: slugOf(paper.id) },
      props: { paper, locale, alternates: await alternatesFor('papers', slugOf(paper.id)) },
    })),
  );
}

export async function seriesPaths(locale: Locale) {
  const items = await getSeries(locale);
  const visible = new Map(items.map((item) => [item.series.id, item]));
  const elsewhere = Object.fromEntries(
    await Promise.all(locales.map(async (other) => [other, new Set((await getSeries(other)).map((item) => item.series.id))] as const)),
  );
  return items.map((item) => ({
    params: { id: item.series.id },
    props: {
      item,
      locale,
      // Only prerequisites that are published themselves.
      prerequisites: item.series.data.prerequisites.flatMap(({ id }) => visible.get(id) ?? []),
      alternates: Object.fromEntries(
        locales.filter((other) => elsewhere[other].has(item.series.id)).map((other) => [other, localize(other, `/series/${item.series.id}/`)]),
      ) as Alternates,
    },
  }));
}

export async function tagPaths(locale: Locale) {
  const posts = await getPosts(locale);
  const elsewhere = Object.fromEntries(
    await Promise.all(locales.map(async (other) => [other, new Set(collectTags(await getPosts(other)).map(({ slug }) => slug))] as const)),
  );
  return collectTags(posts).map(({ tag, slug }) => ({
    params: { tag: slug },
    props: {
      tag,
      locale,
      posts: posts.filter((post) => post.data.tags.some((name) => slugifyTag(name) === slug)),
      alternates: Object.fromEntries(
        locales.filter((other) => elsewhere[other].has(slug)).map((other) => [other, localize(other, `/writing/tags/${slug}/`)]),
      ) as Alternates,
    },
  }));
}

export async function pagePaths(locale: Locale) {
  const pages = await getCollection('pages');
  return inLocale(pages, locale).map((page) => ({
    params: { id: slugOf(page.id) },
    props: {
      page,
      locale,
      alternates: Object.fromEntries(
        locales
          .filter((other) => inLocale(pages, other).some((entry) => slugOf(entry.id) === slugOf(page.id)))
          .map((other) => [other, localize(other, `/${slugOf(page.id)}/`)]),
      ) as Alternates,
    },
  }));
}
