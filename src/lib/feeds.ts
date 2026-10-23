import rss from '@astrojs/rss';
import { site } from '../config/site';
import { languageTag, type Locale } from '../i18n';
import { strings } from '../i18n/strings';
import { entryHref } from './links';
import { getNotes } from './notes';
import { getPapers } from './papers';
import { getPosts } from './posts';

/** Posts and papers in one language, newest first. */
export async function writingFeed(locale: Locale, siteUrl: URL) {
  const entries = [...(await getPosts(locale)), ...(await getPapers(locale))];
  return rss({
    title: site.title,
    description: strings[locale].siteDescription,
    site: siteUrl,
    items: entries
      .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf())
      .map((entry) => ({
        title: entry.data.title,
        description: entry.data.description,
        pubDate: entry.data.date,
        link: entryHref(entry),
      })),
    customData: `<language>${languageTag[locale]}</language>`,
  });
}

/** Notes in one language, newest first. */
export async function notesFeed(locale: Locale, siteUrl: URL) {
  const t = strings[locale];
  return rss({
    title: t.notes.feedTitle(site.title),
    description: t.notes.description,
    site: siteUrl,
    items: (await getNotes(locale)).map((note) => ({
      title: note.data.title,
      pubDate: note.data.date,
      link: entryHref(note),
    })),
    customData: `<language>${languageTag[locale]}</language>`,
  });
}
