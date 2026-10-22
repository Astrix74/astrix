import rss from '@astrojs/rss';
import type { APIContext } from 'astro';
import { site } from '../../config/site';
import { getNotes } from '../../lib/notes';

export async function GET(context: APIContext) {
  const notes = await getNotes();
  return rss({
    title: `${site.title} notes`,
    description: 'Short notes: one formula, one idea, one paper.',
    site: context.site!,
    items: notes.map((note) => ({
      title: note.data.title,
      pubDate: note.data.date,
      link: `/notes/${note.id}/`,
    })),
    customData: `<language>${site.lang}</language>`,
  });
}
