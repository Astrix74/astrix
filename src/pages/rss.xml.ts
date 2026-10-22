import rss from '@astrojs/rss';
import type { APIContext } from 'astro';
import { site } from '../config/site';
import { getPapers } from '../lib/papers';
import { getPosts } from '../lib/posts';

export async function GET(context: APIContext) {
  const posts = (await getPosts()).map((post) => ({
    title: post.data.title,
    description: post.data.description,
    pubDate: post.data.date,
    link: `/writing/${post.id}/`,
  }));
  const papers = (await getPapers()).map((paper) => ({
    title: paper.data.title,
    description: paper.data.description,
    pubDate: paper.data.date,
    link: `/papers/${paper.id}/`,
  }));
  return rss({
    title: site.title,
    description: site.description,
    site: context.site!,
    items: [...posts, ...papers].sort((a, b) => b.pubDate.valueOf() - a.pubDate.valueOf()),
    customData: `<language>${site.lang}</language>`,
  });
}
