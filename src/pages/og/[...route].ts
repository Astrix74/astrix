import { OGImageRoute } from 'astro-og-canvas';
import { site } from '../../config/site';
import { formatDate, getPosts } from '../../lib/posts';

// One share-preview image per post, plus /og/default.png for every other page.
// A post with a `cover` image uses that instead, so it gets no generated card.
const posts = (await getPosts()).filter((post) => !post.data.cover);

const pages: Record<string, { title: string; description: string }> = {
  default: { title: site.title, description: site.description },
  ...Object.fromEntries(
    posts.map((post) => [
      post.id,
      { title: post.data.title, description: `${formatDate(post.data.date)} · ${site.title}` },
    ]),
  ),
};

export const { getStaticPaths, GET } = await OGImageRoute({
  pages,
  getSlug: (path) => `${path}.png`,
  getImageOptions: (_path, page) => ({
    title: page.title,
    description: page.description,
    bgGradient: [[251, 250, 247]],
    padding: 80,
    border: { color: [31, 78, 140], width: 16, side: 'block-end' },
    font: {
      title: { families: ['Newsreader 16pt 16pt SemiBold'], weight: 'SemiBold', size: 72, lineHeight: 1.15, color: [28, 27, 25] },
      description: { families: ['Newsreader 16pt 16pt'], weight: 'Normal', size: 34, lineHeight: 1.4, color: [106, 103, 95] },
    },
    fonts: [
      './src/assets/fonts/newsreader-latin-600-normal.ttf',
      './src/assets/fonts/newsreader-latin-400-normal.ttf',
    ],
  }),
});
