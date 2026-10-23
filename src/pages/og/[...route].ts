import { OGImageRoute } from 'astro-og-canvas';
import { site } from '../../config/site';
import { strings } from '../../i18n/strings';
import { getPapers } from '../../lib/papers';
import { formatDate, getPosts } from '../../lib/posts';
import { getSeries } from '../../lib/series';

// One share-preview image per post, paper and series, plus /og/default.png for every other page.
// A post with a `cover` image uses that instead, so it gets no generated card. Cards are made from the
// English entries only: the card font has no Chinese or Japanese glyphs, so translations reuse them.
const posts = (await getPosts()).filter((post) => !post.data.cover);
const allSeries = await getSeries();
const seriesTitles = new Map(allSeries.map(({ series }) => [series.id, series.data.title]));

const pages: Record<string, { title: string; description: string }> = {
  default: { title: site.title, description: strings.en.siteDescription },
  ...Object.fromEntries(
    posts.map((post) => {
      const seriesTitle = post.data.series && seriesTitles.get(post.data.series.id);
      const subtitle = seriesTitle ? `Chapter ${post.data.chapter} · ${seriesTitle}` : formatDate(post.data.date);
      return [post.id, { title: post.data.title, description: `${subtitle} · ${site.title}` }];
    }),
  ),
  ...Object.fromEntries(
    allSeries.map(({ series }) => [`series/${series.id}`, { title: series.data.title, description: `Series · ${site.title}` }]),
  ),
  ...Object.fromEntries(
    (await getPapers()).map((paper) => [
      `papers/${paper.id}`,
      { title: paper.data.title, description: `Paper · ${formatDate(paper.data.date)} · ${site.title}` },
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
