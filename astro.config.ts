import { defineConfig } from 'astro/config';
import { satteri } from '@astrojs/markdown-satteri';
import sitemap from '@astrojs/sitemap';
import { alertPlugin, figurePlugin, mathPlugin } from './src/lib/markdown';

// SITE_URL is the production origin, for example https://example.com (no trailing slash).
// Canonical links, the RSS feed and the sitemap are built from it.
const site = process.env.SITE_URL ?? 'http://localhost:4321';

export default defineConfig({
  site,
  trailingSlash: 'always',
  integrations: [sitemap()],
  // Keep the small page script as a file instead of inlining it, so the Content-Security-Policy
  // in public/_headers does not need to allow inline scripts.
  vite: { build: { assetsInlineLimit: 0 } },
  markdown: {
    processor: satteri({ features: { math: true }, mdastPlugins: [mathPlugin, alertPlugin, figurePlugin] }),
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
      defaultColor: false,
    },
  },
});
