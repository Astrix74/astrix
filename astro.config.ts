import { rename, rmdir } from 'node:fs/promises';
import { defineConfig } from 'astro/config';
import type { AstroIntegration } from 'astro';
import { satteri } from '@astrojs/markdown-satteri';
import sitemap from '@astrojs/sitemap';
import { defaultLocale, locales } from './src/i18n';
import { alertPlugin, figurePlugin, mathPlugin } from './src/lib/markdown';

// SITE_URL is the production origin, for example https://example.com (no trailing slash).
// Canonical links, the RSS feed and the sitemap are built from it.
const site = process.env.SITE_URL ?? 'http://localhost:4321';

// Cloudflare answers a missing address with the nearest 404.html, but Astro only writes the root error
// page under that name. This moves each language's page from <locale>/404/index.html to <locale>/404.html.
const localizedNotFound: AstroIntegration = {
  name: 'localized-not-found',
  hooks: {
    'astro:build:done': async ({ dir }) => {
      for (const locale of locales.filter((locale) => locale !== defaultLocale)) {
        await rename(new URL(`${locale}/404/index.html`, dir), new URL(`${locale}/404.html`, dir));
        await rmdir(new URL(`${locale}/404/`, dir));
      }
    },
  },
};

export default defineConfig({
  site,
  trailingSlash: 'always',
  // Each URL lists its translations; English URLs have no prefix, Chinese ones start with /zh/.
  integrations: [sitemap({ i18n: { defaultLocale: 'en', locales: { en: 'en', zh: 'zh-CN' } } }), localizedNotFound],
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
