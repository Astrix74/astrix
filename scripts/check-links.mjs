// Checks every internal link and #fragment in a built site, and that pages in a translation link to
// translated pages where they exist. Links meant to change language carry an hreflang attribute.
// Usage: node scripts/check-links.mjs [dist]
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';

const dist = process.argv[2] ?? 'dist';
const translations = ['zh'];

const entities = { amp: '&', lt: '<', gt: '>', quot: '"', '#39': "'" };
const decode = (value) => value.replace(/&(amp|lt|gt|quot|#39);/g, (_, name) => entities[name]);

function* htmlFiles(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) yield* htmlFiles(path);
    else if (entry.name.endsWith('.html')) yield path;
  }
}

const pages = new Map();
for (const file of htmlFiles(dist)) {
  const html = readFileSync(file, 'utf8');
  const ids = new Set([...html.matchAll(/\sid="([^"]*)"/g)].map((match) => decode(match[1])));
  const links = [...html.matchAll(/<(?:a|link|img|script|source)\b[^>]*>/g)].flatMap(([tag]) => {
    const url = /\s(?:href|src)="([^"]*)"/.exec(tag)?.[1];
    return url ? [{ url: decode(url), crossLanguage: /\shreflang="/.test(tag) }] : [];
  });
  pages.set(file, { ids, links });
}

if (pages.size === 0) {
  console.log(`No pages found in ${dist}. Build the site first.`);
  process.exit(1);
}

const fileFor = (path) => join(dist, path.endsWith('/') ? `${path}index.html` : path);
const problems = [];
let checked = 0;
for (const [file, { links }] of pages) {
  const from = relative(dist, file);
  const language = translations.find((locale) => from.startsWith(`${locale}/`));
  for (const { url, crossLanguage } of links) {
    if (/^(https?:|mailto:|data:)/.test(url)) continue;
    checked++;
    const [pathAndQuery, fragment] = url.split('#');
    const path = pathAndQuery.split('?')[0];
    const target = path ? fileFor(path) : file;
    if (!existsSync(target)) {
      problems.push(`${from} -> ${url} (no such page)`);
      continue;
    }
    if (fragment && target.endsWith('.html') && !pages.get(target)?.ids.has(decodeURIComponent(fragment))) {
      problems.push(`${from} -> ${url} (no such #fragment)`);
    }
    if (language && path.startsWith('/') && !path.startsWith(`/${language}/`) && !crossLanguage && existsSync(fileFor(`/${language}${path}`))) {
      problems.push(`${from} -> ${url} (a translation exists at /${language}${path})`);
    }
  }
}

console.log(`${pages.size} pages, ${checked} internal links checked, ${problems.length} problems`);
for (const problem of problems) console.log(`  ${problem}`);
process.exit(problems.length > 0 ? 1 : 0);
