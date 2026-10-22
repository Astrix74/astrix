// Creates a draft post, note or paper, or a series, with the frontmatter filled in.
// Usage: pnpm new post "Title of the post"   (or: note, paper, series)
import { existsSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const [kind, ...words] = process.argv.slice(2);
const title = words.join(' ').trim();

if (!['post', 'note', 'paper', 'series'].includes(kind) || !title) {
  console.error('Usage: pnpm new <post|note|paper|series> "Title"');
  process.exit(1);
}

const slug = title
  .toLowerCase()
  .normalize('NFKD')
  .replace(/[^\p{L}\p{N}]+/gu, '-')
  .replace(/^-+|-+$/g, '');

if (!slug) {
  console.error('The title needs at least one letter or digit.');
  process.exit(1);
}

const file = join('src', 'content', kind === 'series' ? 'series' : `${kind}s`, `${slug}.md`);
if (existsSync(file)) {
  console.error(`${file} already exists.`);
  process.exit(1);
}

const today = new Date().toLocaleDateString('en-CA');
const frontmatter = {
  post: [
    '---',
    `title: ${JSON.stringify(title)}`,
    'description: ""',
    `date: ${today}`,
    'tags: []',
    'references: []   # citation keys from src/content/references.yaml',
    '# series: series-file-name',
    '# chapter: 1',
    'draft: true',
    '---',
  ],
  note: ['---', `title: ${JSON.stringify(title)}`, `date: ${today}`, 'references: []', 'draft: true', '---'],
  paper: [
    '---',
    `title: ${JSON.stringify(title)}`,
    'description: ""',
    'abstract: ""',
    `date: ${today}`,
    'references: []   # citation keys from src/content/references.yaml',
    'draft: true',
    '---',
  ],
  series: ['---', `title: ${JSON.stringify(title)}`, 'description: ""', 'order: 1', 'prerequisites: []', 'planned: []', '---'],
}[kind];

writeFileSync(file, `${frontmatter.join('\n')}\n\n`);
console.log(
  kind === 'series'
    ? `Created ${file}. It appears on the site once one of its chapters is published.`
    : `Created ${file}. Remove "draft: true" when it is ready to publish.`,
);
