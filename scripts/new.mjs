// Creates a draft post or note with the frontmatter filled in.
// Usage: pnpm new post "Title of the post"   or   pnpm new note "Title of the note"
import { existsSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const [kind, ...words] = process.argv.slice(2);
const title = words.join(' ').trim();

if (!['post', 'note'].includes(kind) || !title) {
  console.error('Usage: pnpm new <post|note> "Title"');
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

const file = join('src', 'content', `${kind}s`, `${slug}.md`);
if (existsSync(file)) {
  console.error(`${file} already exists.`);
  process.exit(1);
}

const today = new Date().toLocaleDateString('en-CA');
const frontmatter =
  kind === 'post'
    ? ['---', `title: ${JSON.stringify(title)}`, 'description: ""', `date: ${today}`, 'tags: []', 'draft: true', '---']
    : ['---', `title: ${JSON.stringify(title)}`, `date: ${today}`, 'draft: true', '---'];

writeFileSync(file, `${frontmatter.join('\n')}\n\n`);
console.log(`Created ${file}. Remove "draft: true" when it is ready to publish.`);
