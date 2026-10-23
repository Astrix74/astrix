// Compares every translation with its English original, so a translation can change words but not facts:
// code, formulas, numbers, links, footnotes, figures, callouts and the heading outline must all match.
// Frontmatter that must match (date, series, chapter, tags, references) is checked by the build.
// With a built site, it also looks for emphasis markers that failed to render, which happens in Chinese
// when ** sits between punctuation and a character.
// Usage: node scripts/check-translations.mjs [dist]
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';

const content = 'src/content';
const translations = ['zh'];
const dist = process.argv[2];

function parse(file) {
  const source = readFileSync(file, 'utf8');
  const body = source.replace(/^---\n[\s\S]*?\n---\n/, '');
  const found = { code: [], inlineCode: [], math: [], urls: [], internal: [], images: [], footnotes: [], callouts: [], headings: [], tableRows: 0, digits: [] };
  let rest = body.replace(/^(`{3,}|~{3,})[^\n]*\n([\s\S]*?)^\1[ \t]*$/gm, (_, _fence, code) => {
    found.code.push(code);
    return '\n';
  });
  rest = rest.replace(/`([^`\n]+)`/g, (_, code) => {
    found.inlineCode.push(code);
    return ' ';
  });
  rest = rest.replace(/\$\$([\s\S]+?)\$\$/g, (_, math) => {
    found.math.push(math.trim());
    return ' ';
  });
  rest = rest.replace(/(?<![\\$])\$([^$\n]+?)\$/g, (_, math) => {
    found.math.push(math.trim());
    return ' ';
  });
  rest = rest.replace(/(!?)\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (_, image, text, url) => {
    if (image) found.images.push(url);
    else if (/^https?:/.test(url)) found.urls.push(url);
    else found.internal.push(url.replace(/^\/[a-z]{2}\//, '/'));
    return ` ${text} `;
  });
  rest = rest.replace(/https?:\/\/[^\s)>\]]+/g, (url) => {
    found.urls.push(url);
    return ' ';
  });
  rest = rest.replace(/\[\^([^\]]+)\](:?)/g, (_, label, definition) => {
    found.footnotes.push(`${label}${definition}`);
    return ' ';
  });
  for (const line of rest.split('\n')) {
    const heading = /^(#{1,6})\s/.exec(line);
    if (heading) found.headings.push(heading[1].length);
    const callout = /^>\s*\[!(\w+)\]/.exec(line);
    if (callout) found.callouts.push(callout[1].toUpperCase());
    if (/^\|.*\|\s*$/.test(line)) found.tableRows++;
  }
  // Chinese counts large numbers in 万 (10^4) and 亿 (10^8), so 70 million is 7000 万 and 1.3 billion is 13 亿.
  // Trillion maps directly to 万亿 and keeps its number.
  const scales = { million: 100, billion: 10 };
  rest = rest.replace(/(\d+(?:\.\d+)?)[ -](million|billion)\b/g, (_, n, scale) => ` ${Number((n * scales[scale]).toPrecision(12))} `);
  // Month names count as their numbers, since a translation writes "October 2026" as 2026 年 10 月.
  const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  rest = rest.replace(new RegExp(`\\b(${months.join('|')})\\b`, 'g'), (month) => ` ${months.indexOf(month) + 1} `);
  found.digits = rest.match(/\d+(?:\.\d+)?/g) ?? [];
  return found;
}

const sorted = (list) => [...list].sort();
const sameList = (a, b) => a.length === b.length && a.every((value, i) => value === b[i]);
const describe = (a, b) => {
  const missing = sorted(a).filter((value, i, all) => a.filter((x) => x === value).length > b.filter((x) => x === value).length && all.indexOf(value) === i);
  const extra = sorted(b).filter((value, i, all) => b.filter((x) => x === value).length > a.filter((x) => x === value).length && all.indexOf(value) === i);
  return [missing.length && `missing ${JSON.stringify(missing)}`, extra.length && `extra ${JSON.stringify(extra)}`].filter(Boolean).join(', ');
};

const problems = [];
let compared = 0;
for (const collection of readdirSync(content, { withFileTypes: true }).filter((entry) => entry.isDirectory())) {
  for (const locale of translations) {
    const folder = join(content, collection.name, locale);
    if (!existsSync(folder)) continue;
    for (const name of readdirSync(folder).filter((file) => file.endsWith('.md'))) {
      const file = join(folder, name);
      const originalFile = join(content, collection.name, name);
      if (!existsSync(originalFile)) {
        problems.push(`${file}: no original at ${originalFile}`);
        continue;
      }
      compared++;
      const original = parse(originalFile);
      const translation = parse(file);
      const report = (what, detail) => problems.push(`${file}: ${what}${detail ? ` (${detail})` : ''}`);
      // Order matters for these.
      for (const key of ['code', 'images', 'callouts', 'headings']) {
        if (!sameList(original[key], translation[key])) report(`${key} differ from the original`, key === 'code' ? '' : describe(original[key], translation[key]));
      }
      // These only need the same contents; translation may reorder a sentence.
      for (const key of ['inlineCode', 'math', 'urls', 'internal', 'footnotes', 'digits']) {
        if (!sameList(sorted(original[key]), sorted(translation[key]))) report(`${key} differ from the original`, describe(original[key], translation[key]));
      }
      if (original.tableRows !== translation.tableRows) report(`table rows differ (${original.tableRows} vs ${translation.tableRows})`);
    }
  }
}

let pages = 0;
if (dist) {
  function* htmlFiles(dir) {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) yield* htmlFiles(path);
      else if (entry.name.endsWith('.html')) yield path;
    }
  }
  for (const locale of translations) {
    if (!existsSync(join(dist, locale))) continue;
    for (const file of htmlFiles(join(dist, locale))) {
      pages++;
      const text = readFileSync(file, 'utf8')
        .replace(/<(pre|code|script|style)\b[\s\S]*?<\/\1>/g, '')
        .replace(/<annotation\b[\s\S]*?<\/annotation>/g, '');
      const stray = text.match(/[^\s<>]{0,12}(\*\*|__|[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]\*|\*[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef])[^\s<>]{0,12}/g);
      if (stray) problems.push(`${relative(dist, file)}: emphasis that did not render: ${JSON.stringify(stray.slice(0, 3))}`);
    }
  }
}

console.log(`${compared} translations compared with their originals${dist ? `, ${pages} translated pages scanned` : ''}, ${problems.length} problems`);
for (const problem of problems) console.log(`  ${problem}`);
process.exit(problems.length > 0 ? 1 : 0);
