// Markdown plugins for the Sätteri processor, registered in astro.config.ts.
import katex from 'katex';
import type { SatteriProcessorOptions } from '@astrojs/markdown-satteri';

type MdastPlugin = Extract<NonNullable<SatteriProcessorOptions['mdastPlugins']>[number], { name: string }>;

function renderMath(tex: string, displayMode: boolean, file: URL | undefined): string {
  try {
    return katex.renderToString(tex, { displayMode, throwOnError: true, output: 'htmlAndMathml' });
  } catch (error) {
    // Fail the build rather than publish a formula KaTeX could not read.
    const where = file ? ` in ${decodeURIComponent(file.pathname)}` : '';
    throw new Error(`Could not render the formula "${tex}"${where}: ${(error as Error).message}`);
  }
}

/** `$…$` and `$$…$$` become KaTeX HTML when the site is built, so pages need its CSS but no script. */
export const mathPlugin: MdastPlugin = {
  name: 'katex',
  math: (node, ctx) => ({ type: 'html', value: renderMath(node.value, true, ctx.fileURL) }),
  inlineMath: (node, ctx) => ({ type: 'html', value: renderMath(node.value, false, ctx.fileURL) }),
};

/** An image alone in its paragraph, with a title, becomes a figure: `![alt](./chart.svg "Figure 1. What it shows.")`. */
export const figurePlugin: MdastPlugin = {
  name: 'figures',
  paragraph(node, ctx) {
    const image = node.children[0];
    if (node.children.length !== 1 || image?.type !== 'image' || !image.title) return;
    ctx.appendChild(node, { type: 'emphasis', data: { hName: 'figcaption' }, children: [{ type: 'text', value: image.title }] });
    ctx.setProperty(image, 'title', null);
    ctx.setProperty(node, 'data', { hName: 'figure' });
  },
};

const alertLabels: Record<string, string> = {
  note: 'Note',
  tip: 'Tip',
  important: 'Important',
  warning: 'Warning',
  caution: 'Caution',
};

// Default labels for translations, chosen by the language folder the file is in (content/posts/zh/…).
const translatedAlertLabels: Record<string, Record<string, string>> = {
  zh: { note: '注', tip: '提示', important: '重要', warning: '警告', caution: '注意' },
};

function labelsFor(file: URL | undefined): Record<string, string> {
  const folder = file && /\/content\/[^/]+\/([a-z]{2})\//.exec(file.pathname)?.[1];
  return (folder && translatedAlertLabels[folder]) || alertLabels;
}

/**
 * GitHub-style alerts: a quote whose first line is `[!NOTE]`, `[!TIP]`, `[!IMPORTANT]`, `[!WARNING]`
 * or `[!CAUTION]` becomes a callout. Text after the marker replaces the default label.
 */
export const alertPlugin: MdastPlugin = {
  name: 'alerts',
  blockquote(node, ctx) {
    const paragraph = node.children[0];
    if (paragraph?.type !== 'paragraph') return;
    const lead = paragraph.children[0];
    if (lead?.type !== 'text') return;
    const match = /^\[!(\w+)\][ \t]*(.*)(?:\n|$)/.exec(lead.value);
    if (!match) return;

    const kind = match[1].toLowerCase();
    if (!(kind in alertLabels)) {
      const where = ctx.fileURL ? ` in ${decodeURIComponent(ctx.fileURL.pathname)}` : '';
      throw new Error(`Unknown alert [!${match[1]}]${where}. Use one of: ${Object.keys(alertLabels).join(', ')}.`);
    }

    const rest = lead.value.slice(match[0].length);
    if (rest || paragraph.children.length > 1) ctx.setProperty(lead, 'value', rest);
    else ctx.removeNode(paragraph);

    ctx.prependChild(node, {
      type: 'paragraph',
      data: { hProperties: { className: ['callout-label'] } },
      children: [{ type: 'text', value: match[2].trim() || labelsFor(ctx.fileURL)[kind] }],
    });
    ctx.setProperty(node, 'data', { hName: 'aside', hProperties: { className: ['callout'], dataKind: kind } });
  },
};
