import type { Locale } from './index';

// Every piece of interface text, in every language. The Chinese table must have exactly the
// shape of the English one, so a missing string is a type error, not a gap on the page.
// Functions that take `p` receive a path localizer for the links they contain.
type Localize = (path: string) => string;

const en = {
  /** The site's description: meta tags, feeds and the default share card. */
  siteDescription:
    "Astrix's notebook on how AI works: deep learning, transformers and agents, with the math, the papers and code you can run.",
  nav: { writing: 'writing', series: 'series', papers: 'papers', notes: 'notes', projects: 'projects', about: 'about' },
  skipToContent: 'Skip to content',
  menuLabel: 'Primary',
  moreLabel: 'More',
  languageLabel: 'Language',
  homeLead: (author: string) =>
    `I'm ${author}, a developer who loves AI and building things with it. Coding agents like Claude Code and Codex ` +
    `are part of my everyday work, and right now I'm digging into how deep learning works underneath: ` +
    `backpropagation, transformers, and the harnesses that turn models into agents. This site is where that work collects.`,
  labels: { series: 'Series', papers: 'Papers', writing: 'Writing', projects: 'Projects', elsewhere: 'Elsewhere' },
  allPapers: 'All papers',
  allPosts: 'All posts',
  allProjects: 'All projects',
  nothingPublished: 'Nothing published yet.',
  nothingHere: 'Nothing here yet.',
  draft: 'draft',
  draftSuffix: ' (draft)',
  minRead: (n: number) => `${n} min read`,
  updatedOn: (date: string) => `updated ${date}`,
  updatedCapital: (date: string) => `Updated ${date}`,
  contents: 'Contents',
  comments: 'Comments',
  writing: {
    title: 'Writing',
    description: 'Every article, newest first, including the chapters of each series.',
    intro: (p: Localize) =>
      `Articles, newest first, including every chapter of the <a href="${p('/series/')}">series</a>. Shorter pieces are in ` +
      `<a href="${p('/notes/')}">notes</a>, and longer original analysis is in <a href="${p('/papers/')}">papers</a>.`,
    browseByTag: 'Browse by tag',
    back: '← writing',
  },
  post: {
    chapter: (n: number) => `Chapter ${n}`,
    references: 'References',
    related: 'Related',
    project: 'Project',
    relatedProjectLabel: 'Related project',
    shareOnX: 'Share on X',
    morePosts: 'More posts',
    chapters: 'Chapters',
    previousChapter: (n: number, title: string) => `← Ch ${n}: ${title}`,
    nextChapter: (n: number, title: string) => `Ch ${n}: ${title} →`,
    allChapters: 'All chapters →',
  },
  tags: {
    title: 'Tags',
    description: 'Posts grouped by tag.',
    none: 'No tags yet.',
    tagged: (tag: string) => `Tagged ${tag}`,
    taggedDescription: (tag: string) => `Posts tagged ${tag}.`,
    back: '← tags',
  },
  series: {
    title: 'Series',
    description: 'Step-by-step series, meant to be read in order.',
    intro:
      'Each series builds one topic chapter by chapter, from first principles to the papers it rests on. They are ' +
      'listed in a suggested reading order, and each page says which series to read first.',
    readingBehind: 'The reading list behind them',
    chapters: (n: number) => `${n} ${n === 1 ? 'chapter' : 'chapters'}`,
    readFirst: ' · read first: ',
    back: '← series',
    chaptersLabel: 'Chapters',
    startWith: (n: number) => `Start with chapter ${n}`,
    chapterShort: (n: number) => `Ch ${n}`,
    coming: 'coming',
  },
  papers: {
    title: 'Papers',
    description: 'Long-form analysis: a question, the derivation, and experiments you can rerun.',
    intro:
      'Longer pieces of analysis, each built around one question: the derivation, experiments you can rerun from the ' +
      'code in the paper, and what it does and does not show. They are not peer-reviewed.',
    back: '← papers',
    notPeerReviewed: 'not peer-reviewed',
    abstract: 'Abstract',
  },
  notes: {
    title: 'Notes',
    description: 'Short notes: one formula, one idea, one paper.',
    intro: (p: Localize) =>
      `Short pieces: one formula, one idea or one paper at a time. Longer explanations are in ` +
      `<a href="${p('/writing/')}">writing</a>, and original analysis is in <a href="${p('/papers/')}">papers</a>.`,
    back: '← notes',
    noteDescription: (date: string) => `A short note from ${date}.`,
    feedTitle: (site: string) => `${site} notes`,
  },
  reading: {
    title: 'Reading list',
    description: 'Every paper and article cited on this site, oldest first.',
    intro:
      'Every paper and article cited here, oldest first, so the list also reads as a short history of the field. ' +
      'Each one links to the pieces that work through it.',
    discussedIn: 'Discussed in ',
  },
  projects: {
    title: 'Projects',
    description: "Things I've built.",
    pushed: (month: string) => `pushed ${month}`,
  },
  footer: { readingList: 'Reading list' },
  byline: {
    writtenBy: (who: string) => `Written by ${who}`,
    howItWorks: 'how this site works',
    aboutAnchor: '#how-this-site-is-written',
  },
  // English is the original language, so it never shows a translation notice.
  translation: {
    notice: (_originalHref: string) => '',
    figures: '',
  },
  notFound: { title: 'Not found', body: 'There is nothing at this address.', home: 'Back home' },
};

export type Strings = typeof en;

const zh: Strings = {
  siteDescription: 'Astrix 关于 AI 如何运作的笔记：深度学习、Transformer 和智能体，附带数学推导、论文出处和可以运行的代码。',
  nav: { writing: '文章', series: '系列', papers: '论文', notes: '笔记', projects: '项目', about: '关于' },
  skipToContent: '跳到正文',
  menuLabel: '主导航',
  moreLabel: '更多',
  languageLabel: '语言',
  homeLead: (author: string) =>
    `我是 ${author}，一名热爱 AI、也喜欢用它做东西的开发者。Claude Code、Codex 这样的编程智能体是我日常工作的一部分。` +
    `眼下我正在深入研究深度学习的底层原理：反向传播、Transformer，以及把模型变成智能体的 harness。这个网站收录的就是这些工作。`,
  labels: { series: '系列', papers: '论文', writing: '文章', projects: '项目', elsewhere: '链接' },
  allPapers: '全部论文',
  allPosts: '全部文章',
  allProjects: '全部项目',
  nothingPublished: '还没有发布内容。',
  nothingHere: '这里还没有内容。',
  draft: '草稿',
  draftSuffix: '（草稿）',
  minRead: (n: number) => `阅读约 ${n} 分钟`,
  updatedOn: (date: string) => `更新于 ${date}`,
  updatedCapital: (date: string) => `更新于 ${date}`,
  contents: '目录',
  comments: '评论',
  writing: {
    title: '文章',
    description: '全部文章，按时间倒序排列，包括各个系列的章节。',
    intro: (p: Localize) =>
      `全部文章按时间倒序排列，包括各个<a href="${p('/series/')}">系列</a>的每一章。` +
      `更短的内容在<a href="${p('/notes/')}">笔记</a>里，篇幅更长的原创分析在<a href="${p('/papers/')}">论文</a>里。`,
    browseByTag: '按标签浏览',
    back: '← 文章',
  },
  post: {
    chapter: (n: number) => `第 ${n} 章`,
    references: '参考文献',
    related: '相关文章',
    project: '项目',
    relatedProjectLabel: '相关项目',
    shareOnX: '分享到 X',
    morePosts: '更多文章',
    chapters: '章节',
    previousChapter: (n: number, title: string) => `← 第 ${n} 章：${title}`,
    nextChapter: (n: number, title: string) => `第 ${n} 章：${title} →`,
    allChapters: '全部章节 →',
  },
  tags: {
    title: '标签',
    description: '按标签分类的文章。',
    none: '还没有标签。',
    tagged: (tag: string) => `标签：${tag}`,
    taggedDescription: (tag: string) => `带有“${tag}”标签的文章。`,
    back: '← 标签',
  },
  series: {
    title: '系列',
    description: '循序渐进的系列，适合按顺序阅读。',
    intro:
      '每个系列一章一章地讲透一个主题，从第一性原理一直讲到它所依据的论文。下面按建议的阅读顺序排列，' +
      '每个系列的页面都会注明应该先读哪个系列。',
    readingBehind: '这些系列背后的阅读清单',
    chapters: (n: number) => `${n} 章`,
    readFirst: ' · 先读：',
    back: '← 系列',
    chaptersLabel: '章节',
    startWith: (n: number) => `从第 ${n} 章开始`,
    chapterShort: (n: number) => `第 ${n} 章`,
    coming: '即将推出',
  },
  papers: {
    title: '论文',
    description: '长篇分析：一个问题、完整的推导，以及可以复现的实验。',
    intro:
      '篇幅更长的分析，每篇围绕一个问题展开：完整的推导、可以用文中代码复现的实验，以及它能说明什么、' +
      '不能说明什么。这些论文未经同行评审。',
    back: '← 论文',
    notPeerReviewed: '未经同行评审',
    abstract: '摘要',
  },
  notes: {
    title: '笔记',
    description: '短笔记：一个公式、一个想法、一篇论文。',
    intro: (p: Localize) =>
      `短小的文章，每篇只讲一个公式、一个想法或一篇论文。更长的讲解在<a href="${p('/writing/')}">文章</a>里，` +
      `原创分析在<a href="${p('/papers/')}">论文</a>里。`,
    back: '← 笔记',
    noteDescription: (date: string) => `${date} 的一则短笔记。`,
    feedTitle: (site: string) => `${site} 笔记`,
  },
  reading: {
    title: '阅读清单',
    description: '本站引用过的全部论文和文章，按时间从早到晚排列。',
    intro:
      '这里列出本站引用过的每一篇论文和文章，按时间从早到晚排列，所以它也可以当作这个领域的一部简史来读。' +
      '每一项都链接到讨论它的文章。',
    discussedIn: '讨论见：',
  },
  projects: {
    title: '项目',
    description: '我做过的东西。',
    pushed: (month: string) => `最近提交 ${month}`,
  },
  footer: { readingList: '阅读清单' },
  byline: {
    writtenBy: (who: string) => `由 ${who} 撰写`,
    howItWorks: '本站内容是怎么写成的',
    aboutAnchor: '#本站内容是怎么写成的',
  },
  translation: {
    notice: (originalHref: string) =>
      `由 Claude 译自<a href="${originalHref}" hreflang="en">英文原文</a>，两者如有出入，以英文版为准。`,
    figures: '插图中的文字仍为英文。',
  },
  notFound: { title: '页面不存在', body: '这个地址上什么也没有。', home: '返回首页' },
};

export const strings: Record<Locale, Strings> = { en, zh };
