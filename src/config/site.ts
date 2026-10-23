// Everything you are likely to change on the site lives in this file.
// Interface text in every language lives in src/i18n/strings.ts.
import type { Locale } from '../i18n';
import type { Strings } from '../i18n/strings';

export interface NavItem {
  /** The menu label's key in src/i18n/strings.ts. */
  key: keyof Strings['nav'];
  /** The English path; other languages get it under their prefix. */
  href: string;
}

export interface Social {
  label: string;
  /** An empty href hides the link everywhere it would appear. */
  href: string;
}

export interface SiteConfig {
  title: string;
  author: string;
  /** Who writes the posts, notes and papers, in each language, shown under each title. An empty string hides the line. */
  writtenBy: Record<Locale, string>;
  /** GitHub username. Projects are read from this account at build time. */
  github: string;
  nav: NavItem[];
  socials: Social[];
  /** Cloudflare Web Analytics. Paste the site token from the dashboard; empty turns it off. */
  analytics: { cloudflareToken: string };
  /** giscus comments on posts, stored in GitHub Discussions. An empty repoId or categoryId turns them off. */
  comments: { repo: string; repoId: string; category: string; categoryId: string };
  projects: {
    /** Repos always shown, in this order: "name" for your own, "owner/name" for someone else's. Forks are allowed here. */
    include: string[];
    /** Any other public repo of yours carrying this GitHub topic is shown too. */
    topic: string;
    /** Repos hidden even when they carry the topic. */
    exclude: string[];
    includeForks: boolean;
    /** How many projects the home page shows. */
    homeCount: number;
  };
}

const github = 'Astrix74';

export const site: SiteConfig = {
  title: 'Astrix',
  author: 'Astrix',
  writtenBy: { en: 'Claude, an AI model by Anthropic', zh: 'Anthropic 的 AI 模型 Claude' },
  github,
  nav: [
    { key: 'writing', href: '/writing/' },
    { key: 'series', href: '/series/' },
    { key: 'papers', href: '/papers/' },
    { key: 'notes', href: '/notes/' },
    { key: 'projects', href: '/projects/' },
    { key: 'about', href: '/about/' },
  ],
  socials: [
    { label: 'GitHub', href: `https://github.com/${github}` },
    { label: 'X', href: '' },
    { label: 'RSS', href: '/rss.xml' },
    { label: 'Email', href: '' },
  ],
  analytics: { cloudflareToken: '' },
  comments: { repo: `${github}/astrix`, repoId: '', category: 'Announcements', categoryId: '' },
  projects: {
    include: [],
    topic: 'showcase',
    exclude: [],
    includeForks: false,
    homeCount: 3,
  },
};

export const socials = site.socials.filter((social) => social.href);
