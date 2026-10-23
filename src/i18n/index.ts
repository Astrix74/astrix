// Languages the site is published in. English is the default and keeps unprefixed URLs;
// every other language lives under /<locale>/.
export const locales = ['en', 'zh'] as const;
export type Locale = (typeof locales)[number];
export const defaultLocale: Locale = 'en';

/** BCP 47 tags, used for html lang and hreflang. */
export const languageTag: Record<Locale, string> = { en: 'en', zh: 'zh-CN' };
/** Open Graph locale codes. */
export const ogLocale: Record<Locale, string> = { en: 'en_US', zh: 'zh_CN' };
/** Each language's name in itself, for the language switcher. */
export const languageName: Record<Locale, string> = { en: 'English', zh: '中文' };

/**
 * A translation lives in a folder named after its locale, next to the original:
 * posts/zh/scaling-laws.md has the id "zh/scaling-laws" and translates posts/scaling-laws.md.
 */
export function localeOf(id: string): Locale {
  const prefix = id.split('/')[0];
  return locales.find((locale) => locale !== defaultLocale && locale === prefix) ?? defaultLocale;
}

/** The id without its locale folder: the part shared by an entry and its translations. */
export function slugOf(id: string): string {
  return localeOf(id) === defaultLocale ? id : id.slice(id.indexOf('/') + 1);
}

export function idFor(locale: Locale, slug: string): string {
  return locale === defaultLocale ? slug : `${locale}/${slug}`;
}

/** A site path in the given locale: "/writing/x/" becomes "/zh/writing/x/". */
export function localize(locale: Locale, path: string): string {
  return locale === defaultLocale ? path : `/${locale}${path}`;
}

/** The entries written in one locale. */
export function inLocale<T extends { id: string }>(entries: T[], locale: Locale): T[] {
  return entries.filter((entry) => localeOf(entry.id) === locale);
}

/** Paths of the same page in each locale that has it, keyed by locale. */
export type Alternates = Partial<Record<Locale, string>>;

/** A page that exists in every locale, such as an index page. */
export function everyLocale(path: string): Alternates {
  return Object.fromEntries(locales.map((locale) => [locale, localize(locale, path)]));
}
