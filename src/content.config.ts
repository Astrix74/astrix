import { defineCollection, reference } from 'astro:content';
import { file, glob } from 'astro/loaders';
import { z } from 'astro/zod';
import { githubProjects } from './lib/github';

const posts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/posts' }),
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      description: z.string(),
      date: z.coerce.date(),
      updated: z.coerce.date().optional(),
      tags: z.array(z.string()).default([]),
      /** The project this post is about: a repo name from the projects list, or "owner/name". */
      project: z.string().optional(),
      /** The series this post is a chapter of, by file name in src/content/series. Needs `chapter`. */
      series: reference('series').optional(),
      /** The chapter number within the series, starting at 1. */
      chapter: z.number().int().positive().optional(),
      /** Citation keys from src/content/references.yaml, listed under the post. */
      references: z.array(reference('references')).default([]),
      /** Overrides site.writtenBy for this post. An empty string hides the line. */
      writtenBy: z.string().optional(),
      /** Used as the share preview instead of the generated image. */
      cover: image().optional(),
      /** Set when the post was first published somewhere else. */
      canonical: z.url().optional(),
      /** Drafts show in development only. */
      draft: z.boolean().default(false),
    }),
});

const notes = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/notes' }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    /** Citation keys from src/content/references.yaml, listed under the note. */
    references: z.array(reference('references')).default([]),
    /** Overrides site.writtenBy for this note. An empty string hides the line. */
    writtenBy: z.string().optional(),
    /** Drafts show in development only. */
    draft: z.boolean().default(false),
  }),
});

/** Long-form original analysis: a question, derivations and experiments the reader can rerun. */
const papers = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/papers' }),
  schema: z.object({
    title: z.string(),
    /** One sentence for lists, feeds and search results. */
    description: z.string(),
    /** The question, what was done and what was found, shown at the top of the paper. Plain text. */
    abstract: z.string(),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    /** Citation keys from src/content/references.yaml, listed under the paper. */
    references: z.array(reference('references')).default([]),
    /** Overrides site.writtenBy for this paper. An empty string hides the line. */
    writtenBy: z.string().optional(),
    /** Drafts show in development only. */
    draft: z.boolean().default(false),
  }),
});

const series = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/series' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    /** Place in the reading path: series 1 is meant to be read first. */
    order: z.number(),
    /** Series worth reading before this one. */
    prerequisites: z.array(reference('series')).default([]),
    /** Chapters not written yet, listed as coming. */
    planned: z.array(z.string()).default([]),
  }),
});

/** Works cited by posts, notes and papers: the reading list. */
const references = defineCollection({
  loader: file('src/content/references.yaml'),
  schema: z.object({
    title: z.string(),
    authors: z.string(),
    year: z.number().int(),
    /** Where it was formally published, if anywhere. */
    venue: z.string().optional(),
    url: z.url(),
  }),
});

const pages = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/pages' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    updated: z.coerce.date().optional(),
    /** Adds a link to this page in the footer. */
    footer: z.boolean().default(false),
  }),
});

const projects = defineCollection({
  loader: githubProjects(),
  schema: z.object({
    name: z.string(),
    fullName: z.string(),
    description: z.string().default(''),
    url: z.url(),
    homepage: z.string().optional(),
    language: z.string().optional(),
    stars: z.number(),
    topics: z.array(z.string()).default([]),
    pushedAt: z.coerce.date().optional(),
    order: z.number(),
  }),
});

export const collections = { posts, notes, papers, series, references, pages, projects };
