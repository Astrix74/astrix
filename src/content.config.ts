import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
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
    /** Drafts show in development only. */
    draft: z.boolean().default(false),
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

export const collections = { posts, notes, pages, projects };
