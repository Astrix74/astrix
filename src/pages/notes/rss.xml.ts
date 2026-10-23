import type { APIContext } from 'astro';
import { notesFeed } from '../../lib/feeds';

export const GET = (context: APIContext) => notesFeed('en', context.site!);
