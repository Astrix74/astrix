import type { APIContext } from 'astro';
import { writingFeed } from '../lib/feeds';

export const GET = (context: APIContext) => writingFeed('en', context.site!);
