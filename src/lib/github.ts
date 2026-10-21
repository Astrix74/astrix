import type { Loader } from 'astro/loaders';
import { site } from '../config/site';

interface ApiRepo {
  name: string;
  full_name: string;
  owner: { login: string };
  description: string | null;
  html_url: string;
  homepage: string | null;
  language: string | null;
  stargazers_count: number;
  topics?: string[];
  pushed_at: string | null;
  fork: boolean;
  archived: boolean;
  private: boolean;
}

async function api<T>(path: string): Promise<T> {
  const headers: Record<string, string> = {
    Accept: 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
    'User-Agent': `${site.github}-site`,
  };
  const token = process.env.GITHUB_TOKEN ?? import.meta.env.GITHUB_TOKEN;
  if (token) headers.Authorization = `Bearer ${token}`;

  const response = await fetch(`https://api.github.com${path}`, { headers });
  if (!response.ok) throw new Error(`GitHub API answered ${response.status} for ${path}`);
  return (await response.json()) as T;
}

async function ownedRepos(): Promise<ApiRepo[]> {
  const all: ApiRepo[] = [];
  for (let page = 1; ; page++) {
    const batch = await api<ApiRepo[]>(
      `/users/${site.github}/repos?type=owner&sort=pushed&per_page=100&page=${page}`,
    );
    all.push(...batch);
    if (batch.length < 100) return all;
  }
}

/** Repos to show: the `include` list first, then your public repos tagged with the topic, newest push first. */
async function selectRepos(warn: (message: string) => void): Promise<ApiRepo[]> {
  const { include, topic, exclude, includeForks } = site.projects;
  const key = (value: string) => value.toLowerCase();
  const owned = await ownedRepos();
  const picked = new Map<string, ApiRepo>();

  for (const ref of include) {
    const repo =
      owned.find((r) => key(r.name) === key(ref) || key(r.full_name) === key(ref)) ??
      (ref.includes('/') ? await api<ApiRepo>(`/repos/${ref}`) : undefined);
    if (!repo) warn(`Project "${ref}" was not found among ${site.github}'s public repos`);
    else if (repo.private) warn(`Project "${ref}" is private, so it is not shown`);
    else picked.set(key(repo.full_name), repo);
  }

  const hidden = new Set(exclude.map(key));
  for (const repo of owned) {
    if (!repo.topics?.includes(topic)) continue;
    if (hidden.has(key(repo.name)) || hidden.has(key(repo.full_name))) continue;
    if (repo.archived || (repo.fork && !includeForks)) continue;
    if (!picked.has(key(repo.full_name))) picked.set(key(repo.full_name), repo);
  }

  return [...picked.values()];
}

export function githubProjects(): Loader {
  return {
    name: 'github-projects',
    load: async ({ store, parseData, generateDigest, logger }) => {
      let repos: ApiRepo[];
      try {
        repos = await selectRepos((message) => logger.warn(message));
      } catch (error) {
        logger.warn(`Could not reach GitHub, keeping the previous project data. ${(error as Error).message}`);
        return;
      }

      store.clear();
      for (const [order, repo] of repos.entries()) {
        const id = key(repo.owner.login) === key(site.github) ? repo.name : repo.full_name;
        const data = await parseData({
          id,
          data: {
            name: repo.name,
            fullName: repo.full_name,
            description: repo.description ?? '',
            url: repo.html_url,
            homepage: repo.homepage || undefined,
            language: repo.language ?? undefined,
            stars: repo.stargazers_count,
            topics: repo.topics ?? [],
            pushedAt: repo.pushed_at ?? undefined,
            order,
          },
        });
        store.set({ id, data, digest: generateDigest(data) });
      }
      logger.info(`Loaded ${repos.length} project(s) from GitHub`);
    },
  };
}

function key(value: string) {
  return value.toLowerCase();
}
