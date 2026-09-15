// Resolves a path in `public/` against Vite's configured base URL, so
// assets keep working when the site is served from a subpath (GitHub Pages
// project sites) instead of the domain root.
export function asset(path: string): string {
  const base = import.meta.env.BASE_URL
  return `${base}${path.replace(/^\//, '')}`
}
