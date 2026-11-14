// The arithmetic behind the similarity matrix and the query spotlight,
// kept apart from the drawing so it can be tested: reading a cosine out
// of the upper triangle the server sends, shading within a document's
// own range, the diagonal blocks, and the items nearest a query.

/** The cosine between positions `a` and `b` of `n` items, from the strict upper triangle in row-major order; an item with itself is 1. */
export function cosineAt(upper: number[], n: number, a: number, b: number): number {
  if (a === b) return 1;
  const [i, j] = a < b ? [a, b] : [b, a];
  return upper[(i * (2 * n - i - 1)) / 2 + (j - i - 1)];
}

/** The least and greatest of `values`; nothing gives [0, 1]. */
export function extent(values: number[]): [number, number] {
  if (!values.length) return [0, 1];
  let lo = values[0];
  let hi = values[0];
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return [lo, hi];
}

/** Where `v` sits in [lo, hi], from 0 to 1; a flat range is the middle. */
export function share(v: number, lo: number, hi: number): number {
  if (hi === lo) return 0.5;
  return Math.min(1, Math.max(0, (v - lo) / (hi - lo)));
}

/**
 * The diagonal blocks: for each group of leaf positions, the first and
 * last matrix rows it covers. `rowOf` gives a leaf's row, and a leaf
 * without one (no vector) is skipped; a group left with fewer than two
 * rows draws nothing.
 */
export function blocks(groups: number[][], rowOf: Map<number, number>): { from: number; to: number }[] {
  const out: { from: number; to: number }[] = [];
  for (const g of groups) {
    const rows = g.flatMap((leaf) => (rowOf.has(leaf) ? [rowOf.get(leaf)!] : []));
    if (rows.length >= 2) out.push({ from: Math.min(...rows), to: Math.max(...rows) });
  }
  return out;
}

/** The `k` positions with the greatest cosine, greatest first, the earlier of equals first; `undefined` entries take no part. */
export function nearest(cosines: (number | undefined)[], k: number): { at: number; cosine: number }[] {
  const all = cosines.flatMap((c, at) => (c === undefined ? [] : [{ at, cosine: c }]));
  return all.sort((a, b) => b.cosine - a.cosine || a.at - b.at).slice(0, Math.max(0, k));
}

/** Which labels to draw along an axis of `n` rows so they stay apart: every row up to 24, then every 5th, 10th, 25th or 50th, with the first and the last. */
export function labelled(n: number): number[] {
  const every = n <= 24 ? 1 : n <= 60 ? 5 : n <= 140 ? 10 : n <= 300 ? 25 : 50;
  const out: number[] = [];
  for (let i = 0; i < n; i++) if (i === 0 || (i + 1) % every === 0) out.push(i);
  return out;
}

type Lab = [number, number, number];

function channel(hex: string, at: number): number {
  const v = parseInt(hex.slice(at, at + 2), 16) / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

function toLab(hex: string): Lab | null {
  const h = hex.trim();
  if (h.length !== 7 || h[0] !== "#" || Number.isNaN(parseInt(h.slice(1), 16))) return null;
  const [r, g, b] = [channel(h, 1), channel(h, 3), channel(h, 5)];
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}

function byte(linear: number): number {
  const v = linear <= 0.0031308 ? 12.92 * linear : 1.055 * Math.pow(linear, 1 / 2.4) - 0.055;
  return Math.round(255 * Math.min(1, Math.max(0, v)));
}

/**
 * The colour `t` of the way from `from` to `to` (both `#rrggbb`), mixed
 * in OKLab so equal steps look equal, as `rgb(r g b)`. This is what
 * `color-mix(in oklab, ...)` gives, for a canvas, which takes no
 * custom properties. A colour in another form gives a mid grey.
 */
export function mix(from: string, to: string, t: number): string {
  const [a, b] = [toLab(from), toLab(to)];
  if (!a || !b) return "rgb(128 128 128)";
  const [L, A, B] = [0, 1, 2].map((i) => a[i] + (b[i] - a[i]) * Math.min(1, Math.max(0, t)));
  const l = Math.pow(L + 0.3963377774 * A + 0.2158037573 * B, 3);
  const m = Math.pow(L - 0.1055613458 * A - 0.0638541728 * B, 3);
  const s = Math.pow(L - 0.0894841775 * A - 1.291485548 * B, 3);
  return `rgb(${byte(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s)} ${byte(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s)} ${byte(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s)})`;
}
