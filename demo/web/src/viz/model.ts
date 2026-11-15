// The arithmetic behind the document views (meaning map, seismograph,
// term river), kept apart from the drawing so it can be tested: which
// units are the points and which are regions, convex hulls, the terms
// that tell a chunk from the rest, stream stacking, and cutting the
// text into linked pieces. Offsets are UTF-16 units, as the page asks
// the server for them, so they index a JavaScript string directly.

/** A projected unit: a chunk at some level, with its span and place. */
export type Unit = { item: number; start: number; end: number; level: number; x: number; y: number; z: number; empty: boolean };

/** One term of a stack. */
export type Term = { term: string; weight: number; occStart?: number[]; occEnd?: number[] };

/** The units of the finest level present, in reading order. */
export function leaves(units: Unit[]): Unit[] {
  if (!units.length) return [];
  const finest = Math.min(...units.map((u) => u.level));
  return units.filter((u) => u.level === finest).sort((a, b) => a.start - b.start || a.end - b.end);
}

/**
 * The coarser units worth drawing as a region: each with the positions
 * (in `leaves` order) of the leaves inside its span. A unit holding
 * fewer than two leaves, or all of them, is left out: it would circle
 * one point or the whole picture.
 */
export function regions(units: Unit[], leafUnits: Unit[]): { unit: Unit; members: number[] }[] {
  if (!leafUnits.length) return [];
  const finest = leafUnits[0].level;
  const out: { unit: Unit; members: number[] }[] = [];
  for (const u of units) {
    if (u.level === finest) continue;
    const members = leafUnits.flatMap((l, i) => (l.start >= u.start && l.end <= u.end && !l.empty ? [i] : []));
    if (members.length >= 2 && members.length < leafUnits.filter((l) => !l.empty).length) out.push({ unit: u, members });
  }
  return out.sort((a, b) => a.unit.start - b.unit.start);
}

/** The convex hull of `pts`, counter-clockwise, by the monotone chain. Fewer than three distinct points come back as they are. */
export function hull(pts: [number, number][]): [number, number][] {
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]).filter((q, i, all) => i === 0 || q[0] !== all[i - 1][0] || q[1] !== all[i - 1][1]);
  if (p.length < 3) return p;
  const cross = (o: [number, number], a: [number, number], b: [number, number]): number => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (src: [number, number][]): [number, number][] => {
    const h: [number, number][] = [];
    for (const q of src) {
      while (h.length >= 2 && cross(h[h.length - 2], h[h.length - 1], q) <= 0) h.pop();
      h.push(q);
    }
    h.pop();
    return h;
  };
  return [...half(p), ...half([...p].reverse())];
}

/** A linear map from [lo, hi] onto [a, b]; a flat domain lands in the middle. */
export function scale(lo: number, hi: number, a: number, b: number): (v: number) => number {
  return hi === lo ? () => (a + b) / 2 : (v) => a + ((v - lo) / (hi - lo)) * (b - a);
}

/** In how many of `chunks` each term is listed. */
export function spread(chunks: Term[][]): Map<string, number> {
  const seen = new Map<string, number>();
  for (const c of chunks) for (const t of new Set(c.map((x) => x.term))) seen.set(t, (seen.get(t) ?? 0) + 1);
  return seen;
}

// How much a term tells one chunk from the rest: its share of the chunk
// times the log of how much larger that is than its share of the
// document, times the log of how few of the `count` chunks list it. A
// word every chunk uses scores nothing, in any language, with no stop
// list. A term the document's stack does not list is rarer than
// everything it does list.
function telling(here: number, there: number, listedIn: number, count: number): number {
  const rare = count > 1 ? Math.log(count / Math.max(1, listedIn)) : 1;
  return here * Math.log(here / there) * rare;
}

/** The `k` terms that tell a chunk from its document; `listed` is `spread` over the `count` chunks. */
export function distinctive(chunk: Term[], chunkTotal: number, doc: Term[], docTotal: number, k: number, listed: Map<string, number> = new Map(), count = 1): string[] {
  if (!chunk.length || chunkTotal <= 0 || docTotal <= 0) return [];
  const shares = new Map(doc.map((t) => [t.term, t.weight / docTotal]));
  const floor = doc.length ? Math.min(...doc.map((t) => t.weight / docTotal)) : 1;
  return chunk
    .map((t) => ({ term: t.term, score: telling(t.weight / chunkTotal, shares.get(t.term) ?? floor, listed.get(t.term) ?? 1, count) }))
    .filter((t) => t.score > 0)
    .sort((a, b) => b.score - a.score || (a.term < b.term ? -1 : 1))
    .slice(0, k)
    .map((t) => t.term);
}

/**
 * The document's terms that the river draws, at most `k`, in the order
 * of the chunk each is heaviest in, so the streams hand off from left
 * to right. With `uneven` they are the terms that best tell some chunk
 * from the rest; otherwise the heaviest. `chunks` are the leaf chunks'
 * stacks in reading order.
 */
export function riverTerms(doc: Term[], docTotal: number, chunks: { terms: Term[]; total: number }[], k: number, uneven: boolean): string[] {
  let picked: string[];
  if (uneven) {
    const shares = new Map(doc.map((t) => [t.term, t.weight / docTotal]));
    const listed = spread(chunks.map((c) => c.terms));
    const best = new Map<string, number>();
    for (const c of chunks) {
      for (const t of c.terms) {
        const there = shares.get(t.term);
        if (there === undefined || c.total <= 0) continue;
        best.set(t.term, Math.max(best.get(t.term) ?? 0, telling(t.weight / c.total, there, listed.get(t.term) ?? 1, chunks.length)));
      }
    }
    picked = [...best].filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, k).map(([t]) => t);
  } else {
    picked = [...doc].sort((a, b) => b.weight - a.weight).slice(0, k).map((t) => t.term);
  }
  const peak = (term: string): number => {
    const w = chunks.map((c) => c.terms.find((t) => t.term === term)?.weight ?? 0);
    return w.indexOf(Math.max(...w));
  };
  return picked.map((t, i) => ({ t, i, at: peak(t) })).sort((a, b) => a.at - b.at || a.i - b.i).map((x) => x.t);
}

/**
 * Streams stacked about a centre line. `weights[t][c]` is term `t` in
 * chunk `c`; the answer has, for each term, its lower and upper edge in
 * each chunk, in the weights' own unit, with the stack centred on 0.
 */
export function streams(weights: number[][]): { lo: number[]; hi: number[] }[] {
  const cols = weights[0]?.length ?? 0;
  const out = weights.map(() => ({ lo: new Array<number>(cols).fill(0), hi: new Array<number>(cols).fill(0) }));
  for (let c = 0; c < cols; c++) {
    let at = -weights.reduce((s, w) => s + w[c], 0) / 2;
    weights.forEach((w, t) => {
      out[t].lo[c] = at;
      at += w[c];
      out[t].hi[c] = at;
    });
  }
  return out;
}

/** A piece of the text: inside leaf chunk `chunk` (its position in reading order) or between chunks, marked or not. */
export type Piece = { text: string; start: number; chunk: number | null; marked: boolean };

/**
 * The text cut at every chunk edge and every mark edge, in order.
 * Chunks do not overlap; marks may, and may cross a chunk edge.
 */
export function pieces(text: string, chunks: { start: number; end: number }[], marks: [number, number][]): Piece[] {
  const clamp = (v: number): number => Math.max(0, Math.min(text.length, v));
  const edges = new Set<number>([0, text.length]);
  for (const c of chunks) edges.add(clamp(c.start)).add(clamp(c.end));
  for (const [s, e] of marks) edges.add(clamp(s)).add(clamp(e));
  const cuts = [...edges].sort((a, b) => a - b);
  const out: Piece[] = [];
  for (let i = 0; i + 1 < cuts.length; i++) {
    const [s, e] = [cuts[i], cuts[i + 1]];
    if (e <= s) continue;
    const chunk = chunks.findIndex((c) => s >= c.start && e <= c.end);
    out.push({ text: text.slice(s, e), start: s, chunk: chunk < 0 ? null : chunk, marked: marks.some(([a, b]) => s >= a && e <= b) });
  }
  return out;
}

/** The text before and after the gap at `gaps[i]`, each reaching to the neighbouring gap (or the text's edge), trimmed. */
export function aroundGap(text: string, gaps: number[], i: number): [string, string] {
  const at = gaps[i];
  const from = i > 0 ? gaps[i - 1] : 0;
  const to = i + 1 < gaps.length ? gaps[i + 1] : text.length;
  return [text.slice(from, at).trim(), text.slice(at, to).trim()];
}

/** Why the chunker may cut at a gap, by the code the server sends. */
export function cutReason(code: number): string {
  return ["document start", "sentence end", "section", "heading", "paragraph", "topic shift", "clause", "list item", "inside a sentence", "overlap"][code] ?? `reason ${code}`;
}
