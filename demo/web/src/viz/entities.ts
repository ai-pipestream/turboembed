// The arithmetic behind the entity views: mentions gathered into
// entities, relations between mentions turned into edges between
// entities, a small deterministic layout for the graph, and the spans
// that make the chunk band. Kept apart from the drawing so it can be
// tested.

export type Mention = { start: number; end: number; kind: string; canonical: string; text: string };
export type Entity = { name: string; kind: string; mentions: number[] };
export type Relation = { subject: number; object: number; kind: string; score?: number };
export type Edge = { from: number; to: number; kind: string; support: number; score?: number };

// Two mentions are one entity when their kind and their name agree,
// whatever the case and however the letters are composed. The name is
// the canonical form when the annotator gave one, else the text.
function keyOf(m: Mention): string {
  const name = (m.canonical || m.text).normalize("NFKC").toLocaleLowerCase().split(" ").filter((w) => w.length).join(" ");
  return `${m.kind}\u0000${name}`;
}

/**
 * The entities of `mentions`, most mentioned first (ties in reading
 * order), and for each mention the position of its entity.
 */
export function gather(mentions: Mention[]): { entities: Entity[]; of: number[] } {
  const groups = new Map<string, number[]>();
  mentions.forEach((m, i) => groups.set(keyOf(m), [...(groups.get(keyOf(m)) ?? []), i]));
  const ordered = [...groups.values()].sort((a, b) => b.length - a.length || a[0] - b[0]);
  const of = new Array<number>(mentions.length).fill(-1);
  const entities = ordered.map((members, e) => {
    for (const i of members) of[i] = e;
    const first = mentions[members[0]];
    // The spelling used most, the earliest on a tie.
    const counts = new Map<string, number>();
    for (const i of members) counts.set(mentions[i].text, (counts.get(mentions[i].text) ?? 0) + 1);
    const common = [...counts].sort((a, b) => b[1] - a[1])[0][0];
    return { name: first.canonical || common, kind: first.kind, mentions: members };
  });
  return { entities, of };
}

/**
 * Relations between mentions as edges between entities: the same pair
 * and kind met again adds to `support` and keeps the best score. A
 * relation naming a mention that is not there, or an entity with
 * itself, is left out.
 */
export function link(relations: Relation[], of: number[]): Edge[] {
  const edges = new Map<string, Edge>();
  for (const r of relations) {
    const from = of[r.subject];
    const to = of[r.object];
    if (from === undefined || to === undefined || from < 0 || to < 0 || from === to) continue;
    const key = `${from}\u0000${to}\u0000${r.kind}`;
    const e = edges.get(key);
    if (e) {
      e.support += 1;
      if (r.score !== undefined) e.score = Math.max(e.score ?? r.score, r.score);
    } else edges.set(key, { from, to, kind: r.kind, support: 1, score: r.score });
  }
  return [...edges.values()];
}

/**
 * Places `n` nodes in a box `w` by `h`: they start on a circle, linked
 * nodes pull together and all nodes push apart, for a fixed number of
 * steps. The same input always gives the same picture.
 */
export function layout(n: number, edges: { from: number; to: number }[], w: number, h: number, pad = 40): [number, number][] {
  if (n === 0) return [];
  if (n === 1) return [[w / 2, h / 2]];
  const rx = w / 2 - pad;
  const ry = h / 2 - pad;
  const at = Array.from({ length: n }, (_, i): [number, number] => [w / 2 + rx * Math.cos((2 * Math.PI * i) / n - Math.PI / 2), h / 2 + ry * Math.sin((2 * Math.PI * i) / n - Math.PI / 2)]);
  const ideal = 0.8 * Math.sqrt((rx * ry * 4) / n);
  for (let step = 0; step < 200; step++) {
    const heat = (1 - step / 200) * ideal * 0.5;
    const move = at.map((): [number, number] => [0, 0]);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = at[i][0] - at[j][0];
        const dy = at[i][1] - at[j][1];
        const d = Math.max(1, Math.hypot(dx, dy));
        const push = (ideal * ideal) / d;
        move[i][0] += (dx / d) * push;
        move[i][1] += (dy / d) * push;
        move[j][0] -= (dx / d) * push;
        move[j][1] -= (dy / d) * push;
      }
    }
    for (const e of edges) {
      const dx = at[e.from][0] - at[e.to][0];
      const dy = at[e.from][1] - at[e.to][1];
      const d = Math.max(1, Math.hypot(dx, dy));
      const pull = (d * d) / (ideal * 2.5);
      move[e.from][0] -= (dx / d) * pull;
      move[e.from][1] -= (dy / d) * pull;
      move[e.to][0] += (dx / d) * pull;
      move[e.to][1] += (dy / d) * pull;
    }
    for (let i = 0; i < n; i++) {
      // A spring to the middle keeps unlinked nodes together as a cloud, not pressed to the edges.
      const cx = w / 2 - at[i][0];
      const cy = (h / 2 - at[i][1]) * (rx / ry);
      const far = Math.hypot(cx, cy);
      move[i][0] += (cx * far) / (ideal * 6);
      move[i][1] += (cy * far) / (ideal * 6);
      const d = Math.max(1e-9, Math.hypot(move[i][0], move[i][1]));
      const by = Math.min(d, heat);
      at[i][0] = Math.max(pad, Math.min(w - pad, at[i][0] + (move[i][0] / d) * by));
      at[i][1] = Math.max(pad, Math.min(h - pad, at[i][1] + (move[i][1] / d) * by));
    }
  }
  // Last, nodes that ended too close for their names to be read are eased apart.
  const gap = Math.min(64, ideal);
  for (let step = 0; step < 40; step++) {
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = at[i][0] - at[j][0] || 0.5 - ((i * 7 + j) % 2);
        const dy = at[i][1] - at[j][1] || 0.5 - ((i + j * 3) % 2);
        const d = Math.hypot(dx, dy);
        if (d >= gap) continue;
        const by = (gap - d) / 2;
        at[i] = [Math.max(pad, Math.min(w - pad, at[i][0] + (dx / d) * by)), Math.max(pad, Math.min(h - pad, at[i][1] + (dy / d) * by))];
        at[j] = [Math.max(pad, Math.min(w - pad, at[j][0] - (dx / d) * by)), Math.max(pad, Math.min(h - pad, at[j][1] - (dy / d) * by))];
      }
    }
  }
  return at;
}

/**
 * The pairs of entities mentioned in one sentence, with how many
 * sentences hold both: what the text itself says about who goes with
 * whom, when no relation model has spoken. `from` is the lower index.
 */
export function together(mentions: Mention[], of: number[], sentences: { start: number; end: number }[]): { from: number; to: number; count: number }[] {
  const pairs = new Map<string, { from: number; to: number; count: number }>();
  for (const s of sentences) {
    const inside = [...new Set(mentions.flatMap((m, i) => (m.start >= s.start && m.end <= s.end && of[i] >= 0 ? [of[i]] : [])))].sort((a, b) => a - b);
    for (let i = 0; i < inside.length; i++) {
      for (let j = i + 1; j < inside.length; j++) {
        const key = `${inside[i]}\u0000${inside[j]}`;
        const p = pairs.get(key) ?? { from: inside[i], to: inside[j], count: 0 };
        p.count += 1;
        pairs.set(key, p);
      }
    }
  }
  return [...pairs.values()];
}

/** The spans that hold no other span: the finest cut of a layer whose spans nest. In reading order. */
export function finest(spans: { start: number; end: number }[]): { start: number; end: number }[] {
  return spans
    .filter((a) => a.end > a.start && !spans.some((b) => b !== a && b.end > b.start && b.start >= a.start && b.end <= a.end && (b.start !== a.start || b.end !== a.end)))
    .filter((a, i, all) => all.findIndex((b) => b.start === a.start && b.end === a.end) === i)
    .sort((a, b) => a.start - b.start);
}

/** The span among `spans` that holds `offset`, as text; without one, the text around it. */
export function sentenceAt(text: string, spans: { start: number; end: number }[], offset: number): string {
  const s = spans.find((x) => offset >= x.start && offset < x.end);
  return s ? text.slice(s.start, s.end).trim() : text.slice(Math.max(0, offset - 60), offset + 60).trim();
}

/** The kinds in the order they first appear; past `cap` they share one "other" slot. */
export function kindSlots(kinds: string[], cap: number): Map<string, number> {
  const slots = new Map<string, number>();
  for (const k of kinds) if (!slots.has(k)) slots.set(k, Math.min(slots.size, cap - 1));
  return slots;
}
