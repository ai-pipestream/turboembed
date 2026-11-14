// The meaning map: every chunk vector as a dot at its two main
// directions of meaning, joined in reading order, coloured from the
// first chunk to the last, each named by the terms that tell it apart.
// A coarser unit that holds some of the dots is a shaded region.

import { el } from "../dom.js";
import type { Link } from "../viz/link.js";
import { extent, nearest, share } from "../viz/matrix.js";
import { hull, regions, scale, type Unit } from "../viz/model.js";
import { figure, n, ordered, s } from "../viz/svg.js";
import { short } from "./embed.js";

const W = 640;
const H = 340;
const PAD = { l: 34, r: 16, t: 16, b: 34 };

export type MapInput = {
  /** Every projected unit. */
  units: Unit[];
  /** The finest level, in reading order. */
  chunks: Unit[];
  /** The share of the variance each axis carries. */
  variance: number[];
  /** The terms that name chunk `i` (its position in `chunks`). */
  terms: (i: number) => string[];
  /** The cosine between chunk `i` and the next, when the server gave one. */
  toNext: (i: number) => number | undefined;
  text: string;
  /** Three components asked for: the third is the dot's size. */
  third: boolean;
  /** A query placed among the chunks, when one was asked. */
  spot?: Spot;
};

/** A query in the map's own space: where it lands, and its cosine to chunk `i` (its position in `chunks`). */
export type Spot = { text: string; x: number; y: number; cosine: (i: number) => number | undefined };

export function meaningMap(m: MapInput, link: Link): HTMLElement {
  const placed = m.chunks.map((c, i) => ({ c, i })).filter((p) => !p.c.empty);
  // The query is part of the picture, so the axes make room for it.
  const xs = [...placed.map((p) => p.c.x), ...(m.spot ? [m.spot.x] : [])];
  const ys = [...placed.map((p) => p.c.y), ...(m.spot ? [m.spot.y] : [])];
  const zs = placed.map((p) => p.c.z);
  const sx = scale(Math.min(...xs), Math.max(...xs), PAD.l + 24, W - PAD.r - 24);
  const sy = scale(Math.min(...ys), Math.max(...ys), H - PAD.b - 24, PAD.t + 24);
  const sz = scale(Math.min(...zs), Math.max(...zs), 5, 11);
  const at = new Map(placed.map((p) => [p.i, [sx(p.c.x), sy(p.c.y)] as [number, number]]));
  const pct = (v: number | undefined): string => (v === undefined ? "" : ` (${(v * 100).toFixed(0)}% of the variance)`);
  const near = m.spot ? m.chunks.map((_, i) => m.spot!.cosine(i)) : [];
  const [lo, hi] = extent(near.flatMap((c) => (c === undefined ? [] : [c])));
  const svg = figure(W, H, `Meaning map of ${placed.length} chunks${m.spot ? `, with the query "${short(m.spot.text, 60)}" placed among them` : ""}`);
  svg.append(
    s("rect", { x: PAD.l, y: PAD.t, width: W - PAD.l - PAD.r, height: H - PAD.t - PAD.b, class: "frame" }),
    s("text", { x: (PAD.l + W - PAD.r) / 2, y: H - 10, class: "axis", "text-anchor": "middle" }, `first direction of meaning${pct(m.variance[0])}`),
    s("text", { x: 12, y: (PAD.t + H - PAD.b) / 2, class: "axis", "text-anchor": "middle", transform: `rotate(-90 12 ${(PAD.t + H - PAD.b) / 2})` }, `second direction${pct(m.variance[1])}`),
  );
  // Regions first, so the dots sit on them.
  for (const r of regions(m.units, m.chunks)) {
    const pts = r.members.flatMap((i) => (at.has(i) ? [at.get(i)!] : []));
    const h = hull(pts);
    const g = s("g", { class: "region" });
    if (h.length >= 3) g.append(s("polygon", { points: h.map(([x, y]) => `${n(x)},${n(y)}`).join(" ") }));
    else if (h.length === 2) g.append(s("line", { x1: n(h[0][0]), y1: n(h[0][1]), x2: n(h[1][0]), y2: n(h[1][1]) }));
    g.append(s("title", {}, `Level ${r.unit.level} unit ${r.unit.item}: chunks ${r.members[0] + 1} to ${r.members[r.members.length - 1] + 1}`));
    svg.append(g);
  }
  if (placed.length > 1) svg.append(s("polyline", { class: "trail", points: placed.map((p) => at.get(p.i)!.map(n).join(",")).join(" ") }));
  if (m.spot) {
    // Rays to the nearest few by cosine, which is not always the nearest on the page.
    const [qx, qy] = [sx(m.spot.x), sy(m.spot.y)];
    for (const hit of nearest(near, Math.min(3, Math.max(1, Math.floor(placed.length / 2))))) {
      const to = at.get(hit.at);
      if (to) svg.append(s("line", { x1: n(qx), y1: n(qy), x2: n(to[0]), y2: n(to[1]), class: "ray" }));
    }
  }
  const info = el("p", { class: "info", role: "status", text: "Point at a dot to read its chunk; the same chunk lights up in the text." });
  const dots = new Map<number, SVGGElement>();
  const few = placed.length <= 14;
  for (const p of placed) {
    const [x, y] = at.get(p.i)!;
    const r = m.third ? sz(p.c.z) : 7;
    const names = m.terms(p.i);
    const left = x > W * 0.68;
    const label = `${p.i + 1}${few && names.length ? ` · ${names.slice(0, 2).join(", ")}` : ""}`;
    const cos = near[p.i];
    const warm = cos === undefined ? 0 : share(cos, lo, hi);
    const g = s(
      "g",
      { class: "dot", tabindex: 0, role: "button", "aria-label": `Chunk ${p.i + 1} of ${m.chunks.length}${names.length ? `: ${names.join(", ")}` : ""}${cos === undefined ? "" : `, cosine to the query ${cos.toFixed(3)}`}` },
      // A target larger than the mark.
      s("circle", { cx: n(x), cy: n(y), r: 18, class: "hit" }),
      // The glow: wider and stronger the closer the chunk is to the query.
      cos === undefined ? null : s("circle", { cx: n(x), cy: n(y), r: n(r + 3 + 9 * warm), class: "glow", "fill-opacity": n(0.12 + 0.6 * warm) }),
      s("circle", { cx: n(x), cy: n(y), r: n(r), class: "mark", style: `fill: ${ordered(p.i, m.chunks.length)}` }),
      s("text", { x: n(x + (left ? -r - 5 : r + 5)), y: n(y + 4), class: "label", "text-anchor": left ? "end" : "start" }, label),
    );
    const point = (): void => link.chunk(p.i);
    g.addEventListener("mouseenter", point);
    g.addEventListener("focus", point);
    g.addEventListener("mouseleave", () => link.chunk(null));
    g.addEventListener("blur", () => link.chunk(null));
    dots.set(p.i, g);
    svg.append(g);
  }
  if (m.spot) {
    const [qx, qy] = [sx(m.spot.x), sy(m.spot.y)];
    const left = qx > W * 0.68;
    svg.append(
      s(
        "g",
        { class: "query", role: "img", "aria-label": `The query: ${short(m.spot.text, 80)}` },
        s("path", { d: `M ${n(qx)} ${n(qy - 10)} L ${n(qx + 10)} ${n(qy)} L ${n(qx)} ${n(qy + 10)} L ${n(qx - 10)} ${n(qy)} Z`, class: "mark" }),
        s("text", { x: n(qx + (left ? -14 : 14)), y: n(qy + 4), class: "label", "text-anchor": left ? "end" : "start" }, "query"),
        s("title", {}, m.spot.text),
      ),
    );
  }
  link.on((f) => {
    for (const [i, g] of dots) g.classList.toggle("lit", i === f.chunk);
    svg.classList.toggle("dim", f.chunk !== null);
    if (f.chunk === null) return;
    const c = m.chunks[f.chunk];
    const cos = m.toNext(f.chunk);
    const names = m.terms(f.chunk);
    info.textContent = `Chunk ${f.chunk + 1} of ${m.chunks.length}, characters ${c.start} to ${c.end}${names.length ? `, told apart by ${names.join(", ")}` : ""}${cos === undefined ? "" : `, cosine to the next ${cos.toFixed(3)}`}${near[f.chunk] === undefined ? "" : `, cosine to the query ${near[f.chunk]!.toFixed(3)}`}: ${short(m.text.slice(c.start, c.end), 140)}`;
  });
  const others = m.units.length - m.chunks.length;
  const legend = el(
    "div",
    { class: "legend" },
    el("span", { class: "swatch ramp-key" }, "a chunk, coloured from the first ", el("i", {}), " to the last"),
    el("span", { class: "swatch trail-key" }, el("i", {}), " reading order"),
    m.third ? el("span", { class: "muted", text: "size: the third direction" }) : null,
    m.spot ? el("span", { class: "swatch query-key" }, el("i", {}), " the query") : null,
    m.spot ? el("span", { class: "swatch glow-key" }, el("i", {}), ` glow: cosine to the query, from ${lo.toFixed(3)} to ${hi.toFixed(3)}; lines join it to the closest`) : null,
    others > 0 ? el("span", { class: "muted", text: `${others} coarser ${others === 1 ? "unit is" : "units are"} not drawn as dots; one holding some of the chunks is a shaded region` }) : null,
  );
  return el("div", { class: "figure" }, el("div", { class: "scroll" }, svg), legend, info);
}
