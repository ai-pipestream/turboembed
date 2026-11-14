// The topic seismograph: what the chunker saw at each gap between
// sentences, along the document. With sentence vectors that is the
// cosine across the gap (a valley is a topic change); without them it
// is the cost of cutting there. The gaps the chunker cut at are marked,
// the chunks run as a band above, and each chunking budget is a row of
// ticks below, so where a budget splits what the profile kept whole
// reads off directly.

import { el } from "../dom.js";
import type { Link } from "../viz/link.js";
import { aroundGap, cutReason, scale, type Unit } from "../viz/model.js";
import { figure, n, ordered, s } from "../viz/svg.js";
import { short } from "./embed.js";

const W = 640;
const GUTTER = 150;
const RIGHT = 12;
const BAND = 14;
const CURVE = 120;
const ROW = 20;

export type Gap = { offset: number; raw?: number; smoothed?: number; depth: number; cost: number; reason: number; cut: boolean };
export type Strategy = { name: string; chunks: number; shared: number; apart: number; cuts: number[] };

export type SeismoInput = { gaps: Gap[]; chunks: Unit[]; strategies: Strategy[]; text: string; window: number; smoothing: number };

export function seismograph(m: SeismoInput, link: Link): HTMLElement {
  const len = Math.max(1, m.text.length);
  const x = scale(0, len, GUTTER, W - RIGHT);
  const top = 8;
  const curveTop = top + BAND + 10;
  const base = curveTop + CURVE;
  const rowsTop = base + 26;
  const H = rowsTop + (m.strategies.length + 1) * ROW + 6;
  const hasCosine = m.gaps.some((g) => g.smoothed !== undefined);
  const svg = figure(W, H, `Topic seismograph: ${m.gaps.length} sentence gaps, ${m.gaps.filter((g) => g.cut).length} cuts`);

  // The chunks, as a band.
  svg.append(s("text", { x: GUTTER - 8, y: top + BAND - 3, class: "row-name", "text-anchor": "end" }, "chunks"));
  const bands = m.chunks.map((c, i) => {
    const r = s("rect", { x: n(x(c.start) + 1), y: top, width: n(Math.max(1, x(c.end) - x(c.start) - 2)), height: BAND, rx: 3, class: "band", style: `fill: ${ordered(i, m.chunks.length)}` }, s("title", {}, `Chunk ${i + 1}`));
    r.addEventListener("mouseenter", () => link.chunk(i));
    r.addEventListener("mouseleave", () => link.chunk(null));
    return r;
  });
  svg.append(...bands);
  link.on((f) => bands.forEach((b, i) => b.classList.toggle("lit", i === f.chunk)));

  // The curve.
  const values = m.gaps.map((g) => (hasCosine ? g.smoothed : g.cost));
  const known = values.filter((v): v is number => v !== undefined);
  const lo = hasCosine ? Math.min(...known) : 0;
  const hi = Math.max(...known, lo + 1e-9);
  const y = scale(lo, hi, base, curveTop + 6);
  svg.append(
    s("line", { x1: GUTTER, x2: W - RIGHT, y1: base, y2: base, class: "axis-line" }),
    s("text", { x: GUTTER - 8, y: curveTop + 10, class: "tick", "text-anchor": "end" }, hasCosine ? hi.toFixed(2) : String(Math.round(hi))),
    s("text", { x: GUTTER - 8, y: base, class: "tick", "text-anchor": "end" }, hasCosine ? lo.toFixed(2) : "0"),
    s("text", { x: GUTTER - 8, y: (curveTop + base) / 2 + 4, class: "row-name", "text-anchor": "end" }, hasCosine ? "cosine across the gap" : "cost of a cut here"),
  );
  if (hasCosine) {
    const pts = m.gaps.flatMap((g) => (g.smoothed === undefined ? [] : [[x(g.offset), y(g.smoothed)]]));
    if (pts.length > 1) {
      const line = pts.map(([px, py], i) => `${i ? "L" : "M"}${n(px)} ${n(py)}`).join(" ");
      svg.append(s("path", { d: `${line} L${n(pts[pts.length - 1][0])} ${base} L${n(pts[0][0])} ${base} Z`, class: "area" }), s("path", { d: line, class: "line" }));
    }
    const raw = m.gaps.flatMap((g) => (g.raw === undefined ? [] : [[x(g.offset), y(Math.max(lo, Math.min(hi, g.raw)))]]));
    if (raw.length > 1) svg.append(s("path", { d: raw.map(([px, py], i) => `${i ? "L" : "M"}${n(px)} ${n(py)}`).join(" "), class: "line raw" }));
  } else {
    for (const g of m.gaps) svg.append(s("line", { x1: n(x(g.offset)), x2: n(x(g.offset)), y1: base, y2: n(y(g.cost)), class: g.cut ? "stem cut" : "stem" }));
  }
  // The cuts: a dashed line and a diamond, so they are not told by colour alone.
  for (const g of m.gaps) {
    if (!g.cut) continue;
    const gx = x(g.offset);
    const gy = y(hasCosine ? (g.smoothed ?? lo) : g.cost);
    svg.append(s("line", { x1: n(gx), x2: n(gx), y1: curveTop, y2: base, class: "cut-line" }), s("path", { d: `M${n(gx)} ${n(gy - 6)} L${n(gx + 5)} ${n(gy)} L${n(gx)} ${n(gy + 6)} L${n(gx - 5)} ${n(gy)} Z`, class: "cut-mark" }));
  }

  // One row of ticks per chunking, the profile's own first.
  const profile = m.gaps.filter((g) => g.cut).map((g) => g.offset);
  const rows: { name: string; note: string; cuts: number[] }[] = [
    { name: "this profile", note: `${m.chunks.length} chunks`, cuts: m.chunks.slice(1).map((c) => c.start) },
    ...m.strategies.map((t) => ({ name: t.name, note: `${t.chunks} chunks, ${t.shared} cuts shared, ${t.apart} not`, cuts: t.cuts.filter((c) => c > 0) })),
  ];
  rows.forEach((r, i) => {
    const ry = rowsTop + i * ROW;
    const near = (c: number): boolean => profile.some((p) => Math.abs(p - c) <= 2) || m.chunks.some((k) => k.start === c);
    svg.append(
      s("text", { x: GUTTER - 8, y: ry + 12, class: "row-name", "text-anchor": "end" }, short(r.name, 20), s("title", {}, `${r.name}: ${r.note}`)),
      s("rect", { x: GUTTER, y: ry + 2, width: W - RIGHT - GUTTER, height: ROW - 6, rx: 3, class: "barcode" }),
      ...r.cuts.map((c) => s("rect", { x: n(x(c) - 1), y: ry + 2, width: 2, height: ROW - 6, class: i === 0 || near(c) ? "tick-shared" : "tick-apart" })),
    );
  });

  // Pointing anywhere on the curve picks the nearest gap.
  const cross = s("line", { y1: curveTop, y2: base, class: "cross", visibility: "hidden" });
  const pad = s("rect", { x: GUTTER, y: curveTop, width: W - RIGHT - GUTTER, height: CURVE, class: "pad", tabindex: 0, role: "slider", "aria-valuemin": 1, "aria-valuemax": m.gaps.length, "aria-label": "Sentence gaps; point or use the arrow keys" });
  svg.append(cross, pad);
  const info = el("p", { class: "info", role: "status", text: "Point at the curve to see the two sentences on either side of a gap." });
  const offsets = m.gaps.map((g) => g.offset);
  let picked = -1;
  const pick = (i: number): void => {
    if (i < 0 || i >= m.gaps.length) return;
    picked = i;
    const g = m.gaps[i];
    cross.setAttribute("x1", n(x(g.offset)));
    cross.setAttribute("x2", n(x(g.offset)));
    cross.setAttribute("visibility", "visible");
    const [before, after] = aroundGap(m.text, offsets, i);
    const value = g.smoothed !== undefined ? `cosine ${g.smoothed.toFixed(3)}${g.depth > 0 ? `, depth ${g.depth.toFixed(3)}` : ""}` : `cost ${g.cost}`;
    info.replaceChildren(
      el("b", { text: `Gap ${i + 1} of ${m.gaps.length}${g.cut ? " (cut)" : ""}` }),
      `, ${cutReason(g.reason)}, ${value}. `,
      el("span", { class: "quote", text: short(before, 110) }),
      el("span", { class: "muted", text: g.cut ? "  ✂  " : "  ·  " }),
      el("span", { class: "quote", text: short(after, 110) || "(the end)" }),
    );
    pad.setAttribute("aria-valuenow", String(i + 1));
    pad.setAttribute("aria-valuetext", `Gap ${i + 1} of ${m.gaps.length}`);
    // A gap that lies in no chunk points at none.
    const at = m.chunks.findIndex((c) => g.offset > c.start && g.offset <= c.end);
    link.chunk(at < 0 ? null : at);
  };
  pad.addEventListener("mousemove", (ev) => {
    const box = svg.getBoundingClientRect();
    const at = ((ev.clientX - box.left) / box.width) * W;
    let best = 0;
    m.gaps.forEach((g, i) => {
      if (Math.abs(x(g.offset) - at) < Math.abs(x(m.gaps[best].offset) - at)) best = i;
    });
    pick(best);
  });
  pad.addEventListener("keydown", (ev) => {
    const step = ev.key === "ArrowRight" ? 1 : ev.key === "ArrowLeft" ? -1 : 0;
    if (!step) return;
    ev.preventDefault();
    pick(Math.max(0, Math.min(m.gaps.length - 1, picked + step)));
  });
  pad.addEventListener("mouseleave", () => link.chunk(null));

  const cuts = m.gaps.filter((g) => g.cut).length;
  const legend = el(
    "div",
    { class: "legend" },
    el("span", { class: "swatch cut-key" }, el("i", {}), ` a gap the chunker cut at (${cuts})`),
    m.strategies.length ? el("span", { class: "swatch shared-key" }, el("i", {}), " a cut this profile also makes") : null,
    m.strategies.length ? el("span", { class: "swatch apart-key" }, el("i", {}), " a cut only that budget makes") : null,
  );
  const how = hasCosine
    ? `The line is the cosine between the sentences on each side of a gap, smoothed over ${m.smoothing} (window ${m.window}); the thin line is the raw value. A valley is a change of topic.`
    : "This graph holds no sentence vectors, so there is no cosine to draw: each stem is what the chunker charges to cut at that gap, and a short stem is an easy cut (a paragraph or a heading).";
  return el("div", { class: "figure" }, el("p", { class: "muted", text: how }), el("div", { class: "scroll" }, svg), legend, info);
}
