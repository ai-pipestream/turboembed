// The term river: a handful of the document's terms as streams from
// its start to its end. A stream is as thick, over a chunk, as the
// term's weight in that chunk's stack, so the hand-off from one subject
// to the next shows as one stream thinning while another swells.

import { el } from "../dom.js";
import type { Link } from "../viz/link.js";
import { scale, streams, type Unit } from "../viz/model.js";
import { figure, n, s } from "../viz/svg.js";

const W = 640;
const H = 210;
const PAD = { l: 12, r: 12, t: 22, b: 10 };

export type RiverInput = {
  /** The terms drawn, at most eight, in the order they keep their colour. */
  terms: string[];
  /** `weights[t][c]`: term `t` in chunk `c`. */
  weights: number[][];
  chunks: Unit[];
  /** How many times each term occurs in the document. */
  occurrences: (term: string) => number;
  length: number;
};

export function river(m: RiverInput, link: Link): HTMLElement {
  const x = scale(0, Math.max(1, m.length), PAD.l, W - PAD.r);
  const edges = streams(m.weights);
  const reach = Math.max(1e-9, ...edges.flatMap((e) => e.hi), ...edges.flatMap((e) => e.lo.map((v) => -v)));
  const y = scale(-reach, reach, H - PAD.b, PAD.t);
  const svg = figure(W, H, `Term river: ${m.terms.join(", ")}`);
  // A stream holds its width across a chunk's middle half and eases to the next between.
  const stops = m.chunks.flatMap((c) => [x(c.start + (c.end - c.start) * 0.25), x(c.start + (c.end - c.start) * 0.75)]);
  const along = (v: number[]): [number, number][] => {
    const pts = v.flatMap((w, c) => [[stops[2 * c], y(w)], [stops[2 * c + 1], y(w)]] as [number, number][]);
    // Flat out to the document's two ends.
    return pts.length ? [[x(m.chunks[0].start), pts[0][1]], ...pts, [x(m.chunks[m.chunks.length - 1].end), pts[pts.length - 1][1]]] : pts;
  };
  const curve = (pts: [number, number][], first: "M" | "L"): string =>
    pts.map(([px, py], i) => {
      if (i === 0) return `${first}${n(px)} ${n(py)}`;
      const [qx, qy] = pts[i - 1];
      const mid = (qx + px) / 2;
      return `C${n(mid)} ${n(qy)} ${n(mid)} ${n(py)} ${n(px)} ${n(py)}`;
    }).join(" ");
  m.chunks.forEach((c, i) => {
    if (i) svg.append(s("line", { x1: n(x(c.start)), x2: n(x(c.start)), y1: PAD.t - 4, y2: H - PAD.b, class: "cut-line" }));
    svg.append(s("text", { x: n(x((c.start + c.end) / 2)), y: 12, class: "tick", "text-anchor": "middle" }, String(i + 1)));
  });
  const paths = m.terms.map((term, t) => {
    const up = along(edges[t].hi);
    const down = along(edges[t].lo).reverse();
    const p = s("path", { d: `${curve(up, "M")} ${curve(down, "L")} Z`, class: "stream", style: `fill: var(--m${t})`, tabindex: 0, role: "button", "aria-label": `${term}, ${m.occurrences(term)} occurrences` }, s("title", {}, term));
    const on = (): void => link.term(term);
    p.addEventListener("mouseenter", on);
    p.addEventListener("focus", on);
    svg.append(p);
    // The name, where the stream is widest, when it fits.
    const widest = m.weights[t].indexOf(Math.max(...m.weights[t]));
    const thick = y(edges[t].lo[widest]) - y(edges[t].hi[widest]);
    if (thick >= 14 && m.chunks[widest]) {
      const c = m.chunks[widest];
      svg.append(s("text", { x: n(x((c.start + c.end) / 2)), y: n((y(edges[t].lo[widest]) + y(edges[t].hi[widest])) / 2 + 4), class: "stream-name", "text-anchor": "middle" }, term));
    }
    return p;
  });
  const info = el("p", { class: "info", role: "status", text: "Point at a stream or a name to mark that term in the text." });
  const keys = m.terms.map((term, t) => {
    const b = el("button", { type: "button", class: "swatch key", style: `--mark: var(--m${t})` }, el("i", {}), ` ${term} `, el("span", { class: "muted", text: String(m.occurrences(term)) }));
    b.addEventListener("mouseenter", () => link.term(term));
    b.addEventListener("focus", () => link.term(term));
    b.addEventListener("click", () => link.term(term));
    return b;
  });
  const clear = el("button", { type: "button", class: "small", text: "Clear" });
  clear.addEventListener("click", () => link.term(null));
  link.on((f) => {
    const at = f.term === null ? -1 : m.terms.indexOf(f.term);
    paths.forEach((p, t) => p.classList.toggle("faded", at >= 0 && t !== at));
    keys.forEach((k, t) => k.classList.toggle("on", t === at));
    clear.hidden = at < 0;
    if (at >= 0) {
      const where = m.weights[at].flatMap((w, c) => (w > 0 ? [c + 1] : []));
      info.textContent = `"${m.terms[at]}" occurs ${m.occurrences(m.terms[at])} times; it is among the listed terms of ${where.length === 1 ? "chunk" : "chunks"} ${where.join(", ")}. Its occurrences are marked in the text.`;
    }
  });
  return el("div", { class: "figure" }, el("div", { class: "scroll" }, svg), el("div", { class: "legend" }, keys, clear), info);
}
