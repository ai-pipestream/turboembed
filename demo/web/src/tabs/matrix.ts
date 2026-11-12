// The similarity matrix: the cosine between every pair of chunk vectors,
// shaded within the document's own range, with the chunks a coarser
// unit holds together outlined along the diagonal. The cells are drawn
// on a canvas (a long document has tens of thousands); the outlines and
// the cross that follows the pointer are an SVG over it.

import { el } from "../dom.js";
import type { Link } from "../viz/link.js";
import { blocks, cosineAt, extent, labelled, mix, share } from "../viz/matrix.js";
import type { Unit } from "../viz/model.js";
import { s } from "../viz/svg.js";
import { short } from "./embed.js";
import { take } from "./take.js";

/** The side of the drawing in CSS pixels, at most. */
const SIDE = 520;

export type MatrixInput = {
  /** The finest level, in reading order. */
  chunks: Unit[];
  /** For each matrix row, its chunk's position in `chunks`. */
  rows: number[];
  /** The strict upper triangle of cosines over the rows, row-major. */
  upper: number[];
  /** The positions (in `chunks`) each coarser unit holds. */
  groups: number[][];
  text: string;
};

export function similarityMatrix(m: MatrixInput, link: Link): HTMLElement {
  const n = m.rows.length;
  const [lo, hi] = extent(m.upper);
  const cell = Math.max(1, Math.min(44, Math.floor(SIDE / n)));
  const side = n * cell;
  const canvas = el("canvas", { role: "img", "aria-label": `Similarity matrix of ${n} chunks: cosines from ${lo.toFixed(3)} to ${hi.toFixed(3)}` }) as HTMLCanvasElement;
  const draw = (): void => {
    const ratio = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
    const px = Math.max(1, Math.round(cell * ratio));
    canvas.width = n * px;
    canvas.height = n * px;
    const g = canvas.getContext("2d");
    if (!g) return;
    const css = getComputedStyle(document.documentElement);
    const [from, to] = [css.getPropertyValue("--seq-from"), css.getPropertyValue("--seq-to")];
    // 64 steps are finer than the eye tells apart, and far fewer strings than cells.
    const steps = Array.from({ length: 64 }, (_, i) => mix(from, to, i / 63));
    for (let a = 0; a < n; a++) {
      for (let b = 0; b < n; b++) {
        g.fillStyle = steps[a === b ? 63 : Math.round(63 * share(cosineAt(m.upper, n, a, b), lo, hi))];
        g.fillRect(b * px, a * px, px, px);
      }
    }
  };
  draw();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);

  const over = s("svg", { viewBox: `0 0 ${n} ${n}`, preserveAspectRatio: "none", class: "viz over", "aria-hidden": "true" });
  const rowOf = new Map(m.rows.map((leaf, row) => [leaf, row]));
  for (const b of blocks(m.groups, rowOf)) over.append(s("rect", { x: b.from, y: b.from, width: b.to - b.from + 1, height: b.to - b.from + 1, class: "block" }));
  const across = s("rect", { x: 0, width: n, height: 1, class: "cross-band" });
  const down = s("rect", { y: 0, width: 1, height: n, class: "cross-band" });
  const pick = s("rect", { width: 1, height: 1, class: "cross-cell" });
  const cross = s("g", { class: "cross-on", visibility: "hidden" }, across, down, pick);
  over.append(cross);

  const name = (row: number): string => `chunk ${m.rows[row] + 1}`;
  const said = (row: number): string => {
    const c = m.chunks[m.rows[row]];
    return short(m.text.slice(c.start, c.end), 90);
  };
  const rest = `Point at a cell to read its two chunks. Cosines here run from ${lo.toFixed(3)} to ${hi.toFixed(3)}.`;
  const info = el("p", { class: "info", role: "status", text: rest });
  let at: [number, number] | null = null;
  const show = (row: number, col: number): void => {
    at = [row, col];
    across.setAttribute("y", String(row));
    down.setAttribute("x", String(col));
    pick.setAttribute("x", String(col));
    pick.setAttribute("y", String(row));
    cross.setAttribute("visibility", "visible");
    info.textContent =
      row === col
        ? `${name(row)} with itself: ${said(row)}`
        : `${name(row)} and ${name(col)}: cosine ${cosineAt(m.upper, n, row, col).toFixed(3)}. ${name(row)}: ${said(row)} / ${name(col)}: ${said(col)}`;
    link.chunk(m.rows[row]);
  };
  const leave = (): void => {
    at = null;
    cross.setAttribute("visibility", "hidden");
    info.textContent = rest;
    link.chunk(null);
  };
  const plot = el("div", { class: "plot", tabindex: "0", role: "application", "aria-label": "Similarity matrix: the arrow keys move from cell to cell", style: `width: ${side}px` }, canvas);
  plot.append(over);
  plot.addEventListener("mousemove", (e) => {
    const box = plot.getBoundingClientRect();
    const to = (v: number, size: number): number => Math.min(n - 1, Math.max(0, Math.floor((v / size) * n)));
    show(to(e.clientY - box.top, box.height), to(e.clientX - box.left, box.width));
  });
  plot.addEventListener("mouseleave", leave);
  plot.addEventListener("blur", leave);
  plot.addEventListener("keydown", (e) => {
    const step: Record<string, [number, number]> = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    const d = step[e.key];
    if (!d) return;
    e.preventDefault();
    const [row, col] = at ?? [0, 0];
    show(at ? Math.min(n - 1, Math.max(0, row + d[0])) : 0, at ? Math.min(n - 1, Math.max(0, col + d[1])) : Math.min(1, n - 1));
  });
  // A chunk pointed at in another view shows as its row and column here.
  link.on((f) => {
    if (at) return;
    const row = f.chunk === null ? undefined : rowOf.get(f.chunk);
    cross.setAttribute("visibility", row === undefined ? "hidden" : "visible");
    if (row === undefined) return;
    across.setAttribute("y", String(row));
    down.setAttribute("x", String(row));
    pick.setAttribute("x", String(row));
    pick.setAttribute("y", String(row));
  });

  const marks = labelled(n);
  const top = el("div", { class: "ticks top", "aria-hidden": "true", style: `width: ${side}px` }, ...marks.map((i) => el("span", { style: `left: ${((i + 0.5) / n) * 100}%`, text: String(m.rows[i] + 1) })));
  const left = el("div", { class: "ticks left", "aria-hidden": "true" }, ...marks.map((i) => el("span", { style: `top: ${((i + 0.5) / n) * 100}%`, text: String(m.rows[i] + 1) })));
  const outlined = over.querySelector(".block") !== null;
  const without = m.chunks.length - n;
  const legend = el(
    "div",
    { class: "legend" },
    el("span", { class: "swatch ramp-key" }, `cosine ${lo.toFixed(3)} `, el("i", {}), ` ${hi.toFixed(3)}; a chunk with itself takes the far end`),
    outlined ? el("span", { class: "swatch block-key" }, el("i", {}), " chunks one coarser unit holds") : null,
    without > 0 ? el("span", { class: "muted", text: `${without} ${without === 1 ? "chunk has" : "chunks have"} no vector and no row` }) : null,
  );
  const rows = (): string[][] => {
    const out = [["chunk", "other chunk", "cosine"]];
    for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) out.push([String(m.rows[a] + 1), String(m.rows[b] + 1), String(cosineAt(m.upper, n, a, b))]);
    return out;
  };
  return el("div", { class: "figure matrix" }, el("div", { class: "scroll" }, el("div", { class: "grid" }, el("div", {}), top, left, plot)), legend, info, take("Similarity matrix", rows));
}
