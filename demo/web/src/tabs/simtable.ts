// A cosine similarity matrix as a table: colour scaled to the matrix's
// own range by default, with a switch to the absolute 0..1 scale, and
// each row's best match marked.

import { el } from "../dom.js";
import { best, place, range, strength } from "../heat.js";
import { take } from "./take.js";

// The scale picked last, kept for the next table drawn.
let own = true;

/** A cell's background for a place from 0 to 1 on the scale in use. */
export function shade(t: number): string {
  return `background: rgba(var(--heat), ${strength(t).toFixed(3)})`;
}

/**
 * The matrix `sim` over `labels` (each a short label and its full text),
 * with the scale switch above it and the controls to take it away
 * (`name` names the file).
 */
export function simTable(labels: { short: string; full: string }[], sim: number[][], name = "Cosine similarity"): HTMLElement {
  const [lo, hi] = range(sim);
  const cells: HTMLElement[][] = sim.map((row, i) => {
    const top = best(sim, i);
    return row.map((x, j) => el("td", { class: j === top ? "cell best" : "cell", title: j === top ? "this row's best match" : undefined, text: x.toFixed(3) }));
  });
  const paint = (): void => {
    sim.forEach((row, i) => row.forEach((x, j) => (cells[i][j].style.cssText = shade(own ? place(x, lo, hi) : place(x, 0, 1)))));
  };
  const pick = el("input", { type: "checkbox" });
  pick.checked = own;
  pick.addEventListener("change", () => {
    own = pick.checked;
    paint();
  });
  paint();
  const table = el(
    "table",
    { class: "sim" },
    el("tr", {}, el("th", {}), labels.map((_, j) => el("th", { text: `#${j + 1}` }))),
    labels.map((l, i) => el("tr", {}, el("td", { class: "text", title: l.full, text: `#${i + 1} ${l.short}` }), cells[i])),
  );
  const scale = el(
    "div",
    { class: "scale" },
    el("label", {}, pick, ` colour by this matrix's own range, ${lo.toFixed(3)} to ${hi.toFixed(3)}`),
    el("span", { class: "muted", text: "outlined: each row's best match" }),
  );
  const rows = (): string[][] => [["", ...labels.map((l) => l.full)], ...sim.map((row, i) => [labels[i].full, ...row.map((x) => x.toFixed(6))])];
  return el("div", {}, scale, table, take(name, rows));
}
