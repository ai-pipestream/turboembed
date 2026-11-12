// SVG building for the document views. Text is set as text, never markup.

const NS = "http://www.w3.org/2000/svg";

export type Attrs = Record<string, string | number | undefined>;

/** An SVG element with attributes and children; a string child is its text. */
export function s<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Attrs = {}, ...children: (SVGElement | string | null)[]): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined) e.setAttribute(k, String(v));
  for (const c of children) if (c !== null) e.append(c);
  return e;
}

/** The root of a figure `w` by `h` units, scaled to its box, named for a screen reader. */
export function figure(w: number, h: number, label: string): SVGSVGElement {
  return s("svg", { viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": label, class: "viz" });
}

/** A number for a path or an attribute: two decimals are below a pixel. */
export function n(v: number): string {
  return String(Math.round(v * 100) / 100);
}

/** The colour of position `i` of `count` in reading order: one hue, from the light end to the dark end. */
export function ordered(i: number, count: number): string {
  const share = count > 1 ? Math.round((100 * i) / (count - 1)) : 50;
  return `color-mix(in oklab, var(--seq-from), var(--seq-to) ${share}%)`;
}
