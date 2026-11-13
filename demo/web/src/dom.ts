// DOM helpers. Text is always set as text, never as markup.

import type { Explained } from "./errors.js";

export type Child = Node | string | null | undefined | false | Child[];

export function $(id: string): HTMLElement {
  const e = document.getElementById(id);
  if (!e) throw new Error(`no element #${id}`);
  return e;
}

export function input(id: string): HTMLInputElement {
  return $(id) as HTMLInputElement;
}

/** An element with attributes and children. `class`, `text` and `style` are properties. */
export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Record<string, string | undefined> = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k === "style") e.style.cssText = v;
    else e.setAttribute(k, v);
  }
  append(e, children);
  return e;
}

function append(e: Element, children: Child[]): void {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    if (Array.isArray(c)) append(e, c);
    else e.append(c);
  }
}

/** A small labelled figure. */
export function stat(value: string, key: string): HTMLElement {
  return el("div", { class: "stat" }, el("div", { class: "v", text: value }), el("div", { class: "k", text: key }));
}

/** A table from a header row and rows of cells; with no rows, a line saying so in place of bare headings. */
export function table(head: string[], rows: Child[][], cls = "", none = "None."): HTMLElement {
  if (!rows.length) return el("p", { class: "muted empty", text: none });
  return el(
    "table",
    { class: cls || undefined },
    el("tr", {}, head.map((h) => el("th", { text: h }))),
    rows.map((r) => el("tr", {}, r.map((c) => el("td", {}, c)))),
  );
}

/**
 * A select's options from values, with labels when given. With no values
 * and a `none`, one empty option says why there is nothing to pick.
 */
export function options(select: HTMLSelectElement, values: string[], labels?: string[], none = ""): void {
  const keep = select.value;
  if (!values.length && none) return void select.replaceChildren(el("option", { value: "", text: none }));
  select.replaceChildren(...values.map((v, i) => el("option", { value: v, text: labels ? labels[i] : v })));
  if (values.includes(keep)) select.value = keep;
}

/** The children that show a failure: its sentence, then its details behind a toggle. */
function failure(x: Explained): Child[] {
  if (!x.details) return [x.text];
  return [x.text, " ", el("details", { class: "why" }, el("summary", { text: "details" }), el("code", { text: x.details }))];
}

/** A status line: plain, busy or an error. A failure from `explain` is an error with its details behind a toggle. */
export function say(box: HTMLElement, what: string | Explained, kind: "" | "busy" | "error" = ""): void {
  if (typeof what === "string") {
    box.textContent = what;
    box.className = kind ? `note ${kind}` : "note";
    box.hidden = what === "";
    return;
  }
  box.replaceChildren();
  append(box, failure(what));
  // A call the visitor stopped is said plainly, not as an error.
  box.className = what.text === "Cancelled." ? "note" : "note error";
  box.hidden = false;
}

/** A failure from `explain` as a line of its own. */
export function failed(x: Explained): HTMLElement {
  return el("p", { class: "error" }, failure(x));
}

/** The legend of a row of heat cells: what a cell is, and the scale from low to high. */
export function heatKey(what: string): HTMLElement {
  return el("p", { class: "heat-key muted" }, what, " ", el("span", { class: "ramp", role: "img", "aria-label": "colour scale from low to high" }, "low ", el("i", {}), " high"));
}

/**
 * Where the layout is one column (a phone), the results sit below the
 * form: brings `box` into view, smoothly unless the visitor asked for
 * less motion. On a wider layout it does nothing.
 *
 * A result that grows after its first part (a stream) can call the
 * returned function to bring the box to the top once more; it does
 * nothing once the visitor has scrolled on their own.
 */
export function reveal(box: Element | null): () => void {
  if (!box || !matchMedia("(max-width: 820px)").matches) return () => {};
  const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const show = (): void => box.scrollIntoView({ behavior: calm ? "auto" : "smooth", block: "start" });
  let moved = false;
  for (const ev of ["wheel", "touchmove", "keydown"]) {
    addEventListener(ev, () => (moved = true), { once: true, passive: true });
  }
  show();
  return () => {
    if (!moved) show();
  };
}

/** Bytes as a short human size. */
export function size(n: number | bigint): string {
  const v = Number(n);
  if (v < 1024) return `${v} B`;
  if (v < 1 << 20) return `${(v / 1024).toFixed(1)} KiB`;
  if (v < 1 << 30) return `${(v / (1 << 20)).toFixed(1)} MiB`;
  return `${(v / (1 << 30)).toFixed(2)} GiB`;
}

/** Elapsed time as the buttons show it: `3.4 s`, `2 min 05 s`. */
export function elapsed(ms: number): string {
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`;
}

/** Offers `data` as a file download. */
export function download(name: string, data: BlobPart[], type: string): void {
  const url = URL.createObjectURL(new Blob(data, { type }));
  const a = el("a", { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** The part of `s` after `prefix`, or `s`. */
export function after(s: string, prefix: string): string {
  return s.startsWith(prefix) ? s.slice(prefix.length) : s;
}

/** A panel that renders when opened and again when its input changes while open. */
export function lazy(details: HTMLDetailsElement, render: () => void): () => void {
  let stale = true;
  details.addEventListener("toggle", () => {
    if (details.open && stale) {
      stale = false;
      render();
    }
  });
  return () => {
    stale = true;
    if (details.open) {
      stale = false;
      render();
    }
  };
}
