// DOM helpers. Text is always set as text, never as markup.

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

/** A table from a header row and rows of cells. */
export function table(head: string[], rows: Child[][], cls = ""): HTMLTableElement {
  return el(
    "table",
    { class: cls || undefined },
    el("tr", {}, head.map((h) => el("th", { text: h }))),
    rows.map((r) => el("tr", {}, r.map((c) => el("td", {}, c)))),
  );
}

/** A select's options from values, with labels when given. */
export function options(select: HTMLSelectElement, values: string[], labels?: string[]): void {
  const keep = select.value;
  select.replaceChildren(...values.map((v, i) => el("option", { value: v, text: labels ? labels[i] : v })));
  if (values.includes(keep)) select.value = keep;
}

/** A status line: plain, busy or an error. */
export function say(box: HTMLElement, text: string, kind: "" | "busy" | "error" = ""): void {
  box.textContent = text;
  box.className = kind ? `note ${kind}` : "note";
  box.hidden = text === "";
}

/** Bytes as a short human size. */
export function size(n: number | bigint): string {
  const v = Number(n);
  if (v < 1024) return `${v} B`;
  if (v < 1 << 20) return `${(v / 1024).toFixed(1)} KiB`;
  if (v < 1 << 30) return `${(v / (1 << 20)).toFixed(1)} MiB`;
  return `${(v / (1 << 30)).toFixed(2)} GiB`;
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
