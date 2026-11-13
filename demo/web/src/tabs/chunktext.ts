// The document's text with each chunk underlined in its reading-order
// colour and numbered, linked with the views: pointing at a chunk here
// lights it in the map, and a term picked in the river is marked here.

import { el } from "../dom.js";
import type { Link } from "../viz/link.js";
import { pieces, type Unit } from "../viz/model.js";
import { ordered } from "../viz/svg.js";

/** `marks` gives the spans to mark for the term in focus. */
export function chunkText(text: string, chunks: Unit[], link: Link, marks: (term: string) => [number, number][]): HTMLElement {
  const box = el("div", { class: "doc chunked", tabindex: "0", "aria-label": "The document, by chunk" });
  let drawn: string | null | undefined;
  link.on((f) => {
    if (f.term !== drawn) {
      drawn = f.term;
      const seen = new Set<number>();
      box.replaceChildren(
        ...pieces(text, chunks, f.term === null ? [] : marks(f.term)).map((p) => {
          const body = p.marked ? el("mark", { class: "occ", text: p.text }) : p.text;
          if (p.chunk === null) return el("span", {}, body);
          const first = !seen.has(p.chunk);
          seen.add(p.chunk);
          const span = el("span", { class: "ck", "data-chunk": String(p.chunk), style: `--ck: ${ordered(p.chunk, chunks.length)}` }, first ? el("sup", { text: String(p.chunk + 1) }) : null, body);
          span.addEventListener("mouseenter", () => link.chunk(p.chunk));
          return span;
        }),
      );
    }
    for (const span of box.querySelectorAll<HTMLElement>(".ck")) span.classList.toggle("lit", span.dataset.chunk === String(f.chunk));
  });
  box.addEventListener("mouseleave", () => link.chunk(null));
  return box;
}

/** Brings the lit chunk into view inside the text box, when a view pointed at it. */
export function follow(box: HTMLElement, link: Link): void {
  link.on((f) => {
    if (f.chunk === null || box.matches(":hover")) return;
    const span = box.querySelector<HTMLElement>(`.ck[data-chunk="${f.chunk}"]`);
    if (span) box.scrollTop = Math.max(0, span.offsetTop - box.offsetTop - 12);
  });
}
