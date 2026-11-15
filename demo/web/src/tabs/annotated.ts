// The analyzed text with its annotations marked: each mark's colour says
// what it is (marks.ts), a legend names the colours, and a mark can be
// clicked or focused to list everything annotated on it.

import { el, table } from "../dom.js";
import { colourOf, found, leadOf, legend, painted, type Ann, type Seg } from "../marks.js";

// How many colours the stylesheet defines (--m0 to --m7).
const COLOURS = 8;

function colour(index: number): string {
  return `var(--m${index % COLOURS})`;
}

function tip(anns: Ann[]): string {
  return anns.map((a) => `${a.layer}${a.instance ? ` (${a.instance})` : ""}${a.label ? `: ${a.label}` : ""}${a.score ? ` ${a.score.toFixed(3)}` : ""}`).join("\n");
}

/** What is annotated on one stretch, as the panel beside the text shows it. */
function inspect(text: string, anns: Ann[]): HTMLElement[] {
  return [
    el("p", {}, el("span", { class: "quote", text })),
    table(["layer", "instance", "label", "score", "item"], anns.map((a) => [a.layer, a.instance, a.label, a.score ? a.score.toFixed(3) : "", String(a.item)])),
  ];
}

/**
 * The text of `segs` under `filter` (layer keys; empty for all), its
 * legend, and a panel that lists the annotations of the mark picked.
 */
export function annotated(segs: Seg[], filter: ReadonlySet<string>): HTMLElement {
  const any = leadOf(segs);
  const swatches = legend(segs, filter);
  const index = new Map(swatches.map((s) => [s.key, s.index]));
  const panel = el("div", { class: "inspect", "aria-live": "polite" }, el("p", { class: "muted", text: "Pick a marked stretch of text to list what is annotated on it." }));
  const text = el("div", { class: "doc" });
  let picked: HTMLElement | null = null;
  for (const s of segs) {
    const paint = painted(s.annotations, filter, any);
    if (!paint.length) {
      text.append(s.text);
      continue;
    }
    const lead = paint[0];
    const second = paint.find((a) => colourOf(a).key !== colourOf(lead).key);
    const style = `--mark: ${colour(index.get(colourOf(lead).key) ?? 0)}${second ? `; --under: ${colour(index.get(colourOf(second).key) ?? 0)}` : ""}`;
    // Neighbouring items of one colour alternate in strength, so each can be told from the next.
    const cls = `${lead.item % 2 ? "alt" : ""}${second ? " under" : ""}`.trim();
    const mark = el("mark", { class: cls || undefined, style, title: tip(s.annotations), tabindex: "0", role: "button", text: s.text });
    const show = (): void => {
      picked?.classList.remove("picked");
      picked = mark;
      mark.classList.add("picked");
      panel.replaceChildren(...inspect(s.text, s.annotations));
    };
    mark.addEventListener("click", show);
    mark.addEventListener("focus", show);
    text.append(mark);
  }
  const key = swatches.length
    ? el("div", { class: "legend" }, swatches.map((s) => el("span", { class: "swatch", style: `--mark: ${colour(s.index)}` }, el("i", {}), ` ${s.name} `, el("span", { class: "muted", text: String(s.items) }))))
    : el("p", { class: "muted", text: "Nothing is marked: pick a layer above to mark it in the text." });
  return el("div", { class: "annotated" }, key, text, panel);
}

/** What the naming layers found, grouped by label; nothing when they found nothing. */
export function foundList(segs: Seg[]): HTMLElement | null {
  const groups = found(segs);
  if (!groups.length) return null;
  return el(
    "div",
    { class: "found" },
    el("h3", { text: "Found" }),
    el("dl", {}, groups.map((g) => [el("dt", { text: `${g.label} (${g.texts.length})` }), el("dd", { text: g.texts.join(", ") })])),
  );
}
