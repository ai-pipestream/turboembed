// How the annotated text is coloured: a mark's colour says what it is.
// Layers that name things (entities, PII, numbers, language spans) are
// coloured by label; the layers that cut the text (sentences, tokens,
// chunks) by layer and instance. One annotation leads each stretch of
// text and gives its colour; a second, if any, underlines it.

/** One annotation on a stretch of text, as InspectDocumentWindow gives it. */
export interface Ann {
  layer: string;
  instance: string;
  item: number;
  label: string;
  score: number;
}

/** A stretch of text and what is annotated on it. */
export interface Seg {
  text: string;
  annotations: Ann[];
}

/** A layer and its instance as one key, which the layer filter is made of. */
export function layerKey(layer: string, instance: string): string {
  return `${layer}#${instance}`;
}

/** A layer as its chip names it: the instance too when it has one, so two instances of a layer read differently. */
export function layerName(layer: string, instance: string): string {
  return instance ? `${layer} [${instance}]` : layer;
}

const NAMING = ["entities", "pii", "numbers", "language"];

/** Whether `layer` names things, so that its marks are coloured by label. */
export function names(layer: string): boolean {
  const l = layer.toLowerCase();
  return NAMING.some((n) => l.includes(n));
}

// Finer cuts lead coarser ones, so a token shows inside its sentence.
const ORDER = ["subwords", "tokens", "sentences", "chunks", "structure"];

/** Lower leads: a naming layer first, then the cutting layers from fine to coarse. */
export function rank(a: Ann): number {
  if (names(a.layer) && a.label) return 0;
  const at = ORDER.findIndex((n) => a.layer.toLowerCase().includes(n));
  return at < 0 ? ORDER.length + 1 : at + 1;
}

/** What a mark's colour stands for, and the words the legend shows for it. */
export function colourOf(a: Ann): { key: string; name: string } {
  if (names(a.layer) && a.label) return { key: `label:${a.label}`, name: a.label };
  return { key: `layer:${layerKey(a.layer, a.instance)}`, name: layerName(a.layer, a.instance) };
}

/** The annotations of `anns` the filter keeps: all of them with no filter, else those of the picked layers. */
export function kept(anns: Ann[], filter: ReadonlySet<string>): Ann[] {
  return filter.size ? anns.filter((a) => filter.has(layerKey(a.layer, a.instance))) : anns;
}

/** What an unfiltered text paints: its naming layers, else its sentences, else every layer. */
export type Lead = "naming" | "sentences" | "all";

/**
 * The annotations that paint a stretch, the leading one first. With no
 * filter only naming layers paint, or, in a document that has none, its
 * sentences, so an unfiltered text is not one block of colour. A
 * document with neither paints all it has.
 */
export function painted(anns: Ann[], filter: ReadonlySet<string>, lead: Lead): Ann[] {
  let show = kept(anns, filter);
  if (!filter.size && lead !== "all") show = show.filter((a) => (lead === "naming" ? rank(a) === 0 : sentence(a)));
  return [...show].sort((x, y) => rank(x) - rank(y));
}

function sentence(a: Ann): boolean {
  return a.layer.toLowerCase().includes("sentences");
}

/** What `segs` paint with no filter. */
export function leadOf(segs: Seg[]): Lead {
  if (segs.some((s) => s.annotations.some((a) => rank(a) === 0))) return "naming";
  return segs.some((s) => s.annotations.some(sentence)) ? "sentences" : "all";
}

/** One entry of the legend: a colour's key, its name, its place in the palette and how many items carry it. */
export interface Swatch {
  key: string;
  name: string;
  index: number;
  items: number;
}

/** The legend for what `segs` paint under `filter`, in order of first appearance. */
export function legend(segs: Seg[], filter: ReadonlySet<string>): Swatch[] {
  const any = leadOf(segs);
  const out = new Map<string, Swatch>();
  const seen = new Set<string>();
  for (const s of segs) {
    for (const a of painted(s.annotations, filter, any)) {
      const c = colourOf(a);
      let sw = out.get(c.key);
      if (!sw) {
        sw = { key: c.key, name: c.name, index: out.size, items: 0 };
        out.set(c.key, sw);
      }
      const item = `${layerKey(a.layer, a.instance)}#${a.item}`;
      if (!seen.has(`${c.key} ${item}`)) {
        seen.add(`${c.key} ${item}`);
        sw.items++;
      }
    }
  }
  return [...out.values()];
}

/** The things the naming layers found, by label: each item's text once, in reading order. */
export function found(segs: Seg[]): { label: string; texts: string[] }[] {
  const items = new Map<string, { label: string; text: string }>();
  for (const s of segs) {
    for (const a of s.annotations) {
      if (rank(a) !== 0) continue;
      const key = `${layerKey(a.layer, a.instance)}#${a.item}`;
      const it = items.get(key);
      if (it) it.text += s.text;
      else items.set(key, { label: a.label, text: s.text });
    }
  }
  const groups = new Map<string, string[]>();
  for (const it of items.values()) {
    const text = it.text.trim();
    const list = groups.get(it.label) ?? [];
    if (text && !list.includes(text)) list.push(text);
    groups.set(it.label, list);
  }
  return [...groups].map(([label, texts]) => ({ label, texts }));
}
