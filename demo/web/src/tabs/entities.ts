// The entity views in Analyze, from InspectDocumentGraph's typed layers:
// the entities as a graph (a node per entity, sized by its mentions,
// coloured by its kind, with the relations between them as labelled
// arrows), and mention strips (a row per entity, a mark at each
// mention along the document, under the chunks).

import { LayerWireStatus, OffsetMode, type TypedDocumentResult, type TypedLayer } from "../gen/turbo_typed_annotations_pb.js";
import { $, count, el, failed, lazy, options, table } from "../dom.js";
import { doc, onDoc } from "../doc.js";
import { latest } from "../latest.js";
import { analysis, explain } from "../rpc.js";
import { finest, gather, kindSlots, layout, link, sentenceAt, together, type Edge, type Entity, type Mention, type Relation } from "../viz/entities.js";
import { scale } from "../viz/model.js";
import { figure, n, s } from "../viz/svg.js";
import { short } from "./embed.js";
import { take } from "./take.js";

const W = 640;
const MOST = 30;
const KINDS = 8;

const box = () => $("entities-out");
const picker = () => $("entities-layer") as HTMLSelectElement;
const keyOf = (l: TypedLayer): string => `${l.id}#${l.instance}`;
const nameOf = (l: TypedLayer): string => `${l.id}${l.instance ? ` (${l.instance})` : ""}`;
const has = (l: TypedLayer, kind: string): boolean => l.status === LayerWireStatus.OK && l.annotations.some((a) => a.value.case === kind);

/** The relations that index into `entities`: its relation layers, else its edge layers. */
function relationsOf(r: TypedDocumentResult, entities: TypedLayer): Relation[] {
  const into = (l: TypedLayer): boolean => l.indexesLayer.length === 0 || l.indexesLayer.some((b) => b.layer === entities.id && b.instance === entities.instance);
  const out: Relation[] = [];
  for (const l of r.layers.filter((x) => has(x, "relation") && into(x))) {
    for (const a of l.annotations) if (a.value.case === "relation") out.push({ subject: a.value.value.subjectEntity, object: a.value.value.objectEntity, kind: a.value.value.kind, score: a.probability });
  }
  if (out.length) return out;
  for (const l of r.layers.filter((x) => has(x, "edge") && x.indexesLayer.length > 0 && into(x))) {
    for (const a of l.annotations) if (a.value.case === "edge") out.push({ subject: a.value.value.fromIndex, object: a.value.value.toIndex, kind: a.value.value.kind, score: a.value.value.score });
  }
  return out;
}

type Seen = { entities: Entity[]; mentions: Mention[]; edges: Edge[]; pairs: { from: number; to: number; count: number }[]; slots: Map<string, number>; text: string; sentences: { start: number; end: number }[]; chunks: { start: number; end: number }[] };

function constellation(v: Seen, info: HTMLElement, light: (e: number | null) => void, onLight: (l: (e: number | null) => void) => void): HTMLElement {
  const H = v.entities.length > 12 ? 420 : 320;
  // Relations place the nodes when there are any; otherwise what is mentioned together does.
  const at = layout(v.entities.length, v.edges.length ? v.edges : v.pairs, W, H, 46);
  const most = Math.max(...v.entities.map((e) => e.mentions.length));
  const size = (e: Entity): number => 7 + 13 * Math.sqrt(e.mentions.length / most);
  const svg = figure(W, H, `Entity graph: ${count(v.entities.length, "entity", "entities")}, ${count(v.edges.length, "relation")}`);
  svg.append(s("defs", {}, s("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, s("path", { d: "M0 0 L10 5 L0 10 Z", class: "arrow" }))));
  const most2 = Math.max(1, ...v.pairs.map((p) => p.count));
  const dotted = v.pairs.map((p) => {
    const l = s("line", { x1: n(at[p.from][0]), y1: n(at[p.from][1]), x2: n(at[p.to][0]), y2: n(at[p.to][1]), class: "together", "stroke-width": n(1 + (2 * p.count) / most2) }, s("title", {}, `${v.entities[p.from].name} and ${v.entities[p.to].name}: together in ${count(p.count, "sentence")}`));
    svg.append(l);
    return l;
  });
  const lines = v.edges.map((e) => {
    const [ax, ay] = at[e.from];
    const [bx, by] = at[e.to];
    const d = Math.max(1, Math.hypot(bx - ax, by - ay));
    // From rim to rim, so the arrow head is not under the node.
    const [ra, rb] = [size(v.entities[e.from]) + 2, size(v.entities[e.to]) + 4];
    const g = s(
      "g",
      { class: "edge" },
      s("line", { x1: n(ax + ((bx - ax) / d) * ra), y1: n(ay + ((by - ay) / d) * ra), x2: n(bx - ((bx - ax) / d) * rb), y2: n(by - ((by - ay) / d) * rb), "marker-end": "url(#arrow)" }),
      s("text", { x: n((ax + bx) / 2), y: n((ay + by) / 2 - 4), class: "edge-name", "text-anchor": "middle" }, e.kind),
      s("title", {}, `${v.entities[e.from].name} ${e.kind} ${v.entities[e.to].name}${e.score === undefined ? "" : `, score ${e.score.toFixed(3)}`}${e.support > 1 ? `, found ${e.support} times` : ""}`),
    );
    svg.append(g);
    return g;
  });
  const nodes = v.entities.map((e, i) => {
    const [x, y] = at[i];
    const r = size(e);
    const g = s(
      "g",
      { class: "node", tabindex: 0, role: "button", "aria-label": `${e.name}, ${e.kind}, ${count(e.mentions.length, "mention")}` },
      s("circle", { cx: n(x), cy: n(y), r: n(Math.max(r, 16)), class: "hit" }),
      s("circle", { cx: n(x), cy: n(y), r: n(r), class: "mark", style: `fill: var(--m${v.slots.get(e.kind) ?? KINDS - 1})` }),
      s("text", { x: n(x), y: n(y + r + 13), class: "label", "text-anchor": "middle" }, short(e.name, 24)),
    );
    g.addEventListener("mouseenter", () => light(i));
    g.addEventListener("focus", () => light(i));
    g.addEventListener("mouseleave", () => light(null));
    g.addEventListener("blur", () => light(null));
    svg.append(g);
    return g;
  });
  onLight((e) => {
    svg.classList.toggle("dim", e !== null);
    const near = new Set<number>(e === null ? [] : [e]);
    lines.forEach((l, k) => {
      const on = e !== null && (v.edges[k].from === e || v.edges[k].to === e);
      l.classList.toggle("lit", on);
      if (on) near.add(v.edges[k].from).add(v.edges[k].to);
    });
    dotted.forEach((l, k) => {
      const on = e !== null && (v.pairs[k].from === e || v.pairs[k].to === e);
      l.classList.toggle("lit", on);
      if (on) near.add(v.pairs[k].from).add(v.pairs[k].to);
    });
    nodes.forEach((g, i) => g.classList.toggle("lit", near.has(i)));
    if (e === null) return;
    const withThem = v.pairs.filter((x) => x.from === e || x.to === e).sort((a, b) => b.count - a.count).map((x) => v.entities[x.from === e ? x.to : x.from].name);
    const ent = v.entities[e];
    const told = v.edges.filter((x) => x.from === e || x.to === e).map((x) => `${v.entities[x.from].name} ${x.kind} ${v.entities[x.to].name}`);
    info.textContent = `${ent.name} (${ent.kind}), ${count(ent.mentions.length, "mention")}${told.length ? `: ${told.join("; ")}` : ""}${withThem.length ? `. In a sentence with ${withThem.join(", ")}` : ""}.`;
  });
  return el("div", { class: "scroll" }, svg);
}

function strips(v: Seen, info: HTMLElement, light: (e: number | null) => void, onLight: (l: (e: number | null) => void) => void): HTMLElement {
  const GUTTER = 150;
  const ROW = 18;
  const top = v.chunks.length ? 26 : 8;
  const H = top + v.entities.length * ROW + 6;
  const x = scale(0, Math.max(1, v.text.length), GUTTER, W - 12);
  const svg = figure(W, H, `Mention strips for ${count(v.entities.length, "entity", "entities")}`);
  if (v.chunks.length) {
    svg.append(s("text", { x: GUTTER - 8, y: 17, class: "row-name", "text-anchor": "end" }, "chunks"));
    v.chunks.forEach((c, i) => svg.append(s("rect", { x: n(x(c.start) + 1), y: 6, width: n(Math.max(1, x(c.end) - x(c.start) - 2)), height: 12, rx: 3, class: "band plain" }, s("title", {}, `Chunk ${i + 1}`))));
  }
  const rows = v.entities.map((e, i) => {
    const y = top + i * ROW;
    const g = s(
      "g",
      { class: "strip" },
      s("text", { x: GUTTER - 8, y: y + 12, class: "row-name", "text-anchor": "end" }, short(e.name, 22), s("title", {}, `${e.name} (${e.kind})`)),
      s("rect", { x: GUTTER, y: y + 2, width: W - 12 - GUTTER, height: ROW - 6, rx: 3, class: "barcode" }),
    );
    for (const mi of e.mentions) {
      const m = v.mentions[mi];
      const mark = s("rect", { x: n(Math.min(x(m.start), W - 16)), y: y + 1, width: n(Math.max(4, x(m.end) - x(m.start))), height: ROW - 4, rx: 2, class: "mention", tabindex: 0, role: "button", "aria-label": `${e.name} at ${m.start}`, style: `fill: var(--m${v.slots.get(e.kind) ?? KINDS - 1})` });
      const show = (): void => {
        light(i);
        info.replaceChildren(el("b", { text: `${e.name} (${e.kind})` }), ` at ${m.start}: `, el("span", { class: "quote", text: short(sentenceAt(v.text, v.sentences, m.start), 200) }));
      };
      mark.addEventListener("mouseenter", show);
      mark.addEventListener("focus", show);
      mark.addEventListener("mouseleave", () => light(null));
      g.append(mark);
    }
    svg.append(g);
    return g;
  });
  onLight((e) => rows.forEach((g, i) => g.classList.toggle("lit", i === e)));
  return el("div", { class: "scroll" }, svg);
}

function show(r: TypedDocumentResult): void {
  const layers = r.layers.filter((l) => has(l, "entity"));
  const keep = picker().value;
  options(picker(), layers.map(keyOf), layers.map((l) => `${nameOf(l)} ${l.annotations.length}`), "no entity layer");
  $("entities-pick").hidden = layers.length < 2;
  if (!layers.length) {
    return void box().replaceChildren(el("p", { class: "muted", text: "No entities in this graph. Analyze with an open-span model and labels, or with a profile that has an entity step (default-v1 finds place names)." }));
  }
  // The layer picked before, else the one with the most to show.
  const layer = layers.find((l) => keyOf(l) === keep) ?? [...layers].sort((a, b) => b.annotations.length - a.annotations.length)[0];
  picker().value = keyOf(layer);
  // Positions in `mentions` are positions in the layer, which relations index.
  const mentions: Mention[] = layer.annotations.map((a) => {
    const start = a.spans[0]?.start ?? 0;
    const end = a.spans[a.spans.length - 1]?.end ?? start;
    const e = a.value.case === "entity" ? a.value.value : null;
    return { start, end, kind: e?.kind || "entity", canonical: e?.canonical ?? "", text: r.rawText.slice(start, end) };
  });
  const all = gather(mentions);
  const edgesAll = link(relationsOf(r, layer), all.of);
  // The most mentioned entities are drawn; the rest stay in the table.
  const entities = all.entities.slice(0, MOST);
  const edges = edgesAll.filter((e) => e.from < MOST && e.to < MOST);
  const slots = kindSlots(all.entities.map((e) => e.kind), KINDS);
  const spansOf = (kind: string): { start: number; end: number }[] =>
    r.layers.filter((l) => has(l, kind)).flatMap((l) => l.annotations.flatMap((a) => (a.spans.length ? [{ start: a.spans[0].start, end: a.spans[a.spans.length - 1].end }] : [])));
  const sentences = spansOf("sentence");
  const pairs = together(mentions, all.of, sentences).filter((p) => p.from < MOST && p.to < MOST);
  const seen: Seen = { entities, mentions, edges, pairs, slots, text: r.rawText, sentences, chunks: finest(spansOf("chunk")) };
  const listeners: ((e: number | null) => void)[] = [];
  const light = (e: number | null): void => listeners.forEach((l) => l(e));
  const onLight = (l: (e: number | null) => void): void => void listeners.push(l);
  const info = el("p", { class: "info", role: "status", text: "Point at an entity to see its relations, or at a mark to read its sentence." });
  const shownKinds = [...slots].filter(([, slot], i, list) => list.findIndex(([, x]) => x === slot) === i);
  const legend = el(
    "div",
    { class: "legend" },
    shownKinds.map(([kind, slot]) => el("span", { class: "swatch", style: `--mark: var(--m${slot})` }, el("i", {}), ` ${slot === KINDS - 1 && slots.size > KINDS ? "other kinds" : kind}`)),
    el("span", { class: "muted", text: "size: how often it is mentioned" }),
    edges.length ? el("span", { class: "swatch arrow-key" }, el("i", {}), " a relation, from subject to object") : null,
    pairs.length ? el("span", { class: "swatch together-key" }, el("i", {}), " mentioned in the same sentence") : null,
  );
  box().replaceChildren(
    el("p", { class: "muted", text: `${count(all.entities.length, "entity", "entities")} from ${count(mentions.length, "mention")} in ${nameOf(layer)}${edgesAll.length ? `, ${count(edgesAll.length, "relation")}` : "; no relation layer indexes it, so there are no arrows (a relation model adds them)"}.${all.entities.length > MOST ? ` The ${MOST} most mentioned are drawn.` : ""}` }),
    el("h3", { text: "Entity constellation" }),
    el("div", { class: "figure" }, constellation(seen, info, light, onLight), legend),
    el("h3", { text: "Mention strips" }),
    el("div", { class: "figure" }, strips(seen, info, light, onLight), info),
    el(
      "details",
      { class: "tables" },
      el("summary", { text: "As tables" }),
      table(["entity", "kind", "mentions", "at"], all.entities.map((e) => [e.name, e.kind, String(e.mentions.length), e.mentions.map((i) => String(mentions[i].start)).join(", ")])),
      take("Entities", () => [["entity", "kind", "mentions", "starts"], ...all.entities.map((e) => [e.name, e.kind, String(e.mentions.length), e.mentions.map((i) => String(mentions[i].start)).join(" ")])]),
      edgesAll.length ? table(["subject", "relation", "object", "score", "found"], edgesAll.map((e) => [all.entities[e.from].name, e.kind, all.entities[e.to].name, e.score === undefined ? "" : e.score.toFixed(3), String(e.support)])) : "",
    ),
  );
}

let last: TypedDocumentResult | null = null;

// Only the latest call draws (latest.ts).
const freshRender = latest();
async function render(): Promise<void> {
  const mine = freshRender();
  const d = doc();
  if (!d) return void box().replaceChildren(el("p", { class: "muted", text: "Analyze a document first." }));
  try {
    const r = await analysis.inspectDocumentGraph({ docId: "page", graph: d.graph, view: 0, offsets: OffsetMode.UTF16 });
    if (!mine()) return;
    last = r;
    show(last);
  } catch (e) {
    if (!mine()) return;
    last = null;
    box().replaceChildren(failed(explain(e)));
  }
}

export function init(): void {
  const again = lazy($("analyze-entities") as HTMLDetailsElement, () => void render());
  onDoc(() => again());
  picker().addEventListener("change", () => last && show(last));
}
