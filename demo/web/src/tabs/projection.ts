// The document views in Analyze, from ProjectChunks over the analyzed
// graph: the meaning map with the text beside it, the similarity
// matrix, the topic seismograph, the term river, and the same numbers
// as tables. The views share one focus (viz/link.ts): a chunk pointed
// at in one lights up in the others and in the text. The matrix and a
// query placed on the map are calls of their own, so a document too
// long for one, or a model that is not loaded for the other, leaves
// the rest standing.

import { OffsetMode } from "../gen/turbo_typed_annotations_pb.js";
import type { ProjectChunksResponse } from "../gen/turbo_analysis_pb.js";
import type { MessageInitShape } from "@bufbuild/protobuf";
import type { BudgetSpecSchema } from "../gen/turbo_analysis_pb.js";
import { busy } from "../act.js";
import { $, el, failed, input, lazy, say, table } from "../dom.js";
import { doc, onDoc } from "../doc.js";
import { analysis, explain } from "../rpc.js";
import { linker } from "../viz/link.js";
import { nearest } from "../viz/matrix.js";
import { distinctive, leaves, regions, riverTerms, spread, type Term, type Unit } from "../viz/model.js";
import { chunkText, follow } from "./chunktext.js";
import { short } from "./embed.js";
import { latest } from "../latest.js";
import { similarityMatrix } from "./matrix.js";
import { meaningMap } from "./meaningmap.js";
import { river } from "./river.js";
import { seismograph } from "./seismograph.js";
import { take } from "./take.js";

const box = () => $("projection-out");

/** The most chunks the matrix is asked for: the server's own ceiling. */
const MATRIX_ROWS = 512;
/** How many of the chunks closest to a query are listed and marked. */
const CLOSEST = 5;

/** What is on show: the answer, then what the two further calls added. */
let seen: { r: ProjectChunksResponse; text: string; graph: Uint8Array } | null = null;
let pairs: { items: number[]; upper: number[] } | null = null;
/** Why the matrix call was refused, when it was. */
let unpaired: string | null = null;
let spot: { text: string; x: number; y: number; byItem: Map<number, number>; model: string } | null = null;
/** Counts the queries asked, so a slow answer to an old one is dropped. */
let asked = 0;

/** Budgets as `name:max:target`, comma separated; a bad entry is left out. */
export function budgets(s: string): MessageInitShape<typeof BudgetSpecSchema>[] {
  const out: MessageInitShape<typeof BudgetSpecSchema>[] = [];
  for (const item of s.split(",")) {
    const [name, max, target] = item.split(":").map((x) => x.trim());
    const m = parseInt(max ?? "", 10);
    const t = parseInt(target ?? "", 10);
    if (name && m > 0) out.push({ name, maxTokens: m, targetTokens: t > 0 ? Math.min(t, m) : m });
  }
  return out;
}

const micro = (v: bigint | undefined): number | undefined => (v === undefined ? undefined : Number(v) / 1e6);

function show(r: ProjectChunksResponse, text: string): void {
  const units: Unit[] = (r.projection?.points ?? []).map((q) => ({ item: q.item, start: q.start, end: q.end, level: q.level, x: q.x, y: q.y, z: q.z, empty: q.empty }));
  const chunks = leaves(units);
  const link = linker();
  // A stack's unit is its chunk's item.
  const stacks = new Map(r.chunkTermStacks.map((k) => [k.unit, k]));
  const docTerms: Term[] = r.documentTermStack?.segments ?? [];
  const docTotal = r.documentTermStack?.totalWeight ?? 0;
  const listed = spread(chunks.map((c) => stacks.get(c.item)?.segments ?? []));
  const named = chunks.map((c) => {
    const k = stacks.get(c.item);
    return k ? distinctive(k.segments, k.totalWeight, docTerms, docTotal, 3, listed, chunks.length) : [];
  });
  const cosines = new Map((r.projection?.points ?? []).map((q) => [q.item, q.adjacentCosine]));
  const parts: (HTMLElement | string)[] = [];

  if (chunks.some((c) => !c.empty)) {
    const model = r.projection?.model;
    const toQuery = spot ? chunks.map((c) => (c.empty ? undefined : spot!.byItem.get(c.item))) : [];
    // At most half the chunks, so "closest" still tells some from the rest.
    const close = nearest(toQuery, Math.min(CLOSEST, Math.max(1, Math.floor(toQuery.filter((c) => c !== undefined).length / 2))));
    const reader = chunkText(
      text,
      chunks,
      link,
      (term) => {
        const t = docTerms.find((d) => d.term === term);
        return (t?.occStart ?? []).map((a, i) => [a, t?.occEnd?.[i] ?? a] as [number, number]);
      },
      new Map(close.map((h, rank) => [h.at, `Closest to the query: number ${rank + 1}, cosine ${h.cosine.toFixed(3)}`])),
    );
    follow(reader, link);
    parts.push(
      el("h3", { text: "Meaning map" }),
      el("p", { class: "muted", text: `Each dot is one chunk's vector from ${model ? `${model.id}@${model.revision}` : "the graph"}, placed by its two main directions of meaning. The line follows reading order: a document that wanders draws a long trail, and one that stays on a topic stays in one place.` }),
      el("div", { class: "map-and-text" }, meaningMap({ units, chunks, variance: r.projection?.explainedVariance ?? [], terms: (i) => named[i], toNext: (i) => cosines.get(chunks[i].item), text, third: r.projection?.pcaDims === 3, spot: spot ? { text: spot.text, x: spot.x, y: spot.y, cosine: (i) => toQuery[i] } : undefined }, link), el("div", {}, el("h4", { text: "The text, by chunk" }), reader)),
    );
    if (spot && close.length) {
      const list = el("ol", { class: "closest", id: "projection-closest" });
      for (const h of close) {
        const c = chunks[h.at];
        const row = el("li", { tabindex: "0", "data-chunk": String(h.at) }, el("b", { text: `chunk ${h.at + 1}` }), ` ${h.cosine.toFixed(3)} `, el("span", { class: "muted", text: short(text.slice(c.start, c.end), 110) }));
        row.addEventListener("mouseenter", () => link.chunk(h.at));
        row.addEventListener("focus", () => link.chunk(h.at));
        row.addEventListener("mouseleave", () => link.chunk(null));
        row.addEventListener("blur", () => link.chunk(null));
        list.append(row);
      }
      parts.push(el("h4", { text: `Closest to "${short(spot.text, 60)}" by cosine` }), list, el("p", { class: "muted", text: "The glow and this list are the true cosines between the query's vector and each chunk's. Where the query sits is its shadow in the map's two directions, so the closest chunk is not always the nearest dot." }));
    }
    const place = new Map(chunks.map((c, i) => [c.item, i]));
    const rows = pairs ? pairs.items.flatMap((item) => (place.has(item) ? [place.get(item)!] : [])) : [];
    if (pairs && rows.length === pairs.items.length && rows.length > 1) {
      parts.push(
        el("h3", { text: "Similarity matrix" }),
        el("p", { class: "muted", text: "Every chunk against every other: a cell is the cosine between two chunks' vectors, shaded within this document's own range, so the picture has contrast even when every pair is close. A square in the diagonal's own colour around it is a run of chunks that say similar things; a cell of that colour far from the diagonal is the text coming back to something it said before." }),
        similarityMatrix({ chunks, rows, upper: pairs.upper, groups: regions(units, chunks).map((g) => g.members), text }, link),
      );
    } else if (unpaired !== null) {
      parts.push(el("h3", { text: "Similarity matrix" }), el("p", { class: "muted", id: "matrix-none", text: `Not drawn: ${unpaired}` }));
    } else if (chunks.filter((c) => !c.empty).length > MATRIX_ROWS) {
      parts.push(el("h3", { text: "Similarity matrix" }), el("p", { class: "muted", id: "matrix-none", text: `Not drawn: this document has ${chunks.filter((c) => !c.empty).length} chunks with a vector, and the server compares at most ${MATRIX_ROWS}.` }));
    }
  } else {
    parts.push(el("h3", { text: "Meaning map" }), el("p", { class: "muted", text: "No vectors in this graph: analyze with a profile that embeds, such as default-v1, and an embedding model." }));
  }

  if (r.driftCurve?.gaps.length) {
    const d = r.driftCurve;
    parts.push(
      el("h3", { text: "Topic seismograph" }),
      seismograph(
        {
          gaps: d.gaps.map((g) => ({ offset: g.gapOffset, raw: micro(g.rawCosineMicro), smoothed: micro(g.smoothedCosineMicro), depth: Number(g.depthScoreMicro) / 1e6, cost: g.cost, reason: g.reason, cut: g.isChunkCut })),
          chunks,
          strategies: (r.strategyMatrix?.strategies ?? []).map((t) => ({ name: t.name, chunks: t.chunkCount, shared: t.matchedCutsWithBaseline, apart: t.disagreementsWithBaseline, cuts: t.cutStarts })),
          text,
          window: d.window,
          smoothing: d.smoothing,
        },
        link,
      ),
    );
  }

  const withStack = chunks.filter((c) => stacks.has(c.item));
  if (docTerms.length && withStack.length > 1) {
    const uneven = input("projection-uneven").checked;
    const terms = riverTerms(docTerms, docTotal, withStack.map((c) => ({ terms: stacks.get(c.item)!.segments, total: stacks.get(c.item)!.totalWeight })), 8, uneven);
    const weights = terms.map((t) => withStack.map((c) => stacks.get(c.item)!.segments.find((x) => x.term === t)?.weight ?? 0));
    if (terms.length) {
      parts.push(
        el("h3", { text: "Term river" }),
        el("p", { class: "muted", text: uneven ? "The terms whose weight is least evenly spread over the chunks, so words every chunk uses alike are left out. A stream is as thick over a chunk as the term's weight there." : "The document's heaviest terms. A stream is as thick over a chunk as the term's weight there." }),
        river({ terms, weights, chunks: withStack, occurrences: (t) => docTerms.find((d) => d.term === t)?.occStart?.length ?? 0, length: text.length }, link),
      );
    }
  }

  // The same numbers as tables, for reading exactly and for taking away.
  const m = r.strategyMatrix;
  const rows = chunks.map((c, i) => [String(i + 1), `${c.start}..${c.end}`, named[i].join(", "), (stacks.get(c.item)?.segments ?? []).slice(0, 6).map((t) => `${t.term} ${t.weight.toFixed(1)}`).join(", "), short(text.slice(c.start, c.end), 60)]);
  parts.push(
    el(
      "details",
      { class: "tables" },
      el("summary", { text: "As tables" }),
      chunks.length ? table(["chunk", "characters", "told apart by", "heaviest terms", "text"], rows) : "",
      chunks.length ? take("Chunks", () => [["chunk", "start", "end", "told apart by", "x", "y", "text"], ...chunks.map((c, i) => [String(i + 1), String(c.start), String(c.end), named[i].join(" "), String(c.x), String(c.y), text.slice(c.start, c.end)])]) : "",
      m?.strategies.length ? el("h4", { text: "Chunking budgets" }) : "",
      m?.strategies.length
        ? table(
            ["strategy", "max", "target", "chunks", "over budget", "cost", "cuts shared", "disagreements"],
            m.strategies.map((t) => [t.name, String(t.budget?.maxTokens ?? ""), String(t.budget?.targetTokens ?? ""), String(t.chunkCount), String(t.overBudgetCount), String(t.totalCost), String(t.matchedCutsWithBaseline), String(t.disagreementsWithBaseline)]),
          )
        : "",
    ),
  );
  box().replaceChildren(...parts);
}

const dims = (): number => (($("projection-dims") as HTMLSelectElement).value === "3" ? 3 : 2);

/** Asks for the matrix of what is on show, and draws it in when it comes. */
async function compare(): Promise<void> {
  const now = seen;
  if (!now) return;
  const points = (now.r.projection?.points ?? []).filter((q) => !q.empty);
  if (!points.length) return;
  const level = Math.min(...points.map((q) => q.level));
  const count = points.filter((q) => q.level === level).length;
  if (count < 2 || count > MATRIX_ROWS) return;
  try {
    const r = await analysis.projectChunks({ graph: now.graph, offsets: OffsetMode.UTF16, pcaDims: dims(), maxStackSegments: 1, pairwise: { level, maxItems: MATRIX_ROWS } });
    if (seen !== now || !r.pairwise) return;
    pairs = { items: r.pairwise.items, upper: r.pairwise.cosines };
    show(now.r, now.text);
  } catch (e) {
    // The map stands without its matrix, and says why there is none.
    if (seen !== now) return;
    unpaired = explain(e).text;
    show(now.r, now.text);
  }
}

/** Places the query box's text on the map of what is on show; an empty box takes the query off. */
async function ask(): Promise<void> {
  const now = seen;
  const out = $("projection-query-out");
  const text = input("projection-query").value.trim();
  const mine = ++asked;
  if (!now || !text) {
    out.replaceChildren();
    if (spot && now) {
      spot = null;
      show(now.r, now.text);
    }
    spot = null;
    return;
  }
  if (!(now.r.projection?.points ?? []).some((q) => !q.empty)) return void say(out, "This graph has no vectors to place a query among.");
  say(out, "Placing the query\u2026", "busy");
  try {
    const r = await analysis.projectChunks({ graph: now.graph, offsets: OffsetMode.UTF16, pcaDims: dims(), maxStackSegments: 1, query: text });
    if (mine !== asked || seen !== now) return;
    const q = r.query;
    if (!q) return void say(out, "The server answered without a query point: it is older than this page.", "error");
    const order = r.projection?.points ?? [];
    spot = { text, x: q.x, y: q.y, byItem: new Map(order.map((p, i) => [p.item, q.cosines[i]])), model: q.modelName };
    show(now.r, now.text);
    say(out, `Embedded by ${q.modelName} as a query and placed among the chunks.`);
  } catch (e) {
    if (mine !== asked) return;
    out.replaceChildren(failed(explain(e)));
  }
}

// Only the latest call draws (latest.ts).
const freshRender = latest();
async function render(): Promise<void> {
  const mine = freshRender();
  const d = doc();
  if (!d) return void box().replaceChildren(el("p", { class: "muted", text: "Analyze a document first." }));
  try {
    const r = await analysis.projectChunks({
      graph: d.graph,
      offsets: OffsetMode.UTF16,
      pcaDims: dims(),
      // Enough of each stack to tell a chunk's own terms from the document's.
      maxStackSegments: 32,
      includeDriftCurve: true,
      includeStrategyMatrix: true,
      strategyBudgets: budgets(input("projection-budgets").value),
    });
    if (!mine()) return;
    seen = { r, text: d.source, graph: d.graph };
    pairs = null;
    unpaired = null;
    spot = null;
    show(r, d.source);
    await Promise.all([compare(), ask()]);
  } catch (e) {
    if (!mine()) return;
    seen = null;
    box().replaceChildren(failed(explain(e)));
  }
}

export function init(): void {
  const again = lazy($("analyze-projection") as HTMLDetailsElement, () => void render());
  onDoc(() => again());
  const go = $("projection-run") as HTMLButtonElement;
  go.addEventListener("click", () => void busy(go, render));
  $("projection-uneven").addEventListener("change", () => void render());
  // A query is placed on Enter or the button, and a little after the typing stops.
  const query = input("projection-query");
  const place = $("projection-ask") as HTMLButtonElement;
  let wait: ReturnType<typeof setTimeout> | undefined;
  const now = (): void => {
    clearTimeout(wait);
    void busy(place, ask);
  };
  place.addEventListener("click", now);
  query.addEventListener("keydown", (e) => {
    if (e.key === "Enter") now();
  });
  query.addEventListener("input", () => {
    clearTimeout(wait);
    wait = setTimeout(() => void ask(), 500);
  });
}
