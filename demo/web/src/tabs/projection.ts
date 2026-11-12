// The document views in Analyze, all from one ProjectChunks call over
// the analyzed graph: the meaning map with the text beside it, the
// topic seismograph, the term river, and the same numbers as tables.
// The views share one focus (viz/link.ts): a chunk pointed at in one
// lights up in the others and in the text.

import { OffsetMode } from "../gen/turbo_typed_annotations_pb.js";
import type { ProjectChunksResponse } from "../gen/turbo_analysis_pb.js";
import type { MessageInitShape } from "@bufbuild/protobuf";
import type { BudgetSpecSchema } from "../gen/turbo_analysis_pb.js";
import { busy } from "../act.js";
import { $, el, failed, input, lazy, table } from "../dom.js";
import { doc, onDoc } from "../doc.js";
import { analysis, explain } from "../rpc.js";
import { linker } from "../viz/link.js";
import { distinctive, leaves, riverTerms, spread, type Term, type Unit } from "../viz/model.js";
import { chunkText, follow } from "./chunktext.js";
import { short } from "./embed.js";
import { latest } from "../latest.js";
import { meaningMap } from "./meaningmap.js";
import { river } from "./river.js";
import { seismograph } from "./seismograph.js";
import { take } from "./take.js";

const box = () => $("projection-out");

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
    const reader = chunkText(text, chunks, link, (term) => {
      const t = docTerms.find((d) => d.term === term);
      return (t?.occStart ?? []).map((a, i) => [a, t?.occEnd?.[i] ?? a] as [number, number]);
    });
    follow(reader, link);
    parts.push(
      el("h3", { text: "Meaning map" }),
      el("p", { class: "muted", text: `Each dot is one chunk's vector from ${model ? `${model.id}@${model.revision}` : "the graph"}, placed by its two main directions of meaning. The line follows reading order: a document that wanders draws a long trail, and one that stays on a topic stays in one place.` }),
      el("div", { class: "map-and-text" }, meaningMap({ units, chunks, variance: r.projection?.explainedVariance ?? [], terms: (i) => named[i], toNext: (i) => cosines.get(chunks[i].item), text, third: r.projection?.pcaDims === 3 }, link), el("div", {}, el("h4", { text: "The text, by chunk" }), reader)),
    );
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
      pcaDims: ($("projection-dims") as HTMLSelectElement).value === "3" ? 3 : 2,
      // Enough of each stack to tell a chunk's own terms from the document's.
      maxStackSegments: 32,
      includeDriftCurve: true,
      includeStrategyMatrix: true,
      strategyBudgets: budgets(input("projection-budgets").value),
    });
    if (!mine()) return;
    show(r, d.source);
  } catch (e) {
    if (!mine()) return;
    box().replaceChildren(failed(explain(e)));
  }
}

export function init(): void {
  const again = lazy($("analyze-projection") as HTMLDetailsElement, () => void render());
  onDoc(() => again());
  const go = $("projection-run") as HTMLButtonElement;
  go.addEventListener("click", () => void busy(go, render));
  $("projection-uneven").addEventListener("change", () => void render());
}
