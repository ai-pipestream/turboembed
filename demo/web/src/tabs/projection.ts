// Chunk projection: ProjectChunks over the analyzed document, the chunk
// and sentence vectors in 2 or 3 principal components, the sentence
// boundary drift curve with the chunk cuts, ranked term-vector stacks,
// and how chunking budgets compare with the profile's cuts.

import { OffsetMode } from "../gen/turbo_typed_annotations_pb.js";
import type { BoundaryDriftCurveProto, ProjectChunksResponse, TermVectorStackProto, VectorProjectionProto } from "../gen/turbo_analysis_pb.js";
import type { MessageInitShape } from "@bufbuild/protobuf";
import type { BudgetSpecSchema } from "../gen/turbo_analysis_pb.js";
import { $, el, input, lazy, table } from "../dom.js";
import { doc, onDoc } from "../doc.js";
import { analysis, describe } from "../rpc.js";
import { short } from "./embed.js";

const W = 640;
const H = 360;
const PAD = 24;

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

function hue(level: number): string {
  return `hsl(${(level * 67 + 210) % 360}, 65%, 50%)`;
}

function canvas(): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = el("canvas", { width: String(W), height: String(H), class: "plot" });
  return [c, c.getContext("2d")!];
}

function scatter(p: VectorProjectionProto, text: string): HTMLElement {
  const [c, g] = canvas();
  const pts = p.points.filter((q) => !q.empty);
  const xs = pts.map((q) => q.x);
  const ys = pts.map((q) => q.y);
  const zs = pts.map((q) => q.z);
  const span = (v: number[]) => [Math.min(...v), Math.max(...v) - Math.min(...v) || 1];
  const [x0, xw] = span(xs);
  const [y0, yw] = span(ys);
  const [z0, zw] = span(zs);
  const at = pts.map((q) => [PAD + ((q.x - x0) / xw) * (W - 2 * PAD), H - PAD - ((q.y - y0) / yw) * (H - 2 * PAD)]);
  g.lineWidth = 1;
  for (let i = 1; i < pts.length; i++) {
    if (pts[i].level !== pts[i - 1].level) continue;
    g.strokeStyle = hue(pts[i].level);
    g.globalAlpha = 0.35;
    g.beginPath();
    g.moveTo(at[i - 1][0], at[i - 1][1]);
    g.lineTo(at[i][0], at[i][1]);
    g.stroke();
  }
  pts.forEach((q, i) => {
    g.globalAlpha = p.pcaDims === 3 ? 0.35 + 0.65 * ((q.z - z0) / zw) : 0.9;
    g.fillStyle = hue(q.level);
    g.beginPath();
    g.arc(at[i][0], at[i][1], 5, 0, 2 * Math.PI);
    g.fill();
  });
  const tip = el("p", { class: "muted", text: "Hover a point." });
  c.addEventListener("mousemove", (ev) => {
    const r = c.getBoundingClientRect();
    const mx = ((ev.clientX - r.left) * W) / r.width;
    const my = ((ev.clientY - r.top) * H) / r.height;
    let best = -1;
    let dist = 100;
    at.forEach(([x, y], i) => {
      const d = (x - mx) ** 2 + (y - my) ** 2;
      if (d < dist) [best, dist] = [i, d];
    });
    if (best < 0) return;
    const q = pts[best];
    const cos = q.adjacentCosine === undefined ? "" : `, cosine to the next ${q.adjacentCosine.toFixed(3)}`;
    tip.textContent = `item ${q.item}, level ${q.level}, ${q.start}..${q.end}${cos}: ${short(text.slice(q.start, q.end), 120)}`;
  });
  const levels = [...new Set(pts.map((q) => q.level))].sort((a, b) => a - b);
  const legend = el("div", { class: "chips" }, levels.map((l) => el("span", { class: "chip", style: `border-color: ${hue(l)}; color: ${hue(l)}`, text: `level ${l}` })));
  const ev = p.explainedVariance.map((v) => `${(v * 100).toFixed(1)}%`).join(", ");
  return el(
    "div",
    {},
    el("p", { class: "muted", text: `${pts.length} vectors from ${p.model ? `${p.model.id}@${p.model.revision}` : "the graph"}; explained variance ${ev}${p.pcaDims === 3 ? "; the third component is the opacity" : ""}.` }),
    legend,
    c,
    tip,
  );
}

function drift(d: BoundaryDriftCurveProto): HTMLElement {
  const [c, g] = canvas();
  const n = d.gaps.length;
  const x = (i: number) => PAD + (n > 1 ? (i / (n - 1)) * (W - 2 * PAD) : (W - 2 * PAD) / 2);
  const y = (v: number) => H - PAD - Math.max(0, Math.min(1, v)) * (H - 2 * PAD);
  const micro = (v: bigint | undefined) => (v === undefined ? NaN : Number(v) / 1e6);
  g.fillStyle = "rgba(127, 127, 127, 0.25)";
  d.gaps.forEach((p, i) => {
    const h = Math.max(0, Math.min(1, micro(p.depthScoreMicro)));
    g.fillRect(x(i) - 3, y(h), 6, H - PAD - y(h));
  });
  for (const [field, color] of [["rawCosineMicro", "hsl(210, 20%, 60%)"], ["smoothedCosineMicro", "hsl(210, 70%, 50%)"]] as const) {
    g.strokeStyle = color;
    g.beginPath();
    let on = false;
    d.gaps.forEach((p, i) => {
      const v = micro(p[field]);
      if (Number.isNaN(v)) return void (on = false);
      if (on) g.lineTo(x(i), y(v));
      else g.moveTo(x(i), y(v));
      on = true;
    });
    g.stroke();
  }
  g.strokeStyle = "hsl(10, 75%, 50%)";
  d.gaps.forEach((p, i) => {
    if (!p.isChunkCut) return;
    g.beginPath();
    g.moveTo(x(i), PAD);
    g.lineTo(x(i), H - PAD);
    g.stroke();
  });
  const cuts = d.gaps.filter((p) => p.isChunkCut).length;
  return el(
    "div",
    {},
    el("p", { class: "muted", text: `${n} sentence gaps (window ${d.window}, smoothing ${d.smoothing}), ${cuts} chunk cuts in red. Lines: cosine across each gap, raw and smoothed; bars: depth score.` }),
    c,
  );
}

function stack(s: TermVectorStackProto, label: string): HTMLElement {
  const bar = el(
    "div",
    { class: "stack" },
    s.segments.map((t) => el("span", { title: `${t.term}: ${t.frequency}x, weight ${t.weight.toFixed(3)}${t.isExpanded ? ", expanded" : ""}`, style: `flex: ${Math.max(t.share, 0.001)}`, text: t.term })),
    s.otherTermCount ? el("span", { class: "other", title: `${s.otherTermCount} more terms`, style: `flex: ${Math.max(1 - s.segments.reduce((a, t) => a + t.share, 0), 0.001)}`, text: "\u2026" }) : null,
  );
  return el("div", { class: "lane" }, el("span", { class: "lane-name", text: label }), bar);
}

function show(r: ProjectChunksResponse, text: string): void {
  const m = r.strategyMatrix;
  box().replaceChildren(
    el("h3", { text: "Projection" }),
    r.projection?.points.length ? scatter(r.projection, text) : el("p", { class: "muted", text: "No vectors in this graph: analyze with a profile that embeds, such as default-v1, and an embedding model." }),
    r.driftCurve ? el("h3", { text: "Boundary drift" }) : "",
    r.driftCurve ? drift(r.driftCurve) : "",
    el("h3", { text: "Term vector stacks" }),
    r.documentTermStack ? stack(r.documentTermStack, "document") : "",
    ...r.chunkTermStacks.map((s) => stack(s, `chunk ${s.unit}`)),
    m?.strategies.length ? el("h3", { text: "Strategies" }) : "",
    m?.strategies.length
      ? table(
          ["strategy", "max", "target", "chunks", "over budget", "cost", "cuts shared", "disagreements"],
          m.strategies.map((s) => [s.name, String(s.budget?.maxTokens ?? ""), String(s.budget?.targetTokens ?? ""), String(s.chunkCount), String(s.overBudgetCount), String(s.totalCost), String(s.matchedCutsWithBaseline), String(s.disagreementsWithBaseline)]),
        )
      : "",
  );
}

async function render(): Promise<void> {
  const d = doc();
  if (!d) return void box().replaceChildren(el("p", { class: "muted", text: "Analyze a document first." }));
  try {
    const r = await analysis.projectChunks({
      graph: d.graph,
      offsets: OffsetMode.UTF16,
      pcaDims: ($("projection-dims") as HTMLSelectElement).value === "3" ? 3 : 2,
      maxStackSegments: 8,
      includeDriftCurve: input("projection-drift").checked,
      includeStrategyMatrix: true,
      strategyBudgets: budgets(input("projection-budgets").value),
    });
    show(r, d.source);
  } catch (e) {
    box().replaceChildren(el("p", { class: "error", text: describe(e) }));
  }
}

export function init(): void {
  const again = lazy($("analyze-projection") as HTMLDetailsElement, () => void render());
  onDoc(() => again());
  $("projection-run").addEventListener("click", () => void render());
}
