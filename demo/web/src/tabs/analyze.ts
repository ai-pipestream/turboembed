// Analyze: AnalyzeDocumentProgressive as a stream of step events with
// their timings and layers, then the finished document through
// InspectDocumentWindow (paged text with its annotations, heatmap lanes)
// and FormatDocument (export). The finished graph goes to doc.ts for the
// X-Ray, annotation drawer, projection and wire panels.

import { LayerErrorKind, LayerWireStatus, OffsetMode } from "../gen/turbo_typed_annotations_pb.js";
import type { DocumentWindowResponse, LayerInfo } from "../gen/turbo_analysis_pb.js";
import type { ViewInfo } from "../gen/turbo_typed_annotations_pb.js";
import { $, download, el, options, say, table } from "../dom.js";
import { doc, onDoc, setDoc } from "../doc.js";
import { analysis, describe } from "../rpc.js";
import { bundles, capabilities, onCapabilities } from "../state.js";

let running: AbortController | null = null;
let page = 0;
let filter = new Set<string>();
let openSpanModels = new Set<string>();
let relationModels = new Set<string>();

const select = (id: string) => $(id) as HTMLSelectElement;

function statusName(s: LayerWireStatus): string {
  return LayerWireStatus[s] ?? String(s);
}

function layerRows(layers: Map<string, LayerInfo>): HTMLTableElement {
  return table(
    ["layer", "instance", "status", "items", "annotator"],
    [...layers.values()].map((l) => [
      l.layer,
      l.instance,
      el("span", { class: l.status === LayerWireStatus.OK ? "" : "error", text: statusName(l.status), title: l.error ? LayerErrorKind[l.error.kind] : "" }),
      String(l.items),
      `${l.annotator} ${l.version}`.trim(),
    ]),
  );
}

/** Stops the analysis in flight, if any; the server sees it cancelled. */
export function stop(): void {
  running?.abort();
  running = null;
}

async function run(): Promise<void> {
  stop();
  const ac = new AbortController();
  running = ac;
  const note = $("analyze-note");
  const steps = $("analyze-steps");
  const layersBox = $("analyze-layers");
  steps.replaceChildren();
  layersBox.replaceChildren();
  $("analyze-window").replaceChildren();
  $("analyze-export-out").replaceChildren();
  setDoc(null);
  let graph: Uint8Array | null = null;
  let views: ViewInfo[] = [];
  let source = "";
  const layers = new Map<string, LayerInfo>();
  const t0 = performance.now();
  say(note, "Analyzing\u2026", "busy");
  try {
    const model = select("analyze-model").value;
    const open = openSpanModels.has(model);
    const relationModel = select("analyze-relation-model").value;
    const labels = ($("analyze-labels") as HTMLInputElement).value.split(",").map((label) => label.trim()).filter(Boolean);
    const relationLabels = ($("analyze-relation-labels") as HTMLInputElement).value.split(",").map((label) => label.trim()).filter(Boolean);
    const threshold = ($("analyze-threshold") as HTMLInputElement).valueAsNumber;
    const relationThreshold = ($("analyze-relation-threshold") as HTMLInputElement).valueAsNumber;
    const guidelineText = ($("analyze-guidelines") as HTMLTextAreaElement).value.trim();
    const guidelineMap = new Map<string, string>();
    for (const line of guidelineText ? guidelineText.split("\n") : []) {
      const at = line.indexOf("=");
      if (at < 1 || at === line.length - 1) throw new Error("Use label=prompt on each guideline line.");
      const label = line.slice(0, at).trim();
      if (guidelineMap.has(label)) throw new Error(`Duplicate guideline for ${label}.`);
      guidelineMap.set(label, line.slice(at + 1).trim());
    }
    const prompts = guidelineText ? labels.map((label) => guidelineMap.get(label) ?? label) : [];
    if (open && labels.length === 0) throw new Error("Enter at least one open-span label.");
    if (open && !Number.isFinite(threshold)) throw new Error("Enter a threshold from 0 to 1.");
    if (relationModel && !open) throw new Error("Select an open-span entity model before running relations.");
    if (relationModel && relationLabels.length === 0) throw new Error("Enter at least one relation label.");
    if (relationModel && !Number.isFinite(relationThreshold)) throw new Error("Enter a relation threshold from 0 to 1.");
    const pipelineSteps = open || relationModel ? ["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped"] : [];
    if (open) pipelineSteps.push("turbo:gliner");
    if (relationModel) pipelineSteps.push("turbo:zero-shot-relations");
    const req = {
      docId: "page",
      text: (document.getElementById("analyze-text") as HTMLTextAreaElement).value,
      profile: select("analyze-profile").value,
      model,
      relationModel,
      offsets: OffsetMode.UTF16,
      steps: pipelineSteps,
      openSpan: open ? { labels, prompts, threshold, nested: ($("analyze-nested") as HTMLInputElement).checked } : undefined,
      relations: relationModel
        ? {
            labels: relationLabels.map((name) => ({ name })),
            threshold: relationThreshold,
            withinSentence: ($("analyze-relation-sentence") as HTMLInputElement).checked,
          }
        : undefined,
    };
    for await (const ev of analysis.analyzeDocumentProgressive(req, { signal: ac.signal })) {
      const ms = (performance.now() - t0).toFixed(1);
      const e = ev.event;
      if (e.case === "started") {
        steps.append(el("li", { class: "muted", text: `started: ${e.value.stepIds.join(" \u2192 ")}` }));
      } else if (e.case === "layersReady") {
        const b = e.value;
        for (const l of b.layers) layers.set(`${l.layer}#${l.instance}`, l);
        steps.append(
          el(
            "li",
            {},
            el("span", { class: "badge", text: `${(Number(b.durationUs) / 1000).toFixed(2)} ms` }),
            ` ${b.completedStep} (stage ${b.stage}, ${b.layers.length} layers, at ${ms} ms)`,
          ),
        );
        if (b.graph.length) graph = b.graph;
        layersBox.replaceChildren(layerRows(layers));
      } else if (e.case === "stepFailed") {
        const f = e.value;
        steps.append(
          el("li", { class: "error", text: `${f.stepId} failed: ${f.failure?.message ?? ""}; skipped ${f.skippedDependents.join(", ") || "nothing"}` }),
        );
      } else if (e.case === "complete") {
        graph = e.value.graph;
        views = e.value.views;
        source = e.value.sourceText;
        for (const l of e.value.layers) layers.set(`${l.layer}#${l.instance}`, l);
        layersBox.replaceChildren(layerRows(layers));
        steps.append(el("li", { class: "muted", text: `complete at ${ms} ms, ${graph.length} graph bytes` }));
      }
    }
    say(note, "");
    if (graph) setDoc({ graph, views, source });
  } catch (e) {
    say(note, describe(e), "error");
  } finally {
    if (running === ac) running = null;
  }
}

function segmentsView(w: DocumentWindowResponse): HTMLElement {
  const text = el("div", { class: "doc" });
  for (const s of w.segments) {
    if (!s.annotations.length) {
      text.append(s.text);
      continue;
    }
    const tip = s.annotations.map((a) => `${a.layer}${a.label ? `: ${a.label}` : ""}${a.score ? ` ${a.score.toFixed(3)}` : ""}`).join("\n");
    const level = Math.min(4, s.annotations.length);
    text.append(el("mark", { class: `l${level}`, title: tip, text: s.text }));
  }
  return text;
}

function lanesView(w: DocumentWindowResponse): HTMLElement {
  const box = el("div", { class: "lanes" });
  for (const lane of w.heatmapLanes) {
    const top = Math.max(1, ...lane.buckets.map((b) => Math.abs(b.score) || b.count));
    box.append(
      el(
        "div",
        { class: "lane" },
        el("span", { class: "lane-name", text: `${lane.id} (${lane.kind})` }),
        el(
          "span",
          { class: "cells" },
          lane.buckets.map((b) => {
            const v = (Math.abs(b.score) || b.count) / top;
            return el("span", { class: "hcell", title: `${b.start}..${b.end}: ${b.count} items, score ${b.score.toFixed(3)}`, style: `opacity: ${(0.1 + 0.9 * v).toFixed(2)}` });
          }),
        ),
      ),
    );
  }
  return box;
}

async function showWindow(): Promise<void> {
  const graph = doc()?.graph;
  const box = $("analyze-window");
  if (!graph) return void box.replaceChildren();
  try {
    const base = { graph, view: 0, offsets: OffsetMode.UTF16, pageSize: 16000, filterLayers: [...filter] };
    const [w, heat] = await Promise.all([
      analysis.inspectDocumentWindow({ ...base, page }),
      analysis.inspectDocumentWindow({ ...base, page: 0, heatmapBuckets: 96 }),
    ]);
    const win = w.window;
    const chips = (w.summary?.layers ?? []).map((l) => {
      const on = filter.has(l.layer);
      const b = el("button", { type: "button", class: on ? "chip on" : "chip", text: `${l.layer} ${l.items}` });
      b.addEventListener("click", () => {
        if (on) filter.delete(l.layer);
        else filter.add(l.layer);
        void showWindow();
      });
      return b;
    });
    const nav = el("div", { class: "nav" });
    const prev = el("button", { type: "button", text: "\u2190 page" });
    const next = el("button", { type: "button", text: "page \u2192" });
    prev.toggleAttribute("disabled", !win || win.page === 0);
    next.toggleAttribute("disabled", !win || win.page + 1 >= win.pageCount);
    prev.addEventListener("click", () => ((page = Math.max(0, page - 1)), void showWindow()));
    next.addEventListener("click", () => ((page += 1), void showWindow()));
    nav.append(prev, el("span", { class: "muted", text: win ? ` page ${win.page + 1} of ${win.pageCount}, units ${win.start}..${win.end} ` : "" }), next);
    const s = w.summary;
    box.replaceChildren(
      el("h3", { text: "Document" }),
      s ? el("p", { class: "muted", text: `${s.totalBytes} bytes, ${s.totalUnits} UTF-16 units, ${s.viewCount} views, ${s.layerCount} layers. Pick layers to show only them.` }) : "",
      el("div", { class: "chips" }, chips),
      nav,
      segmentsView(w),
      el("h3", { text: "Heatmap" }),
      lanesView(heat),
    );
  } catch (e) {
    box.replaceChildren(el("p", { class: "error", text: describe(e) }));
  }
}

async function exportDoc(): Promise<void> {
  const out = $("analyze-export-out");
  const graph = doc()?.graph;
  if (!graph) {
    out.replaceChildren(el("p", { class: "error", text: "Analyze a document first." }));
    return;
  }
  try {
    const r = await analysis.formatDocument({ docId: "page", graph, formatId: select("analyze-format").value, offsets: OffsetMode.UTF16 });
    const name = `document${r.fileExtension.startsWith(".") ? "" : "."}${r.fileExtension}`;
    const save = el("button", { type: "button", text: `Download ${name}` });
    save.addEventListener("click", () => download(name, [r.content as Uint8Array<ArrayBuffer>], r.mediaType));
    const text = r.mediaType.startsWith("text/") || r.mediaType.includes("csv") || r.mediaType.includes("ld+");
    out.replaceChildren(save, text ? el("pre", { class: "export", text: new TextDecoder().decode(r.content) }) : el("p", { class: "muted", text: `${r.content.length} bytes of ${r.mediaType}` }));
  } catch (e) {
    out.replaceChildren(el("p", { class: "error", text: describe(e) }));
  }
}

async function loadFormats(): Promise<void> {
  try {
    const f = await analysis.listOutputFormats({});
    options(select("analyze-format"), f.formats.map((x) => x.formatId), f.formats.map((x) => `${x.displayName} (${x.mediaType})`));
  } catch (e) {
    say($("analyze-note"), describe(e), "error");
  }
}

export function init(): void {
  $("analyze-run").addEventListener("click", run);
  $("analyze-stop").addEventListener("click", stop);
  $("analyze-export").addEventListener("click", exportDoc);
  onDoc(() => {
    page = 0;
    filter = new Set();
    void showWindow();
  });
  onCapabilities((c) => {
    openSpanModels = new Set(bundles(c, "TASK_OPEN_SPAN"));
    relationModels = new Set(bundles(c, "TASK_RELATION"));
    const models = [...bundles(c, "TASK_EMBED"), ...openSpanModels];
    options(select("analyze-profile"), ["", ...c.profiles.map((p) => p.name)], ["default", ...c.profiles.map((p) => `${p.name} v${p.version}`)]);
    options(select("analyze-model"), ["", ...models], ["none", ...models]);
    options(select("analyze-relation-model"), ["", ...relationModels], ["none", ...relationModels]);
  });
  capabilities().catch(() => undefined);
  void loadFormats();
}
