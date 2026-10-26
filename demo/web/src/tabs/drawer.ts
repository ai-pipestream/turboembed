// The typed annotation drawer: InspectDocumentGraph's TypedDocumentResult
// for a view, one chip per layer with its status, and the picked layer's
// annotations with their spans, text, probability and every set field of
// the annotation, read through the generated descriptors.

import { LayerWireStatus, OffsetMode, TypedAnnotationSchema, type TypedDocumentResult, type TypedLayer } from "../gen/turbo_typed_annotations_pb.js";
import type { DescMessage } from "@bufbuild/protobuf";
import { brief } from "../brief.js";
import { $, el, lazy, options, table } from "../dom.js";
import { doc, onDoc } from "../doc.js";
import { analysis, describe } from "../rpc.js";
import { viewName } from "./xray.js";

const PAGE = 200;
let result: TypedDocumentResult | null = null;
let picked = "";
let shown = PAGE;

const box = () => $("drawer-out");
const viewSelect = () => $("drawer-view") as HTMLSelectElement;

function key(l: TypedLayer): string {
  return `${l.id}#${l.instance}`;
}

function kindOf(name: string | undefined): DescMessage | undefined {
  return TypedAnnotationSchema.fields.find((f) => f.localName === name)?.message;
}

function rows(r: TypedDocumentResult, l: TypedLayer): HTMLElement {
  const list = l.annotations.slice(0, shown).map((a, i) => {
    const spans = a.spans.map((s) => `${s.start}..${s.end}`).join(" ");
    const text = a.spans.map((s) => r.rawText.slice(s.start, s.end)).join(" \u2026 ");
    const kind = kindOf(a.value.case);
    const detail = kind && a.value.value ? brief(kind, a.value.value as unknown as Record<string, unknown>) : "";
    return [String(i), spans, el("span", { class: "quote", text }), a.value.case ?? "", a.probability === undefined ? "" : a.probability.toFixed(3), el("span", { class: "vec", text: detail })];
  });
  const more = el("button", { type: "button", text: `Show ${Math.min(PAGE, l.annotations.length - shown)} more` });
  more.addEventListener("click", () => ((shown += PAGE), draw()));
  return el(
    "div",
    {},
    el("p", { class: "muted", text: `${l.annotations.length} annotations on ${l.scope === 0 ? "text" : l.scope === 1 ? "an aligned layer" : "the document"}${l.anchorLayer ? `, anchored to ${l.anchorLayer}` : ""}${l.modelRefs.length ? `, models ${l.modelRefs.join(", ")}` : ""}.` }),
    table(["#", "span", "text", "kind", "p", "fields"], list, "drawer"),
    l.annotations.length > shown ? more : "",
  );
}

function draw(): void {
  const r = result;
  if (!r) return;
  const chips = r.layers.map((l) => {
    const bad = l.status !== LayerWireStatus.OK;
    const b = el("button", {
      type: "button",
      class: `chip${key(l) === picked ? " on" : ""}${bad ? " bad" : ""}`,
      title: bad ? `${LayerWireStatus[l.status]}: ${l.statusDetail}` : "",
      text: `${l.id}${l.instance ? `#${l.instance}` : ""} ${l.annotations.length}`,
    });
    b.addEventListener("click", () => {
      picked = key(l);
      shown = PAGE;
      draw();
    });
    return b;
  });
  const layer = r.layers.find((l) => key(l) === picked);
  box().replaceChildren(
    el("p", { class: "muted", text: `${r.layers.length} layers on view ${r.view}. Pick one to open its annotations.` }),
    el("div", { class: "chips" }, chips),
    layer ? rows(r, layer) : "",
    r.models.length ? el("h3", { text: "Models" }) : "",
    r.models.length
      ? table(
          ["ref", "annotator", "version", "bundle", "ran on"],
          r.models.map((m) => [m.modelRef, m.annotator, m.version, m.bundle ? `${m.bundle.id}@${m.bundle.revision}` : "", m.execution ? `${m.execution.backend} ${m.execution.device}`.trim() : ""]),
        )
      : "",
  );
}

async function render(): Promise<void> {
  const d = doc();
  if (!d) return void box().replaceChildren(el("p", { class: "muted", text: "Analyze a document first." }));
  try {
    result = await analysis.inspectDocumentGraph({ docId: "page", graph: d.graph, view: parseInt(viewSelect().value, 10) || 0, offsets: OffsetMode.UTF16 });
    if (!result.layers.some((l) => key(l) === picked)) {
      const first = result.layers.find((l) => l.annotations.length);
      picked = first ? key(first) : "";
    }
    shown = PAGE;
    draw();
  } catch (e) {
    result = null;
    box().replaceChildren(el("p", { class: "error", text: describe(e) }));
  }
}

export function init(): void {
  const again = lazy($("analyze-drawer") as HTMLDetailsElement, () => void render());
  onDoc((d) => {
    const views = d?.views ?? [];
    options(viewSelect(), views.map((_, i) => String(i)), views.map((v, i) => viewName(v.instance, i)));
    viewSelect().value = "0";
    again();
  });
  viewSelect().addEventListener("change", () => void render());
}
