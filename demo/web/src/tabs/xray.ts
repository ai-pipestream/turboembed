// Normalization X-Ray: InspectDocumentWindow with xray_target_view, the
// edit runs that turn one view's text into another's (equal, replaced,
// inserted, deleted) and the normalization steps between them.

import { OffsetMode } from "../gen/turbo_typed_annotations_pb.js";
import { XrayEditKindProto, type NormalizationXrayProto } from "../gen/turbo_analysis_pb.js";
import { $, el, lazy, options, table } from "../dom.js";
import { doc, onDoc } from "../doc.js";
import { analysis, describe } from "../rpc.js";

const box = () => $("xray-out");
const from = () => $("xray-from") as HTMLSelectElement;
const to = () => $("xray-to") as HTMLSelectElement;

export function viewName(instance: string, i: number): string {
  return `${i}: ${instance || "source"}`;
}

function runs(x: NormalizationXrayProto): HTMLElement {
  const text = el("div", { class: "doc xray" });
  for (const r of x.runs) {
    const where = `${r.parentStart}..${r.parentEnd} \u2192 ${r.childStart}..${r.childEnd}`;
    if (r.kind === XrayEditKindProto.EQUAL) text.append(r.childText);
    else if (r.kind === XrayEditKindProto.INSERTED) text.append(el("ins", { title: `inserted ${where}`, text: r.childText }));
    else if (r.kind === XrayEditKindProto.DELETED) text.append(el("del", { title: `deleted ${where}`, text: r.parentText }));
    else text.append(el("del", { title: `replaced ${where}`, text: r.parentText }), el("ins", { title: `replaced ${where}`, text: r.childText }));
  }
  return text;
}

async function render(): Promise<void> {
  const d = doc();
  if (!d) return void box().replaceChildren(el("p", { class: "muted", text: "Analyze a document first." }));
  if (d.views.length < 2) {
    return void box().replaceChildren(el("p", { class: "muted", text: "This document has one view: the profile ran no normalization step." }));
  }
  const a = parseInt(from().value, 10) || 0;
  const b = parseInt(to().value, 10) || 0;
  try {
    const r = await analysis.inspectDocumentWindow({ graph: d.graph, view: a, offsets: OffsetMode.UTF16, pageSize: 1, page: 0, xrayTargetView: b });
    const x = r.xray;
    if (!x) return void box().replaceChildren(el("p", { class: "muted", text: "The server gave no X-Ray for these views." }));
    const edits = x.runs.filter((r) => r.kind !== XrayEditKindProto.EQUAL).length;
    box().replaceChildren(
      el("p", { class: "muted", text: `${edits} edits from view ${x.fromView} to view ${x.toView}, offsets in UTF-16 units. Hover an edit for its spans.` }),
      runs(x),
      el("h3", { text: "Steps" }),
      table(
        ["view", "parent", "instance", "annotator", "edits", "bytes in", "bytes out"],
        x.steps.map((s) => [String(s.view), s.parentView === undefined ? "" : String(s.parentView), s.instance, `${s.annotator} ${s.version}`, String(s.editCount), String(s.parentBytes), String(s.childBytes)]),
      ),
    );
  } catch (e) {
    box().replaceChildren(el("p", { class: "error", text: describe(e) }));
  }
}

export function init(): void {
  const again = lazy($("analyze-xray") as HTMLDetailsElement, () => void render());
  onDoc((d) => {
    const views = d?.views ?? [];
    const values = views.map((_, i) => String(i));
    const labels = views.map((v, i) => viewName(v.instance, i));
    options(from(), values, labels);
    options(to(), values, labels);
    from().value = "0";
    to().value = String(Math.max(0, views.length - 1));
    again();
  });
  from().addEventListener("change", () => void render());
  to().addEventListener("change", () => void render());
}
