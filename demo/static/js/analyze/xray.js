// The Normalization X-Ray: the edit runs between a view and a normalized
// view of it, aligned by the server's offsets (NormalizationXrayProto).

import { h, fmt } from "../dom.js";
import { viewName } from "./viewport.js";

const KIND = ["equal", "replaced", "inserted", "deleted"];

function show(text) {
  return text.replace(/ /g, "\u00b7").replace(/\n/g, "\u21b5").replace(/\t/g, "\u2192");
}

export function renderXray(el, state, on) {
  const w = state.window;
  const views = (w && w.summary.views) || [];
  if (views.length < 2) {
    el.replaceChildren(h("p", { class: "muted small" }, "The X-Ray compares a view with a normalized one: add a turbo:normalize step to get one."));
    return;
  }
  const target = h(
    "select",
    { "aria-label": "X-Ray target view", onchange: (ev) => on.target(ev.target.value === "" ? null : Number(ev.target.value)) },
    h("option", { value: "" }, "off"),
    views.map((_, i) => {
      const o = h("option", { value: i, disabled: i === w.window.view }, viewName(views, i));
      o.selected = i === state.xrayTarget;
      return o;
    }),
  );
  const x = w.xray;
  const head = h("div", { class: "row wrap" }, h("h3", {}, "Normalization X-Ray"), h("label", {}, "Compare with ", target));
  if (!x) {
    el.replaceChildren(head);
    return;
  }
  const edits = x.runs.filter((r) => r.kind !== 0);
  const align = h(
    "div",
    { class: "xray" },
    h("div", { class: "lane-label muted small" }, viewName(views, x.from_view)),
    h(
      "div",
      { class: "xline" },
      x.runs.map((r) => h("span", { class: `run k${r.kind}`, title: `${KIND[r.kind]} ${fmt(r.parent_start)} to ${fmt(r.parent_end)}`, onclick: () => on.jump(r.parent_start) }, show(r.parent_text) || "\u2205")),
    ),
    h("div", { class: "lane-label muted small" }, viewName(views, x.to_view)),
    h(
      "div",
      { class: "xline" },
      x.runs.map((r) => h("span", { class: `run k${r.kind}`, title: `${KIND[r.kind]} ${fmt(r.child_start)} to ${fmt(r.child_end)}` }, show(r.child_text) || "\u2205")),
    ),
  );
  el.replaceChildren(
    head,
    h(
      "p",
      { class: "small" },
      `${edits.length} edits in this window. `,
      (x.steps || []).map((s) => h("span", { class: "chip" }, `${s.annotator} v${s.version}: ${fmt(s.edit_count)} edits, ${fmt(s.parent_bytes)} to ${fmt(s.child_bytes)} bytes`)),
    ),
    align,
    h(
      "div",
      { class: "legend small" },
      KIND.map((k, i) => h("span", { class: `run k${i}` }, k)),
    ),
  );
}
