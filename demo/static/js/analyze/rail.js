// The layer rail: each DAG step as the stream reports it, and each layer it
// wrote with its item count and a switch to show or hide its highlights.

import { h, hue, fmt } from "../dom.js";

const MARK = { pending: "waiting", done: "done", failed: "failed", skipped: "skipped" };

export function renderRail(el, state, onToggle) {
  const steps = [...state.steps].sort((a, b) => (a.stage ?? 99) - (b.stage ?? 99));
  const done = steps.filter((s) => s.status === "done").length;
  el.replaceChildren(
    h("h3", {}, "Layers ", h("span", { class: "muted" }, `${done} of ${steps.length} steps`)),
    h(
      "ol",
      { class: "rail" },
      steps.map((s) =>
        h(
          "li",
          { class: `step ${s.status}` },
          h(
            "div",
            { class: "step-head" },
            h("span", { class: "dot", title: MARK[s.status] }),
            h("code", {}, s.id),
            s.stage !== null ? h("span", { class: "muted small" }, ` stage ${s.stage}`) : null,
            s.duration !== null ? h("span", { class: "muted small" }, ` ${fmt(s.duration / 1000, 2)} ms`) : null,
          ),
          s.failure ? h("p", { class: "error small" }, `${s.failure.code || ""} ${s.failure.message || JSON.stringify(s.failure)}`) : null,
          h(
            "ul",
            {},
            [...state.layers.values()]
              .filter((l) => l.step === s.id)
              .map((l) => layerRow(l, state, onToggle)),
          ),
        ),
      ),
    ),
  );
}

function layerRow(l, state, onToggle) {
  const box = h("input", { type: "checkbox", "aria-label": `Show ${l.layer}` });
  box.checked = !state.hidden.has(l.layer);
  box.addEventListener("change", () => onToggle(l.layer, box.checked));
  const swatch = h("span", { class: "swatch" });
  swatch.style.background = `hsl(${hue(l.layer)} 70% 55%)`;
  return h(
    "li",
    { class: "layer" },
    h("label", {}, box, swatch, h("code", {}, l.layer), l.instance ? h("span", { class: "muted" }, ` #${l.instance}`) : null),
    h("span", { class: "muted small" }, ` ${fmt(l.items || 0)} items`),
    l.model ? h("span", { class: "tag" }, l.model.id) : null,
  );
}
