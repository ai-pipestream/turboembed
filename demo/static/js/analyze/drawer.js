// The detail drawer: the selected segment's server offsets and text, every
// annotation over it, and its X-Ray run when the X-Ray is on.

import { h, hue, fmt } from "../dom.js";
import { UNIT } from "./state.js";

export function renderDrawer(el, state, onClose) {
  const w = state.window;
  const seg = w && state.selected >= 0 ? w.segments[state.selected] : null;
  el.hidden = !seg;
  if (!seg) return;
  const runs = (w.xray && w.xray.runs) || [];
  const run = runs.find((r) => r.parent_start <= seg.start && seg.start < Math.max(r.parent_end, r.parent_start + 1));
  el.replaceChildren(
    h(
      "div",
      { class: "row" },
      h("h3", {}, "Segment"),
      h("button", { type: "button", class: "close", "aria-label": "Close the drawer", onclick: onClose }, "Close"),
    ),
    h("p", { class: "seg-text" }, seg.text),
    h(
      "dl",
      {},
      h("dt", {}, "Offsets"),
      h("dd", {}, `${fmt(seg.start)} to ${fmt(seg.end)} (${UNIT[state.offsets]}, view ${w.window.view})`),
      h("dt", {}, "Length"),
      h("dd", {}, `${fmt(seg.end - seg.start)} ${UNIT[state.offsets]}`),
    ),
    h(
      "table",
      { class: "grid" },
      h("thead", {}, h("tr", {}, ["Layer", "Item", "Label", "Score"].map((t) => h("th", {}, t)))),
      h(
        "tbody",
        {},
        (seg.annotations || []).map((a) => {
          const sw = h("span", { class: "swatch" });
          sw.style.background = `hsl(${hue(a.layer)} 70% 55%)`;
          return h(
            "tr",
            { class: state.hidden.has(a.layer) ? "dim" : "" },
            h("td", {}, sw, h("code", {}, a.layer), a.instance ? ` #${a.instance}` : ""),
            h("td", {}, fmt(a.item)),
            h("td", {}, a.label),
            h("td", {}, fmt(a.score)),
          );
        }),
      ),
    ),
    run
      ? h(
          "p",
          { class: "small" },
          `X-Ray: ${["equal", "replaced", "inserted", "deleted"][run.kind] || run.kind} run, view ${w.xray.from_view} `,
          `${fmt(run.parent_start)} to ${fmt(run.parent_end)} is view ${w.xray.to_view} ${fmt(run.child_start)} to ${fmt(run.child_end)}`,
        )
      : null,
    h("p", { class: "muted small" }, "Arrow keys move between annotated segments, Escape closes."),
  );
}
