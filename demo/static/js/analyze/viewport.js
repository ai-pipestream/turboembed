// The windowed document viewport: one page of a view from
// InspectDocumentWindow (api/analyze/window), its segments drawn as the
// server cut them. Every offset shown is the server's, in the unit the
// analysis asked for; nothing here re-tokenizes or recounts the text.

import { h, hue, fmt, bytes } from "../dom.js";
import { lanes } from "../charts.js";
import { UNIT } from "./state.js";

// The most specific layer wins a segment's color.
const PRIORITY = [
  "opennlp:entities", "turbo:keywords", "turbo:abbreviations", "turbo:sparse-terms<turbo:view>", "opennlp:subwords",
  "turbo:stopwords", "opennlp:tokens", "opennlp:sentences", "turbo:chunks", "turbo:structure",
];

function rank(layer) {
  const i = PRIORITY.indexOf(layer);
  return i < 0 ? PRIORITY.length : i;
}

export function topAnnotation(seg, hidden) {
  let best = null;
  for (const a of seg.annotations || []) {
    if (hidden.has(a.layer)) continue;
    if (!best || rank(a.layer) < rank(best.layer)) best = a;
  }
  return best;
}

function chunkOf(seg) {
  const a = (seg.annotations || []).filter((x) => x.layer === "turbo:chunks");
  return a.length ? a.map((x) => x.item).join(",") : null;
}

export function viewName(views, i) {
  const v = views[i];
  if (!v) return `view ${i}`;
  return v.instance ? `${v.instance} (${v.annotator})` : `source (${v.annotator})`;
}

// Draws the window into `el`; `on` has select(i), page(n), view(n), jump(offset).
export function renderViewport(el, state, on) {
  const w = state.window;
  if (!w) {
    el.replaceChildren(h("p", { class: "muted" }, state.graph ? "Loading the window" : "Analyze a text to see it here."));
    return;
  }
  const win = w.window;
  const sum = w.summary;
  const views = sum.views || [];
  const viewSel = h(
    "select",
    { "aria-label": "View", onchange: (ev) => on.view(Number(ev.target.value)) },
    views.map((_, i) => {
      const o = h("option", { value: i }, viewName(views, i));
      o.selected = i === win.view;
      return o;
    }),
  );
  const nav = h(
    "div",
    { class: "row wrap pager" },
    h("label", {}, "View ", viewSel),
    h("button", { type: "button", disabled: win.page === 0, onclick: () => on.page(win.page - 1) }, "Previous page"),
    h("span", {}, `Page ${win.page + 1} of ${win.page_count}`),
    h("button", { type: "button", disabled: win.page + 1 >= win.page_count, onclick: () => on.page(win.page + 1) }, "Next page"),
    h(
      "span",
      { class: "muted small" },
      `${UNIT[state.offsets]} ${fmt(win.start)} to ${fmt(win.end)} of ${fmt(sum.total_units)}; bytes ${fmt(win.byte_start)} to ${fmt(win.byte_end)} of ${bytes(sum.total_bytes)}`,
    ),
  );
  const chips = (w.document_chips || []).map((c) => h("span", { class: "chip" }, `${c.label} `, h("span", { class: "muted" }, c.layer)));
  const map = lanes(w.heatmap_lanes || [], { height: 12, onPick: (b) => on.jump(b) });
  const doc = h("div", { class: "doc", tabindex: "0", role: "listbox", "aria-label": "Document segments", "aria-activedescendant": state.selected >= 0 ? `seg-${state.selected}` : null });
  let last = null;
  w.segments.forEach((seg, i) => {
    const top = topAnnotation(seg, state.hidden);
    const span = h("span", { id: `seg-${i}`, class: "seg", role: "option", dataset: { i } }, seg.text);
    if (top) {
      span.classList.add("on");
      span.style.setProperty("--h", hue(top.layer));
      span.title = `${top.layer} ${top.label}`;
    }
    const chunk = state.hidden.has("turbo:chunks") ? null : chunkOf(seg);
    if (chunk !== null && last !== null && chunk !== last) span.classList.add("cut");
    if (chunk !== null) last = chunk;
    if (i === state.selected) {
      span.classList.add("sel");
      span.setAttribute("aria-selected", "true");
    }
    doc.append(span);
  });
  doc.addEventListener("click", (ev) => {
    const s = ev.target.closest(".seg");
    if (s) on.select(Number(s.dataset.i));
  });
  doc.addEventListener("keydown", (ev) => keys(ev, state, on));
  el.replaceChildren(nav, chips.length ? h("div", { class: "chips" }, chips) : null, map, doc);
  if (state.selected >= 0) doc.querySelector(`#seg-${state.selected}`)?.scrollIntoView({ block: "nearest" });
}

// Arrow keys walk the annotated segments; Home and End go to the ends;
// Escape clears the selection; PageUp and PageDown turn the page.
function keys(ev, state, on) {
  const segs = state.window.segments;
  const annotated = (i) => topAnnotation(segs[i], state.hidden) !== null;
  let i = state.selected;
  const move = (step) => {
    let j = i;
    do j += step;
    while (j >= 0 && j < segs.length && !annotated(j));
    return j >= 0 && j < segs.length ? j : i;
  };
  const win = state.window.window;
  switch (ev.key) {
    case "ArrowRight":
    case "ArrowDown":
      i = move(1);
      break;
    case "ArrowLeft":
    case "ArrowUp":
      i = i < 0 ? segs.length : i;
      i = move(-1);
      break;
    case "Home":
      i = -1;
      i = move(1);
      break;
    case "End":
      i = segs.length;
      i = move(-1);
      break;
    case "Escape":
      i = -1;
      break;
    case "PageDown":
      if (win.page + 1 < win.page_count) on.page(win.page + 1);
      ev.preventDefault();
      return;
    case "PageUp":
      if (win.page > 0) on.page(win.page - 1);
      ev.preventDefault();
      return;
    default:
      return;
  }
  ev.preventDefault();
  on.select(i, true);
}

// The segment holding the server offset `offset`, or -1.
export function segmentAt(window, offset) {
  return window.segments.findIndex((s) => s.start <= offset && offset < s.end);
}
