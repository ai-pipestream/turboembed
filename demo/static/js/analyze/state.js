// The Analyze tab's state and the progressive stream that fills it. The
// stream (AnalyzeDocumentProgressive) reports each DAG step as it finishes
// with the graph so far; the viewport refetches its window from the latest
// graph, one call at a time, so the document fills in layer by layer.

import { streamNdjson, postJson } from "../api.js";

export const UNIT = { 0: "UTF-8 bytes", 1: "UTF-16 units", 2: "code points" };
export const PAGE_BYTES = 16000;

export function newState() {
  return {
    req: null,
    offsets: 1,
    graph: null,
    done: false,
    steps: [],
    layers: new Map(),
    views: [],
    hidden: new Set(),
    view: 0,
    page: 0,
    xrayTarget: null,
    window: null,
    selected: -1,
    started: 0,
    error: null,
  };
}

function stepFor(state, id) {
  let s = state.steps.find((x) => x.id === id);
  if (!s) {
    s = { id, status: "pending", stage: null, duration: null, layers: [], failure: null };
    state.steps.push(s);
  }
  return s;
}

function mergeLayers(state, layers, step) {
  for (const l of layers || []) {
    const key = l.instance ? `${l.layer}#${l.instance}` : l.layer;
    state.layers.set(key, { ...l, key, step: step ? step.id : l.annotator, stage: step ? step.stage : null });
  }
}

// Applies one AnalyzeProgressiveEvent; answers true when the graph changed.
export function apply(state, line) {
  const ev = line.event || {};
  if (ev.started) {
    state.offsets = ev.started.offsets;
    state.views = ev.started.views || [];
    for (const id of ev.started.step_ids || []) stepFor(state, id);
    return false;
  }
  if (ev.layers_ready) {
    const b = ev.layers_ready;
    const s = stepFor(state, b.completed_step);
    Object.assign(s, { status: "done", stage: b.stage, duration: b.duration_us, layers: (b.layers || []).map((l) => l.layer) });
    mergeLayers(state, b.layers, s);
    if (b.views && b.views.length) state.views = b.views;
    if (b.graph) state.graph = b.graph;
    return !!b.graph;
  }
  if (ev.step_failed) {
    const f = ev.step_failed;
    const s = stepFor(state, f.step_id);
    Object.assign(s, { status: "failed", stage: f.stage, duration: f.duration_us, failure: f.failure });
    for (const d of f.skipped_dependents || []) stepFor(state, d).status = "skipped";
    mergeLayers(state, f.layers, s);
    return false;
  }
  if (ev.complete) {
    const c = ev.complete;
    state.graph = c.graph;
    state.done = true;
    if (c.views && c.views.length) state.views = c.views;
    for (const l of c.layers || []) {
      const key = l.instance ? `${l.layer}#${l.instance}` : l.layer;
      const old = state.layers.get(key) || {};
      state.layers.set(key, { ...old, ...l, key });
    }
    for (const s of state.steps) if (s.status === "pending") s.status = "skipped";
    return true;
  }
  return false;
}

export function runAnalysis(state, req, onChange, signal) {
  return streamNdjson("api/analyze/progressive", req, (line) => onChange(apply(state, line)), signal);
}

// The window request for the state's view and page: offsets in the unit the
// analysis asked for, computed by the server from its offset table.
export function windowRequest(state, extra = {}) {
  const req = {
    graph: state.graph,
    view: state.view,
    offsets: state.offsets,
    page_size: PAGE_BYTES,
    page: state.page,
    heatmap_buckets: 64,
    ...extra,
  };
  if (state.xrayTarget !== null && state.xrayTarget !== state.view) req.xray_target_view = state.xrayTarget;
  return req;
}

// Fetches windows one at a time; a request made while one is in flight
// replaces any other waiting one, so only the newest graph is drawn.
export function windowQueue(state, onWindow, onError) {
  let busy = false;
  let again = false;
  async function pump() {
    if (busy) {
      again = true;
      return;
    }
    busy = true;
    try {
      do {
        again = false;
        if (!state.graph) break;
        const w = await postJson("api/analyze/window", windowRequest(state));
        onWindow(w);
      } while (again);
    } catch (e) {
      onError(e);
    } finally {
      busy = false;
    }
  }
  return pump;
}
