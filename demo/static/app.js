"use strict";

// The page talks to the demo binary's JSON calls; the demo makes the gRPC
// calls to turbo-kserve. Nothing here is fetched from anywhere else.

const $ = (id) => document.getElementById(id);

// A DOM element with properties and children; text is always set as text.
function el(tag, props, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k === "style") e.style.cssText = v;
    else e.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined) continue;
    e.append(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}

const models = new Map(); // name -> { meta, error, checked }

async function getJson(url, init) {
  const r = await fetch(url, init);
  let body;
  try {
    body = await r.json();
  } catch {
    body = { error: { message: `HTTP ${r.status}` } };
  }
  return { ok: r.ok, body };
}

function describeError(e) {
  if (!e) return "unknown error";
  const parts = [];
  if (e.grpc) parts.push(e.grpc);
  if (e.turbo_code) parts.push(`turbo-code ${e.turbo_code}`);
  if (e.turbo_field && e.turbo_field !== "0") parts.push(`field ${e.turbo_field}`);
  return (parts.length ? parts.join(", ") + ": " : "") + (e.message || "");
}

function tier(meta) {
  const p = meta && meta.properties && meta.properties["session_info.precision"];
  return p ? p.replace(/^PRECISION_/, "") : "?";
}

function facts(meta) {
  const p = meta.properties || {};
  const out = [];
  if (p["model_info.model_id"]) out.push(p["model_info.model_id"]);
  if (p["model_info.dim"]) out.push(`dim ${p["model_info.dim"]}`);
  if (p["session_info.compute_dtype"]) out.push(p["session_info.compute_dtype"].replace(/^DTYPE_/, ""));
  const dev = [p["device_info.backend"], p["device_info.name"]].filter(Boolean).join(" ");
  if (dev) out.push(dev);
  return out.join(" · ");
}

async function loadStatus() {
  const { body } = await getJson("api/status");
  const s = $("server");
  if (!body.reachable) {
    s.textContent = `Server ${body.server} is not reachable: ${body.message || ""}`;
    s.className = "error";
  } else {
    const state = body.ready ? "ready" : body.live ? "live, still loading" : "not live";
    s.textContent = `${body.name || "server"} ${body.version || ""} at ${body.server}: ${state}`;
    s.className = "muted";
  }
  for (const m of body.models || []) {
    if (!models.has(m.name)) models.set(m.name, { checked: true });
  }
  await Promise.all([...models.keys()].map(loadMeta));
  renderModels();
}

async function loadMeta(name) {
  const m = models.get(name);
  const { ok, body } = await getJson(`api/models/${encodeURIComponent(name)}`);
  if (ok) {
    m.meta = body;
    m.error = null;
  } else {
    m.meta = null;
    m.error = describeError(body.error);
    m.checked = false;
  }
}

function renderModels() {
  const box = $("models");
  box.replaceChildren();
  if (models.size === 0) {
    box.append(el("p", { class: "muted", text: "No model names yet: add one below." }));
  }
  for (const [name, m] of models) {
    const cb = el("input", { type: "checkbox" });
    cb.checked = m.checked;
    cb.disabled = !m.meta;
    cb.addEventListener("change", () => (m.checked = cb.checked));
    const label = el(
      "label",
      { class: "model" },
      cb,
      el("span", {}, el("span", { class: "name", text: name }), m.meta ? el("span", { class: "tag", text: tier(m.meta) }) : null),
      el("span", { class: m.error ? "facts error" : "facts", text: m.meta ? facts(m.meta) : m.error || "" }),
    );
    box.append(label);
  }
}

$("add").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const name = $("add-name").value.trim();
  if (!name || models.has(name)) return;
  models.set(name, { checked: true });
  await loadMeta(name);
  $("add-name").value = "";
  renderModels();
});

// ---- Math -----------------------------------------------------------------

function cosine(a, b) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na && nb ? d / Math.sqrt(na * nb) : 0;
}

function norm(a) {
  return Math.sqrt(a.reduce((s, x) => s + x * x, 0));
}

function matrix(vectors) {
  return vectors.map((a) => vectors.map((b) => cosine(a, b)));
}

// ---- Running --------------------------------------------------------------

function parameters() {
  const p = {};
  const role = $("prompt_role").value;
  if (role) p.prompt_role = role;
  const max = parseInt($("max_tokens").value, 10);
  if (max > 0) {
    p.max_tokens = max;
    p.truncate = "TRUNCATE_RIGHT";
  }
  return p;
}

async function run() {
  const texts = $("texts").value.split("\n").map((t) => t.trim()).filter((t) => t.length);
  const chosen = [...models].filter(([, m]) => m.checked && m.meta).map(([n]) => n);
  const err = $("error");
  err.hidden = true;
  if (!texts.length || !chosen.length) {
    err.textContent = !texts.length ? "Enter at least one text." : "Pick at least one model.";
    err.hidden = false;
    return;
  }
  const btn = $("run");
  btn.disabled = true;
  btn.textContent = "Embedding…";
  const results = [];
  // One request at a time, so no call waits behind another.
  for (const name of chosen) {
    const t0 = performance.now();
    const { ok, body } = await getJson("api/infer", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: name, texts, parameters: parameters() }),
    });
    const ms = performance.now() - t0;
    results.push({ name, meta: models.get(name).meta, ok, body, ms });
  }
  btn.disabled = false;
  btn.textContent = "Embed";
  render(texts, results);
}

$("run").addEventListener("click", run);

// ---- Results --------------------------------------------------------------

function heat(v) {
  const a = Math.max(0, Math.min(1, v));
  return `background: rgba(var(--heat), ${(a * a * 0.85).toFixed(3)})`;
}

function short(t, n = 28) {
  return t.length > n ? t.slice(0, n - 1) + "…" : t;
}

function stat(v, k) {
  return el("div", { class: "stat" }, el("div", { class: "v", text: v }), el("div", { class: "k", text: k }));
}

function card(texts, r) {
  const head = el("div", { class: "head" }, el("h3", {}, r.name, el("span", { class: "tag", text: tier(r.meta) })));
  if (!r.ok) {
    return el("div", { class: "card" }, head, el("p", { class: "error", text: describeError(r.body.error) }),
      stat(`${r.ms.toFixed(1)} ms`, "browser round trip"));
  }
  const v = r.body.vectors;
  const s = r.body.summary || {};
  const sim = matrix(v);
  const table = el("table", { class: "sim" },
    el("tr", {}, el("th", {}), texts.map((_, j) => el("th", { text: `#${j + 1}` }))),
    texts.map((t, i) =>
      el("tr", {}, el("td", { class: "text", title: t, text: `#${i + 1} ${short(t, 22)}` }),
        sim[i].map((x) => el("td", { class: "cell", style: heat(x), text: x.toFixed(3) })))));
  const previews = el("table", {},
    el("tr", {}, el("th", { class: "text", text: "text" }), el("th", { text: "norm" }), el("th", { class: "text", text: "first values" })),
    v.map((row, i) => el("tr", {},
      el("td", { class: "text", title: texts[i], text: `#${i + 1} ${short(texts[i], 18)}` }),
      el("td", { text: norm(row).toFixed(4) }),
      el("td", { class: "vec", text: row.slice(0, 6).map((x) => (x < 0 ? "" : " ") + x.toFixed(4)).join(" ") + (row.length > 6 ? "  …" : "") }))));
  const device = [s.backend, s.arch].filter(Boolean).join(" ");
  return el("div", { class: "card" }, head,
    el("div", { class: "stats" },
      stat(String(r.body.shape[1]), "dimensions"),
      stat(String(r.body.shape[0]), "texts"),
      stat(`${r.ms.toFixed(1)} ms`, "browser round trip"),
      device ? stat(device, "ran on") : null,
      s.compute_dtype ? stat(String(s.compute_dtype).replace(/^DTYPE_/, ""), "computed in") : null),
    el("h2", { text: "Cosine similarity" }), table,
    el("details", {}, el("summary", { text: "Vectors" }), previews));
}

// The same texts through every model: each pair's similarity side by side,
// and how far each model's vectors are from the first model's.
function comparison(texts, ok) {
  const sims = ok.map((r) => matrix(r.body.vectors));
  const rows = [];
  let widest = 0;
  for (let i = 0; i < texts.length; i++) {
    for (let j = i + 1; j < texts.length; j++) {
      const vals = sims.map((m) => m[i][j]);
      const spread = Math.max(...vals) - Math.min(...vals);
      widest = Math.max(widest, spread);
      rows.push(el("tr", {},
        el("td", { class: "text", title: `${texts[i]}\n${texts[j]}`, text: `#${i + 1} · #${j + 1}` }),
        vals.map((x) => el("td", { class: "cell", style: heat(x), text: x.toFixed(4) })),
        el("td", { text: spread === 0 ? "0" : spread.toExponential(1) })));
    }
  }
  const base = ok[0];
  const agree = ok.map((r) => {
    if (r.body.vectors[0].length !== base.body.vectors[0].length) return "—";
    const c = r.body.vectors.map((row, i) => cosine(row, base.body.vectors[i]));
    return Math.min(...c).toFixed(6);
  });
  const table = el("table", { class: "sim cmp" },
    el("tr", {}, el("th", { class: "text", text: "pair" }), ok.map((r) => el("th", { text: `${r.name} (${tier(r.meta)})` })),
      el("th", { text: "spread" })),
    rows,
    el("tr", {}, el("th", { class: "text", text: `lowest cosine to ${base.name}, same text` }),
      agree.map((a) => el("td", { text: a })), el("td", {})));
  return el("div", { class: "card" },
    el("h3", { text: "Side by side" }),
    el("p", { class: "muted", text: "Each pair of texts, scored by every model picked. The spread is how much the tiers disagree; the last row is how close each model's vector for a text is to the first model's." }),
    table,
    widest === 0
      ? el("p", { class: "note", text: "Every model gave the same numbers: on this device these tiers compute alike. A device that runs FASTEST in a narrower type shows the difference here." })
      : null);
}

function render(texts, results) {
  const out = $("results");
  out.replaceChildren();
  const ok = results.filter((r) => r.ok);
  if (ok.length > 1 && texts.length > 1) out.append(comparison(texts, ok));
  out.append(el("div", { class: "cards" }, results.map((r) => card(texts, r))));
}

loadStatus().catch((e) => {
  $("server").textContent = `Could not reach the demo: ${e}`;
  $("server").className = "error";
});
