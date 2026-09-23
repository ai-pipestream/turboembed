// The page over /api/v1. Every refusal is shown exactly as the server sent it,
// including the TURBO_E_* status name and the field index when libturbo named
// one; nothing here retries, clamps or hides a failure.
"use strict";

const $ = (id) => document.getElementById(id);
const state = { models: [], byTask: {} };

/* ---------------------------------------------------------------- helpers */

function fmt(ms) {
  return ms >= 100 ? ms.toFixed(0) : ms.toFixed(ms >= 10 ? 1 : 2);
}

function deviceOf(d) {
  return `${d.name} (${d.provider_id}:${d.ordinal})`;
}

function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

/** The message a refusal carries: the server's own words plus its code. */
function refusal(body, fallback) {
  if (!body || typeof body !== "object") return fallback;
  let text = body.error || fallback;
  if (body.status && !text.includes(body.status)) text = `${body.status}: ${text}`;
  if (body.field) text += ` (field ${body.field})`;
  return text;
}

async function call(path, payload) {
  const t0 = performance.now();
  const response = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const text = await response.text();
  let body = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch (e) {
      throw new Error(`${path} answered ${response.status} with a body that is not JSON: ${text.slice(0, 200)}`);
    }
  }
  if (!response.ok) throw new Error(refusal(body, `${path} answered ${response.status}`));
  return { body, roundTrip: performance.now() - t0 };
}

function showError(box, statusEl, e) {
  if (statusEl) statusEl.textContent = "";
  box.textContent = String(e && e.message ? e.message : e);
  box.hidden = false;
}

function lines(value) {
  return value.split("\n").map((s) => s.trim()).filter(Boolean);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* ------------------------------------------------------------------- tabs */

const tabs = Array.from(document.querySelectorAll('[role="tab"]'));

function select(tab) {
  for (const t of tabs) {
    const on = t === tab;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    $(t.getAttribute("aria-controls")).hidden = !on;
  }
  tab.focus();
  if (tab.id === "tab-devices") loadDevices();
  if (tab.id === "tab-benchmarks") loadBenchmarks();
}

for (const tab of tabs) {
  tab.addEventListener("click", () => select(tab));
  tab.addEventListener("keydown", (e) => {
    const usable = tabs.filter((t) => !t.disabled);
    const i = usable.indexOf(tab);
    if (i < 0) return;
    let next = null;
    if (e.key === "ArrowRight") next = usable[(i + 1) % usable.length];
    else if (e.key === "ArrowLeft") next = usable[(i - 1 + usable.length) % usable.length];
    else if (e.key === "Home") next = usable[0];
    else if (e.key === "End") next = usable[usable.length - 1];
    if (next) {
      e.preventDefault();
      select(next);
    }
  });
}

function runOnCtrlEnter(input, button) {
  input.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && !button.disabled) {
      e.preventDefault();
      button.click();
    }
  });
}

/* ------------------------------------------------------------------- info */

async function loadInfo() {
  const response = await fetch("/api/v1/models");
  if (!response.ok) throw new Error(`GET /api/v1/models answered ${response.status}`);
  state.models = await response.json();
  state.byTask = {};
  for (const m of state.models) {
    if (!state.byTask[m.task]) state.byTask[m.task] = m;
  }

  const embed = state.byTask.EMBED;
  if (embed) {
    $("device").textContent =
      `${embed.model_id} (dim ${embed.dim}, max_seq ${embed.max_seq}, batch ${embed.max_batch})` +
      ` on ${embed.device.name} (${embed.device.provider_id}:${embed.device.ordinal} ${embed.device.kind}),` +
      ` runtime ${embed.device.runtime_version}` +
      (embed.fully_accelerated ? "" : `; host stages: ${hostStages(embed)}`);
  } else {
    $("device").textContent = "no embedding bundle configured (start with --turbo.bundle=<dir>)";
    disable("tab-embed", "embed");
  }

  const rerank = state.byTask.RERANK;
  if (rerank) {
    $("reranker").textContent =
      `${rerank.model_id} (max_seq ${rerank.max_seq}, batch ${rerank.max_batch}) on ${deviceOf(rerank.device)}`;
  } else {
    $("reranker").textContent = "no reranker bundle configured (start with --turbo.models[n].bundle=<dir>)";
    disable("tab-rerank", "rerank");
  }

  const tokenizerModel = state.byTask.EMBED || state.models[0];
  if (tokenizerModel) {
    $("tokenizer").textContent = `tokenizer of ${tokenizerModel.tokenizer_bundle}`;
  } else {
    $("tokenizer").textContent = "no model is loaded, so there is no tokenizer";
    disable("tab-tokenize", "tokenize");
  }

  const gen = state.byTask.GENERATE;
  if (gen) {
    $("generator").textContent =
      `${gen.model_id} (max_seq ${gen.max_seq}) on ${deviceOf(gen.device)}`;
  } else {
    $("generator").textContent = "no generative bundle configured (start with --turbo.generate-bundle=<dir>)";
    disable("tab-summarize", "summarize");
  }
}

function hostStages(model) {
  return Object.entries(model.stage_placement)
    .filter(([, place]) => place === "host")
    .map(([stage]) => stage)
    .join(", ");
}

function disable(tabId, buttonId) {
  $(tabId).disabled = true;
  $(tabId).tabIndex = -1;
  $(buttonId).disabled = true;
}

/* ------------------------------------------------------------------ embed */

function heat(v) {
  // -1..1 -> red..grey..green, legible against white text in both schemes.
  const t = Math.max(-1, Math.min(1, v));
  const g = t >= 0 ? Math.round(110 + 80 * t) : 110;
  const r = t < 0 ? Math.round(110 + 80 * -t) : 110;
  return `rgb(${r}, ${g}, 110)`;
}

async function embed() {
  const texts = lines($("texts").value);
  $("error").hidden = true;
  $("embed").disabled = true;
  $("status").textContent = `embedding ${texts.length}…`;
  try {
    const { body, roundTrip } = await call("/api/v1/similarity", { texts });
    $("status").textContent =
      `${body.count} × ${body.dim} in ${fmt(body.timings.total_ms)} ms on ${deviceOf(body.device)},` +
      ` placement ${body.placement}, ${fmt(roundTrip)} ms round trip`;

    const table = $("matrix");
    table.replaceChildren();
    const head = table.insertRow();
    head.appendChild(el("th"));
    body.texts.forEach((_, j) => head.appendChild(el("th", null, String(j + 1))));
    body.similarity.forEach((row, i) => {
      const tr = table.insertRow();
      const th = el("th", "text", `${i + 1}. ${body.texts[i]}`);
      th.title = body.texts[i];
      tr.appendChild(th);
      row.forEach((v) => {
        const td = tr.insertCell();
        td.className = "cell";
        td.style.background = heat(v);
        td.textContent = v.toFixed(3);
      });
    });
    $("vectors").textContent = body.vectors
      .map((v, i) => `${i + 1}: [${v.slice(0, 8).map((x) => x.toFixed(4)).join(", ")}${v.length > 8 ? ", …" : ""}]`)
      .join("\n");
    $("result").hidden = false;
  } catch (e) {
    showError($("error"), $("status"), e);
  } finally {
    $("embed").disabled = false;
  }
}

/* ----------------------------------------------------------------- rerank */

async function rerank() {
  const query = $("query").value.trim();
  const documents = lines($("documents").value);
  $("rerank-error").hidden = true;
  $("rerank").disabled = true;
  $("rerank-status").textContent = `scoring ${documents.length}…`;
  try {
    const payload = { query, documents, options: { return_sorted: $("rerank-sorted").checked } };
    const { body, roundTrip } = await call("/api/v1/rerank", payload);
    $("rerank-status").textContent =
      `${plural(body.results.length, "document")} in ${fmt(body.timings.total_ms)} ms on ${deviceOf(body.device)},` +
      ` placement ${body.placement}, ${fmt(roundTrip)} ms round trip`;

    const rows = body.sorted ? body.sorted.map((i) => body.results[i]) : body.results.slice();
    const table = $("ranking");
    table.replaceChildren();
    const head = table.createTHead().insertRow();
    ["#", "score", "document"].forEach((t) => head.appendChild(el("th", t === "document" ? "text" : null, t)));
    const tbody = table.createTBody();
    rows.forEach((hit, position) => {
      const tr = tbody.insertRow();
      tr.appendChild(el("td", "num", String(body.sorted ? position + 1 : hit.index + 1)));
      tr.appendChild(el("td", "num", hit.score.toFixed(4)));
      const doc = el("td", "text", hit.document);
      doc.title = hit.document;
      tr.appendChild(doc);
    });
    $("rerank-result").hidden = false;
  } catch (e) {
    showError($("rerank-error"), $("rerank-status"), e);
  } finally {
    $("rerank").disabled = false;
  }
}

/* --------------------------------------------------------------- tokenize */

async function tokenize() {
  const texts = lines($("tok-text").value);
  $("tok-error").hidden = true;
  $("tokenize").disabled = true;
  $("tok-status").textContent = `tokenizing ${texts.length}…`;
  try {
    const payload = { texts, add_special_tokens: $("tok-specials").checked };
    const { body, roundTrip } = await call("/api/v1/tokenize", payload);
    const total = body.results.reduce((n, r) => n + r.count, 0);
    $("tok-status").textContent =
      `${plural(body.results.length, "text")}, ${plural(total, "token")} in ${fmt(body.timings.total_ms)} ms` +
      ` with the ${body.tokenizer.kind} tokenizer (vocab ${body.tokenizer.vocab_size}, max_seq` +
      ` ${body.tokenizer.max_seq}), ${fmt(roundTrip)} ms round trip`;

    const out = $("tokens");
    out.replaceChildren();
    for (const row of body.results) {
      const card = el("div", "tokrow");
      card.appendChild(el("p", "src", `${row.index + 1}. ${row.text} (${plural(row.count, "token")})`));
      const strip = el("div", "toks");
      row.ids.forEach((id, i) => {
        const piece = row.tokens[i];
        const chip = el("span", /^\[.*\]$/.test(piece) || /^<.*>$/.test(piece) ? "tok special" : "tok");
        chip.appendChild(document.createTextNode(piece));
        chip.appendChild(el("small", null, String(id)));
        strip.appendChild(chip);
      });
      card.appendChild(strip);
      out.appendChild(card);
    }
    $("tok-result").hidden = false;
  } catch (e) {
    showError($("tok-error"), $("tok-status"), e);
  } finally {
    $("tokenize").disabled = false;
  }
}

/* --------------------------------------------------------------- generate */

let generation = null;

async function summarize() {
  const text = $("document").value;
  $("gen-error").hidden = true;
  $("summary").textContent = "";
  $("summary").hidden = true; // shown with the first chunk
  $("summarize").disabled = true;
  $("stop").hidden = false;
  $("gen-status").textContent = "generating…";
  const t0 = performance.now();
  let generated = 0;
  generation = new AbortController();
  try {
    const response = await fetch("/api/v1/generate/stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        messages: [
          { role: "system", content: "You summarize text. Reply with a summary of at most three sentences and nothing else." },
          { role: "user", content: text },
        ],
        options: { max_tokens: 160 },
      }),
      signal: generation.signal,
    });
    if (!response.ok) {
      const raw = await response.text();
      let body = null;
      try {
        body = JSON.parse(raw);
      } catch (e) {
        throw new Error(`POST /api/v1/generate/stream answered ${response.status}: ${raw.slice(0, 200)}`);
      }
      throw new Error(refusal(body, `POST /api/v1/generate/stream answered ${response.status}`));
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        let event = "message";
        let data = "";
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (!data) continue;
        const payload = JSON.parse(data);
        if (event === "chunk") {
          $("summary").hidden = false;
          $("summary").textContent += payload.text;
          generated = payload.generated;
        } else if (event === "done") {
          const seconds = (performance.now() - t0) / 1000;
          $("gen-status").textContent =
            `${payload.generated_tokens} tokens in ${seconds.toFixed(1)} s` +
            ` (${(payload.generated_tokens / seconds).toFixed(1)} tok/s), finish ${payload.finish_reason},` +
            ` prompt ${payload.prompt_tokens} tokens, on ${deviceOf(payload.device)}`;
        } else if (event === "error") {
          throw new Error(refusal(payload, "the generation failed"));
        }
      }
    }
  } catch (e) {
    if (e && e.name === "AbortError") {
      $("gen-status").textContent = `stopped after ${generated} tokens`;
    } else {
      $("gen-status").textContent = generated ? `stopped after ${generated} tokens` : "";
      showError($("gen-error"), null, e);
    }
  } finally {
    generation = null;
    $("stop").hidden = true;
    $("summarize").disabled = false;
  }
}

function stopGeneration() {
  if (generation) generation.abort();
}

/* ---------------------------------------------------------------- devices */

let devicesLoaded = false;

async function loadDevices() {
  if (devicesLoaded) return;
  try {
    const response = await fetch("/api/v1/devices");
    if (!response.ok) throw new Error(`GET /api/v1/devices answered ${response.status}`);
    const devices = await response.json();
    const out = $("devices");
    out.replaceChildren();
    for (const d of devices) {
      const card = el("div", "card");
      card.appendChild(el("h4", null, `${d.index}. ${d.name} (${d.provider_id}:${d.ordinal} ${d.kind})`));
      card.appendChild(el("p", "muted",
        `provider ${d.provider_version}, runtime ${d.runtime_version}` +
        (d.driver_version ? `, driver ${d.driver_version}` : "") +
        (d.memory_total ? `, memory ${(d.memory_total / 1073741824).toFixed(1)} GiB` : "") +
        (d.models.length ? `, serving ${d.models.join(", ")}` : ", no model loaded here")));
      const chips = el("div", "chips");
      for (const feature of d.features) chips.appendChild(el("span", "chip on", feature));
      card.appendChild(chips);
      const grid = el("div", "grid");
      for (const cell of d.capabilities) {
        if (cell.status === "UNSUPPORTED") continue;
        const box = el("div", `cap ${cell.status}`);
        box.appendChild(el("b", null, `${cell.task} / ${cell.modality}`));
        box.appendChild(document.createTextNode(
          `${cell.status}${cell.dtype ? `, ${cell.dtype}` : ""}${cell.deterministic ? ", deterministic" : ""}`));
        box.title = cell.notes || "";
        grid.appendChild(box);
      }
      if (!grid.childElementCount) grid.appendChild(el("div", "cap UNSUPPORTED", "no task is offered on this device"));
      card.appendChild(grid);
      out.appendChild(card);
    }

    const models = $("models");
    models.replaceChildren();
    for (const m of state.models) {
      const card = el("div", "card");
      card.appendChild(el("h4", null, `${m.name}: ${m.model_id}`));
      card.appendChild(el("p", "muted",
        `${m.task} / ${m.kind} / ${m.modality}, dim ${m.dim}, max_seq ${m.max_seq}, batch ${m.max_batch}` +
        `, dtype ${m.dtype}, ${m.fully_accelerated ? "fully accelerated" : `host stages: ${hostStages(m)}`}` +
        ` on ${deviceOf(m.device)}`));
      const chips = el("div", "chips");
      if (m.pooling) chips.appendChild(el("span", "chip", `pooling ${m.pooling}`));
      if (m.normalize) chips.appendChild(el("span", "chip", `normalize ${m.normalize}`));
      for (const label of m.labels) chips.appendChild(el("span", "chip on", label));
      if (m.prefix_query) chips.appendChild(el("span", "chip", `query prefix "${m.prefix_query}"`));
      if (m.prefix_document) chips.appendChild(el("span", "chip", `document prefix "${m.prefix_document}"`));
      card.appendChild(chips);
      models.appendChild(card);
    }
    devicesLoaded = true;
  } catch (e) {
    showError($("devices-error"), null, e);
  }
}

/* ------------------------------------------------------------- benchmarks */

let benchmarksLoaded = false;

/** A throughput figure, grouped, or "none" when the receipt carries none. */
function rate(value) {
  return value === null || value === undefined ? "none" : Math.round(value).toLocaleString("en-US");
}

function ms(value) {
  return value === null || value === undefined ? "none" : fmt(value);
}

function ratioText(ratio) {
  return `${ratio.toFixed(2)}x`;
}

/** The ratio bar: 1.0 is the centre line, 4x is the right edge and 0.25x the left. */
function ratioBar(ratio) {
  const bar = el("span", "bar");
  const fill = el("i");
  const f = Math.max(-1, Math.min(1, Math.log2(ratio) / 2));
  const width = Math.abs(f) * 50;
  fill.style.left = `${f >= 0 ? 50 : 50 - width}%`;
  fill.style.width = `${width}%`;
  bar.appendChild(fill);
  return bar;
}

/** The cell with the highest and the one with the lowest ratio. */
function extremes(cells) {
  let best = cells[0];
  let worst = cells[0];
  for (const c of cells) {
    if (c.ratio > best.ratio) best = c;
    if (c.ratio < worst.ratio) worst = c;
  }
  return { best, worst };
}

/** "libturbo at 1.58x of onnxruntime-cuda", or a range when the cells differ. */
function headline(comparison) {
  const { best, worst } = extremes(comparison.cells);
  const span = best.ratio === worst.ratio
    ? ratioText(best.ratio)
    : `${ratioText(worst.ratio)} to ${ratioText(best.ratio)}`;
  return `libturbo at ${span} of ${comparison.native_provider}`;
}

function row(parent, cells) {
  const tr = parent.insertRow();
  for (const cell of cells) tr.appendChild(cell);
  return tr;
}

function th(text, className) {
  return el("th", className, text);
}

function td(text, className) {
  return el("td", className, text);
}

function table(caption) {
  const t = document.createElement("table");
  if (caption) {
    const cap = el("caption", "receipt", caption);
    t.appendChild(cap);
  }
  return t;
}

function head(t, labels) {
  const thead = t.createTHead();
  const tr = thead.insertRow();
  for (const [text, className] of labels) tr.appendChild(th(text, className));
  return t.createTBody();
}

function benchSummary(report) {
  const out = $("bench-summary");
  out.replaceChildren();
  const t = table(null);
  t.id = "bench-table";
  t.className = "bench-summary";
  const body = head(t, [
    ["device", "text"], ["task", "text"], ["libturbo vs the runtime alone", "text"],
    ["best cell", "text"], ["worst cell", "text"], ["verdict", "text"], ["receipt", "text"],
  ]);
  for (const c of report.comparisons) {
    const { best, worst } = extremes(c.cells);
    const under = c.cells.filter((cell) => !cell.within_floor);
    const tr = row(body, [
      td(c.device, "text"),
      td(c.task, "text"),
      td(`${c.provider} vs ${c.native_provider}`, "text"),
      td(`${ratioText(best.ratio)} ${best.cell}`, "text"),
      td(`${ratioText(worst.ratio)} ${worst.cell}`, "text"),
      td("", "text"),
      td(c.file, "text receipt"),
    ]);
    tr.className = `bench-row ${c.verdict}`;
    tr.dataset.file = c.file;
    const verdict = tr.cells[5];
    verdict.appendChild(el("span", `chip ${c.verdict}`, c.verdict));
    if (under.length) {
      verdict.appendChild(el("span", "under note",
        `${under.length} of ${c.cells.length} ${under.length === 1 ? "cell" : "cells"} under ${c.floor}`));
    }
    if (c.unmatched.length) {
      verdict.appendChild(el("span", "under note", plural(c.unmatched.length, "unmatched cell")));
    }
    tr.cells[3].title = best.measure;
    tr.cells[4].title = worst.measure;
    tr.cells[2].title = `${c.runtime} against ${c.native_runtime}`;
  }
  out.appendChild(t);
}

function benchCells(report) {
  const out = $("bench-cells");
  out.replaceChildren();
  for (const c of report.comparisons) {
    const box = document.createElement("details");
    box.className = "bench-detail";
    box.dataset.file = c.file;
    box.appendChild(el("summary", null, `${c.device}, ${c.task}: ${headline(c)} (${c.verdict})`));
    box.appendChild(el("p", "muted",
      `${c.provider} ${c.provider_version} on ${c.runtime}` +
      (c.driver ? `, driver ${c.driver}` : "") +
      ` against ${c.native_provider} on ${c.native_runtime}` +
      (c.native_driver ? `, driver ${c.native_driver}` : "") +
      `; bundle ${c.model_id}; ${c.date}, commit ${c.commit}`));
    const t = table(`receipt ${c.file}`);
    const body = head(t, [
      ["cell", "text"], ["measure", "text"], ["libturbo"], ["the runtime alone"], ["ratio", "text"],
    ]);
    for (const cell of c.cells) {
      const ratio = td("", "text ratio");
      ratio.appendChild(ratioBar(cell.ratio));
      ratio.appendChild(document.createTextNode(ratioText(cell.ratio)));
      if (!cell.within_floor) ratio.appendChild(el("span", "under", ` under ${c.floor}`));
      const tr = row(body, [
        td(cell.cell, "text"),
        td(cell.measure, "text"),
        td(cell.turbo.toFixed(cell.turbo >= 100 ? 1 : 3)),
        td(cell.native.toFixed(cell.native >= 100 ? 1 : 3)),
        ratio,
      ]);
      if (!cell.within_floor) tr.className = "under";
    }
    box.appendChild(t);
    for (const missing of c.unmatched) box.appendChild(el("p", "under", missing));
    out.appendChild(box);
  }
}

function embedTable(run) {
  const t = table(`receipt ${run.file}, ${run.date}, commit ${run.commit}`);
  t.dataset.file = run.file;
  const body = head(t, [
    ["cell", "text"], ["text p50 ms"], ["text rows/s"], ["text tok/s"],
    ["prepared p50 ms"], ["prepared rows/s"], ["prepared tok/s"],
  ]);
  for (const cell of run.embed) {
    const prepared = cell.prepared_tokens || null;
    const tr = row(body, [
      td(`${cell.batch}x${cell.seq}`, "text"),
      td(ms(cell.text.p50_ms)),
      td(rate(cell.text.rows_per_s)),
      td(rate(cell.text.tokens_per_s)),
      td(prepared ? ms(prepared.p50_ms) : "none"),
      td(prepared ? rate(prepared.rows_per_s) : "none"),
      td(prepared ? rate(prepared.tokens_per_s) : "none"),
    ]);
    tr.dataset.cell = `${cell.batch}x${cell.seq}`;
    if (!prepared && cell.prepared_tokens_note) tr.cells[4].title = cell.prepared_tokens_note;
    tr.cells[0].title = `${cell.live_tokens_per_row} live tokens per row` +
      (cell.token_count_source ? `, counted by ${cell.token_count_source}` : "");
  }
  return t;
}

function rerankTable(run) {
  const t = table(`receipt ${run.file}, ${run.date}, commit ${run.commit}`);
  t.dataset.file = run.file;
  const body = head(t, [["cell", "text"], ["p50 ms"], ["documents/s"], ["iterations"]]);
  const r = run.rerank;
  row(body, [
    td(`rerank ${r.docs}x${r.seq}`, "text"),
    td(ms(r.text.p50_ms)),
    td(rate(r.text.rows_per_s)),
    td(String(r.text.iters)),
  ]);
  return t;
}

function generateTable(run) {
  const t = table(`receipt ${run.file}, ${run.date}, commit ${run.commit}`);
  t.dataset.file = run.file;
  const body = head(t, [
    ["cell", "text"], ["ttft p50 ms"], ["decode tok/s"], ["total p50 ms"], ["prompt tokens"], ["iterations"],
  ]);
  const g = run.generate;
  row(body, [
    td(`generate ${g.new_tokens_requested}`, "text"),
    td(ms(g.time_to_first_token_ms_p50)),
    td(rate(g.decode_tokens_per_s_p50)),
    td(ms(g.total_ms_p50)),
    td(String(g.prompt_tokens)),
    td(String(g.iters)),
  ]);
  return t;
}

function benchThroughput(report) {
  const out = $("bench-throughput");
  out.replaceChildren();
  // One card per device as one provider sees it: the same silicon under two
  // providers is two runtimes, so it is two cards.
  const byDevice = new Map();
  for (const run of report.turbo) {
    const key = `${run.provider}\u0000${run.device}`;
    if (!byDevice.has(key)) byDevice.set(key, []);
    byDevice.get(key).push(run);
  }
  for (const runs of byDevice.values()) {
    const first = runs[0];
    const card = el("div", "card bench-device");
    card.dataset.device = first.device;
    card.dataset.provider = first.provider;
    card.appendChild(el("h4", null, `${first.device} (${first.provider}:${first.ordinal} ${first.device_kind})`));
    card.appendChild(el("p", "muted",
      `${first.hostname}, runtime ${first.runtime}` + (first.driver ? `, driver ${first.driver}` : "")));
    for (const run of runs) {
      card.appendChild(el("p", "muted", `${run.task}, ${run.model_id}`));
      if (run.embed.length) card.appendChild(embedTable(run));
      if (run.rerank) card.appendChild(rerankTable(run));
      if (run.generate) card.appendChild(generateTable(run));
    }
    out.appendChild(card);
  }
}

function renderBenchmarks(report) {
  $("bench-dir").textContent =
    `${plural(report.receipt_count, "receipt")} from ${report.directory}`;
  const floors = [...new Set(report.comparisons.map((c) => c.floor))].sort((a, b) => a - b);
  $("bench-rule").textContent = floors.length
    ? `SUPPORTED: every cell at ${floors.join(" or ")} of the runtime alone or better.`
    : "";
  benchSummary(report);
  benchCells(report);
  benchThroughput(report);
}

async function loadBenchmarks() {
  if (benchmarksLoaded) return;
  try {
    const response = await fetch("/api/v1/benchmarks");
    const text = await response.text();
    let body = null;
    if (text) {
      try {
        body = JSON.parse(text);
      } catch (e) {
        throw new Error(`GET /api/v1/benchmarks answered ${response.status} with a body that is not JSON`);
      }
    }
    if (!response.ok) throw new Error(refusal(body, `GET /api/v1/benchmarks answered ${response.status}`));
    renderBenchmarks(body);
    benchmarksLoaded = true;
  } catch (e) {
    $("bench-dir").textContent = "";
    showError($("bench-error"), null, e);
  }
}

/* ------------------------------------------------------------------- wire */

$("embed").addEventListener("click", embed);
$("rerank").addEventListener("click", rerank);
$("tokenize").addEventListener("click", tokenize);
$("summarize").addEventListener("click", summarize);
$("stop").addEventListener("click", stopGeneration);
runOnCtrlEnter($("texts"), $("embed"));
runOnCtrlEnter($("documents"), $("rerank"));
runOnCtrlEnter($("query"), $("rerank"));
runOnCtrlEnter($("tok-text"), $("tokenize"));
runOnCtrlEnter($("document"), $("summarize"));

loadInfo().catch((e) => {
  $("device").textContent = String(e && e.message ? e.message : e);
  for (const tab of tabs) tab.disabled = true;
});
