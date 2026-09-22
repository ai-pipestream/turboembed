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
