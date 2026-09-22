// The page talks to /api/info and /api/embed only; errors are shown verbatim.
const $ = (id) => document.getElementById(id);

async function loadInfo() {
  const r = await fetch("/api/info");
  if (!r.ok) throw new Error(`GET /api/info: ${r.status}`);
  const i = await r.json();
  $("device").textContent =
    `${i.modelId} (dim ${i.dim}, max_seq ${i.maxSeq}) on ${i.deviceName} — ${i.providerId}:${i.ordinal} ${i.deviceKind}, runtime ${i.runtimeVersion}` +
    (i.fullyAccelerated ? "" : " — host stages: tokenize" );
}

function heat(v) {
  // -1..1 -> red..white..green, readable in both color schemes.
  const t = Math.max(-1, Math.min(1, v));
  const g = t >= 0 ? Math.round(110 + 80 * t) : 110;
  const r = t < 0 ? Math.round(110 + 80 * -t) : 110;
  return `rgb(${r}, ${g}, 110)`;
}

async function embed() {
  const texts = $("texts").value.split("\n").map((s) => s.trim()).filter(Boolean);
  $("error").hidden = true;
  $("embed").disabled = true;
  $("status").textContent = `embedding ${texts.length}…`;
  const t0 = performance.now();
  try {
    const r = await fetch("/api/embed", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ texts }) });
    const body = await r.json();
    if (!r.ok) throw new Error(body.error || `POST /api/embed: ${r.status}`);
    const ms = (performance.now() - t0).toFixed(1);
    $("status").textContent = `${body.texts.length} × ${body.dim} in ${ms} ms`;
    const table = $("matrix");
    table.replaceChildren();
    const head = table.insertRow();
    head.appendChild(document.createElement("th"));
    body.texts.forEach((_, j) => { const th = document.createElement("th"); th.textContent = j + 1; head.appendChild(th); });
    body.similarity.forEach((row, i) => {
      const tr = table.insertRow();
      const th = document.createElement("th"); th.className = "text"; th.textContent = `${i + 1}. ${body.texts[i]}`; tr.appendChild(th);
      row.forEach((v) => { const td = tr.insertCell(); td.className = "cell"; td.style.background = heat(v); td.textContent = v.toFixed(3); });
    });
    $("vectors").textContent = body.vectors.map((v, i) => `${i + 1}: [${v.slice(0, 8).map((x) => x.toFixed(4)).join(", ")}${v.length > 8 ? ", …" : ""}]`).join("\n");
    $("result").hidden = false;
  } catch (e) {
    $("status").textContent = "";
    $("error").textContent = String(e.message || e);
    $("error").hidden = false;
  } finally {
    $("embed").disabled = false;
  }
}

// Summaries stream as server-sent events over a POST, parsed by hand
// (EventSource is GET-only): "event: chunk" lines carry text, "event: done"
// ends the stream with the finish reason, "event: error" carries the message.
async function summarize() {
  const text = $("document").value;
  $("gen-error").hidden = true;
  $("summary").textContent = "";
  $("summary").hidden = false;
  $("summarize").disabled = true;
  $("gen-status").textContent = "generating…";
  const t0 = performance.now();
  let generated = 0;
  try {
    const r = await fetch("/api/summarize", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text, maxNewTokens: 160 }) });
    if (!r.ok) { const body = await r.json(); throw new Error(body.error || `POST /api/summarize: ${r.status}`); }
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, idx); buffer = buffer.slice(idx + 2);
        let event = "message", data = "";
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (!data) continue;
        const payload = JSON.parse(data);
        if (event === "chunk") { $("summary").textContent += payload.text; generated = payload.generated; }
        else if (event === "done") {
          const s = (performance.now() - t0) / 1000;
          $("gen-status").textContent = `${payload.generated} tokens in ${s.toFixed(1)} s (${(payload.generated / s).toFixed(1)} tok/s), finish ${payload.finish}, prompt ${payload.promptTokens} tokens`;
        } else if (event === "error") { throw new Error(payload.error); }
      }
    }
  } catch (e) {
    $("gen-status").textContent = generated ? `stopped after ${generated} tokens` : "";
    $("gen-error").textContent = String(e.message || e);
    $("gen-error").hidden = false;
  } finally {
    $("summarize").disabled = false;
  }
}

async function loadGenerator() {
  const r = await fetch("/api/info");
  const i = await r.json();
  if (i.generate) {
    $("generator").textContent = `${i.generate.modelId} (max_seq ${i.generate.maxSeq}) on ${i.generate.deviceName} — ${i.generate.providerId}:${i.generate.ordinal}`;
  } else {
    $("generator").textContent = "no generative bundle configured (start with --turbo.generate-bundle=<dir>)";
    $("summarize").disabled = true;
  }
}

$("embed").addEventListener("click", embed);
$("summarize").addEventListener("click", summarize);
loadInfo().catch((e) => { $("device").textContent = String(e.message || e); });
loadGenerator().catch((e) => { $("generator").textContent = String(e.message || e); });
