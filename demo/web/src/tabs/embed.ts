// Embeddings: the same texts through each served embedding model, with
// cosine similarity, over GRPCInferenceService ModelMetadata and
// ModelInfer.

import { create } from "@bufbuild/protobuf";
import {
  InferParameterSchema,
  type InferParameter,
  type ModelInferResponse,
  type ModelMetadataResponse,
} from "../gen/open_inference_grpc_pb.js";
import { $, after, el, input, say, stat } from "../dom.js";
import { describe, inference } from "../rpc.js";
import { bundles, capabilities, onCapabilities } from "../state.js";

type Model = { meta: ModelMetadataResponse | null; error: string; checked: boolean };
type Result = { name: string; meta: ModelMetadataResponse | null; ms: number; vectors?: number[][]; reply?: ModelInferResponse; error?: string };

const models = new Map<string, Model>();

function tier(meta: ModelMetadataResponse | null): string {
  const p = meta?.properties["session_info.precision"];
  return p ? after(p, "PRECISION_") : "?";
}

function facts(meta: ModelMetadataResponse): string {
  const p = meta.properties;
  const out: string[] = [];
  if (p["model_info.model_id"]) out.push(p["model_info.model_id"]);
  if (p["model_info.dim"]) out.push(`dim ${p["model_info.dim"]}`);
  if (p["session_info.compute_dtype"]) out.push(after(p["session_info.compute_dtype"], "DTYPE_"));
  const dev = [p["device_info.backend"], p["device_info.name"]].filter(Boolean).join(" ");
  if (dev) out.push(dev);
  return out.join(" \u00b7 ");
}

async function loadMeta(name: string): Promise<void> {
  const m = models.get(name)!;
  try {
    m.meta = await inference.modelMetadata({ name });
    m.error = "";
  } catch (e) {
    m.meta = null;
    m.error = describe(e);
    m.checked = false;
  }
}

function renderModels(): void {
  const box = $("embed-models");
  box.replaceChildren();
  if (models.size === 0) box.append(el("p", { class: "muted", text: "No embedding model is served: add a name below." }));
  for (const [name, m] of models) {
    const cb = el("input", { type: "checkbox" });
    cb.checked = m.checked;
    cb.disabled = !m.meta;
    cb.addEventListener("change", () => (m.checked = cb.checked));
    box.append(
      el(
        "label",
        { class: "model" },
        cb,
        el("span", {}, el("span", { class: "name", text: name }), m.meta ? el("span", { class: "tag", text: tier(m.meta) }) : null),
        el("span", { class: m.error ? "facts error" : "facts", text: m.meta ? facts(m.meta) : m.error }),
      ),
    );
  }
}

async function addModels(names: string[]): Promise<void> {
  const fresh = names.filter((n) => !models.has(n));
  for (const n of fresh) models.set(n, { meta: null, error: "", checked: true });
  await Promise.all(fresh.map(loadMeta));
  renderModels();
}

// ---- Math -----------------------------------------------------------------

export function cosine(a: number[], b: number[]): number {
  let d = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na && nb ? d / Math.sqrt(na * nb) : 0;
}

function norm(a: number[]): number {
  return Math.sqrt(a.reduce((s, x) => s + x * x, 0));
}

export function matrix(vectors: number[][]): number[][] {
  return vectors.map((a) => vectors.map((b) => cosine(a, b)));
}

/** A half-precision float from its bits. */
function half(bits: number): number {
  const s = bits & 0x8000 ? -1 : 1;
  const e = (bits >> 10) & 0x1f;
  const f = bits & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

/** The reply's vectors, rows of the output's shape, from its raw little-endian bytes. */
export function vectorsOf(reply: ModelInferResponse): number[][] {
  const out = reply.outputs[0];
  const raw = reply.rawOutputContents[0];
  if (!out || !raw) throw new Error("the reply has no vectors");
  const [rows, dim] = out.shape.map(Number);
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const width = out.datatype === "FP32" ? 4 : out.datatype === "FP16" || out.datatype === "BF16" ? 2 : 0;
  if (!width) throw new Error(`vectors in ${out.datatype} are not shown`);
  if (raw.byteLength !== rows * dim * width) throw new Error(`${raw.byteLength} bytes for a ${rows}x${dim} ${out.datatype} tensor`);
  const v: number[][] = [];
  for (let r = 0; r < rows; r++) {
    const row: number[] = [];
    for (let c = 0; c < dim; c++) {
      const at = (r * dim + c) * width;
      if (out.datatype === "FP32") row.push(dv.getFloat32(at, true));
      else if (out.datatype === "FP16") row.push(half(dv.getUint16(at, true)));
      else {
        const f = new DataView(new ArrayBuffer(4));
        f.setUint16(0, dv.getUint16(at, true), false);
        row.push(f.getFloat32(0, false));
      }
    }
    v.push(row);
  }
  return v;
}

// ---- Running --------------------------------------------------------------

function param(p: InferParameter["parameterChoice"]): InferParameter {
  return create(InferParameterSchema, { parameterChoice: p });
}

function parameters(): Record<string, InferParameter> {
  const p: Record<string, InferParameter> = {};
  const role = (document.getElementById("embed-role") as HTMLSelectElement).value;
  if (role) p.prompt_role = param({ case: "stringParam", value: role });
  const max = parseInt(input("embed-max").value, 10);
  if (max > 0) {
    p.max_tokens = param({ case: "int64Param", value: BigInt(max) });
    p.truncate = param({ case: "stringParam", value: "TRUNCATE_RIGHT" });
  }
  return p;
}

async function run(): Promise<void> {
  const texts = (document.getElementById("embed-texts") as HTMLTextAreaElement).value
    .split("\n")
    .map((t) => t.trim())
    .filter((t) => t.length);
  const chosen = [...models].filter(([, m]) => m.checked && m.meta).map(([n]) => n);
  const err = $("embed-error");
  say(err, "");
  if (!texts.length || !chosen.length) {
    say(err, !texts.length ? "Enter at least one text." : "Pick at least one model.", "error");
    return;
  }
  const btn = $("embed-run") as HTMLButtonElement;
  btn.disabled = true;
  btn.textContent = "Embedding\u2026";
  const encoder = new TextEncoder();
  const results: Result[] = [];
  // One request at a time, so no call waits behind another.
  for (const name of chosen) {
    const t0 = performance.now();
    const meta = models.get(name)!.meta;
    try {
      const reply = await inference.modelInfer({
        modelName: name,
        parameters: parameters(),
        inputs: [{ name: "texts", datatype: "BYTES", shape: [BigInt(texts.length)], contents: { bytesContents: texts.map((t) => encoder.encode(t)) } }],
      });
      results.push({ name, meta, ms: performance.now() - t0, reply, vectors: vectorsOf(reply) });
    } catch (e) {
      results.push({ name, meta, ms: performance.now() - t0, error: describe(e) });
    }
  }
  btn.disabled = false;
  btn.textContent = "Embed";
  render(texts, results);
}

// ---- Results --------------------------------------------------------------

export function heat(v: number): string {
  const a = Math.max(0, Math.min(1, v));
  return `background: rgba(var(--heat), ${(a * a * 0.85).toFixed(3)})`;
}

// Cuts `t` to at most `n` characters, the last an ellipsis when it was cut.
// A character is a grapheme cluster where the browser has Intl.Segmenter,
// else a code point, so a cut never splits a surrogate pair (and, with
// graphemes, never an emoji sequence or a letter from its combining mark).
export function short(t: string, n = 28): string {
  const chars = typeof Intl !== "undefined" && Intl.Segmenter
    ? Array.from(new Intl.Segmenter(undefined, { granularity: "grapheme" }).segment(t), (g) => g.segment)
    : Array.from(t);
  return chars.length > n ? chars.slice(0, n - 1).join("") + "\u2026" : t;
}

function card(texts: string[], r: Result): HTMLElement {
  const head = el("div", { class: "head" }, el("h3", {}, r.name, el("span", { class: "tag", text: tier(r.meta) })));
  if (!r.vectors || !r.reply) {
    return el("div", { class: "card" }, head, el("p", { class: "error", text: r.error ?? "" }), stat(`${r.ms.toFixed(1)} ms`, "browser round trip"));
  }
  const v = r.vectors;
  const p = r.reply.parameters;
  const str = (k: string) => (p[k]?.parameterChoice.case === "stringParam" ? String(p[k].parameterChoice.value) : "");
  const sim = matrix(v);
  const simTable = el(
    "table",
    { class: "sim" },
    el("tr", {}, el("th", {}), texts.map((_, j) => el("th", { text: `#${j + 1}` }))),
    texts.map((t, i) =>
      el(
        "tr",
        {},
        el("td", { class: "text", title: t, text: `#${i + 1} ${short(t, 22)}` }),
        sim[i].map((x) => el("td", { class: "cell", style: heat(x), text: x.toFixed(3) })),
      ),
    ),
  );
  const previews = el(
    "table",
    {},
    el("tr", {}, el("th", { class: "text", text: "text" }), el("th", { text: "norm" }), el("th", { class: "text", text: "first values" })),
    v.map((row, i) =>
      el(
        "tr",
        {},
        el("td", { class: "text", title: texts[i], text: `#${i + 1} ${short(texts[i], 18)}` }),
        el("td", { text: norm(row).toFixed(4) }),
        el("td", {
          class: "vec",
          text: row.slice(0, 6).map((x) => (x < 0 ? "" : " ") + x.toFixed(4)).join(" ") + (row.length > 6 ? "  \u2026" : ""),
        }),
      ),
    ),
  );
  const device = [str("backend"), str("arch")].filter(Boolean).join(" ");
  return el(
    "div",
    { class: "card" },
    head,
    el(
      "div",
      { class: "stats" },
      stat(String(v[0]?.length ?? 0), "dimensions"),
      stat(String(v.length), "texts"),
      stat(`${r.ms.toFixed(1)} ms`, "browser round trip"),
      device ? stat(device, "ran on") : null,
      str("compute_dtype") ? stat(after(str("compute_dtype"), "DTYPE_"), "computed in") : null,
    ),
    el("h2", { text: "Cosine similarity" }),
    simTable,
    el("details", {}, el("summary", { text: "Vectors" }), previews),
  );
}

// The same texts through every model: each pair's similarity side by side,
// and how far each model's vectors are from the first model's.
function comparison(texts: string[], ok: Result[]): HTMLElement {
  const sims = ok.map((r) => matrix(r.vectors!));
  const rows: HTMLElement[] = [];
  let widest = 0;
  for (let i = 0; i < texts.length; i++) {
    for (let j = i + 1; j < texts.length; j++) {
      const vals = sims.map((m) => m[i][j]);
      const spread = Math.max(...vals) - Math.min(...vals);
      widest = Math.max(widest, spread);
      rows.push(
        el(
          "tr",
          {},
          el("td", { class: "text", title: `${texts[i]}\n${texts[j]}`, text: `#${i + 1} \u00b7 #${j + 1}` }),
          vals.map((x) => el("td", { class: "cell", style: heat(x), text: x.toFixed(4) })),
          el("td", { text: spread === 0 ? "0" : spread.toExponential(1) }),
        ),
      );
    }
  }
  const baseV = ok[0].vectors!;
  const agree = ok.map((r) => {
    const v = r.vectors!;
    if (v[0].length !== baseV[0].length) return "n/a";
    return Math.min(...v.map((row, i) => cosine(row, baseV[i]))).toFixed(6);
  });
  const tableEl = el(
    "table",
    { class: "sim cmp" },
    el("tr", {}, el("th", { class: "text", text: "pair" }), ok.map((r) => el("th", { text: `${r.name} (${tier(r.meta)})` })), el("th", { text: "spread" })),
    rows,
    el("tr", {}, el("th", { class: "text", text: `lowest cosine to ${ok[0].name}, same text` }), agree.map((a) => el("td", { text: a })), el("td", {})),
  );
  return el(
    "div",
    { class: "card" },
    el("h3", { text: "Side by side" }),
    el("p", {
      class: "muted",
      text: "Each pair of texts, scored by every model picked. The spread is how much the tiers disagree; the last row is how close each model's vector for a text is to the first model's.",
    }),
    tableEl,
    widest === 0
      ? el("p", { class: "note", text: "Every model gave the same numbers: on this device these tiers compute alike. A device that runs FASTEST in a narrower type shows the difference here." })
      : null,
  );
}

function render(texts: string[], results: Result[]): void {
  const out = $("embed-results");
  out.replaceChildren();
  const ok = results.filter((r) => r.vectors);
  if (ok.length > 1 && texts.length > 1) out.append(comparison(texts, ok));
  out.append(el("div", { class: "cards" }, results.map((r) => card(texts, r))));
}

export function init(): void {
  $("embed-add").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const name = input("embed-add-name").value.trim();
    if (!name) return;
    await addModels([name]);
    input("embed-add-name").value = "";
  });
  $("embed-run").addEventListener("click", run);
  onCapabilities((c) => void addModels(bundles(c, "TASK_EMBED")));
  capabilities().catch((e) => say($("embed-error"), describe(e), "error"));
}
