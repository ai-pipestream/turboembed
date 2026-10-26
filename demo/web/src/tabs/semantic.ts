// Semantic: one query against a few passages, through every served task.
// An embedding bundle gives the passages' similarity matrix and the
// query's nearest neighbours by cosine; a rerank bundle scores each
// (query, passage) pair as a cross-encoder; an extractive QA bundle
// finds answer spans in the passages. All of it is ModelInfer.

import { create } from "@bufbuild/protobuf";
import { InferParameterSchema, type InferParameter, type ModelInferResponse } from "../gen/open_inference_grpc_pb.js";
import type { CapabilitiesResponse } from "../gen/turbo_analysis_pb.js";
import { $, el, input, options, say, table } from "../dom.js";
import { describe, inference } from "../rpc.js";
import { capabilities, onCapabilities } from "../state.js";
import { cosine, heat, matrix, short, vectorsOf } from "./embed.js";

const utf8 = new TextEncoder();
const select = (id: string) => $(id) as HTMLSelectElement;
let caps: CapabilitiesResponse | null = null;

function param(p: InferParameter["parameterChoice"]): InferParameter {
  return create(InferParameterSchema, { parameterChoice: p });
}

function texts(name: string, values: string[]) {
  return { name, datatype: "BYTES", shape: [BigInt(values.length)], contents: { bytesContents: values.map((t) => utf8.encode(t)) } };
}

/** Output `name` of `reply` as a DataView over its raw bytes, with its shape. */
export function output(reply: ModelInferResponse, name: string): { dv: DataView; shape: number[]; datatype: string } {
  const i = reply.outputs.findIndex((o) => o.name === name);
  const raw = reply.rawOutputContents[i];
  if (i < 0 || !raw) throw new Error(`the reply has no ${name}`);
  return { dv: new DataView(raw.buffer, raw.byteOffset, raw.byteLength), shape: reply.outputs[i].shape.map(Number), datatype: reply.outputs[i].datatype };
}

function role(bundle: string, r: string): Record<string, InferParameter> {
  const b = caps?.bundles.find((x) => x.name === bundle);
  return b?.roles.includes(r) ? { prompt_role: param({ case: "stringParam", value: r }) } : {};
}

async function embed(model: string, query: string, passages: string[]): Promise<void> {
  const out = $("sem-sim");
  // In turns: a server started with one session refuses a second call at once.
  const q = await inference.modelInfer({ modelName: model, parameters: role(model, "PROMPT_QUERY"), inputs: [texts("texts", [query])] });
  const p = await inference.modelInfer({ modelName: model, parameters: role(model, "PROMPT_DOCUMENT"), inputs: [texts("texts", passages)] });
  const qv = vectorsOf(q)[0];
  const pv = vectorsOf(p);
  const sim = matrix(pv);
  const k = Math.max(1, parseInt(input("sem-k").value, 10) || 3);
  const near = pv.map((v, i) => ({ i, s: cosine(qv, v) })).sort((a, b) => b.s - a.s);
  out.replaceChildren(
    el("h3", { text: `Nearest neighbours (${model})` }),
    table(["rank", "passage", "cosine"], near.slice(0, k).map((n, r) => [String(r + 1), el("span", { class: "quote", title: passages[n.i], text: `#${n.i + 1} ${short(passages[n.i], 60)}` }), n.s.toFixed(4)])),
    el("h3", { text: "Passage similarity" }),
    el(
      "table",
      { class: "sim" },
      el("tr", {}, el("th", {}), passages.map((_, j) => el("th", { text: `#${j + 1}` }))),
      sim.map((row, i) => el("tr", {}, el("td", { class: "text", title: passages[i], text: `#${i + 1} ${short(passages[i], 22)}` }), row.map((x) => el("td", { class: "cell", style: heat(x), text: x.toFixed(3) })))),
    ),
  );
}

async function rerank(model: string, query: string, passages: string[]): Promise<void> {
  // One call scores at most the sessions' batch of pairs; more go in turns.
  const meta = await inference.modelMetadata({ name: model });
  const limits = ["model_info.max_batch", "session_info.max_batch"].map((k) => parseInt(meta.properties[k] ?? "", 10)).filter((n) => n > 0);
  const batch = limits.length ? Math.min(...limits) : passages.length;
  const logits: number[] = [];
  for (let at = 0; at < passages.length; at += batch) {
    const part = passages.slice(at, at + batch);
    const r = await inference.modelInfer({ modelName: model, inputs: [texts("query", [query]), texts("passages", part)] });
    const { dv } = output(r, "scores");
    part.forEach((_, i) => logits.push(dv.getFloat32(i * 4, true)));
  }
  const scored = logits.map((s, i) => ({ i, s })).sort((a, b) => b.s - a.s);
  $("sem-rerank-out").replaceChildren(
    el("h3", { text: `Reranked (${model})` }),
    el("p", { class: "muted", text: "Relevance logits from the cross-encoder, one per (query, passage) pair; higher is more relevant." }),
    table(["rank", "passage", "logit", "sigmoid"], scored.map((x, r) => [String(r + 1), el("span", { class: "quote", title: passages[x.i], text: `#${x.i + 1} ${short(passages[x.i], 60)}` }), x.s.toFixed(4), (1 / (1 + Math.exp(-x.s))).toFixed(4)])),
  );
}

async function answer(model: string, question: string, context: string): Promise<void> {
  const topK = Math.max(1, parseInt(input("sem-topk").value, 10) || 3);
  const r = await inference.modelInfer({
    modelName: model,
    parameters: { top_k: param({ case: "int64Param", value: BigInt(topK) }) },
    inputs: [texts("question", [question]), texts("context", [context])],
  });
  const spans = output(r, "spans");
  const scores = output(r, "scores");
  const probs = output(r, "probabilities");
  const bytes = utf8.encode(context);
  const dec = new TextDecoder();
  const rows = [];
  for (let i = 0; i < spans.shape[0]; i++) {
    const [s, e] = [spans.dv.getInt32(i * 8, true), spans.dv.getInt32(i * 8 + 4, true)];
    rows.push([String(i + 1), el("span", { class: "quote", text: dec.decode(bytes.subarray(s, e)) }), `${s}..${e}`, scores.dv.getFloat64(i * 8, true).toFixed(3), probs.dv.getFloat64(i * 8, true).toFixed(4)]);
  }
  const p = r.parameters;
  const abstained = p.abstained?.parameterChoice.case === "boolParam" && p.abstained.parameterChoice.value;
  const nullP = p.null_probability?.parameterChoice.case === "doubleParam" ? p.null_probability.parameterChoice.value : NaN;
  $("sem-qa-out").replaceChildren(
    el("h3", { text: `Answers (${model})` }),
    el("p", { class: abstained ? "error" : "muted", text: `${abstained ? "The model abstains: no answer clears the null score." : "Spans in the passages, as UTF-8 byte offsets."}${Number.isNaN(nullP) ? "" : ` No-answer probability ${nullP.toFixed(4)}.`}` }),
    rows.length ? table(["#", "answer", "bytes", "score", "probability"], rows) : el("p", { class: "muted", text: "No span." }),
  );
}

async function run(): Promise<void> {
  const note = $("sem-note");
  const query = input("sem-query").value.trim();
  const passages = ($("sem-passages") as HTMLTextAreaElement).value.split("\n").map((t) => t.trim()).filter((t) => t.length);
  const [e, rr, qa] = [select("sem-embed").value, select("sem-rerank").value, select("sem-qa").value];
  for (const id of ["sem-sim", "sem-rerank-out", "sem-qa-out"]) $(id).replaceChildren();
  if (!query || !passages.length) return say(note, "Enter a query and at least one passage.", "error");
  if (!e && !rr && !qa) return say(note, "Pick at least one model.", "error");
  say(note, "Running\u2026", "busy");
  const jobs: [string, string, () => Promise<void>][] = [
    [e, "sem-sim", () => embed(e, query, passages)],
    [rr, "sem-rerank-out", () => rerank(rr, query, passages)],
    [qa, "sem-qa-out", () => answer(qa, query, passages.join("\n"))],
  ];
  await Promise.all(
    jobs.filter(([m]) => m).map(([, box, job]) => job().catch((err) => $(box).replaceChildren(el("p", { class: "error", text: describe(err) })))),
  );
  say(note, "");
}

export function init(): void {
  $("sem-run").addEventListener("click", () => void run());
  onCapabilities((c) => {
    caps = c;
    for (const [id, task] of [["sem-embed", "TASK_EMBED"], ["sem-rerank", "TASK_RERANK"], ["sem-qa", "TASK_EXTRACTIVE_QA"]]) {
      const names = c.bundles.filter((b) => b.task === task).map((b) => b.name);
      const first = select(id).options.length === 0;
      options(select(id), ["", ...names], ["none", ...names]);
      if (first && names.length) select(id).value = names[0];
    }
  });
  capabilities().catch((err) => say($("sem-note"), describe(err), "error"));
}
