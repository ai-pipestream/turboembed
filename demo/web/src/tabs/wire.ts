// The DocumentGraph wire inspector: where the analyzed document's binary
// graph spends its bytes (source, views, each layer's provenance and
// codec body), read in the page by graphwire.ts; RoundTripDocumentGraph
// to check the server re-exports the same bytes; and a graph file opened
// from disk, saved, or handed to the other panels.

import { $, download, el, input, lazy, say, size } from "../dom.js";
import { doc, onDoc, setDoc } from "../doc.js";
import { WireError, hex, walkGraph, type Part } from "../graphwire.js";
import { analysis, describe } from "../rpc.js";

const box = () => $("wire-out");
const note = () => $("wire-note");

function bar(p: Part, total: number): HTMLElement {
  const pct = total ? (100 * p.len) / total : 0;
  return el("span", { class: "share", title: `${pct.toFixed(1)}%` }, el("span", { style: `width: ${pct.toFixed(2)}%` }));
}

function tree(bytes: Uint8Array, p: Part, total: number): HTMLElement {
  const line = el(
    "span",
    { class: "wire", title: `bytes ${p.at}..${p.at + p.len}: ${hex(bytes, p)}` },
    el("b", { text: p.name }),
    p.value ? el("span", { class: "muted", text: ` ${p.value}` }) : "",
    el("span", { class: "size", text: size(p.len) }),
    bar(p, total),
  );
  if (!p.parts?.length) return el("div", { class: "leaf" }, line);
  return el("details", {}, el("summary", {}, line), el("div", { class: "kids" }, p.parts.map((c) => tree(bytes, c, total))));
}

function render(): void {
  const d = doc();
  if (!d) return void box().replaceChildren(el("p", { class: "muted", text: "Analyze a document or open a graph file." }));
  try {
    const g = walkGraph(d.graph);
    const layers = g.parts?.find((p) => p.name === "layers")?.parts ?? [];
    const bodies = layers.reduce((n, l) => n + (l.parts?.find((p) => p.name === "body")?.len ?? 0), 0);
    box().replaceChildren(
      el("p", { class: "muted", text: `${size(d.graph.length)} in all, ${size(bodies)} of it layer bodies. Open a part for its fields; hover for its offsets and first bytes.` }),
      el("div", { class: "kids top" }, (g.parts ?? []).map((p) => tree(d.graph, p, g.len))),
    );
  } catch (e) {
    const where = e instanceof WireError ? ` (byte ${e.at})` : "";
    box().replaceChildren(el("p", { class: "error", text: `${e instanceof Error ? e.message : String(e)}${where}` }));
  }
}

/** Sends the graph through RoundTripDocumentGraph and says whether the bytes came back the same. */
async function roundTrip(): Promise<void> {
  const d = doc();
  if (!d) return say(note(), "Analyze a document or open a graph file.", "error");
  try {
    const r = await analysis.roundTripDocumentGraph({ graph: d.graph });
    const same = r.graph.length === d.graph.length && r.graph.every((b, i) => b === d.graph[i]);
    say(note(), `The server re-exported ${size(r.graph.length)}, ${r.layers.length} layers and ${r.views.length} views: ${same ? "byte for byte the same" : "different bytes"}.`, same ? "" : "error");
  } catch (e) {
    say(note(), describe(e), "error");
  }
}

/** Reads a graph file, has the server import it, and holds it for every panel. */
async function open(): Promise<void> {
  const f = input("wire-file").files?.[0];
  if (!f) return;
  try {
    const graph = new Uint8Array(await f.arrayBuffer());
    const r = await analysis.roundTripDocumentGraph({ graph });
    setDoc({ graph, views: r.views, source: r.sourceText });
    say(note(), `Opened ${f.name}: ${r.layers.length} layers, ${r.views.length} views.`);
  } catch (e) {
    say(note(), describe(e), "error");
  }
}

export function init(): void {
  const again = lazy($("analyze-wire") as HTMLDetailsElement, render);
  onDoc(() => again());
  $("wire-roundtrip").addEventListener("click", () => void roundTrip());
  $("wire-file").addEventListener("change", () => void open());
  $("wire-save").addEventListener("click", () => {
    const d = doc();
    if (d) download("document.graph", [d.graph as Uint8Array<ArrayBuffer>], "application/octet-stream");
  });
}
