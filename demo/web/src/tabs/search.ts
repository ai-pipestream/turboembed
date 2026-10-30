// Search: indexes (list, create from an embedding bundle, delete), adding
// documents through the IndexDocuments upload, and hybrid search with
// each hit's signal scores, matched spans and the document heat.

import { DistanceMetric, NormalizationMode, TurboSearchService, VectorStorageKind, type SearchResponse } from "../gen/turbo_search_pb.js";
import { $, el, input, options, say, size, table } from "../dom.js";
import { indexBody, type Text } from "../frames.js";
import { describe, search } from "../rpc.js";
import { capabilities, onCapabilities } from "../state.js";
import { put } from "../upload.js";
import type { CapabilitiesResponse } from "../gen/turbo_analysis_pb.js";

let caps: CapabilitiesResponse | null = null;
let indexing: AbortController | null = null;

const select = (id: string) => $(id) as HTMLSelectElement;

async function list(): Promise<void> {
  const box = $("search-indexes");
  try {
    const r = await search.listIndexes({});
    options(select("search-index"), r.indexes.map((i) => i.indexName));
    box.replaceChildren(
      r.indexes.length
        ? table(
            ["index", "model", "dim", "metric", "storage", "documents", "chunks", ""],
            r.indexes.map((i) => [
              i.indexName,
              i.space?.model ? `${i.space.model.id}@${i.space.model.revision}` : "",
              String(i.space?.dim ?? ""),
              DistanceMetric[i.metric] ?? "",
              VectorStorageKind[i.storage] ?? "",
              String(i.documentCount),
              String(i.chunkCount),
              deleteButton(i.indexName),
            ]),
          )
        : el("p", { class: "muted", text: "No index yet: create one from an embedding bundle." }),
    );
  } catch (e) {
    box.replaceChildren(el("p", { class: "error", text: describe(e) }));
  }
}

function deleteButton(name: string): HTMLElement {
  const b = el("button", { type: "button", text: "Delete" });
  b.addEventListener("click", async () => {
    try {
      await search.deleteIndex({ indexName: name });
      await list();
    } catch (e) {
      say($("search-create-note"), describe(e), "error");
    }
  });
  return b;
}

async function createIndex(ev: Event): Promise<void> {
  ev.preventDefault();
  const note = $("search-create-note");
  const name = input("search-create-name").value.trim();
  const bundle = caps?.bundles.find((b) => b.name === select("search-create-bundle").value);
  if (!name || !bundle?.model) {
    say(note, "Name the index and pick an embedding bundle.", "error");
    return;
  }
  try {
    await search.createIndex({
      spec: {
        indexName: name,
        space: { model: { id: bundle.model.id, revision: bundle.model.revision, sha256: bundle.model.sha256 }, dim: bundle.widths[0] ?? 0, normalization: NormalizationMode.UNIT },
        metric: DistanceMetric.COSINE,
        storage: VectorStorageKind.FLAT_F32,
        embedBundle: bundle.name,
        collectionName: input("search-create-collection").value.trim(),
        chunkBudgetTokens: 0,
      },
    });
    say(note, `Created ${name}.`);
    await list();
    select("search-index").value = name;
  } catch (e) {
    say(note, describe(e), "error");
  }
}

async function addDocuments(): Promise<void> {
  const note = $("search-add-note");
  const index = select("search-index").value;
  const typed = (document.getElementById("search-docs") as HTMLTextAreaElement).value
    .split("\n\n")
    .map((t) => t.trim())
    .filter((t) => t.length);
  const files = [...(input("search-files").files ?? [])];
  const docs: { id: string; text: Text }[] = [
    ...typed.map((t, i) => ({ id: `typed-${Date.now()}-${i + 1}`, text: t })),
    ...files.map((f) => ({ id: f.name, text: f as Blob })),
  ];
  if (!index || !docs.length) {
    say(note, !index ? "Create or pick an index first." : "Type paragraphs (a blank line between) or pick files.", "error");
    return;
  }
  indexing?.abort();
  const ac = new AbortController();
  indexing = ac;
  try {
    const body = indexBody(index, docs);
    say(note, `Uploading ${docs.length} documents, ${size(body.size)}\u2026`, "busy");
    const r = await put(TurboSearchService.method.indexDocuments, body, ac.signal);
    say(note, `Indexed ${r.indexedDocuments} documents as ${r.indexedChunks} chunks.`);
    await list();
  } catch (e) {
    say(note, describe(e), "error");
  } finally {
    if (indexing === ac) indexing = null;
  }
}

function heatView(r: SearchResponse): HTMLElement {
  return el(
    "div",
    {},
    r.documentHeat.map((d) =>
      el(
        "div",
        { class: "lane" },
        el("span", { class: "lane-name", text: `${d.docId} (max ${d.maxScore.toFixed(3)})` }),
        el(
          "span",
          { class: "cells" },
          d.segments.map((s) =>
            el("span", {
              class: "hcell",
              title: `${s.chunkId} ${s.span?.start ?? 0}..${s.span?.end ?? 0}: ${s.score.toFixed(3)}`,
              style: `opacity: ${(0.1 + 0.9 * (d.maxScore ? s.score / d.maxScore : 0)).toFixed(2)}`,
            }),
          ),
        ),
      ),
    ),
  );
}

async function runSearch(ev: Event): Promise<void> {
  ev.preventDefault();
  const out = $("search-results");
  try {
    const r = await search.search({
      indexName: select("search-index").value,
      naturalQuery: input("search-query").value,
      topK: Math.max(1, parseInt(input("search-k").value, 10) || 10),
      rerankBundle: select("search-rerank").value,
    });
    const hits = r.hits.map((h, i) => {
      const a = h.attribution;
      const signals = a ? Object.entries(a.signalScores).map(([k, v]) => `${k} ${v.toFixed(3)}`).join(", ") : "";
      const spans = (a?.matchedSpans ?? []).map((m) => `${m.source}${m.label ? `:${m.label}` : ""} ${m.span?.start ?? 0}..${m.span?.end ?? 0}`);
      return el(
        "div",
        { class: "hit" },
        el("div", { class: "head" }, el("b", { text: `${i + 1}. ${h.docId}` }), el("span", { class: "tag", text: h.score.toFixed(4) }), el("span", { class: "muted", text: ` ${h.chunkId}` })),
        el("p", { text: h.text }),
        signals ? el("p", { class: "muted", text: `signals: ${signals}` }) : null,
        spans.length ? el("p", { class: "muted", text: `matched: ${spans.join("; ")}` }) : null,
      );
    });
    out.replaceChildren(
      el("p", { class: "muted", text: `${r.hits.length} hits in ${r.resolvedIndex}; query terms: ${r.analyzedQueryTerms.join(" ") || "none"}` }),
      ...hits,
      r.documentHeat.length ? el("h3", { text: "Document heat" }) : "",
      heatView(r),
    );
  } catch (e) {
    out.replaceChildren(el("p", { class: "error", text: describe(e) }));
  }
}

export function stop(): void {
  indexing?.abort();
  indexing = null;
}

export function init(): void {
  $("search-create").addEventListener("submit", createIndex);
  $("search-add").addEventListener("click", addDocuments);
  $("search-form").addEventListener("submit", runSearch);
  $("search-refresh").addEventListener("click", list);
  onCapabilities((c) => {
    caps = c;
    const embed = c.bundles.filter((b) => b.task === "TASK_EMBED");
    options(select("search-create-bundle"), embed.map((b) => b.name), embed.map((b) => `${b.name} (dim ${b.widths[0] ?? "?"})`));
    const rerank = c.bundles.filter((b) => b.task === "TASK_RERANK").map((b) => b.name);
    options(select("search-rerank"), ["", ...rerank], ["no rerank", ...rerank]);
  });
  capabilities().catch(() => undefined);
  void list();
}
