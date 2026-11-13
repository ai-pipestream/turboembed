// Search: indexes (list, create from an embedding bundle, delete), adding
// documents through the IndexDocuments upload, and hybrid search with
// each hit's signal scores, matched spans and the document heat.

import { DistanceMetric, NormalizationMode, TurboSearchService, VectorStorageKind, type SearchResponse } from "../gen/turbo_search_pb.js";
import { busy, confirmed, submitter } from "../act.js";
import { $, count, el, failed, heatKey, input, options, reveal, say, sending, table } from "../dom.js";
import { take } from "./take.js";
import { indexBody, type Text } from "../frames.js";
import { pieces, unplaced, type Match } from "../highlight.js";
import { lock, off, onChange } from "../permit.js";
import { describe, explain, search } from "../rpc.js";
import { onCapabilities, onCapabilitiesFailed, onReachable, onServed, reachable, whyNone } from "../state.js";
import { put } from "../upload.js";
import type { CapabilitiesResponse } from "../gen/turbo_analysis_pb.js";
import { latest } from "../latest.js";

let caps: CapabilitiesResponse | null = null;
let indexing: AbortController | null = null;

const select = (id: string) => $(id) as HTMLSelectElement;

// Only the latest call draws (latest.ts).
const freshList = latest();
async function list(): Promise<void> {
  const mine = freshList();
  const box = $("search-indexes");
  try {
    const r = await search.listIndexes({});
    if (!mine()) return;
    options(select("search-index"), r.indexes.map((i) => i.indexName), undefined, "no index yet");
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
    if (!mine()) return;
    const down = reachable() === false;
    options(select("search-index"), [], undefined, down ? "the server is not reachable" : "the indexes could not be listed");
    box.replaceChildren(el("p", { class: "error", text: down ? "The server is not reachable, so its indexes cannot be listed." : describe(e) }));
  }
}

function deleteButton(name: string): HTMLElement {
  return confirmed("Delete", `Delete ${name}?`, async () => {
    await search.deleteIndex({ indexName: name });
    await list();
  }, off(TurboSearchService.method.deleteIndex));
}

async function createIndex(ev: Event): Promise<void> {
  ev.preventDefault();
  const note = $("search-create-note");
  const name = input("search-create-name").value.trim();
  const bundle = caps?.bundles.find((b) => b.name === select("search-create-bundle").value);
  const model = bundle?.model;
  if (!name || !bundle || !model) {
    say(note, "Name the index and pick an embedding bundle.", "error");
    return;
  }
  await busy(submitter(ev), async () => {
    try {
      await search.createIndex({
        spec: {
          indexName: name,
          space: { model: { id: model.id, revision: model.revision, sha256: model.sha256 }, dim: bundle.widths[0] ?? 0, normalization: NormalizationMode.UNIT },
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
      say(note, explain(e));
    }
  });
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
  await busy($("search-add") as HTMLButtonElement, async () => {
    // Inside, so a click while the upload runs (which busy ignores) does not end it.
    indexing?.abort();
    const ac = new AbortController();
    indexing = ac;
    try {
      const body = indexBody(index, docs);
      const r = await put(TurboSearchService.method.indexDocuments, body, ac.signal, sending(note, count(docs.length, "document")));
      say(note, `Indexed ${r.indexedDocuments} documents as ${r.indexedChunks} chunks.`);
      await list();
    } catch (e) {
      say(note, explain(e));
    } finally {
      if (indexing === ac) indexing = null;
    }
  }, $("search-add-stop") as HTMLButtonElement);
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
  // The obvious gaps are said here, before the server is asked.
  const missing = !select("search-index").value ? "Pick or create an index first." : !input("search-query").value.trim() ? "Enter something to search for." : "";
  if (missing) return void out.replaceChildren(el("p", { class: "error", text: missing }));
  await busy(submitter(ev), async () => {
    try {
      const r = await search.search({
        indexName: select("search-index").value,
        naturalQuery: input("search-query").value,
        topK: Math.max(1, parseInt(input("search-k").value, 10) || 10),
        rerankBundle: select("search-rerank").value,
      });
      // One colour per signal, in the order the hits first name them.
      const sources: string[] = [];
      const colour = (source: string): string => {
        if (!sources.includes(source)) sources.push(source);
        return `var(--m${sources.indexOf(source) % 8})`;
      };
      const hits = r.hits.map((h, i) => {
        const a = h.attribution;
        const signals = a ? Object.entries(a.signalScores).map(([k, v]) => `${k} ${v.toFixed(3)}`).join(", ") : "";
        const matches: Match[] = (a?.matchedSpans ?? []).map((m) => ({ source: m.source, label: m.label, start: m.span?.start ?? 0, end: m.span?.end ?? 0 }));
        const at = h.chunkSpan?.start ?? 0;
        const text = pieces(h.text, at, matches).map((p) =>
          p.sources.length
            ? el("mark", { style: `--mark: ${colour(p.sources[0])}`, title: `${p.sources.join(", ")}${p.labels.length ? `: ${p.labels.join(", ")}` : ""}` }, p.text, el("sup", { text: p.sources.join("+") }))
            : p.text,
        );
        const rest = unplaced(h.text, at, matches).map((m) => `${m.source}${m.label ? `:${m.label}` : ""}`);
        return el(
          "div",
          { class: "hit" },
          el("div", { class: "head" }, el("b", { text: `${i + 1}. ${h.docId}` }), el("span", { class: "tag", text: h.score.toFixed(4) }), el("span", { class: "muted", text: ` ${h.chunkId}` })),
          el("p", { class: "doc flat" }, text),
          signals ? el("p", { class: "muted", text: `signals: ${signals}` }) : null,
          rest.length ? el("p", { class: "muted", text: `matched the whole passage: ${[...new Set(rest)].join("; ")}` }) : null,
        );
      });
      const key = sources.length
        ? el("div", { class: "legend" }, el("span", { class: "muted", text: "matched by" }), sources.map((x) => el("span", { class: "swatch", style: `--mark: ${colour(x)}` }, el("i", {}), ` ${x}`)))
        : null;
      out.replaceChildren(
        el("p", { class: "muted", text: `${r.hits.length} hits in ${r.resolvedIndex}; query terms: ${r.analyzedQueryTerms.join(" ") || "none"}` }),
        key ?? "",
        ...hits,
        r.hits.length
          ? take(`Search hits ${r.resolvedIndex}`, () => [
              ["rank", "document", "chunk", "score", "text", "signals"],
              ...r.hits.map((h, i) => [String(i + 1), h.docId, h.chunkId, h.score.toFixed(6), h.text, Object.entries(h.attribution?.signalScores ?? {}).map(([k, v]) => `${k} ${v.toFixed(6)}`).join("; ")]),
            ])
          : "",
        r.documentHeat.length ? el("h3", { text: "Document heat" }) : "",
        r.documentHeat.length ? heatKey("One cell per chunk, in reading order. Stronger colour is a higher score, relative to that document's best (max).") : "",
        heatView(r),
      );
      reveal(out);
    } catch (e) {
      out.replaceChildren(failed(explain(e)));
    }
  });
}

export function stop(): void {
  indexing?.abort();
  indexing = null;
}

export function init(): void {
  $("search-create").addEventListener("submit", createIndex);
  lock($("search-create"), TurboSearchService.method.createIndex);
  lock($("search-add-box"), TurboSearchService.method.indexDocuments);
  $("search-add").addEventListener("click", addDocuments);
  $("search-add-stop").addEventListener("click", stop);
  $("search-form").addEventListener("submit", runSearch);
  const refreshButton = $("search-refresh") as HTMLButtonElement;
  refreshButton.addEventListener("click", () => void busy(refreshButton, list));
  const served = (): void => {
    const embed = caps?.bundles.filter((b) => b.task === "TASK_EMBED") ?? [];
    options(select("search-create-bundle"), embed.map((b) => b.name), embed.map((b) => `${b.name} (dim ${b.widths[0] ?? "?"})`), whyNone("embedding model"));
    // Before the first answer the rerank select says why it is empty too.
    const rerank = caps?.bundles.filter((b) => b.task === "TASK_RERANK").map((b) => b.name) ?? [];
    if (caps) options(select("search-rerank"), ["", ...rerank], ["no rerank", ...rerank]);
    else options(select("search-rerank"), [], undefined, reachable() === false ? "the server is not reachable" : "no rerank");
  };
  onReachable(() => {
    if (!caps) served();
  });
  onServed(() => void list());
  // Each row's Delete is drawn from what the demo forwards.
  onChange(() => void list());
  onCapabilities((c) => {
    caps = c;
    served();
  });
  onCapabilitiesFailed(served);
}
