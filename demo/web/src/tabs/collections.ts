// Collections: the corpus vocabularies indexes feed (SetCollection,
// ListCollections, GetCollection, DeleteCollection) and WatchCollection,
// a live stream of vocabulary drift reports as documents are indexed.

import { $, el, input, say, table } from "../dom.js";
import type { CollectionEvent } from "../gen/turbo_search_pb.js";
import { describe, search } from "../rpc.js";

let watching: AbortController | null = null;

const note = () => $("coll-note");

async function list(): Promise<void> {
  const box = $("coll-list");
  try {
    const r = await search.listCollections({});
    box.replaceChildren(
      r.collections.length
        ? table(
            ["collection", "units", "mean length", "vocabulary", "alert at", "top terms", ""],
            r.collections.map((c) => [
              c.collectionName,
              String(c.totalUnits),
              c.meanLength.toFixed(1),
              String(c.vocabularySize),
              c.oovAlertThreshold.toFixed(2),
              c.topTerms.slice(0, 6).map((t) => `${t.term} ${t.documentFrequency}`).join(", "),
              remove(c.collectionName),
            ]),
          )
        : el("p", { class: "muted", text: "No collection yet: set one, or create an index with a collection name." }),
    );
  } catch (e) {
    box.replaceChildren(el("p", { class: "error", text: describe(e) }));
  }
}

function remove(name: string): HTMLElement {
  const b = el("button", { type: "button", text: "Delete" });
  b.addEventListener("click", async () => {
    try {
      await search.deleteCollection({ collectionName: name });
      await list();
    } catch (e) {
      say(note(), describe(e), "error");
    }
  });
  return b;
}

async function set(ev: Event): Promise<void> {
  ev.preventDefault();
  const name = input("coll-name").value.trim();
  if (!name) return say(note(), "Name the collection.", "error");
  try {
    const r = await search.setCollection({
      collectionName: name,
      baselineTerms: input("coll-baseline").value.split(" ").map((t) => t.trim()).filter((t) => t.length),
      oovAlertThreshold: Math.min(1, Math.max(0, parseFloat(input("coll-threshold").value) || 0)),
    });
    say(note(), `Set ${r.collection?.collectionName ?? name}.`);
    await list();
  } catch (e) {
    say(note(), describe(e), "error");
  }
}

function event(ev: CollectionEvent): HTMLElement {
  const r = ev.report;
  const at = new Date().toLocaleTimeString();
  if (!r) return el("li", { text: `${at} ${ev.collectionName} from ${ev.sourceIndex}: no report` });
  return el(
    "li",
    {},
    el("span", { class: "badge", text: `OOV ${(r.oovRatio * 100).toFixed(1)}%` }),
    ` ${at} ${ev.collectionName} from ${ev.sourceIndex}: coverage ${(r.vocabularyCoverage * 100).toFixed(1)}%, ${r.totalUnits} units`,
    r.newTerms.length ? el("div", { class: "muted", text: `new: ${r.newTerms.slice(0, 24).join(" ")}${r.newTerms.length > 24 ? ` \u2026 ${r.newTerms.length} in all` : ""}` }) : "",
  );
}

async function watch(): Promise<void> {
  stop();
  const name = input("coll-name").value.trim();
  if (!name) return say(note(), "Name the collection to watch.", "error");
  const ac = new AbortController();
  watching = ac;
  const log = $("coll-events");
  say(note(), `Watching ${name}: index documents into an index of this collection to see its drift.`, "busy");
  try {
    for await (const ev of search.watchCollection({ collectionName: name, minOovRatio: Math.max(0, parseFloat(input("coll-min").value) || 0) }, { signal: ac.signal })) {
      log.prepend(event(ev));
      void list();
    }
    say(note(), "The server ended the watch.");
  } catch (e) {
    const why = describe(e);
    say(note(), why === "cancelled" ? "Stopped watching." : why, why === "cancelled" ? "" : "error");
  } finally {
    if (watching === ac) watching = null;
  }
}

/** Ends the watch in flight, if any. */
export function stop(): void {
  watching?.abort();
  watching = null;
}

export function init(): void {
  $("coll-set").addEventListener("submit", set);
  $("coll-refresh").addEventListener("click", () => void list());
  $("coll-watch").addEventListener("click", () => void watch());
  $("coll-stop").addEventListener("click", stop);
  void list();
}
