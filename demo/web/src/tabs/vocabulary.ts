// Vocabulary: dictionary formats and the store's limits, the stored
// dictionaries and vocabularies, ImportDictionary and LearnVocabulary as
// uploads, DownloadVocabulary as a stream to a file, and delete.

import { TurboVocabularyService } from "../gen/turbo_vocabulary_pb.js";
import { $, download, el, input, options, say, size, table } from "../dom.js";
import { importBody, learnBody, type Text } from "../frames.js";
import { describe, vocabulary } from "../rpc.js";
import { sha256Blob } from "../sha256.js";
import { put } from "../upload.js";

let busy: AbortController | null = null;

const select = (id: string) => $(id) as HTMLSelectElement;

function start(): AbortController {
  busy?.abort();
  busy = new AbortController();
  return busy;
}

export function stop(): void {
  busy?.abort();
  busy = null;
}

function actions(id: string): HTMLElement {
  const get = el("button", { type: "button", text: "Download" });
  get.addEventListener("click", () => void fetchArtifact(id));
  const del = el("button", { type: "button", text: "Delete" });
  del.addEventListener("click", async () => {
    try {
      await vocabulary.deleteVocabularyArtifact({ artifactId: id });
      await list();
    } catch (e) {
      say($("vocab-note"), describe(e), "error");
    }
  });
  return el("span", {}, get, " ", del);
}

async function list(): Promise<void> {
  const box = $("vocab-lists");
  try {
    const [f, d, v] = await Promise.all([vocabulary.listDictionaryFormats({}), vocabulary.listDictionaries({}), vocabulary.listVocabularies({})]);
    const importable = f.formats.filter((x) => x.importable);
    options(select("vocab-format"), importable.map((x) => x.formatId), importable.map((x) => `${x.formatId}: ${x.description}`));
    options(select("vocab-seed"), ["", ...d.dictionaries.map((x) => x.artifactId)], ["no seed dictionary", ...d.dictionaries.map((x) => x.name || x.artifactId)]);
    const l = f.limits;
    box.replaceChildren(
      l
        ? el("p", {
            class: "muted",
            text: l.store
              ? `Store on${l.admin ? ", writable" : ", read only"}. Dictionaries up to ${size(l.maxDictionaryUploadBytes)}, learning up to ${size(l.maxLearnBytes)}, n-grams up to ${l.maxNgramSize}.`
              : "The server keeps no vocabulary store.",
          })
        : "",
      el("h3", { text: "Dictionaries" }),
      table(
        ["name", "format", "language", "entries", "size", "artifact", ""],
        d.dictionaries.map((x) => [x.name, x.formatId, x.language, String(x.entryCount), size(x.byteSize), x.artifactId, actions(x.artifactId)]),
      ),
      el("h3", { text: "Vocabularies" }),
      table(
        ["artifact", "language", "documents", "tokens", "terms", "n-grams", "size", ""],
        v.vocabularies.map((x) => [x.artifactId, x.language, String(x.documents), String(x.tokens), String(x.termCount), String(x.ngramCount), size(x.byteSize), actions(x.artifactId)]),
      ),
    );
  } catch (e) {
    box.replaceChildren(el("p", { class: "error", text: describe(e) }));
  }
}

async function importDictionary(ev: Event): Promise<void> {
  ev.preventDefault();
  const note = $("vocab-note");
  const file = input("vocab-file").files?.[0];
  if (!file) {
    say(note, "Pick a dictionary file.", "error");
    return;
  }
  const ac = start();
  try {
    const sha = await sha256Blob(file, (done) => say(note, `Hashing ${size(done)} of ${size(file.size)}\u2026`, "busy"));
    say(note, `Uploading ${size(file.size)}\u2026`, "busy");
    const d = await put(
      TurboVocabularyService.method.importDictionary,
      importBody(
        {
          formatId: select("vocab-format").value,
          language: input("vocab-language").value.trim(),
          name: input("vocab-name").value.trim() || file.name,
          byteSize: BigInt(file.size),
          sha256: sha,
          hasHeaderRow: input("vocab-header").checked,
        },
        file,
      ),
      ac.signal,
    );
    say(note, `Imported ${d.name || d.artifactId}: ${d.entryCount} entries.`);
    await list();
  } catch (e) {
    say(note, describe(e), "error");
  }
}

async function learn(ev: Event): Promise<void> {
  ev.preventDefault();
  const note = $("vocab-note");
  const typed = (document.getElementById("vocab-docs") as HTMLTextAreaElement).value
    .split("\n\n")
    .map((t) => t.trim())
    .filter((t) => t.length);
  const docs: Text[] = [...typed, ...[...(input("vocab-learn-files").files ?? [])]];
  if (!docs.length) {
    say(note, "Type documents (a blank line between) or pick files.", "error");
    return;
  }
  const ac = start();
  try {
    const body = learnBody(
      {
        minTermFrequency: parseInt(input("vocab-min-tf").value, 10) || 1,
        maxTerms: parseInt(input("vocab-max-terms").value, 10) || 0,
        lowercase: input("vocab-lowercase").checked,
        filterStopwords: input("vocab-stopwords").checked,
        includeNgrams: input("vocab-ngrams").checked,
        minNgramSize: 2,
        maxNgramSize: 3,
        dictionaryArtifactId: select("vocab-seed").value,
      },
      docs,
    );
    say(note, `Uploading ${docs.length} documents, ${size(body.size)}\u2026`, "busy");
    const v = await put(TurboVocabularyService.method.learnVocabulary, body, ac.signal);
    say(note, `Learned ${v.artifactId}: ${v.termCount} terms and ${v.ngramCount} n-grams from ${v.documents} documents.`);
    await list();
  } catch (e) {
    say(note, describe(e), "error");
  }
}

async function fetchArtifact(id: string): Promise<void> {
  const note = $("vocab-note");
  const ac = start();
  const parts: Uint8Array<ArrayBuffer>[] = [];
  let name = id;
  let got = 0;
  try {
    for await (const c of vocabulary.downloadVocabulary({ artifactId: id }, { signal: ac.signal })) {
      const f = c.frame;
      if (f.case === "dictionary") name = `${f.value.name || id}.dictionary`;
      else if (f.case === "vocabulary") name = `${id}.vocabulary`;
      else if (f.case === "data") {
        parts.push(f.value as Uint8Array<ArrayBuffer>);
        got += f.value.length;
        say(note, `Downloading ${name}: ${size(got)}\u2026`, "busy");
      }
    }
    download(name, parts, "application/octet-stream");
    say(note, `Downloaded ${name}, ${size(got)}.`);
  } catch (e) {
    say(note, describe(e), "error");
  }
}

export function init(): void {
  $("vocab-import").addEventListener("submit", importDictionary);
  $("vocab-learn").addEventListener("submit", learn);
  $("vocab-refresh").addEventListener("click", list);
  $("vocab-stop").addEventListener("click", stop);
  void list();
}
