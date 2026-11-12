// Vocabulary: dictionary formats and the store's limits, the stored
// dictionaries and vocabularies, ImportDictionary and LearnVocabulary as
// uploads, DownloadVocabulary as a stream to a file, and delete.

import { TurboVocabularyService } from "../gen/turbo_vocabulary_pb.js";
import { busy as working, confirmed, submitter } from "../act.js";
import { $, download, el, input, options, say, size, table } from "../dom.js";
import { importBody, learnBody, type Text } from "../frames.js";
import { lock, off } from "../permit.js";
import { describe, explain, vocabulary } from "../rpc.js";
import { onServed, reachable } from "../state.js";
import { sha256Blob } from "../sha256.js";
import { put } from "../upload.js";

// Every call in flight: an import, a learn and downloads can run side
// by side, each with its own signal, and Stop ends them all.
const inFlight = new Set<AbortController>();

const select = (id: string) => $(id) as HTMLSelectElement;

/** Runs `job` with a signal of its own, which Stop aborts while the job runs. */
async function stoppable(job: (signal: AbortSignal) => Promise<void>): Promise<void> {
  const ac = new AbortController();
  inFlight.add(ac);
  try {
    await job(ac.signal);
  } finally {
    inFlight.delete(ac);
  }
}

export function stop(): void {
  for (const ac of inFlight) ac.abort();
  inFlight.clear();
}

const stopButton = () => $("vocab-stop") as HTMLButtonElement;

function actions(id: string): HTMLElement {
  const get = el("button", { type: "button", text: "Download" });
  get.addEventListener("click", () => void working(get, () => stoppable((signal) => fetchArtifact(id, signal)), stopButton()));
  const del = confirmed("Delete", `Delete ${id}?`, async () => {
    await vocabulary.deleteVocabularyArtifact({ artifactId: id });
    await list();
  }, off(TurboVocabularyService.method.deleteVocabularyArtifact));
  return el("span", {}, get, " ", del);
}

async function list(): Promise<void> {
  const box = $("vocab-lists");
  try {
    const [f, d, v] = await Promise.all([vocabulary.listDictionaryFormats({}), vocabulary.listDictionaries({}), vocabulary.listVocabularies({})]);
    const importable = f.formats.filter((x) => x.importable);
    options(select("vocab-format"), importable.map((x) => x.formatId), importable.map((x) => `${x.formatId}: ${x.description}`), "the server imports no format");
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
    const down = reachable() === false;
    // A select is never left blank: it says why there is nothing to pick.
    for (const id of ["vocab-format", "vocab-seed"]) options(select(id), [], undefined, down ? "the server is not reachable" : "the store could not be listed");
    box.replaceChildren(el("p", { class: "error", text: down ? "The server is not reachable, so its store cannot be listed." : describe(e) }));
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
  await working(submitter(ev), () => stoppable(async (signal) => {
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
        signal,
      );
      say(note, `Imported ${d.name || d.artifactId}: ${d.entryCount} entries.`);
      await list();
    } catch (e) {
      say(note, explain(e));
    }
  }), stopButton());
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
  await working(submitter(ev), () => stoppable(async (signal) => {
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
      const v = await put(TurboVocabularyService.method.learnVocabulary, body, signal);
      say(note, `Learned ${v.artifactId}: ${v.termCount} terms and ${v.ngramCount} n-grams from ${v.documents} documents.`);
      await list();
    } catch (e) {
      say(note, explain(e));
    }
  }), stopButton());
}

async function fetchArtifact(id: string, signal: AbortSignal): Promise<void> {
  const note = $("vocab-note");
  const parts: Uint8Array<ArrayBuffer>[] = [];
  let name = id;
  let got = 0;
  try {
    for await (const c of vocabulary.downloadVocabulary({ artifactId: id }, { signal })) {
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
    say(note, explain(e));
  }
}

export function init(): void {
  $("vocab-import").addEventListener("submit", importDictionary);
  $("vocab-learn").addEventListener("submit", learn);
  lock($("vocab-import"), TurboVocabularyService.method.importDictionary);
  lock($("vocab-learn"), TurboVocabularyService.method.learnVocabulary);
  const again = $("vocab-refresh") as HTMLButtonElement;
  again.addEventListener("click", () => void working(again, list));
  onServed(() => void list());
  $("vocab-stop").addEventListener("click", stop);
}
