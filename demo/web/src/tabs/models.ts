// Models: the model catalog with licence acceptance (licences.ts), the
// user's model sources (sources.ts), a streaming InstallModel monitor,
// and static model distillation with a streaming TrainStaticModel
// monitor, the teachers it can use and the static models it made.

import { InstallStage, StaticDtype, TrainStage, type CatalogEntry, type CatalogFile, type LicenceTerms } from "../gen/turbo_training_pb.js";
import { kindName, kinds, matches, purpose } from "../catalogfilter.js";
import { busy, confirmed, submitter } from "../act.js";
import { $, el, input, options, say, size, table } from "../dom.js";
import { catalog, describe, training } from "../rpc.js";
import { onCapabilities, onServed, reachable, refresh } from "../state.js";
import { accept, acceptedTable, revoke, termsBlock, who, type Who } from "./licences.js";
import { download, nonCommercial } from "./download.js";
import * as sources from "./sources.js";

let installing: AbortController | null = null;
let saving: AbortController | null = null;
let trainingRun: AbortController | null = null;
// The catalog filter, kept while the catalog is read again after a change.
let filterKind = "";
let filterText = "";

const note = () => $("models-note");

function bar(done: number, total: number): HTMLProgressElement {
  const p = el("progress", { max: String(Math.max(1, total)) });
  p.value = Math.min(done, Math.max(1, total));
  return p;
}

async function act(change: () => Promise<string>): Promise<void> {
  try {
    const done = await change();
    await loadCatalog();
    say(note(), done);
  } catch (e) {
    say(note(), describe(e), "error");
  }
}

/**
 * A catalog entry as a card: its name, what it is for, its licence and
 * size, and one action. With terms still to accept the action is
 * reading them, since Install stays off until they are accepted.
 */
function entryRow(e: CatalogEntry, w: Who): HTMLElement {
  const terms = e.missingTerms.map((t) => termsBlock(e, t, w, (t: LicenceTerms) => act(() => accept(e, t))));
  let install: HTMLElement | string = "";
  if (w.canChange) {
    const b = el("button", { type: "button", class: e.missingTerms.length ? "" : "primary", text: e.installed ? "Reinstall" : "Install" });
    b.disabled = e.missingTerms.length > 0;
    b.title = b.disabled ? "Accept the licence terms first" : "";
    b.addEventListener("click", () => void busy(b, () => installModel(e)));
    install = b;
  }
  const licence = el("span", { class: "licence", text: e.licence }, nonCommercial(e) ? el("strong", { class: "error", text: " Non-commercial use only" }) : null);
  return el(
    "div",
    { class: "entry" },
    el("div", { class: "id", text: e.id }),
    el("div", { class: "purpose", text: purpose(e) }),
    el(
      "div",
      { class: "facts" },
      el("span", { class: "tag kind", text: kindName(e.kind) }),
      e.language ? el("span", { class: "tag", title: "language", text: e.language }) : null,
      e.installed ? el("span", { class: "tag installed", text: "installed" }) : null,
    ),
    el("div", { class: "facts" }, licence, el("span", { class: "muted", text: size(e.totalBytes) })),
    el("div", { class: "acts" }, install, saveButtons(e, w), terms),
  );
}

/**
 * The catalog as cards under a filter: a chip per kind with its count
 * and a box for words, which hide the cards that do not match.
 */
function catalogTable(entries: CatalogEntry[], w: Who): HTMLElement {
  if (!entries.length) return el("p", { class: "muted", text: "The catalog lists nothing." });
  const rows = entries.map((e) => entryRow(e, w));
  const list = el("div", { class: "catalog" }, rows);
  const count = el("span", { class: "muted", id: "models-count", role: "status" });
  const box = el("input", { id: "models-filter", type: "search", placeholder: "filter by name, purpose, kind, language or licence", autocomplete: "off", "aria-label": "Filter the catalog" });
  box.value = filterText;
  if (filterKind && !entries.some((e) => e.kind === filterKind)) filterKind = "";
  const chips: [string, HTMLButtonElement][] = [["", "all", entries.length] as const, ...kinds(entries).map(([k, n]) => [k, kindName(k), n] as const)].map(([k, name, n]) => {
    const chip = el("button", { type: "button", class: "chip", "data-kind": k, text: `${name} ${n}` });
    chip.addEventListener("click", () => {
      filterKind = k;
      apply();
    });
    return [k, chip];
  });
  const none = el("p", { class: "muted", text: "No entry matches.", hidden: "" });
  function apply(): void {
    let shown = 0;
    entries.forEach((e, i) => {
      const on = matches(e, filterKind, filterText);
      rows[i].hidden = !on;
      if (on) shown++;
    });
    for (const [k, chip] of chips) {
      chip.classList.toggle("on", k === filterKind);
      chip.setAttribute("aria-pressed", String(k === filterKind));
    }
    count.textContent = shown === entries.length ? `${entries.length} entries` : `${shown} of ${entries.length} entries`;
    list.hidden = shown === 0;
    none.hidden = shown !== 0;
  }
  box.addEventListener("input", () => {
    filterText = box.value;
    apply();
  });
  apply();
  return el("div", {}, el("div", { class: "row filter" }, box, count), el("div", { class: "chips" }, chips.map(([, c]) => c)), list, none);
}

/** A save button per file, once the user may have the artifact. */
function saveButtons(e: CatalogEntry, w: Who): HTMLElement | null {
  if (!w.user || e.missingTerms.length > 0) return null;
  if (!e.files.length) return null;
  return el(
    "details",
    { class: "files" },
    el("summary", { text: e.files.length === 1 ? "1 file" : `${e.files.length} files` }),
    e.files.map((f) => {
      const b = el("button", { type: "button", text: `Save ${f.path}`, title: size(f.size) });
      b.addEventListener("click", () => void busy(b, () => saveFile(e, f)));
      return b;
    }),
  );
}

async function saveFile(e: CatalogEntry, f: CatalogFile): Promise<void> {
  saving?.abort();
  const ac = new AbortController();
  saving = ac;
  const status = el("p", { class: "muted", text: `Reading ${e.id}/${f.path}\u2026` });
  $("models-install").replaceChildren(el("h3", { text: `Save ${e.id}/${f.path}` }), status);
  try {
    await download(e, f, status, ac.signal);
  } catch (err) {
    status.className = "error";
    status.textContent = describe(err);
  } finally {
    if (saving === ac) saving = null;
  }
}

async function loadCatalog(): Promise<void> {
  const box = $("models-catalog");
  say(note(), "Reading the catalog\u2026", "busy");
  try {
    const [cat, installed, acc] = await Promise.all([catalog.listModelCatalog({}), catalog.listInstalledModels({}), catalog.listLicenceAcceptances({})]);
    const w = who(acc);
    box.replaceChildren(
      el("h3", { text: "Installed" }),
      table(["id", "version", "files", "size", "directory"], installed.models.map((m) => [m.id, m.version, String(m.files.length), size(m.totalBytes), m.dir]), "", "Nothing is installed."),
      el("h3", { text: "Accepted terms" }),
      acceptedTable(acc, w, async (t: LicenceTerms) => {
        const done = await revoke(t);
        await loadCatalog();
        say(note(), done);
      }),
      el("h3", { text: "Catalog" }),
      el("p", { class: "muted", text: w.user ? `Licence terms still to accept are those for ${w.user}; read them before accepting.${w.why ? ` (${w.why})` : ""}` : w.why }),
      catalogTable(cat.entries, w),
    );
    say(note(), "");
  } catch (e) {
    say(note(), reachable() === false ? "The server is not reachable, so its catalog cannot be read." : describe(e), "error");
  }
}

async function installModel(e: CatalogEntry): Promise<void> {
  installing?.abort();
  const ac = new AbortController();
  installing = ac;
  const log = $("models-install");
  const status = el("p", { class: "muted", text: `Installing ${e.id}\u2026` });
  const progress = bar(0, 1);
  log.replaceChildren(el("h3", { text: `Install ${e.id}` }), progress, status);
  try {
    for await (const u of catalog.installModel({ artifactId: e.id }, { signal: ac.signal })) {
      if (u.update.case === "progress") {
        const p = u.update.value;
        progress.max = Math.max(1, p.fileCount);
        progress.value = p.stage === InstallStage.VERIFIED ? p.fileIndex + 1 : p.fileIndex;
        status.textContent = `${InstallStage[p.stage]}: file ${p.fileIndex + 1} of ${p.fileCount} ${p.path} (${size(p.bytes)})`;
      } else if (u.update.case === "installed") {
        const m = u.update.value;
        progress.value = progress.max;
        status.textContent = `Installed ${m.id} ${m.version}: ${m.files.length} files, ${size(m.totalBytes)} in ${m.dir}.`;
      }
    }
    await loadCatalog();
  } catch (err) {
    status.className = "error";
    status.textContent = describe(err);
  } finally {
    if (installing === ac) installing = null;
  }
}

async function loadTraining(): Promise<void> {
  const box = $("models-static");
  try {
    const t = await training.listTeachers({});
    options($("train-teacher") as HTMLSelectElement, t.teachers.map((x) => x.name), t.teachers.map((x) => `${x.name} (hidden ${x.hidden}, ${x.vocabSize} words)`), "no teacher is served");
    let made: HTMLElement;
    try {
      const s = await training.listStaticModels({});
      made = table(
        ["name", "dims", "words", "learned", "dtype", "teacher", "quality", "served", ""],
        s.models.map((m) => [
          m.name,
          String(m.dims),
          String(m.vocabSize),
          String(m.learnedTerms),
          m.dtype,
          m.teacher?.modelId ?? "",
          m.quality ? `top-1 ${m.quality.staticTop1.toFixed(3)} of ${m.quality.baseTop1.toFixed(3)}, rho ${m.quality.similaritySpearman.toFixed(3)}` : "",
          m.served ? "yes" : m.problem,
          deleteButton(m.name),
        ]),
      );
    } catch (e) {
      made = el("p", { class: "error", text: reachable() === false ? "The server is not reachable, so its static models cannot be listed." : describe(e) });
    }
    box.replaceChildren(el("h3", { text: "Teachers" }), table(["name", "model", "hidden", "words"], t.teachers.map((x) => [x.name, `${x.modelId}@${x.revision}`, String(x.hidden), String(x.vocabSize)])), el("h3", { text: "Static models" }), made);
  } catch (e) {
    const down = reachable() === false;
    options($("train-teacher") as HTMLSelectElement, [], undefined, down ? "the server is not reachable" : "the teachers could not be listed");
    box.replaceChildren(el("p", { class: "error", text: down ? "The server is not reachable, so its teachers and static models cannot be listed." : describe(e) }));
  }
}

function deleteButton(name: string): HTMLElement {
  return confirmed("Delete", `Delete ${name}?`, async () => {
    await training.deleteStaticModel({ name });
    await Promise.all([loadTraining(), refresh()]);
  });
}

async function train(ev: Event): Promise<void> {
  ev.preventDefault();
  await busy(submitter(ev), async () => {
    // Inside, so a submit while a run is on (which busy ignores) does not end it.
    stopTraining();
    const ac = new AbortController();
    trainingRun = ac;
    const steps = $("train-steps");
    const progress = bar(0, 1);
    steps.replaceChildren();
    $("train-progress").replaceChildren(progress);
    say($("train-note"), "Training\u2026", "busy");
    const t0 = performance.now();
    let last = -1;
    try {
      const req = {
        name: input("train-name").value.trim(),
        teacher: ($("train-teacher") as HTMLSelectElement).value,
        pcaDims: parseInt(input("train-dims").value, 10) || 0,
        dtype: ($("train-dtype") as HTMLSelectElement).value === "F16" ? StaticDtype.F16 : StaticDtype.F32,
        vocabularyArtifactId: input("train-vocabulary").value.trim(),
        terms: ($("train-terms") as HTMLTextAreaElement).value.split("\n").map((t) => t.trim()).filter((t) => t.length),
      };
      for await (const u of training.trainStaticModel(req, { signal: ac.signal })) {
        const ms = `${((performance.now() - t0) / 1000).toFixed(1)} s`;
        if (u.update.case === "progress") {
          const p = u.update.value;
          progress.max = Math.max(1, Number(p.totalItems));
          progress.value = Number(p.completedItems);
          const facts = [
            p.totalItems ? `${p.completedItems} of ${p.totalItems}` : "",
            p.addedTerms || p.refusedTerms ? `${p.addedTerms} terms added, ${p.refusedTerms} refused` : "",
            p.explainedVariance ? `explained variance ${(p.explainedVariance * 100).toFixed(1)}%` : "",
            p.quality ? `top-1 ${p.quality.staticTop1.toFixed(3)} (teacher ${p.quality.baseTop1.toFixed(3)}), rho ${p.quality.similaritySpearman.toFixed(3)}` : "",
          ].filter(Boolean);
          const line = el("li", {}, el("span", { class: "badge", text: ms }), ` ${TrainStage[p.stage]}${facts.length ? `: ${facts.join(", ")}` : ""}`);
          if (p.stage === last && steps.lastElementChild) steps.lastElementChild.replaceWith(line);
          else steps.append(line);
          last = p.stage;
        } else if (u.update.case === "trained") {
          const t = u.update.value;
          progress.value = progress.max;
          const refused = t.refusedTerms.map((r) => `${r.term} (${r.reason})`).join(", ");
          steps.append(el("li", {}, el("span", { class: "badge", text: ms }), ` trained ${t.model?.name ?? ""}: ${t.model?.dims ?? 0} dims, ${t.model?.vocabSize ?? 0} words${t.addedTerms.length ? `, added ${t.addedTerms.join(" ")}` : ""}${refused ? `; refused ${refused}` : ""}`));
        }
      }
      await Promise.all([loadTraining(), refresh()]);
      say($("train-note"), "");
    } catch (e) {
      say($("train-note"), describe(e), "error");
    } finally {
      if (trainingRun === ac) trainingRun = null;
    }
  }, $("train-stop") as HTMLButtonElement);
}

function stopTraining(): void {
  trainingRun?.abort();
  trainingRun = null;
}

/** Cancels an install or training in flight; the server sees it cancelled. */
export function stop(): void {
  installing?.abort();
  installing = null;
  saving?.abort();
  saving = null;
  stopTraining();
}

export function init(): void {
  const read = $("models-catalog-load") as HTMLButtonElement;
  read.addEventListener("click", () => void busy(read, loadCatalog));
  $("train-form").addEventListener("submit", train);
  $("train-stop").addEventListener("click", stopTraining);
  const again = $("train-refresh") as HTMLButtonElement;
  again.addEventListener("click", () => void busy(again, loadTraining));
  // The teachers are the served models: read them with every other
  // list, and again whenever what is served was read again (a bundle
  // loaded or unloaded on the Server tab, a model trained here).
  // One read at a time; a call for another while one runs reads once more after it.
  let reading = false;
  let more = false;
  const teachers = (): void => {
    if (reading) return void (more = true);
    reading = true;
    void loadTraining().finally(() => {
      reading = false;
      if (more) {
        more = false;
        teachers();
      }
    });
  };
  onServed(teachers);
  onCapabilities(teachers);
  sources.init();
}
