// Models: the model catalog with licence acceptance (licences.ts), the
// user's model sources (sources.ts), a streaming InstallModel monitor,
// and static model distillation with a streaming TrainStaticModel
// monitor, the teachers it can use and the static models it made.

import { InstallStage, StaticDtype, TrainStage, type CatalogEntry, type CatalogFile, type LicenceTerms } from "../gen/turbo_training_pb.js";
import { $, el, input, options, say, size, table } from "../dom.js";
import { catalog, describe, training } from "../rpc.js";
import { refresh } from "../state.js";
import { accept, acceptedTable, revoke, termsBlock, who, type Who } from "./licences.js";
import { download, nonCommercial } from "./download.js";
import * as sources from "./sources.js";

let installing: AbortController | null = null;
let saving: AbortController | null = null;
let trainingRun: AbortController | null = null;

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

function entryRow(e: CatalogEntry, w: Who): HTMLElement[] {
  const terms = e.missingTerms.map((t) => termsBlock(e, t, w, (t: LicenceTerms) => act(() => accept(e, t))));
  let install: HTMLElement | string = "";
  if (w.canChange) {
    const b = el("button", { type: "button", text: e.installed ? "Reinstall" : "Install" });
    b.disabled = e.missingTerms.length > 0;
    b.title = b.disabled ? "Accept the licence terms first" : "";
    b.addEventListener("click", () => void installModel(e));
    install = b;
  }
  const licence = el("span", { text: e.licence }, nonCommercial(e) ? el("strong", { class: "error", text: " Non-commercial use only" }) : null);
  return [el("span", { text: e.id }), el("span", { text: `${e.kind} ${e.language}` }), licence, el("span", { text: size(e.totalBytes) }), el("span", { text: e.installed ? "yes" : "" }), el("div", {}, terms, install, saveButtons(e, w))];
}

/** A save button per file, once the user may have the artifact. */
function saveButtons(e: CatalogEntry, w: Who): HTMLElement | null {
  if (!w.user || e.missingTerms.length > 0) return null;
  return el(
    "div",
    {},
    e.files.map((f) => {
      const b = el("button", { type: "button", text: `Save ${f.path}` });
      b.addEventListener("click", () => void saveFile(e, f));
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
      el("h3", { text: "Catalog" }),
      el("p", { class: "muted", text: w.user ? `Licence terms still to accept are those for ${w.user}; read them before accepting.${w.why ? ` (${w.why})` : ""}` : w.why }),
      table(["id", "kind", "licence", "size", "installed", ""], cat.entries.map((e) => entryRow(e, w))),
      el("h3", { text: "Accepted terms" }),
      acceptedTable(acc, w, (t: LicenceTerms) => act(() => revoke(t))),
      el("h3", { text: "Installed" }),
      table(["id", "version", "files", "size", "directory"], installed.models.map((m) => [m.id, m.version, String(m.files.length), size(m.totalBytes), m.dir])),
    );
    say(note(), "");
  } catch (e) {
    say(note(), describe(e), "error");
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
    options($("train-teacher") as HTMLSelectElement, t.teachers.map((x) => x.name), t.teachers.map((x) => `${x.name} (hidden ${x.hidden}, ${x.vocabSize} words)`));
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
      made = el("p", { class: "error", text: describe(e) });
    }
    box.replaceChildren(el("h3", { text: "Teachers" }), table(["name", "model", "hidden", "words"], t.teachers.map((x) => [x.name, `${x.modelId}@${x.revision}`, String(x.hidden), String(x.vocabSize)])), el("h3", { text: "Static models" }), made);
  } catch (e) {
    box.replaceChildren(el("p", { class: "error", text: describe(e) }));
  }
}

function deleteButton(name: string): HTMLElement {
  const b = el("button", { type: "button", text: "Delete" });
  b.addEventListener("click", async () => {
    try {
      await training.deleteStaticModel({ name });
      await Promise.all([loadTraining(), refresh()]);
    } catch (e) {
      say($("train-note"), describe(e), "error");
    }
  });
  return b;
}

async function train(ev: Event): Promise<void> {
  ev.preventDefault();
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
  $("models-catalog-load").addEventListener("click", () => void loadCatalog());
  $("train-form").addEventListener("submit", train);
  $("train-stop").addEventListener("click", stopTraining);
  $("train-refresh").addEventListener("click", () => void loadTraining());
  void loadTraining();
  sources.init();
}
