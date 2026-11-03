// The Analyze form: the text, the profile picker, the custom step matrix,
// the chunk controls, the model and the offset unit. `read()` gives the
// AnalyzePipelineRequest the form describes.

import { h } from "../dom.js";

// The step ids the server's Analyzer DAG knows, in the order they are sent:
// a normalizer first, since the steps after it read its view. The server
// has no call listing them (only the profiles' steps come from
// Capabilities), so they are listed here.
export const STEP_GROUPS = [
  ["Normalize", ["turbo:normalize", "turbo:normalize:nfkc", "turbo:normalize:whitespace", "turbo:normalize:invisible", "turbo-structure:project"]],
  ["Structure and sentences", ["turbo-structure:blocks", "turbo-sentdetect:rules"]],
  ["Tokens and terms", ["turbo-tokenize:uax29-untyped", "turbo-stem:terms", "turbo-stem:stemmer", "turbo-stopword:bundled"]],
  ["Lexical", ["turbo-abbrev:schwartz-hearst", "turbo-keywords:yake", "turbo-geo:names", "turbo:term-indexer"]],
  ["Boundaries and chunks", ["turbo:structure-boundaries", "turbo:lexical-boundaries", "turbo:chunker", "turbo:level-builder"]],
  ["Model backed", ["turbo:subwords", "turbo:sparse", "turbo:sentence-vectors", "turbo:chunk-vectors", "turbo:sentence-chunk-vectors", "turbo:subword-vectors", "turbo:level-vectors"]],
];

const MODEL_STEPS = new Set(STEP_GROUPS[5][1]);
const ORDER = STEP_GROUPS.flatMap(([, s]) => s);

// The chunk controls, each a set of steps added to the matrix.
export const CHUNKING = {
  sentence: {
    label: "Sentence",
    note: "Sentence spans from the rule detector, one vector a sentence when a model is picked.",
    steps: ["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped", "turbo:sentence-vectors"],
  },
  window: {
    label: "Token window",
    note: "The chunker packs sentences into windows of the model's width (or 256 tokens without one); the Chunks view compares other budgets.",
    steps: ["turbo-structure:blocks", "turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped", "turbo:structure-boundaries", "turbo:chunker", "turbo:level-builder"],
  },
  semantic: {
    label: "Semantic",
    note: "Structure and lexical boundaries scored for drift, then chunk vectors; the Chunks view shows the drift curve and its cuts.",
    steps: [
      "turbo-structure:blocks", "turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped", "turbo-stem:terms", "turbo-stopword:bundled",
      "turbo:structure-boundaries", "turbo:lexical-boundaries", "turbo:chunker", "turbo:level-builder", "turbo:chunk-vectors",
    ],
  },
};

export const SAMPLE = `Dr. Ada Lovelace wrote the first published algorithm in London in 1843. Her notes on the Analytical Engine ran longer than the paper they translated.

Charles Babbage designed the engine; it was never finished. Today the notes are read as the start of programming. Cafe\u0301 owners in Paris still name drinks after her \u{1F600}, and the \uFB01rst edition sells for a small fortune.`;

function option(value, text, selected) {
  const o = h("option", { value }, text);
  o.selected = !!selected;
  return o;
}

export function controls(caps, onRun) {
  const profiles = (caps && caps.profiles) || [];
  const bundles = (caps && caps.bundles) || [];
  const checks = new Map();
  const text = h("textarea", { id: "an-text", rows: 9, spellcheck: "false", "aria-label": "Text to analyze" });
  text.value = SAMPLE;
  const profile = h(
    "select",
    { id: "an-profile", "aria-label": "Profile" },
    profiles.map((p) => option(p.name, `${p.name} (v${p.version})`, p.name === "minimal-v1")),
    option("custom", "custom steps"),
  );
  const model = h(
    "select",
    { id: "an-model", "aria-label": "Model" },
    option("", "no model"),
    bundles.map((b, i) => option(b.name, `${b.name} (${b.task || "?"})`, i === 0)),
  );
  const offsets = h("select", { id: "an-offsets", "aria-label": "Offset unit" }, option("1", "UTF-16", true), option("0", "UTF-8"), option("2", "code points"));
  const facts = h("p", { class: "muted small" });

  const matrix = h(
    "div",
    { class: "matrix" },
    STEP_GROUPS.map(([title, steps]) =>
      h(
        "fieldset",
        {},
        h("legend", {}, title),
        steps.map((s) => {
          const box = h("input", { type: "checkbox", value: s });
          checks.set(s, box);
          box.addEventListener("change", () => {
            profile.value = "custom";
            describe();
          });
          return h("label", { class: MODEL_STEPS.has(s) ? "needs-model" : "" }, box, h("code", {}, s));
        }),
      ),
    ),
  );

  const chunkNote = h("p", { class: "muted small" });
  const chunking = h(
    "div",
    { class: "seg", role: "group", "aria-label": "Chunking" },
    Object.entries(CHUNKING).map(([key, c]) =>
      h("button", { type: "button", dataset: { chunk: key }, onclick: () => applyChunking(key) }, c.label),
    ),
  );

  function selected() {
    return ORDER.filter((s) => checks.get(s).checked);
  }

  function setSteps(steps) {
    const on = new Set(steps);
    for (const [s, box] of checks) box.checked = on.has(s);
  }

  function applyChunking(key) {
    const c = CHUNKING[key];
    const steps = new Set(selected());
    for (const s of c.steps) if (!MODEL_STEPS.has(s) || model.value) steps.add(s);
    setSteps([...steps]);
    profile.value = "custom";
    for (const b of chunking.children) b.setAttribute("aria-pressed", b.dataset.chunk === key ? "true" : "false");
    chunkNote.textContent = c.note;
    describe();
  }

  function describe() {
    const p = profiles.find((x) => x.name === profile.value);
    if (p) {
      setSteps(p.steps);
      const need = p.required_bundles.length ? `needs ${p.required_bundles.join(", ")}` : "needs no model";
      facts.textContent = `${p.steps.length} steps, layers ${p.layers.join(", ")}; ${need}.`;
    } else {
      const n = selected();
      const modelSteps = n.filter((s) => MODEL_STEPS.has(s));
      facts.textContent = `${n.length} custom steps${modelSteps.length && !model.value ? "; the model-backed ones need a model" : ""}.`;
    }
  }

  profile.addEventListener("change", describe);
  model.addEventListener("change", describe);
  describe();

  const run = h("button", { type: "submit", class: "primary", id: "an-run" }, "Analyze");
  const stop = h("button", { type: "button", id: "an-stop", disabled: true }, "Stop");
  const file = h("input", { type: "file", accept: ".txt,.md,.text,text/*", "aria-label": "Load a text file" });
  file.addEventListener("change", async () => {
    if (file.files[0]) text.value = await file.files[0].text();
  });

  const form = h(
    "form",
    { class: "panel an-form", onsubmit: (ev) => (ev.preventDefault(), onRun()) },
    h("div", { class: "row" }, h("h2", {}, "Text"), file),
    text,
    h(
      "div",
      { class: "row wrap" },
      h("label", {}, "Profile ", profile),
      h("label", {}, "Model ", model),
      h("label", {}, "Offsets ", offsets),
    ),
    facts,
    h("details", { class: "steps" }, h("summary", {}, "Step matrix"), matrix),
    h("div", { class: "row wrap" }, h("span", { class: "muted" }, "Chunking"), chunking),
    chunkNote,
    h("div", { class: "row" }, run, stop),
  );

  return {
    form,
    run,
    stop,
    read() {
      const custom = profile.value === "custom";
      return {
        doc_id: "workbench",
        text: text.value,
        offsets: Number(offsets.value),
        profile: custom ? "" : profile.value,
        steps: custom ? selected() : [],
        model: model.value,
      };
    },
  };
}
