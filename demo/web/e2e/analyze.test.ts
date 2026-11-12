// The Analyze tab in a real browser against the real servers that
// scripts/e2e-web.sh starts: the step stream, the document window, the
// annotation drawer, every output format, and the graph wire panel's
// round trip, save and reopen. Nothing is stubbed.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { chromium, type Browser, type Page } from "playwright";

const BASE = process.env.TURBO_E2E_URL ?? "http://127.0.0.1:8080";

/** Opens the Analyze tab, runs the default profile on the sample text and waits for the document. */
async function analyzed(browser: Browser): Promise<Page> {
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, acceptDownloads: true });
  const page = await context.newPage();
  await page.goto(`${BASE}/#analyze`);
  await page.waitForSelector("#analyze-profile option", { state: "attached" });
  await page.waitForSelector("#analyze-format option", { state: "attached" });
  await page.click("#analyze-run");
  await page.waitForSelector("#analyze-window .doc");
  return page;
}

/** The layer table's rows as `layer#instance` keys, each with its item count. */
function layerRows(page: Page): Promise<[string, number][]> {
  return page.locator("#analyze-layers table tr").evaluateAll((rows) =>
    rows
      .map((r) => [...r.querySelectorAll("td")].map((c) => (c.textContent ?? "").trim()))
      .filter((c) => c.length >= 4)
      .map((c): [string, number] => [`${c[0]}#${c[1]}`, Number(c[3])]),
  );
}

test("analyze streams every step of the profile and shows the document it built", async () => {
  const browser = await chromium.launch();
  try {
    const page = await analyzed(browser);
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    const source = await page.inputValue("#analyze-text");

    // The first line names the steps; one line per finished step; the last says complete.
    const steps = await page.locator("#analyze-steps li").allTextContents();
    assert.match(steps[0], /^started: /);
    const planned = steps[0].slice("started: ".length).split(" → ");
    assert.equal(planned.length > 1, true);
    const done = steps.slice(1, -1);
    assert.equal(done.length, planned.length);
    for (const step of planned) assert.equal(done.filter((s) => s.includes(` ${step} (`)).length, 1, `one line for ${step}`);
    assert.match(steps[steps.length - 1], /^complete at [\d.]+ ms, \d+ graph bytes$/);
    assert.equal(await page.locator("#analyze-steps li.error").count(), 0);
    assert.equal(await page.isVisible("#analyze-note.error"), false);

    // Every layer finished, sentences and tokens among them, each listed once.
    const layers = await layerRows(page);
    const keys = layers.map(([k]) => k);
    assert.equal(new Set(keys).size, keys.length);
    assert.equal(keys.includes("turbo:sentences#"), true);
    assert.equal(keys.includes("turbo:tokens#"), true);
    assert.equal(layers.find(([k]) => k === "turbo:sentences#")![1], 3);
    const statuses = await page.locator("#analyze-layers table td:nth-child(3)").allTextContents();
    assert.deepEqual(statuses.filter((s) => s.trim() !== "OK"), []);

    // The window shows the whole source text, its size in UTF-16 units and a chip per layer.
    const shown = await page.locator("#analyze-window .doc").textContent();
    assert.equal(shown, source);
    const summary = await page.locator("#analyze-window p.muted").first().textContent();
    assert.match(summary ?? "", new RegExp(`^\\d+ bytes, ${source.length} UTF-16 units, 1 views, ${layers.length} layers\\.`));
    assert.equal(await page.locator("#analyze-window .chips button").count(), layers.length);
    assert.equal(await page.locator("#analyze-window .doc mark").count() > 0, true);
    assert.equal(await page.locator("#analyze-window .lanes .lane").count() > 0, true);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});

test("the annotation drawer opens a layer's annotations on the source text", async () => {
  const browser = await chromium.launch();
  try {
    const page = await analyzed(browser);
    const source = await page.inputValue("#analyze-text");
    await page.click("#analyze-drawer summary");
    await page.waitForSelector("#drawer-out .chips button");
    assert.deepEqual(await page.locator("#drawer-view option").allTextContents(), ["0: source"]);

    await page.locator("#drawer-out .chips button", { hasText: /^turbo:sentences / }).click();
    const head = page.locator("#drawer-out p.muted", { hasText: /^3 annotations on text, models (m\d+)\.$/ });
    await head.waitFor();
    const rows = await page.locator("#drawer-out table.drawer tr").evaluateAll((trs) =>
      trs.map((r) => [...r.querySelectorAll("td")].map((c) => c.textContent ?? "")).filter((c) => c.length >= 3),
    );
    assert.equal(rows.length, 3);
    // Each row's span, read as UTF-16 offsets, is the text the row quotes.
    for (const [, span, text] of rows) {
      const [start, end] = span.split("..").map(Number);
      assert.equal(source.slice(start, end), text);
    }
    assert.equal(rows.map((r) => r[2]).join(" "), source.replace(/\s+/g, " ").trim());
    // The model the layer names is listed with the annotator that made it.
    const ref = ((await head.textContent()) ?? "").match(/models (m\d+)\.$/)![1];
    const models = await page.locator("#drawer-out h3:has-text('Models') + table tr").evaluateAll((trs) =>
      trs.map((r) => [...r.querySelectorAll("td")].map((c) => c.textContent ?? "")).filter((c) => c.length >= 2),
    );
    assert.equal(models.find((m) => m[0] === ref)?.[1], "turbo-sentdetect:rules");

    // A document with one view has nothing to X-ray, and says so.
    await page.click("#analyze-xray summary");
    await page.waitForSelector("#xray-out p");
    assert.equal(await page.textContent("#xray-out p"), "This document has one view: the profile ran no normalization step.");
  } finally {
    await browser.close();
  }
});

test("every output format downloads the file the page shows", async () => {
  const browser = await chromium.launch();
  try {
    const page = await analyzed(browser);
    const formats = await page.locator("#analyze-format option").evaluateAll((os) => os.map((o) => (o as HTMLOptionElement).value));
    assert.equal(formats.length > 1, true);
    for (const format of formats) {
      await page.selectOption("#analyze-format", format);
      await page.locator("#analyze-export-out").evaluate((e) => e.replaceChildren());
      await page.click("#analyze-export");
      const save = page.locator("#analyze-export-out button");
      await save.waitFor();
      assert.equal(await page.locator("#analyze-export-out .error").count(), 0, `${format} formats`);
      const name = ((await save.textContent()) ?? "").replace(/^Download /, "");
      assert.match(name, /^document\.[a-z0-9]+$/, format);
      const [file] = await Promise.all([page.waitForEvent("download"), save.click()]);
      assert.equal(file.suggestedFilename(), name);
      const bytes = readFileSync(await file.path());
      // Brat standoff lists entities only, and the default profile finds none.
      if (format !== "brat") assert.equal(bytes.length > 0, true, `${format} is empty`);
      // A text format is shown in the page exactly as downloaded.
      if (await page.locator("#analyze-export-out pre").count()) {
        assert.equal(await page.textContent("#analyze-export-out pre"), new TextDecoder().decode(bytes), format);
      }
    }
  } finally {
    await browser.close();
  }
});

test("the graph survives a server round trip, a save and a reopen", async () => {
  const browser = await chromium.launch();
  try {
    const page = await analyzed(browser);
    const source = await page.inputValue("#analyze-text");
    const layers = (await layerRows(page)).length;
    const bytes = Number((await page.locator("#analyze-steps li").last().textContent())?.match(/(\d+) graph bytes$/)?.[1]);

    await page.click("#analyze-wire summary");
    await page.waitForSelector("#wire-out .kids.top");
    const parts = await page.locator("#wire-out .kids.top > * .wire > b").allTextContents();
    for (const part of ["magic", "source text", "views", "layers"]) assert.equal(parts.includes(part), true, `wire part ${part}`);

    await page.click("#wire-roundtrip");
    await page.waitForSelector("#wire-note:not([hidden])");
    assert.match((await page.textContent("#wire-note")) ?? "", new RegExp(`, ${layers} layers and 1 views: byte for byte the same\\.$`));

    const [file] = await Promise.all([page.waitForEvent("download"), page.click("#wire-save")]);
    assert.equal(file.suggestedFilename(), "document.graph");
    const saved = readFileSync(await file.path());
    assert.equal(saved.length, bytes);

    // A fresh page opens the saved graph and shows the same document without analyzing.
    const other = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
    await other.goto(`${BASE}/#analyze`);
    await other.click("#analyze-wire summary");
    await other.setInputFiles("#wire-file", { name: "document.graph", mimeType: "application/octet-stream", buffer: saved });
    await other.waitForSelector("#wire-note:not([hidden])");
    assert.equal(await other.textContent("#wire-note"), `Opened document.graph: ${layers} layers, 1 views.`);
    await other.waitForSelector("#analyze-window .doc");
    assert.equal(await other.locator("#analyze-window .doc").textContent(), source);
  } finally {
    await browser.close();
  }
});

test("a file that is not a graph is refused with a reason", async () => {
  const browser = await chromium.launch();
  try {
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const page = await context.newPage();
    await page.goto(`${BASE}/#analyze`);
    await page.click("#analyze-wire summary");
    await page.setInputFiles("#wire-file", { name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("not a graph at all") });
    await page.waitForSelector("#wire-note.error:not([hidden])");
    assert.equal(((await page.textContent("#wire-note")) ?? "").trim().length > 0, true);
    assert.equal(await page.locator("#analyze-window .doc").count(), 0);
  } finally {
    await browser.close();
  }
});
