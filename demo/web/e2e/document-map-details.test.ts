// What document-map.test.ts leaves out of the document map under
// Analyze, in a real browser against the real servers scripts/e2e-web.sh
// starts: the meaning map's reading order, a picked river term fading the
// other streams, the numbers as tables, the phone layout, and the
// similarity matrix's pairs as CSV and on the clipboard. Nothing is
// stubbed.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { chromium, type Browser, type Page } from "playwright";
import { settled, track } from "./settle.js";

const BASE = process.env.TURBO_E2E_URL ?? "http://127.0.0.1:8080";
const DOCUMENT = [
  "Ada Lovelace wrote the first published program for the Analytical Engine. Charles Babbage designed the engine, and it was never finished. Her notes describe how the engine could compute Bernoulli numbers.",
  "A century later, Grace Hopper wrote one of the first compilers. That idea led to COBOL, and banks still run COBOL programs today. Hopper also popularised the word debugging.",
  "The central bank raised interest rates again this quarter. Higher rates make borrowing more expensive for households. Banks passed the rates on to their customers within a week.",
  "A vector can stand for the meaning of a sentence. Words that are used alike end up close together. Similar sentences have similar vectors, so a search can find them by meaning.",
].join("\n\n");

/** Analyzes DOCUMENT with default-v1 and the tiny BERT model, and waits for the map. */
async function mapped(browser: Browser, width = 1280, height = 900): Promise<Page> {
  const page = await (await browser.newContext({ viewport: { width, height } })).newPage();
  track(page);
  await page.goto(`${BASE}/#analyze`);
  await page.waitForSelector("#analyze-profile option[value=default-v1]", { state: "attached" });
  await page.selectOption("#analyze-profile", "default-v1");
  await page.selectOption("#analyze-model", "tiny-bert-bundle");
  await page.fill("#analyze-text", DOCUMENT);
  await page.click("#analyze-run");
  await page.waitForSelector("#projection-out .map-and-text svg");
  // The other panels under Analyze make their own calls for the new document and can still move this one.
  await settled(page);
  return page;
}

const MAP = "#projection-out .map-and-text .figure";
const TEXT = "#projection-out .doc.chunked";
const RIVER = "#projection-out h3:has-text('Term river') + p + .figure";

test("the meaning map numbers its dots in reading order, and the text points back at them", async () => {
  const browser = await chromium.launch();
  try {
    const page = await mapped(browser);
    const chunks = await page.locator(`${TEXT} .ck sup`).count();
    const dots = page.locator(`${MAP} g.dot`);
    // The trail joins every dot in reading order, and each dot is numbered in that order.
    const trail = (await page.locator(`${MAP} polyline.trail`).getAttribute("points")) ?? "";
    assert.equal(trail.trim().split(" ").length, chunks);
    const labels = await page.locator(`${MAP} g.dot text.label`).allTextContents();
    assert.deepEqual(labels.map((l) => Number(l.split(" ")[0])), labels.map((_, i) => i + 1));
    assert.match((await dots.first().getAttribute("aria-label")) ?? "", new RegExp(`^Chunk 1 of ${chunks}`));
    // Pointing at the last chunk of text lights the last dot, and only it.
    await page.locator(`${TEXT} .ck[data-chunk="${chunks - 1}"]`).first().hover();
    await page.waitForFunction((n) => document.querySelectorAll(".map-and-text g.dot")[n]?.classList.contains("lit"), chunks - 1);
    assert.equal(await page.locator(`${MAP} g.dot.lit`).count(), 1);
  } finally {
    await browser.close();
  }
});

test("a term picked on the river fades the others and is marked in the text, and the numbers are tables too", async () => {
  const browser = await chromium.launch();
  try {
    const page = await mapped(browser);
    const chunks = await page.locator(`${TEXT} .ck sup`).count();
    const fig = page.locator(RIVER);
    const streams = fig.locator("path.stream");
    const n = await streams.count();
    assert.equal(n > 0 && n <= 8, true, `${n} streams`);
    const keys = fig.locator(".legend button.key");

    // Picking a term fades the other streams and marks its occurrences in the text.
    const key = keys.first();
    const [term, count] = ((await key.textContent()) ?? "").trim().split(" ");
    await key.click();
    await page.waitForSelector(`${TEXT} mark.occ`);
    assert.equal(await fig.locator("path.stream.faded").count(), n - 1);
    const marked = await page.locator(`${TEXT} mark.occ`).allTextContents();
    assert.equal(marked.length, Number(count));
    for (const m of marked) assert.equal(m.toLowerCase().startsWith(term.slice(0, 3)), true, `"${m}" for ${term}`);
    // Clear brings the other streams back.
    await fig.locator(".legend button", { hasText: "Clear" }).click();
    assert.equal(await fig.locator("path.stream.faded").count(), 0);

    // The numbers are there as tables too.
    await page.click("#projection-out details.tables summary");
    assert.equal(await page.locator("#projection-out details.tables table").count() >= 2, true);
    assert.equal(await page.locator("#projection-out details.tables table").first().locator("tr").count(), chunks + 1);
  } finally {
    await browser.close();
  }
});

test("on a phone the figures scroll in their own boxes and the page does not", async () => {
  const browser = await chromium.launch();
  try {
    const page = await mapped(browser, 390, 844);
    // The meaning map, the seismograph and the river: each is wider than the phone and scrolls in its own box.
    // (The similarity matrix of a text this short is narrower than the phone; its box scrolls all the same.)
    assert.equal(await page.locator("#projection-out .figure.matrix .scroll").evaluate((b) => getComputedStyle(b).overflowX), "auto");
    const boxes = await page.locator("#projection-out .figure:not(.matrix) .scroll").evaluateAll((all) =>
      all.map((b) => ({ overflow: getComputedStyle(b).overflowX, wider: b.scrollWidth > b.clientWidth })),
    );
    assert.deepEqual(boxes, Array(3).fill({ overflow: "auto", wider: true }));
    // And the page itself does not scroll sideways.
    const sideways = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth);
    assert.equal(sideways, false);
  } finally {
    await browser.close();
  }
});

test("the similarity matrix's CSV has each pair of chunks once, with the cosines the matrix shows", async () => {
  const browser = await chromium.launch();
  try {
    const page = await mapped(browser);
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    const matrix = "#projection-out .figure.matrix";
    await page.waitForSelector(`${matrix} canvas`);
    const chunks = await page.locator(`${TEXT} .ck sup`).count();
    const label = (await page.locator(`${matrix} canvas`).getAttribute("aria-label")) ?? "";
    const range = label.split(": cosines from ")[1]?.split(" to ").map(Number) ?? [];
    assert.equal(range.length === 2 && range[0] <= range[1] && range[1] <= 1, true, label);
    const [lo, hi] = range;

    // Every cell above the diagonal, walked by arrow key from the first (chunks 1 and 2): what the
    // matrix reads out for each pair, to hold the file's rows against.
    const plot = page.locator(`${matrix} .plot`);
    const info = page.locator(`${matrix} .info`);
    await page.mouse.move(0, 0);
    await plot.focus();
    await page.keyboard.press("ArrowRight");
    let [row, col] = [0, 1];
    const shown = new Map<string, string>();
    for (let a = 0; a < chunks; a++) {
      for (let b = a + 1; b < chunks; b++) {
        for (; row < a; row++) await page.keyboard.press("ArrowDown");
        for (; col < b; col++) await page.keyboard.press("ArrowRight");
        for (; col > b; col--) await page.keyboard.press("ArrowLeft");
        const said = (await info.textContent()) ?? "";
        const [names, rest] = said.split(": cosine ");
        const [x, y] = names.split(" and ").map((c) => Number(c.slice("chunk ".length)));
        assert.equal(rest === undefined || Number.isNaN(x) || Number.isNaN(y), false, said);
        shown.set(`${x},${y}`, rest.split(". ")[0]);
      }
    }
    assert.equal(shown.size, (chunks * (chunks - 1)) / 2);

    const [file] = await Promise.all([page.waitForEvent("download"), page.locator(`${matrix} .take button`, { hasText: "Download CSV" }).click()]);
    const rows = readFileSync(await file.path()).toString().trimEnd().split("\r\n");
    assert.equal(rows[0], "chunk,other chunk,cosine");
    const pairs = rows.slice(1).map((r) => r.split(",").map(Number));
    assert.equal(pairs.length, (chunks * (chunks - 1)) / 2);
    const seen = new Set<string>();
    for (const [a, b, c] of pairs) {
      assert.equal(a >= 1 && a < b && b <= chunks, true, `${a},${b}`);
      seen.add(`${a},${b}`);
      // The label rounds the range to three places.
      assert.equal(c >= lo - 0.0005 && c <= hi + 0.0005, true, `${a},${b},${c} outside ${lo}..${hi}`);
    }
    assert.equal(seen.size, pairs.length, "no pair twice");
    // Each row carries the cosine of its own cell.
    for (const [a, b, c] of pairs) assert.equal(c.toFixed(3), shown.get(`${a},${b}`), `the row for chunks ${a} and ${b}`);

    // Copy puts the same rows on the clipboard, tab-separated.
    await page.context().grantPermissions(["clipboard-read", "clipboard-write"], { origin: BASE });
    await page.locator(`${matrix} .take button`, { hasText: "Copy" }).click();
    await page.locator(`${matrix} .take [role=status]`, { hasText: `Copied ${pairs.length} rows.` }).waitFor();
    const copied = (await page.evaluate(() => navigator.clipboard.readText())).trimEnd().split(/\r?\n/);
    assert.deepEqual(copied, rows.map((r) => r.split(",").join("\t")));
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});
