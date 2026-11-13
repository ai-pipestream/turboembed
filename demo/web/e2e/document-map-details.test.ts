// What document-map.test.ts leaves out of the document map under
// Analyze, in a real browser against the real servers scripts/e2e-web.sh
// starts: the meaning map's reading order, a picked river term fading the
// other streams, the numbers as tables, and the phone layout. Nothing is
// stubbed.

import assert from "node:assert/strict";
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
    const boxes = await page.locator("#projection-out .figure .scroll").evaluateAll((all) =>
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
