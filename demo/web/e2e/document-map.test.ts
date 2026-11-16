// The document map in Analyze: the meaning map, the topic seismograph
// and the term river over ProjectChunks, and the pointing that links
// them with the text. A real server and a real demo (the shared ones of
// scripts/e2e-web.sh); nothing is stubbed.

import assert from "node:assert/strict";
import { test } from "node:test";
import { chromium, type Page } from "playwright";

const BASE = process.env.TURBO_E2E_URL ?? "http://127.0.0.1:8080";
const OUT = "#projection-out";
// The figures by their headings: the similarity matrix sits between the map and these two.
const SEISMOGRAPH = `${OUT} h3:has-text('Topic seismograph') + .figure`;
const RIVER = `${OUT} h3:has-text('Term river') + p + .figure`;

// Four paragraphs on three subjects; "COBOL" occurs twice.
const TEXT = [
  "Ada Lovelace wrote the first published program in 1843. Her notes on the Analytical Engine described how the machine could compute Bernoulli numbers. Charles Babbage designed the engine in London, but the engine was never built. Lovelace saw that the machine could work with symbols as well as numbers.",
  "A century later, Grace Hopper wrote one of the first compilers for the UNIVAC. Her compiler translated mathematical notation into machine code. Hopper argued that programs should be written in words people could read. That idea led to COBOL, and banks still run COBOL programs today.",
  "Banks process billions of transactions on that old code every day. When the central bank raises interest rates, those bank systems recalculate millions of loans overnight. Higher interest rates push borrowing costs up for households and firms. Inflation this quarter kept rates high for longer than expected.",
  "Today, search engines turn words into vectors so a machine can compare meaning. A query and a passage that share no words can still sit close together in that vector space. Each vector is a list of numbers, and similar meanings point in similar directions. Lovelace's idea that a machine could work with symbols is the root of that design.",
].join("\n\n");

/** Analyzes TEXT with `profile` and waits for the document. */
async function analyze(page: Page, profile: string): Promise<void> {
  await page.fill("#analyze-text", TEXT);
  await page.selectOption("#analyze-profile", profile);
  await page.click("#analyze-run");
  await page.waitForSelector("#analyze-window .doc");
}

test("the document map draws a dot per chunk and links the views with the text", async () => {
  const browser = await chromium.launch();
  try {
    const page = await (await browser.newContext({ viewport: { width: 1400, height: 1000 } })).newPage();
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`${BASE}/#analyze`);
    await page.waitForSelector("#analyze-profile option", { state: "attached" });
    await analyze(page, "default-v1");
    await page.waitForSelector(`${OUT} svg .dot`);

    // The meaning map: a dot per chunk of the text beside it, a trail, two axes.
    const chunks = await page.locator(`${OUT} .chunked .ck`).evaluateAll((all) => new Set(all.map((e) => (e as HTMLElement).dataset.chunk)).size);
    assert.equal(chunks > 1, true, `${chunks} chunks`);
    assert.equal(await page.locator(`${OUT} svg .dot`).count(), chunks);
    assert.equal(await page.locator(`${OUT} svg .dot[tabindex="0"]`).count(), chunks);
    assert.equal(await page.locator(`${OUT} svg .trail`).count(), 1);
    const axes = await page.locator(`${OUT} svg text.axis`).allTextContents();
    assert.deepEqual(axes.map((a) => a.includes("% of the variance")), [true, true]);

    // Pointing at a dot lights its chunk in the text and says which it is.
    await page.locator(`${OUT} svg .dot`).nth(1).hover();
    await page.waitForSelector(`${OUT} .chunked .ck.lit`);
    assert.deepEqual(await page.locator(`${OUT} .chunked .ck.lit`).evaluateAll((all) => [...new Set(all.map((e) => (e as HTMLElement).dataset.chunk))]), ["1"]);
    assert.equal(((await page.locator(`${OUT} .figure .info`).first().textContent()) ?? "").startsWith(`Chunk 2 of ${chunks}`), true);
    // And the other way: a chunk of the text lights its dot.
    await page.locator(`${OUT} .chunked .ck[data-chunk="0"]`).first().hover();
    await page.waitForFunction((out) => (document.querySelector(`${out} svg .dot.lit .label`)?.textContent ?? "").startsWith("1"), OUT);
    assert.equal(await page.locator(`${OUT} svg .dot.lit`).count(), 1);

    // The seismograph: the chunks as a band, a mark per cut, and the gaps by arrow key.
    const seismograph = page.locator(SEISMOGRAPH);
    assert.equal(await seismograph.locator("svg .band").count(), chunks);
    assert.equal((await seismograph.locator("svg .stem, svg path.line").count()) > 0, true);
    assert.equal(await seismograph.locator("svg .cut-mark").count(), chunks);
    // The profile's own row of ticks, then one per budget of the box (small and large).
    assert.equal(await seismograph.locator("svg .barcode").count(), 3);
    const rows = await seismograph.locator("svg text.row-name title").allTextContents();
    assert.equal(rows.some((t) => t.startsWith("small:") && t.includes("cuts shared")), true, rows.join(" | "));
    const pad = seismograph.locator("svg .pad");
    // Pointing at the curve picks the gap nearest the pointer and shows the sentence on each side.
    await pad.scrollIntoViewIfNeeded();
    await pad.hover();
    await seismograph.locator(".info", { hasText: "Gap " }).waitFor();
    assert.equal(((await seismograph.locator(".info").textContent()) ?? "").startsWith("Gap "), true);
    assert.equal(await seismograph.locator(".info .quote").count(), 2);
    await page.mouse.move(0, 0);
    await pad.focus();
    await page.keyboard.press("ArrowRight");
    const now = Number(await pad.getAttribute("aria-valuenow"));
    const most = Number(await pad.getAttribute("aria-valuemax"));
    for (let i = now; i <= most; i++) await page.keyboard.press("ArrowRight");
    assert.equal(await pad.getAttribute("aria-valuenow"), String(most), "the last gap is the right end");
    assert.equal(((await seismograph.locator(".info").textContent()) ?? "").startsWith(`Gap ${most} of ${most}`), true);
    for (let i = 0; i <= most; i++) await page.keyboard.press("ArrowLeft");
    assert.equal(await pad.getAttribute("aria-valuenow"), "1", "the first gap is the left end");

    // The river: a stream and a legend key per term, and a term marks its occurrences in the text.
    const river = page.locator(RIVER);
    const streams = await river.locator("svg .stream").count();
    assert.equal(streams > 0, true);
    assert.equal(await river.locator("button.key").count(), streams);
    assert.equal(await river.locator("svg .cut-line").count(), chunks - 1);
    const key = river.locator("button.key", { hasText: "cobol" });
    await key.scrollIntoViewIfNeeded();
    await key.hover();
    await page.waitForSelector(`${OUT} .chunked mark.occ`);
    assert.deepEqual(await page.locator(`${OUT} .chunked mark.occ`).allTextContents(), ["COBOL", "COBOL"]);
    assert.equal(((await river.locator(".info").textContent()) ?? "").includes("occurs 2 times"), true);
    // Clear takes the marks off and puts the prompt back.
    await river.locator("button", { hasText: "Clear" }).click();
    await page.waitForSelector(`${OUT} .chunked mark.occ`, { state: "detached" });
    assert.equal(await river.locator(".info").textContent(), "Point at a stream or a name to mark that term in the text.");

    // The switch draws the heaviest terms instead, words every chunk uses among them.
    const skipped = (await river.locator("button.key").allTextContents()).map((k) => k.trim().split(" ")[0]);
    assert.equal(skipped.includes("the"), false, skipped.join(" "));
    await page.uncheck("#projection-uneven");
    await page.waitForFunction((out) => [...document.querySelectorAll(`${out} .figure button.key`)].some((k) => (k.textContent ?? "").trim().startsWith("the ")), OUT);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});

test("a graph with no vectors says how to get them", async () => {
  const browser = await chromium.launch();
  try {
    const page = await (await browser.newContext({ viewport: { width: 1400, height: 1000 } })).newPage();
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`${BASE}/#analyze`);
    await page.waitForSelector("#analyze-profile option", { state: "attached" });
    await analyze(page, "minimal-v1");
    await page.waitForSelector(`${OUT} :text("No vectors in this graph")`);
    assert.equal(await page.locator(`${OUT} svg .dot`).count(), 0);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});
