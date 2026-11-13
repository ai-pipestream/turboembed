// How often the Document map in Analyze asks the server: one
// ProjectChunks call carries the map, the similarity matrix and the query
// in the box, and a refused part is asked for again without it. A real
// server and a real demo (the shared ones of scripts/e2e-web.sh); nothing
// is stubbed.

import assert from "node:assert/strict";
import { test } from "node:test";
import { chromium, type Browser, type BrowserContext, type Page } from "playwright";
import { Hold } from "./hold.js";

const BASE = process.env.TURBO_E2E_URL ?? "http://127.0.0.1:8080";
const OUT = "#projection-out";
const CALL = "AnalysisService/ProjectChunks";

const TEXT = [
  "Ada Lovelace wrote the first published program in 1843. Her notes on the Analytical Engine described how the machine could compute Bernoulli numbers. Charles Babbage designed the engine in London, but the engine was never built. Lovelace saw that the machine could work with symbols as well as numbers.",
  "A century later, Grace Hopper wrote one of the first compilers for the UNIVAC. Her compiler translated mathematical notation into machine code. Hopper argued that programs should be written in words people could read. That idea led to COBOL, and banks still run COBOL programs today.",
  "Banks process billions of transactions on that old code every day. When the central bank raises interest rates, those bank systems recalculate millions of loans overnight. Higher interest rates push borrowing costs up for households and firms. Inflation this quarter kept rates high for longer than expected.",
  "Today, search engines in Zürich turn words into vectors so a machine can compare meaning. A query and a passage that share no words can still sit close together in that vector space. Each vector is a list of numbers, and similar meanings point in similar directions. Lovelace's idea that a machine could work with symbols is the root of that design.",
].join("\n\n");
const BANK = "what happens when the central bank raises interest rates";
const SEARCH = "Zürich search engines compare meaning with vectors";

type Seen = { page: Page; context: BrowserContext; calls: () => number; errors: string[] };

/** A page that counts the ProjectChunks calls it sends. */
async function watch(browser: Browser): Promise<Seen> {
  const context = await browser.newContext({ viewport: { width: 1400, height: 1000 } });
  const page = await context.newPage();
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  let sent = 0;
  page.on("request", (r) => {
    if (new URL(r.url()).pathname.endsWith(CALL)) sent++;
  });
  return { page, context, calls: () => sent, errors };
}

/** Fills the form of Analyze; `query` goes in the map's query box before anything is drawn. */
async function ready(page: Page, query = ""): Promise<void> {
  await page.goto(`${BASE}/#analyze`);
  await page.waitForSelector("#analyze-profile option", { state: "attached" });
  await page.fill("#analyze-text", TEXT);
  await page.selectOption("#analyze-profile", "default-v1");
  if (query) {
    await page.fill("#projection-query", query);
    // The typing's own timer finds nothing on show and asks nothing.
    await page.waitForTimeout(700);
  }
}

async function analyze(page: Page): Promise<void> {
  await page.click("#analyze-run");
  await page.waitForSelector(`${OUT} svg .dot`);
}

const placed = (page: Page, q: string): Promise<unknown> => page.waitForFunction(([out, q]) => document.querySelector(`${out} svg .query title`)?.textContent === q, [OUT, q] as const);

test("one Analyze draws the map, the matrix and the query from one ProjectChunks call", async () => {
  const browser = await chromium.launch();
  try {
    // Without a query: the map and its matrix.
    const plain = await watch(browser);
    await ready(plain.page);
    await analyze(plain.page);
    await plain.page.waitForSelector(`${OUT} .matrix canvas`);
    await plain.page.waitForTimeout(700);
    assert.equal(plain.calls(), 1, "calls for the map and the matrix");
    assert.equal(await plain.page.locator(`${OUT} svg .query`).count(), 0);
    assert.deepEqual(plain.errors, []);

    // With a query in the box: the map, its matrix and the query's place and cosines.
    const asked = await watch(browser);
    await ready(asked.page, BANK);
    assert.equal(asked.calls(), 0);
    await analyze(asked.page);
    await placed(asked.page, BANK);
    await asked.page.waitForSelector(`${OUT} .matrix canvas`);
    await asked.page.waitForTimeout(700);
    assert.equal(asked.calls(), 1, "calls for the map, the matrix and the query");
    assert.equal(((await asked.page.locator("#projection-query-out").textContent()) ?? "").startsWith("Embedded by tiny-bert"), true);
    assert.equal((await asked.page.locator("#projection-closest li").count()) > 0, true);
    assert.equal(await asked.page.locator(`${OUT} svg .dot .glow`).count(), await asked.page.locator(`${OUT} svg .dot`).count());

    // A query typed later is a call of its own.
    await asked.page.fill("#projection-query", SEARCH);
    await placed(asked.page, SEARCH);
    await asked.page.waitForTimeout(700);
    assert.equal(asked.calls(), 2);
    assert.deepEqual(asked.errors, []);
  } finally {
    await browser.close();
  }
});

test("a query typed just before Draw rides on that call and is not asked twice", async () => {
  const browser = await chromium.launch();
  try {
    const { page, calls, errors } = await watch(browser);
    await ready(page);
    await analyze(page);
    await page.waitForSelector(`${OUT} .matrix canvas`);
    assert.equal(calls(), 1);
    // Typed, and Draw pressed before the typing's timer runs out.
    await page.fill("#projection-query", BANK);
    await page.click("#projection-run");
    await placed(page, BANK);
    // Long enough for a timer that was left running to ask.
    await page.waitForTimeout(900);
    assert.equal(calls(), 2, "the typed query was asked once, by Draw");
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});

test("a refused query leaves the map and the matrix standing and says why", async () => {
  const browser = await chromium.launch();
  try {
    // Longer than the server places: the whole call is refused, and asked again without the query.
    const long = "interest rates and central banks ".repeat(300);
    const { page, calls, errors } = await watch(browser);
    await ready(page, long);
    await analyze(page);
    await page.waitForSelector(`${OUT} .matrix canvas`);
    await page.waitForSelector("#projection-query-out .error");
    await page.waitForTimeout(700);
    assert.equal(calls(), 2, "the refused call, and the one without the query");
    assert.equal(await page.locator(`${OUT} svg .query`).count(), 0);
    assert.equal(await page.locator(`${OUT} svg .dot .glow`).count(), 0);
    const why = (await page.locator("#projection-query-out").textContent()) ?? "";
    assert.equal(why.includes("query bytes"), true, why);
    assert.equal((await page.locator(`${OUT} svg .dot`).count()) > 3, true);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});

test("an answer to an older drawing that arrives late is not drawn", async () => {
  const browser = await chromium.launch();
  try {
    const { page, context, errors } = await watch(browser);
    const hold = await Hold.of(context, CALL);
    await ready(page, BANK);
    // The first drawing's answer, with the first query placed, is held back.
    hold.start();
    await page.click("#analyze-run");
    await hold.held(page, 1);
    assert.equal(await page.locator(`${OUT} svg .dot`).count(), 0);

    // The document is analyzed again with another query in the box, and that drawing is answered.
    await page.fill("#projection-query", SEARCH);
    await page.click("#analyze-run");
    await placed(page, SEARCH);
    await page.waitForSelector(`${OUT} .matrix canvas`);
    const closest = await page.locator("#projection-closest li").allTextContents();
    const mark = await page.locator(`${OUT} svg .query path`).getAttribute("d");

    // The first answer arrives now, the older of the two, and changes nothing.
    await hold.release(page);
    assert.equal(await page.locator(`${OUT} svg .query title`).textContent(), SEARCH);
    assert.deepEqual(await page.locator("#projection-closest li").allTextContents(), closest);
    assert.equal(await page.locator(`${OUT} svg .query path`).getAttribute("d"), mark);
    assert.equal(await page.locator(`${OUT} .matrix canvas`).count(), 1);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});
