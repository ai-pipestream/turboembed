// The similarity matrix and the query spotlight in Analyze: the pairwise
// cosines and the placed query of ProjectChunks. A real server and a
// real demo (the shared ones of scripts/e2e-web.sh); nothing is stubbed.

import assert from "node:assert/strict";
import { test } from "node:test";
import { chromium, type Page } from "playwright";

const BASE = process.env.TURBO_E2E_URL ?? "http://127.0.0.1:8080";
const OUT = "#projection-out";

// Four paragraphs on three subjects, one with a name outside ASCII.
const TEXT = [
  "Ada Lovelace wrote the first published program in 1843. Her notes on the Analytical Engine described how the machine could compute Bernoulli numbers. Charles Babbage designed the engine in London, but the engine was never built. Lovelace saw that the machine could work with symbols as well as numbers.",
  "A century later, Grace Hopper wrote one of the first compilers for the UNIVAC. Her compiler translated mathematical notation into machine code. Hopper argued that programs should be written in words people could read. That idea led to COBOL, and banks still run COBOL programs today.",
  "Banks process billions of transactions on that old code every day. When the central bank raises interest rates, those bank systems recalculate millions of loans overnight. Higher interest rates push borrowing costs up for households and firms. Inflation this quarter kept rates high for longer than expected.",
  "Today, search engines in Zürich turn words into vectors so a machine can compare meaning. A query and a passage that share no words can still sit close together in that vector space. Each vector is a list of numbers, and similar meanings point in similar directions. Lovelace's idea that a machine could work with symbols is the root of that design.",
].join("\n\n");

async function open(page: Page): Promise<number> {
  await page.goto(`${BASE}/#analyze`);
  await page.waitForSelector("#analyze-profile option", { state: "attached" });
  await page.fill("#analyze-text", TEXT);
  await page.selectOption("#analyze-profile", "default-v1");
  await page.click("#analyze-run");
  await page.waitForSelector(`${OUT} svg .dot`);
  await page.waitForSelector(`${OUT} .matrix canvas`);
  return page.locator(`${OUT} svg .dot`).count();
}

/** The colour of the matrix cell at (`row`, `col`), read off the canvas. */
function shade(page: Page, row: number, col: number, n: number): Promise<string> {
  return page.evaluate(
    ([out, row, col, n]) => {
      const c = document.querySelector(`${out} .matrix canvas`) as HTMLCanvasElement;
      const px = c.width / (n as number);
      const at = (v: number): number => Math.floor(((v as number) + 0.5) * px);
      return Array.from(c.getContext("2d")!.getImageData(at(col as number), at(row as number), 1, 1).data).join(",");
    },
    [OUT, row, col, n] as const,
  );
}

test("the similarity matrix shades a cell per pair of chunks and says what is pointed at", async () => {
  const browser = await chromium.launch();
  try {
    const page = await (await browser.newContext({ viewport: { width: 1400, height: 1000 } })).newPage();
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    const n = await open(page);
    assert.equal(n > 3, true, `${n} chunks`);

    // One row per chunk; the cells are drawn, symmetric, and not of one shade.
    const label = (await page.locator(`${OUT} .matrix canvas`).getAttribute("aria-label")) ?? "";
    assert.equal(label.startsWith(`Similarity matrix of ${n} chunks: cosines from `), true, label);
    const self = await shade(page, 0, 0, n);
    assert.equal(self.endsWith(",255"), true, `the diagonal is painted: ${self}`);
    const shades = new Set<string>();
    for (let a = 0; a < n; a++) {
      assert.equal(await shade(page, a, a, n), self, `chunk ${a + 1} with itself`);
      for (let b = a + 1; b < n; b++) {
        const cell = await shade(page, a, b, n);
        assert.equal(await shade(page, b, a, n), cell, `cells ${a},${b} and ${b},${a}`);
        shades.add(cell);
      }
    }
    assert.equal(shades.size > 2, true, `${shades.size} shades among the pairs`);

    // Pointing at a cell names its two chunks and their cosine, and lights the row's chunk in the text.
    const plot = page.locator(`${OUT} .matrix .plot`);
    const info = page.locator(`${OUT} .matrix .info`);
    const rest = (await info.textContent()) ?? "";
    assert.equal(rest.startsWith("Point at a cell"), true, rest);
    await plot.scrollIntoViewIfNeeded();
    const box = (await plot.boundingBox())!;
    await page.mouse.move(box.x + (box.width * 2.5) / n, box.y + (box.height * 1.5) / n);
    await page.waitForFunction((out) => (document.querySelector(`${out} .matrix .info`)?.textContent ?? "").startsWith("chunk 2 and chunk 3: cosine "), OUT);
    const said = (await info.textContent()) ?? "";
    const cosine = Number(said.slice("chunk 2 and chunk 3: cosine ".length).split(". ")[0] + "");
    assert.equal(Number.isFinite(cosine) && cosine <= 1 && cosine >= -1, true, said);
    assert.deepEqual(await page.locator(`${OUT} .chunked .ck.lit`).evaluateAll((all) => [...new Set(all.map((e) => (e as HTMLElement).dataset.chunk))]), ["1"]);
    assert.equal(await page.locator(`${OUT} .matrix .cross-on`).getAttribute("visibility"), "visible");
    // Leaving puts the prompt back and takes the light off.
    await page.mouse.move(2, 2);
    await page.waitForFunction(([out, rest]) => document.querySelector(`${out} .matrix .info`)?.textContent === rest, [OUT, rest] as const);
    assert.equal(await page.locator(`${OUT} .chunked .ck.lit`).count(), 0);

    // The arrow keys walk the cells: the first press lands on chunks 1 and 2.
    await plot.focus();
    await page.keyboard.press("ArrowRight");
    assert.equal(((await info.textContent()) ?? "").startsWith("chunk 1 and chunk 2: cosine "), true);
    await page.keyboard.press("ArrowDown");
    assert.equal(((await info.textContent()) ?? "").startsWith("chunk 2 with itself"), true);

    // A chunk pointed at in the text shows as its row and column.
    await page.mouse.move(2, 2);
    await plot.blur();
    await page.locator(`${OUT} .chunked .ck[data-chunk="2"]`).first().hover();
    await page.waitForFunction((out) => document.querySelector(`${out} .matrix .cross-on`)?.getAttribute("visibility") === "visible", OUT);
    assert.equal(await page.locator(`${OUT} .matrix .cross-cell`).getAttribute("x"), "2");

    // The pairs as rows to copy or download: n(n-1)/2 of them.
    assert.equal(await page.locator(`${OUT} [data-take="Similarity matrix"] button`).count(), 2);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});

test("a query is placed on the map, lights the chunks by cosine and lists the closest", async () => {
  const browser = await chromium.launch();
  try {
    const page = await (await browser.newContext({ viewport: { width: 1400, height: 1000 } })).newPage();
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    const n = await open(page);
    assert.equal(await page.locator(`${OUT} svg .query`).count(), 0);
    assert.equal(await page.locator(`${OUT} svg .dot .glow`).count(), 0);

    const placed = (q: string): Promise<unknown> => page.waitForFunction(([out, q]) => document.querySelector(`${out} svg .query title`)?.textContent === q, [OUT, q] as const);
    const listed = async (): Promise<[number, number][]> =>
      (await page.locator("#projection-closest li").allTextContents()).map((t) => {
        const words = t.split(" ");
        return [Number(words[1]), Number(words[2])];
      });

    const first = "what happens when the central bank raises interest rates";
    await page.fill("#projection-query", first);
    await page.click("#projection-ask");
    await placed(first);
    assert.equal(((await page.locator("#projection-query-out").textContent()) ?? "").startsWith("Embedded by tiny-bert"), true);
    const mark = await page.locator(`${OUT} svg .query path`).getAttribute("d");
    // Every chunk glows, and not all alike.
    const glows = await page.locator(`${OUT} svg .dot .glow`).evaluateAll((all) => all.map((g) => Number(g.getAttribute("r"))));
    assert.equal(glows.length, n);
    assert.equal(new Set(glows).size > 1, true, glows.join(" "));
    // The closest are listed greatest first, at most half the chunks, and marked in the text.
    const want = Math.min(5, Math.max(1, Math.floor(n / 2)));
    const list = await listed();
    assert.equal(list.length, want);
    assert.equal(list.every((row, i) => row[1] >= -1 && row[1] <= 1 && (i === 0 || row[1] <= list[i - 1][1])), true, JSON.stringify(list));
    assert.equal(glows.indexOf(Math.max(...glows)) + 1, list[0][0], "the strongest glow is the first of the list");
    const near = await page.locator(`${OUT} .chunked .ck.near`).evaluateAll((all) => [...new Set(all.map((e) => Number((e as HTMLElement).dataset.chunk) + 1))].sort((a, b) => a - b));
    assert.deepEqual(near, list.map((row) => row[0]).sort((a, b) => a - b));
    assert.equal(await page.locator(`${OUT} svg .ray`).count(), Math.min(3, Math.max(1, Math.floor(n / 2))));
    assert.equal(((await page.locator(`${OUT} svg .dot`).first().getAttribute("aria-label")) ?? "").includes("cosine to the query "), true);
    // A row of the list lights its chunk on the map and in the text.
    await page.locator("#projection-closest li").first().hover();
    await page.waitForSelector(`${OUT} svg .dot.lit`);
    assert.deepEqual(await page.locator(`${OUT} .chunked .ck.lit`).evaluateAll((all) => [...new Set(all.map((e) => Number((e as HTMLElement).dataset.chunk) + 1))]), [list[0][0]]);
    await page.mouse.move(2, 2);
    // The matrix is still there beside the query.
    assert.equal(await page.locator(`${OUT} .matrix canvas`).count(), 1);

    // Another query, typed and left alone, moves the marker and scores the chunks again.
    const second = "Zürich search engines compare meaning with vectors";
    await page.fill("#projection-query", second);
    await placed(second);
    assert.notEqual(await page.locator(`${OUT} svg .query path`).getAttribute("d"), mark);
    assert.notEqual(JSON.stringify(await listed()), JSON.stringify(list));

    // An empty box takes the query off the map.
    await page.fill("#projection-query", "");
    await page.press("#projection-query", "Enter");
    await page.waitForSelector(`${OUT} svg .query`, { state: "detached" });
    assert.equal(await page.locator(`${OUT} svg .dot .glow`).count(), 0);
    assert.equal(await page.locator("#projection-closest").count(), 0);
    assert.equal(await page.locator(`${OUT} .chunked .ck.near`).count(), 0);
    assert.equal(((await page.locator("#projection-query-out").textContent()) ?? "").trim(), "");
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});
