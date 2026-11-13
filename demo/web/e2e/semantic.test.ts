// The Semantic and Server tabs in a real browser against the real servers
// that scripts/e2e-web.sh starts: nearest neighbours and the passage
// matrix from a served embedding model, and the bundles and profiles the
// server says it serves. Nothing is stubbed.

import assert from "node:assert/strict";
import { test } from "node:test";
import { chromium, type Page } from "playwright";

const BASE = process.env.TURBO_E2E_URL ?? "http://127.0.0.1:8080";
// The models scripts/e2e-web.sh serves.
const SERVED = ["tiny-bert-bundle", "tiny-bert-fastest", "tiny-bert-exact", "tiny-modernbert-bundle"];

/** The cells of the first table under `selector`, one array of texts per data row. */
function rows(page: Page, selector: string): Promise<string[][]> {
  return page.locator(selector).first().evaluate((t) =>
    [...t.querySelectorAll("tr")].map((r) => [...r.querySelectorAll("td")].map((c) => (c.textContent ?? "").trim())).filter((c) => c.length > 0),
  );
}

test("semantic ranks the passages by cosine and fills a symmetric matrix", async () => {
  const browser = await chromium.launch();
  try {
    const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`${BASE}/#semantic`);
    await page.waitForSelector("#sem-embed option[value=tiny-bert-bundle]", { state: "attached" });
    const passages = (await page.inputValue("#sem-passages")).split("\n").map((p) => p.trim()).filter(Boolean);
    assert.equal(passages.length > 3, true);

    for (const model of ["tiny-bert-bundle", "tiny-modernbert-bundle"]) {
      for (const k of [1, 3]) {
        await page.selectOption("#sem-embed", model);
        await page.fill("#sem-k", String(k));
        await page.locator("#sem-sim").evaluate((e) => e.replaceChildren());
        await page.click("#sem-run");
        await page.waitForSelector("#sem-sim table");
        assert.equal(await page.textContent("#sem-sim h3"), `Nearest neighbours (${model})`);

        const near = await rows(page, "#sem-sim table");
        assert.equal(near.length, k, `${model} k=${k}`);
        assert.deepEqual(near.map((r) => r[0]), near.map((_, i) => String(i + 1)));
        const cosines = near.map((r) => Number(r[2]));
        for (let i = 1; i < cosines.length; i++) assert.equal(cosines[i] <= cosines[i - 1], true, `${model} ranks by cosine`);
        // Every neighbour is a distinct passage named by its number.
        const picked = near.map((r) => Number(r[1].match(/^#(\d+) /)?.[1]));
        assert.equal(new Set(picked).size, k);
        for (const n of picked) assert.equal(n >= 1 && n <= passages.length, true);

        const matrix = (await rows(page, "#sem-sim table.sim")).map((r) => r.slice(1).map(Number));
        assert.equal(matrix.length, passages.length);
        for (let i = 0; i < matrix.length; i++) {
          assert.equal(matrix[i].length, passages.length);
          assert.equal(matrix[i][i], 1);
          for (let j = 0; j < i; j++) assert.equal(matrix[i][j], matrix[j][i], `${model} [${i}][${j}]`);
        }
        // The ranked cosines are the query's, not a row of the matrix, but lie in the same range.
        for (const c of cosines) assert.equal(c >= -1 && c <= 1, true);
      }
    }
    assert.equal(await page.isVisible("#sem-note.error"), false);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});

test("the server tab lists every served bundle and every analysis profile", async () => {
  const browser = await chromium.launch();
  try {
    const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
    await page.goto(`${BASE}/#server`);
    await page.waitForSelector("#server-caps table");
    await page.waitForSelector("#server.status.up");

    const bundles = await rows(page, "#server-caps table");
    assert.deepEqual(bundles.map((r) => r[0]).sort(), [...SERVED].sort());
    assert.deepEqual(bundles.filter((r) => r[1] !== "embed"), []);
    // The two named variants serve the same model as the bundle they load.
    const model = (name: string) => bundles.find((r) => r[0] === name)![2];
    assert.equal(model("tiny-bert-fastest"), model("tiny-bert-bundle"));
    assert.equal(model("tiny-bert-exact"), model("tiny-bert-bundle"));
    assert.notEqual(model("tiny-modernbert-bundle"), model("tiny-bert-bundle"));

    // Each profile the server lists is offered on Analyze, and each embedding model on Embed.
    const profiles = (await rows(page, "#server-caps table:nth-of-type(2)")).map((r) => `${r[0]} v${r[1]}`);
    assert.equal(profiles.length > 0, true);
    await page.click("nav.tabs button[data-tab=analyze]");
    const offered = await page.locator("#analyze-profile option").allTextContents();
    assert.deepEqual(offered, ["default", ...profiles]);
    await page.click("nav.tabs button[data-tab=embed]");
    // Each model is listed once its metadata has been read.
    await page.waitForFunction((n) => document.querySelectorAll("#embed-models label.model").length === n, SERVED.length);
    const embed = await page.locator("#embed-models label.model").allTextContents();
    for (const name of SERVED) assert.equal(embed.some((t) => t.includes(name)), true, `Embed offers ${name}`);
  } finally {
    await browser.close();
  }
});
