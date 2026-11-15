// The header and the tabs follow the server: a server of this file's own
// (stack.ts) is stopped and started under an open page, and a page is
// opened while the server is stopped. Nothing is stubbed.

import assert from "node:assert/strict";
import { test } from "node:test";
import { chromium, type Page } from "playwright";
import { Stack } from "./stack.js";

// The page reads the status every 10 s; allow one interval and the call.
const ONE_CHECK = 15_000;
const DOWN = "The server is not reachable, so its embedding models cannot be listed.";

/** The texts of the options of `select`. */
function choices(page: Page, select: string): Promise<string[]> {
  return page.locator(`${select} option`).allTextContents();
}

test("the header turns red when the server stops and green when it is back, with no reload", async () => {
  const stack = await Stack.start();
  const browser = await chromium.launch();
  try {
    const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    let loads = 0;
    page.on("load", () => (loads += 1));
    await page.goto(`${stack.url}/#server`);
    await page.waitForSelector("#server.status.up");
    await page.waitForSelector("#server-caps table");
    assert.equal(await page.isVisible("#down"), false);

    await stack.stopServer();
    await page.waitForSelector("#server.status.down", { timeout: ONE_CHECK });
    assert.equal(await page.textContent("#server"), "server not reachable");
    assert.equal(await page.isVisible("#down"), true);
    assert.equal(await page.isVisible("#down-retry"), true);
    assert.equal(((await page.textContent("#down-why")) ?? "").trim().length > 0, true);

    await stack.startServer();
    await page.waitForSelector("#server.status.up", { timeout: ONE_CHECK });
    assert.match((await page.textContent("#server")) ?? "", /: ready$/);
    await page.waitForSelector("#down", { state: "hidden" });
    // The tabs read what the server serves again.
    await page.waitForSelector("#server-caps td:has-text('tiny-bert-bundle')");
    assert.deepEqual(await page.locator("#server-caps table").first().locator("td:first-child").allTextContents(), ["tiny-bert-bundle"]);
    await page.click("nav.tabs button[data-tab=embed]");
    await page.waitForSelector("#embed-models label.model");
    assert.equal(loads, 1, "the page was not reloaded");
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
    await stack.close();
  }
});

test("a page opened with the server stopped says why each tab is empty, and fills on Retry", async () => {
  const stack = await Stack.start();
  const browser = await chromium.launch();
  try {
    await stack.stopServer();
    const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
    await page.goto(`${stack.url}/#embed`);
    await page.waitForSelector("#down:not([hidden])");
    assert.equal(await page.isVisible("#server.status.down"), true);

    await page.waitForSelector("#embed-models .empty");
    assert.equal(await page.textContent("#embed-models .empty"), DOWN);
    assert.equal(await page.locator("#embed-models label.model").count(), 0);
    for (const id of ["#sem-embed", "#sem-rerank", "#sem-qa", "#search-index"]) {
      // Each list says so once its own read has failed.
      await page.waitForSelector(`${id} option`, { state: "attached" });
      assert.deepEqual(await choices(page, id), ["the server is not reachable"], id);
    }
    await page.click("nav.tabs button[data-tab=search]");
    await page.waitForSelector("#search-indexes p.error");
    assert.equal(await page.textContent("#search-indexes p.error"), "The server is not reachable, so its indexes cannot be listed.");

    // Retry reads the status at once, without waiting for the next interval.
    await stack.startServer();
    await page.click("#down-retry");
    await page.waitForSelector("#server.status.up", { timeout: 5_000 });
    await page.waitForSelector("#down", { state: "hidden" });
    await page.waitForSelector("#search-indexes p.muted");
    assert.equal(await page.textContent("#search-indexes p.muted"), "No index yet: create one from an embedding bundle.");
    assert.deepEqual(await choices(page, "#search-index"), ["no index yet"]);
    // What the server serves is read after the status.
    await page.waitForSelector("#sem-embed option[value=tiny-bert-bundle]", { state: "attached" });
    assert.deepEqual(await choices(page, "#sem-embed"), ["none", "tiny-bert-bundle"]);
    // Tasks the server does not serve now say so, rather than blaming the server.
    assert.deepEqual(await choices(page, "#sem-rerank"), ["no reranker is served"]);
    await page.click("nav.tabs button[data-tab=embed]");
    await page.waitForSelector("#embed-models label.model");
    assert.equal(await page.locator("#embed-models .empty").count(), 0);
  } finally {
    await browser.close();
    await stack.close();
  }
});
