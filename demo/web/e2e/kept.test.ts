// Kept picks, in a real browser against real servers: a model picked on
// Semantic is back after a reload, and an index the page picks itself
// (the one just created) stays picked when the index list is read again.
// The writes run through a demo of this file's own (stack.ts) that
// forwards them; nothing is stubbed.

import assert from "node:assert/strict";
import { test } from "node:test";
import { chromium, type Browser, type Page } from "playwright";
import { Stack } from "./stack.js";

const BASE = process.env.TURBO_E2E_URL ?? "http://127.0.0.1:8080";

async function opened(browser: Browser, url: string, tab: string): Promise<Page> {
  const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
  await page.goto(`${url}/#${tab}`);
  await page.waitForSelector("#server.status.up");
  return page;
}

const picked = (page: Page, id: string): Promise<string> => page.locator(`#${id}`).evaluate((s) => (s as HTMLSelectElement).value);

test("a model picked on Semantic is back after a reload, and Reset picks the first again", async () => {
  const browser = await chromium.launch();
  try {
    const page = await opened(browser, BASE, "semantic");
    await page.waitForSelector("#sem-embed option[value=tiny-modernbert-bundle]", { state: "attached" });
    const first = await picked(page, "sem-embed");
    assert.notEqual(first, "tiny-modernbert-bundle");
    await page.selectOption("#sem-embed", "tiny-modernbert-bundle");

    await page.reload();
    await page.waitForSelector("#server.status.up");
    await page.waitForSelector("#sem-embed option[value=tiny-modernbert-bundle]", { state: "attached" });
    assert.equal(await picked(page, "sem-embed"), "tiny-modernbert-bundle");

    await page.click("#semantic-reset");
    assert.equal(await picked(page, "sem-embed"), first);
    await page.reload();
    await page.waitForSelector("#server.status.up");
    await page.waitForSelector("#sem-embed option[value=tiny-modernbert-bundle]", { state: "attached" });
    assert.equal(await picked(page, "sem-embed"), first);
  } finally {
    await browser.close();
  }
});

test("a new index stays picked after an upload reads the list again, and after a reload", async () => {
  const stack = await Stack.start({ allow: ["/turbo.search.v1.TurboSearchService/CreateIndex", "/turbo.search.v1.TurboSearchService/IndexDocuments"] });
  const browser = await chromium.launch();
  try {
    const page = await opened(browser, stack.url, "search");
    await page.waitForSelector("#search-create:not(.off)");
    await page.waitForSelector("#search-create-bundle option[value=tiny-bert-bundle]", { state: "attached" });
    for (const name of ["older", "newer"]) {
      await page.fill("#search-create-name", name);
      await page.click("#search-create button[type=submit]");
      await page.waitForSelector(`#search-create-note:has-text('Created ${name}.')`);
    }
    // Picked by hand, so it is the kept pick.
    await page.selectOption("#search-index", "older");
    // A pick that fires no change event, as a page script makes one, is not
    // kept: the shown pick and the kept pick differ. Reading the list again
    // keeps the shown one, since the kept pick goes back only once a load.
    await page.locator("#search-index").evaluate((s) => ((s as HTMLSelectElement).value = "newer"));
    await page.click("#search-refresh");
    await page.waitForSelector("#search-refresh:not([aria-busy])");
    assert.equal(await picked(page, "search-index"), "newer");
    // Then the page picks the next new one.
    await page.fill("#search-create-name", "newest");
    await page.click("#search-create button[type=submit]");
    await page.waitForSelector("#search-create-note:has-text('Created newest.')");
    await page.waitForFunction(() => (document.querySelector("#search-index") as HTMLSelectElement).value === "newest");

    // The upload goes to the picked index and reads the list again after it.
    await page.click("#search-add");
    await page.waitForSelector("#search-add-note:has-text('Indexed')");
    await page.waitForSelector("#search-add:not([aria-busy])");
    assert.equal(await picked(page, "search-index"), "newest");
    const docs = await page.locator("#search-indexes tr", { hasText: "newest" }).locator("td").nth(5).textContent();
    assert.notEqual(docs, "0");

    await page.reload();
    await page.waitForSelector("#server.status.up");
    await page.waitForSelector("#search-index option[value=newest]", { state: "attached" });
    assert.equal(await picked(page, "search-index"), "newest");
  } finally {
    await browser.close();
    await stack.close();
  }
});
