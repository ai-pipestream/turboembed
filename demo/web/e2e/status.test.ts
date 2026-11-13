// The header and the tabs follow the server: a server of this file's own
// (stack.ts) is stopped and started under an open page, and a page is
// opened while the server is stopped. Nothing is stubbed.

import assert from "node:assert/strict";
import { test } from "node:test";
import { chromium, type Page, type Route } from "playwright";
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
    await page.waitForSelector("#server-caps table");
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

test("a late Capabilities answer from an earlier refresh does not overwrite a newer one", async () => {
  const token = "status-stale-order-token";
  const bert = `bundle=testdata/tiny-bert-bundle,device=${process.env.TURBO_E2E_DEVICE ?? "0"},sessions=1`;
  const stack = await Stack.start({
    models: [bert, `name=tiny-bert-spare,${bert}`],
    admin: true,
    allow: ["/turbo.result.v1.AnalysisService/UnloadBundle"],
    token,
  });
  const browser = await chromium.launch();
  let held: Route | undefined;
  try {
    const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
    await page.goto(`${stack.url}/#server`);
    await page.waitForSelector("#server:has-text('token')");
    await page.waitForSelector("#server-caps p.error");

    let releaseHeld!: () => void;
    const firstHeld = new Promise<void>((r) => (releaseHeld = r));
    await page.route("**/turbo.result.v1.AnalysisService/Capabilities", async (route) => {
      if (!held) {
        held = route;
        releaseHeld();
        return;
      }
      await route.continue();
    });

    const setToken = (v: string) =>
      page.locator("#token").evaluate((el: HTMLInputElement, val) => {
        el.value = val;
        el.dispatchEvent(new Event("change"));
      }, v);

    // Older refresh with a wrong token, held before it reaches the demo.
    await setToken("wrong-token");
    await firstHeld;

    // Newer refresh with the real token finishes first and fills the tabs.
    await setToken(token);
    await page.waitForSelector("#server.status.up");
    await page.waitForSelector("#server-caps table");
    assert.deepEqual(await page.locator("#server-caps table").first().locator("td:first-child").allTextContents(), ["tiny-bert-bundle", "tiny-bert-spare"]);

    // Now the older refresh reaches the demo and fails 401; the tabs keep the newer answer.
    const late = page.waitForResponse((r) => r.url().endsWith("/Capabilities") && r.status() === 401);
    await held!.continue();
    held = undefined;
    await late;
    await page.evaluate(() => new Promise((r) => setTimeout(r, 50)));
    assert.equal(await page.locator("#server-caps p.error").count(), 0);
    assert.deepEqual(await page.locator("#server-caps table").first().locator("td:first-child").allTextContents(), ["tiny-bert-bundle", "tiny-bert-spare"]);

    // Hold a 200 Capabilities answer that still lists both bundles while tiny-bert-spare is unloaded.
    await page.unroute("**/turbo.result.v1.AnalysisService/Capabilities");
    let capturedTwo = false;
    let finishOlderTwoBundles!: () => Promise<void>;
    let twoBundlesFetched!: () => void;
    const fetchedTwo = new Promise<void>((r) => (twoBundlesFetched = r));
    await page.route("**/turbo.result.v1.AnalysisService/Capabilities", async (route) => {
      if (!capturedTwo) {
        capturedTwo = true;
        const response = await route.fetch();
        finishOlderTwoBundles = () => route.fulfill({ response });
        twoBundlesFetched();
        return;
      }
      await route.continue();
    });
    await setToken(token);
    await fetchedTwo;

    // Unload tiny-bert-spare: its Capabilities refresh lists only tiny-bert-bundle.
    const spare = page.locator("#server-caps tr", { hasText: "tiny-bert-spare" });
    await spare.locator(".confirm button", { hasText: "Unload" }).click();
    await spare.locator(".confirm button", { hasText: "Yes" }).click();
    await page.waitForSelector("#server-load-note:has-text('Unloaded tiny-bert-spare.')");
    assert.deepEqual(await page.locator("#server-caps table").first().locator("td:first-child").allTextContents(), ["tiny-bert-bundle"]);

    // Deliver the earlier two-bundle answer late: Server, Semantic and Analyze keep the one-bundle answer.
    const lateTwo = page.waitForResponse((r) => r.url().endsWith("/Capabilities") && r.status() === 200);
    await finishOlderTwoBundles();
    await lateTwo;
    await page.evaluate(() => new Promise((r) => setTimeout(r, 50)));
    assert.deepEqual(await page.locator("#server-caps table").first().locator("td:first-child").allTextContents(), ["tiny-bert-bundle"]);
    assert.deepEqual(await choices(page, "#sem-embed"), ["none", "tiny-bert-bundle"]);
    assert.deepEqual(await choices(page, "#analyze-model"), ["none", "tiny-bert-bundle"]);
    await page.click("nav.tabs button[data-tab=embed]");
    await page.waitForSelector("#embed-models label.model");
    assert.equal(await page.locator("#embed-models .empty").count(), 0);
  } finally {
    if (held) await held.abort();
    await browser.close();
    await stack.close();
  }
});
