// An answer that arrives after a newer call was made is not drawn
// (#2325). The page's calls reach a real server through a real demo and
// the answers are the server's own: a held call is asked at once and its
// answer is handed to the page only on release, so by then it is the
// older of two. Nothing is stubbed.

import assert from "node:assert/strict";
import { test } from "node:test";
import { chromium } from "playwright";
import { Hold } from "./hold.js";
import { Stack } from "./stack.js";

const BASE = process.env.TURBO_E2E_URL ?? "http://127.0.0.1:8080";
const DEVICE = process.env.TURBO_E2E_DEVICE ?? "0";
const BERT = `bundle=testdata/tiny-bert-bundle,device=${DEVICE},sessions=1`;

test("a read of what is served, answered after an unload, does not bring the unloaded model back", async () => {
  const stack = await Stack.start({
    admin: true,
    allow: ["/turbo.result.v1.AnalysisService/UnloadBundle"],
    models: [BERT, `name=tiny-bert-spare,${BERT}`],
  });
  const browser = await chromium.launch();
  try {
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const hold = await Hold.of(context, "AnalysisService/Capabilities");
    const page = await context.newPage();
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`${stack.url}/#server`);
    await page.waitForSelector("#server.status.up");
    const row = page.locator("#server-caps table").first().locator("tr", { hasText: "tiny-bert-spare" });
    await row.waitFor();

    // Setting the token (here: none) reads what is served again; that read is held.
    hold.start();
    await page.locator("#token").dispatchEvent("change");
    await hold.held(page, 1);
    await row.locator(".confirm button", { hasText: "Unload" }).click();
    await row.locator(".confirm button", { hasText: "Yes" }).click();
    await page.waitForSelector("#server-load-note:has-text('Unloaded tiny-bert-spare.')");
    await row.waitFor({ state: "detached" });

    await hold.release(page);
    assert.equal(await row.count(), 0);
    assert.deepEqual(await page.locator("#server-caps table").first().locator("td:first-child").allTextContents(), ["tiny-bert-bundle"]);
    assert.equal((await page.locator("#analyze-model option").allTextContents()).includes("tiny-bert-spare"), false);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
    await stack.close();
  }
});

test("the document window of an earlier layer pick, answered after a later pick, is not drawn", async () => {
  const browser = await chromium.launch();
  try {
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const hold = await Hold.of(context, "AnalysisService/InspectDocumentWindow");
    const page = await context.newPage();
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`${BASE}/#analyze`);
    await page.waitForSelector("#analyze-profile option", { state: "attached" });
    await page.click("#analyze-run");
    await page.waitForSelector("#analyze-window .doc");
    const chip = (name: string) => page.locator("#analyze-window .chips button").filter({ hasText: name }).first();
    const legend = () => page.locator("#analyze-window .legend .swatch").count();

    // The first pick asks for the window and its heat map; both answers are held.
    hold.start();
    await chip("turbo:tokens").click();
    await hold.held(page, 2);
    await chip("turbo:sentences").click();
    await page.waitForFunction(() => document.querySelectorAll("#analyze-window .chips button.on").length === 2);
    await page.waitForSelector("#analyze-window .doc mark.under");
    assert.equal(await legend(), 2);

    await hold.release(page);
    assert.equal(await page.locator("#analyze-window .chips button.on").count(), 2);
    assert.equal(await legend(), 2);
    // Text under both layers is still marked as such: the tokens-only answer did not replace it.
    assert.equal((await page.locator("#analyze-window .doc mark.under").count()) > 0, true);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});
