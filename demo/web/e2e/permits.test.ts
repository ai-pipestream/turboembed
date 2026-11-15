// What the demo forwards reaches the page late when the demo wants a
// token: the first Describe is refused, so nothing is off until the
// token is typed. A real server and a real demo started with a token
// and one --allow (stack.ts); nothing is stubbed.

import assert from "node:assert/strict";
import { test } from "node:test";
import { chromium } from "playwright";
import { Stack } from "./stack.js";

const TOKEN = "a-token-for-this-test";

test("typing the token turns off the writes the demo does not forward", async () => {
  const stack = await Stack.start({ admin: true, allow: ["/turbo.search.v1.TurboSearchService/CreateIndex"], token: TOKEN });
  const browser = await chromium.launch();
  try {
    const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`${stack.url}/#search`);
    // Without the token the demo says nothing about its writes, so none is off yet.
    await page.waitForSelector("#server:has-text('token')");
    const set = page.locator("#coll-set button[type=submit]");
    assert.equal(await set.isDisabled(), false);
    assert.equal(await page.locator("#search-add-box").evaluate((e) => (e as HTMLElement).inert), false);

    await page.fill("#token", TOKEN);
    await page.locator("#token").dispatchEvent("change");
    await page.waitForSelector("#server.status.up");
    // The forwarded write stays live; the others turn off and say which --allow enables them.
    await page.waitForFunction(() => (document.querySelector("#search-add-box") as HTMLElement).inert);
    assert.equal(await page.locator("#search-create").evaluate((e) => (e as HTMLElement).inert), false);
    assert.equal(await page.textContent("#search-add-box-off"), "Off: the demo does not forward this write. Start it with --allow turbo.search.v1.TurboSearchService/IndexDocuments");
    await page.waitForFunction(() => (document.querySelector("#coll-set button[type=submit]") as HTMLButtonElement).disabled);
    assert.equal(await set.getAttribute("title"), "Off: the demo does not forward this write. Start it with --allow turbo.search.v1.TurboSearchService/SetCollection");
    // Watching is a read and stays usable.
    assert.equal(await page.locator("#coll-watch").isDisabled(), false);

    // A row's Delete and Unload are off too.
    // The bundle list is read after the status: Create asks for a bundle until it is filled.
    await page.waitForSelector("#search-create-bundle option[value=tiny-bert-bundle]", { state: "attached" });
    await page.click("#search-create button[type=submit]");
    const row = page.locator("#search-indexes tr", { hasText: "demo" });
    await row.waitFor();
    assert.equal(await row.locator(".confirm button", { hasText: "Delete" }).isDisabled(), true);
    await page.click("nav.tabs button[data-tab=server]");
    const unload = page.locator("#server-caps tr", { hasText: "tiny-bert-bundle" }).locator(".confirm button", { hasText: "Unload" });
    await unload.waitFor();
    assert.equal(await unload.isDisabled(), true);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
    await stack.close();
  }
});
