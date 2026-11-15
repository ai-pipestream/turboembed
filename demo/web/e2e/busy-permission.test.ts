// Pause and resume actual requests across a real token/Describe transition.
// The demo and server produce every response; none is intercepted or replaced.
import assert from "node:assert/strict";
import { test } from "node:test";
import { chromium, type Route } from "playwright";
import { Stack } from "./stack.js";

const METHOD = "/turbo.search.v1.TurboSearchService/SetCollection";
const TOKEN = "busy-permission-test-token";

for (const forwarded of [false, true]) {
  test(`collection Set preserves permission and busy state when forwarded=${forwarded}`, async () => {
    const stack = await Stack.start({ admin: true, allow: forwarded ? [METHOD] : [], token: TOKEN });
    const browser = await chromium.launch();
    let held: Route | undefined;
    try {
      const page = await browser.newPage();
      const initial = page.waitForResponse((r) => r.url().endsWith("/Describe"));
      await page.goto(`${stack.url}/#search`);
      assert.equal((await initial).status(), 401);
      const set = page.locator("#coll-set button[type=submit]");
      assert.equal(await set.isDisabled(), false);
      await page.fill("#coll-name", "permission-transition");
      await page.fill("#coll-baseline", "hello world");
      await page.route(`**${METHOD}`, (route) => { held = route; });
      const requested = page.waitForRequest((r) => r.url().endsWith(METHOD));
      await set.click();
      await requested;
      assert.equal(await set.isDisabled(), true);
      assert.equal(await set.getAttribute("aria-busy"), "true");

      const describe = page.waitForResponse((r) => r.url().endsWith("/Describe") && r.status() === 200);
      await page.fill("#token", TOKEN);
      await page.locator("#token").dispatchEvent("change");
      await describe;
      // The list reload follows the permission callback, so this proves it ran.
      await page.locator("#coll-list p", { hasText: "No collection yet" }).waitFor();
      if (process.env.TURBO_E2E_SHOTS) await page.screenshot({ path: `${process.env.TURBO_E2E_SHOTS}/during-${forwarded}.png`, fullPage: true });
      assert.equal(await set.isDisabled(), true, "Describe must not enable an in-flight Set");
      const reason = forwarded ? "" : `Off: the demo does not forward this write. Start it with --allow ${METHOD.slice(1)}`;
      assert.equal(await set.getAttribute("title"), reason);
      assert.equal(await page.locator("#coll-watch").isDisabled(), false, "Watch is a read");

      const completed = page.waitForResponse((r) => r.url().endsWith(METHOD));
      assert.notEqual(held, undefined);
      await held!.continue();
      held = undefined;
      assert.equal((await completed).status(), 401, "the pre-token request reaches the real token gate");
      await page.waitForFunction(() => !document.querySelector("#coll-set button[type=submit]")!.hasAttribute("aria-busy"));
      if (process.env.TURBO_E2E_SHOTS) await page.screenshot({ path: `${process.env.TURBO_E2E_SHOTS}/after-${forwarded}.png`, fullPage: true });
      assert.equal(await set.isDisabled(), !forwarded, "busy completion preserves the current permission");
      assert.equal(await set.getAttribute("title"), reason);

      if (forwarded) {
        await page.unroute(`**${METHOD}`);
        const retry = page.waitForResponse((r) => r.url().endsWith(METHOD));
        await set.click();
        assert.equal((await retry).status(), 200);
        await page.waitForFunction(() => !document.querySelector("#coll-set button[type=submit]")!.hasAttribute("aria-busy"));
        assert.equal(await set.isDisabled(), false, "an allowed action recovers after success");
        await page.click("#coll-watch");
        await page.waitForSelector("#coll-watch[aria-busy=true]");
        assert.equal(await page.locator("#coll-watch").isDisabled(), true);
        assert.equal(await page.locator("#coll-stop").isDisabled(), false);
        await page.click("#coll-stop");
        await page.waitForFunction(() => !document.querySelector("#coll-watch")!.hasAttribute("aria-busy"));
        assert.equal(await page.locator("#coll-watch").isDisabled(), false);
        assert.equal(await page.locator("#coll-stop").isDisabled(), true);
      }
    } finally {
      if (held) await held.abort();
      await browser.close();
      await stack.close();
    }
  });
}
