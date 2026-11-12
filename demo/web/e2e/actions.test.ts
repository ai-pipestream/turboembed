// How the page's controls behave, in a real browser against real
// servers: tab addresses and history, one busy state for every run
// button, and a confirm step before anything that cannot be undone.
// The deletes run against a server and demo of this file's own
// (stack.ts) that forward them; nothing is stubbed.

import assert from "node:assert/strict";
import { test } from "node:test";
import { chromium, type Page } from "playwright";
import { Stack } from "./stack.js";

const BASE = process.env.TURBO_E2E_URL ?? "http://127.0.0.1:8080";
const DEVICE = process.env.TURBO_E2E_DEVICE ?? "0";
const BERT = `bundle=testdata/tiny-bert-bundle,device=${DEVICE},sessions=1`;

/** The tab now shown: its button's name and the one panel not hidden. */
async function shown(page: Page): Promise<[string, string[]]> {
  const selected = (await page.getAttribute("nav.tabs button[aria-selected=true]", "data-tab")) ?? "";
  const panels = await page.locator("section.panel:not([hidden])").evaluateAll((ps) => ps.map((p) => p.id));
  return [selected, panels];
}

test("tab addresses and Back and Forward switch tabs", async () => {
  const browser = await chromium.launch();
  try {
    const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
    await page.goto(`${BASE}/#embed`);
    await page.waitForSelector("#embed-models label.model");
    assert.deepEqual(await shown(page), ["embed", ["tab-embed"]]);

    // An address change on the open page opens the tab it names.
    await page.evaluate(() => (location.hash = "#search"));
    await page.waitForSelector("#tab-search:not([hidden])");
    assert.deepEqual(await shown(page), ["search", ["tab-search"]]);
    // So does a link on the page.
    await page.evaluate(() => {
      const a = document.createElement("a");
      a.href = "#vocabulary";
      document.body.append(a);
      a.click();
      a.remove();
    });
    await page.waitForSelector("#tab-vocabulary:not([hidden])");
    assert.deepEqual(await shown(page), ["vocabulary", ["tab-vocabulary"]]);

    // Each tab click is a history entry.
    await page.click("nav.tabs button[data-tab=models]");
    assert.equal(new URL(page.url()).hash, "#models");
    assert.deepEqual(await shown(page), ["models", ["tab-models"]]);
    await page.goBack();
    await page.waitForSelector("#tab-vocabulary:not([hidden])");
    assert.equal(new URL(page.url()).hash, "#vocabulary");
    await page.goBack();
    await page.waitForSelector("#tab-search:not([hidden])");
    assert.deepEqual(await shown(page), ["search", ["tab-search"]]);
    await page.goForward();
    await page.goForward();
    await page.waitForSelector("#tab-models:not([hidden])");
    assert.deepEqual(await shown(page), ["models", ["tab-models"]]);

    // Clicking the tab already open adds no entry: Back leaves it.
    await page.click("nav.tabs button[data-tab=models]");
    await page.goBack();
    await page.waitForSelector("#tab-vocabulary:not([hidden])");

    // An address that names no tab opens the first one.
    await page.goto(`${BASE}/#nothing-here`);
    await page.waitForSelector("#tab-embed:not([hidden])");
    assert.deepEqual(await shown(page), ["embed", ["tab-embed"]]);
  } finally {
    await browser.close();
  }
});

interface Seen {
  disabled: boolean;
  busy: boolean;
  label: string;
  stopOn: boolean | null;
}

/**
 * Clicks `button` and records the state of it and of `stop` each time the
 * button changes while its call runs, then waits for `done` and returns
 * the records with the state after the call.
 */
async function watchRun(page: Page, button: string, done: () => Promise<unknown>, stop?: string): Promise<{ during: Seen[]; after: Seen }> {
  await page.evaluate(
    ([b, s]) => {
      const button = document.querySelector(b) as HTMLButtonElement;
      const stop = s ? (document.querySelector(s) as HTMLButtonElement) : null;
      const w = window as unknown as { seen: unknown[]; watcher?: MutationObserver };
      w.seen = [];
      w.watcher?.disconnect();
      w.watcher = new MutationObserver(() => {
        if (button.getAttribute("aria-busy") !== "true") return;
        w.seen.push({ disabled: button.disabled, busy: true, label: button.textContent ?? "", stopOn: stop ? !stop.disabled : null });
      });
      w.watcher.observe(button, { attributes: true, childList: true, subtree: true, characterData: true });
    },
    [button, stop ?? ""],
  );
  await page.click(button);
  await done();
  await page.waitForSelector(`${button}:not([aria-busy])`);
  return page.evaluate(
    ([b, s]) => {
      const w = window as unknown as { seen: Seen[]; watcher?: MutationObserver };
      w.watcher?.disconnect();
      const button = document.querySelector(b) as HTMLButtonElement;
      const stop = s ? (document.querySelector(s) as HTMLButtonElement) : null;
      return { during: w.seen, after: { disabled: button.disabled, busy: button.hasAttribute("aria-busy"), label: button.textContent ?? "", stopOn: stop ? !stop.disabled : null } };
    },
    [button, stop ?? ""],
  );
}

test("every run button is off and counts while its call runs, and Stop is on only then", async () => {
  const browser = await chromium.launch();
  try {
    const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`${BASE}/#embed`);
    await page.waitForSelector("#embed-models label.model");
    // Every Stop starts off.
    for (const stop of ["#analyze-stop", "#coll-stop", "#vocab-stop", "#train-stop"]) assert.equal(await page.isDisabled(stop), true, stop);

    const runs: [string, string, string, () => Promise<unknown>, string?][] = [
      ["embed", "#embed-run", "Embed", () => page.waitForSelector("#embed-results .card table.sim")],
      ["semantic", "#sem-run", "Run", () => page.waitForSelector("#sem-sim table")],
      ["analyze", "#analyze-run", "Analyze", () => page.waitForSelector("#analyze-window .doc"), "#analyze-stop"],
      ["analyze", "#analyze-export", "Format", () => page.waitForSelector("#analyze-export-out button")],
      ["analyze", "#wire-roundtrip", "Round trip on the server", () => page.waitForSelector("#wire-note:not([hidden])")],
    ];
    for (const [tab, button, label, done, stop] of runs) {
      await page.click(`nav.tabs button[data-tab=${tab}]`);
      if (button === "#wire-roundtrip") await page.click("#analyze-wire summary");
      const { during, after } = await watchRun(page, button, done, stop);
      assert.equal(during.length > 0, true, `${button} showed a busy state`);
      for (const s of during) {
        assert.equal(s.disabled, true, `${button} is off while it runs`);
        assert.match(s.label, new RegExp(`^${label} \\d+\\.\\d s$`), `${button} counts the time`);
        if (stop) assert.equal(s.stopOn, true, `${stop} is on while ${button} runs`);
      }
      assert.deepEqual(after, { disabled: false, busy: false, label, stopOn: stop ? false : null }, `${button} after its call`);
    }
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});

test("a delete asks in place, and only Yes calls the server", async () => {
  const stack = await Stack.start({
    models: [BERT, `name=tiny-bert-spare,${BERT}`],
    admin: true,
    allow: [
      "/turbo.search.v1.TurboSearchService/CreateIndex",
      "/turbo.search.v1.TurboSearchService/DeleteIndex",
      "/turbo.search.v1.TurboSearchService/SetCollection",
      "/turbo.search.v1.TurboSearchService/DeleteCollection",
      "/turbo.result.v1.AnalysisService/UnloadBundle",
    ],
  });
  const browser = await chromium.launch();
  try {
    const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`${stack.url}/#search`);
    await page.waitForSelector("#search-create-bundle option[value=tiny-bert-bundle]", { state: "attached" });
    await page.click("#search-create button[type=submit]");
    const row = page.locator("#search-indexes tr", { hasText: "demo" });
    await row.waitFor();

    // One click asks and deletes nothing, even after the list is read again.
    await row.locator(".confirm button", { hasText: "Delete" }).click();
    assert.equal(await row.locator(".confirm .ask").textContent(), "Delete demo?");
    assert.equal(await page.evaluate(() => document.activeElement?.textContent), "Cancel");
    await page.click("#search-refresh");
    await page.waitForSelector("#search-refresh:not([aria-busy])");
    await row.waitFor();
    // Cancel and Escape each put Delete back.
    await row.locator(".confirm button", { hasText: "Delete" }).click();
    await row.locator(".confirm button", { hasText: "Cancel" }).click();
    assert.equal(await row.locator(".confirm .ask").count(), 0);
    assert.equal(await page.evaluate(() => document.activeElement?.textContent), "Delete");
    await row.locator(".confirm button", { hasText: "Delete" }).click();
    await page.keyboard.press("Escape");
    assert.equal(await row.locator(".confirm .ask").count(), 0);
    // Yes deletes it.
    await row.locator(".confirm button", { hasText: "Delete" }).click();
    await row.locator(".confirm button", { hasText: "Yes" }).click();
    await page.waitForSelector("#search-indexes p.muted");
    assert.equal(await page.textContent("#search-indexes p.muted"), "No index yet: create one from an embedding bundle.");

    // Deleting a collection asks first too.
    await page.click("#coll-set button[type=submit]");
    const coll = page.locator("#coll-list tr", { hasText: "demo-docs" });
    await coll.waitFor();
    await coll.locator(".confirm button", { hasText: "Delete" }).click();
    assert.equal(await coll.locator(".confirm .ask").textContent(), "Delete demo-docs?");
    await page.click("#coll-refresh");
    await page.waitForSelector("#coll-refresh:not([aria-busy])");
    await coll.waitFor();
    await coll.locator(".confirm button", { hasText: "Delete" }).click();
    await coll.locator(".confirm button", { hasText: "Yes" }).click();
    await page.waitForSelector("#coll-list p.muted");

    // Unload asks, and Yes unloads that bundle only.
    await page.click("nav.tabs button[data-tab=server]");
    const spare = page.locator("#server-caps tr", { hasText: "tiny-bert-spare" });
    await spare.locator(".confirm button", { hasText: "Unload" }).click();
    assert.equal(await spare.locator(".confirm .ask").textContent(), "Unload tiny-bert-spare?");
    assert.equal(await page.locator("#server-caps table").first().locator("td:first-child").count(), 2);
    await spare.locator(".confirm button", { hasText: "Yes" }).click();
    await page.waitForSelector("#server-load-note:has-text('Unloaded tiny-bert-spare.')");
    assert.deepEqual(await page.locator("#server-caps table").first().locator("td:first-child").allTextContents(), ["tiny-bert-bundle"]);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
    await stack.close();
  }
});

test("a refused delete says why beside its own row", async () => {
  // The demo forwards Unload, and the server, started without --admin, refuses it.
  const stack = await Stack.start({
    models: [BERT, `name=tiny-bert-exact,${BERT}`],
    allow: ["/turbo.result.v1.AnalysisService/UnloadBundle"],
  });
  const browser = await chromium.launch();
  try {
    const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
    await page.goto(`${stack.url}/#server`);
    const row = page.locator("#server-caps tr", { hasText: "tiny-bert-exact" });
    await row.locator(".confirm button:not([disabled])", { hasText: "Unload" }).waitFor();
    await row.locator(".confirm button", { hasText: "Unload" }).click();
    await row.locator(".confirm button", { hasText: "Yes" }).click();
    await row.locator(".confirm .error").waitFor();
    // A sentence first; the code stays behind the details toggle.
    const why = (await row.locator(".confirm .error").textContent()) ?? "";
    assert.equal(why.includes("Not served: UnloadBundle is admin-only: start the server with --admin to serve it."), true, why);
    assert.equal(why.trim().startsWith("Not served"), true, why);
    // Only that row reports, Unload is back on it, and nothing was unloaded.
    assert.equal(await page.locator("#server-caps .confirm .error").count(), 1);
    assert.equal(await row.locator(".confirm button", { hasText: "Unload" }).count(), 1);
    await page.reload();
    await page.waitForSelector("#server-caps table");
    assert.equal(await page.locator("#server-caps tr", { hasText: "tiny-bert-exact" }).count(), 1);
  } finally {
    await browser.close();
    await stack.close();
  }
});
