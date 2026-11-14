// What the page says before and after a call, in a real browser against
// real servers: forms whose write the demo does not forward are off and
// say which --allow enables them, a refused call reads as a sentence with
// the protocol's words behind a toggle, every control has a name, and
// options for a kind of model show only when one is served. The demos
// with writes forwarded are this file's own (stack.ts); nothing is
// stubbed.

import assert from "node:assert/strict";
import { test } from "node:test";
import { chromium, type Page } from "playwright";
import { Stack } from "./stack.js";

const BASE = process.env.TURBO_E2E_URL ?? "http://127.0.0.1:8080";
const TABS = ["embed", "semantic", "analyze", "search", "vocabulary", "models", "server"];
// Each form a write needs, with the method that write calls.
const FORMS: [string, string, string][] = [
  ["search", "search-create", "turbo.search.v1.TurboSearchService/CreateIndex"],
  ["search", "search-add-box", "turbo.search.v1.TurboSearchService/IndexDocuments"],
  ["vocabulary", "vocab-import", "turbo.vocabulary.v1.TurboVocabularyService/ImportDictionary"],
  ["vocabulary", "vocab-learn", "turbo.vocabulary.v1.TurboVocabularyService/LearnVocabulary"],
  ["models", "train-form", "turbo.training.v1.TurboTrainingService/TrainStaticModel"],
  ["server", "server-load", "turbo.result.v1.AnalysisService/LoadBundle"],
];

async function opened(url: string, tab: string): Promise<Page> {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
  page.on("close", () => void browser.close());
  await page.goto(`${url}/#${tab}`);
  await page.waitForSelector("#server.status.up");
  return page;
}

/** Whether the form `id` is off: inert, marked, and followed by its reason. */
async function offState(page: Page, id: string): Promise<{ inert: boolean; marked: boolean; why: string }> {
  return page.evaluate((id) => {
    const box = document.getElementById(id)!;
    return { inert: box.inert, marked: box.classList.contains("off"), why: document.getElementById(`${id}-off`)?.textContent ?? "" };
  }, id);
}

test("with no --allow every write form is off and names the --allow that turns it on", async () => {
  const page = await opened(BASE, "search");
  try {
    // The forms are turned off once the demo has said what it forwards.
    await page.waitForSelector("#search-create-off");
    for (const [tab, id, method] of FORMS) {
      await page.click(`nav.tabs button[data-tab=${tab}]`);
      assert.deepEqual(await offState(page, id), { inert: true, marked: true, why: `Off: the demo does not forward this write. Start it with --allow ${method}` }, id);
    }
    // A form that is off takes no click.
    await page.click("nav.tabs button[data-tab=search]");
    await page.locator("#search-create button[type=submit]").click({ timeout: 2_000 }).catch(() => undefined);
    assert.equal(await page.isVisible("#search-create-note"), false);
    // The controls of single writes are off too, and say why when pointed at.
    await page.click("nav.tabs button[data-tab=server]");
    await page.waitForSelector("#server-caps table");
    const unload = page.locator("#server-caps .confirm button", { hasText: "Unload" }).first();
    assert.equal(await unload.isDisabled(), true);
    assert.match((await unload.getAttribute("title")) ?? "", /^Off: .*--allow turbo\.result\.v1\.AnalysisService\/UnloadBundle$/);
  } finally {
    await page.close();
  }
});

test("with --allow for one write only its form is live", async () => {
  const stack = await Stack.start({ allow: ["/turbo.search.v1.TurboSearchService/CreateIndex"] });
  const page = await opened(stack.url, "search");
  try {
    await page.waitForSelector("#search-add-box-off");
    for (const [tab, id] of FORMS) {
      await page.click(`nav.tabs button[data-tab=${tab}]`);
      const s = await offState(page, id);
      if (id === "search-create") assert.deepEqual(s, { inert: false, marked: false, why: "" }, id);
      else assert.equal(s.inert && s.marked && s.why.startsWith("Off: "), true, id);
    }
    await page.click("nav.tabs button[data-tab=search]");
    // The bundle list is read after the status: Create asks for a bundle until it is filled.
    await page.waitForSelector("#search-create-bundle option[value=tiny-bert-bundle]", { state: "attached" });
    await page.click("#search-create button[type=submit]");
    await page.waitForSelector("#search-create-note:has-text('Created demo.')");
  } finally {
    await page.close();
    await stack.close();
  }
});

test("a refused call leads with a sentence and keeps the protocol's words behind a toggle", async () => {
  const stack = await Stack.start({ admin: true, allow: ["/turbo.result.v1.AnalysisService/LoadBundle"] });
  const page = await opened(stack.url, "server");
  try {
    await page.waitForSelector("#server-load:not(.off)");
    await page.fill("#server-load-name", "nope");
    await page.fill("#server-load-path", "testdata/no-such-bundle");
    await page.click("#server-load button[type=submit]");
    const note = page.locator("#server-load-note.error");
    await note.waitFor();
    const sentence = (await note.evaluate((n) => n.firstChild?.textContent ?? "")).trim();
    assert.equal(sentence.length > 0, true);
    // Not the protocol's words: no Connect code, turbo-code or TURBO_E_ name.
    for (const word of ["not_found", "turbo-code", "TURBO_E_"]) assert.equal(sentence.includes(word), false, `${word} in "${sentence}"`);
    assert.match(sentence, /^[A-Z].*\.$/);
    const details = note.locator("details.why");
    assert.equal(await details.getAttribute("open"), null);
    assert.equal(await details.locator("code").isVisible(), false);
    await details.locator("summary").click();
    assert.match((await details.locator("code").textContent()) ?? "", /^not_found, turbo-code \d+.*: TURBO_E_[A-Z_]+: /);
  } finally {
    await page.close();
    await stack.close();
  }
});

test("an empty input is said before the server is asked", async () => {
  const page = await opened(BASE, "search");
  try {
    await page.waitForSelector("#search-index option", { state: "attached" });
    assert.deepEqual(await page.locator("#search-index option").allTextContents(), ["no index yet"]);
    await page.click("#search-form button[type=submit]");
    await page.waitForSelector("#search-results p.error");
    assert.equal(await page.textContent("#search-results p.error"), "Pick or create an index first.");
    assert.equal(await page.locator("#search-results details").count(), 0);
  } finally {
    await page.close();
  }
});

test("every control on every tab has a label or an accessible name", async () => {
  const page = await opened(BASE, "embed");
  try {
    await page.waitForSelector("#embed-models label.model");
    const nameless: string[] = [];
    for (const tab of TABS) {
      await page.click(`nav.tabs button[data-tab=${tab}]`);
      await page.waitForTimeout(300);
      await page.evaluate(() => document.querySelectorAll("details").forEach((d) => (d.open = true)));
      const found = await page.evaluate(() => {
        const out: string[] = [];
        for (const c of document.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>("input, select, textarea")) {
          if (c.type === "hidden" || c.getClientRects().length === 0) continue;
          const named = (c.labels?.length ?? 0) > 0 || !!c.getAttribute("aria-label")?.trim() || !!c.getAttribute("aria-labelledby");
          if (!named) out.push(c.id || `${c.tagName.toLowerCase()} in ${c.closest("[id]")?.id ?? "the page"}`);
        }
        return out;
      });
      for (const f of found) nameless.push(`${tab}: ${f}`);
    }
    assert.deepEqual(nameless, []);
  } finally {
    await page.close();
  }
});

test("options for a kind of model show only when one is served and picked", async () => {
  const page = await opened(BASE, "analyze");
  try {
    // The test bundles serve embedding models only.
    await page.waitForSelector("#analyze-model option[value=tiny-bert-bundle]", { state: "attached" });
    for (const id of ["#analyze-open-span", "#analyze-relation-pick", "#analyze-relation"]) assert.equal(await page.isHidden(id), true, id);
    // Picking an embedding model brings no open-span options.
    await page.selectOption("#analyze-model", "tiny-bert-bundle");
    assert.equal(await page.isHidden("#analyze-open-span"), true);
    await page.click("nav.tabs button[data-tab=semantic]");
    for (const id of ["#sem-rerank-pick", "#sem-qa-pick", "#sem-topk-pick"]) assert.equal(await page.isHidden(id), true, id);
    assert.equal(await page.isVisible("#sem-embed"), true);
    assert.equal(await page.isVisible("#sem-k"), true);
    await page.click("#sem-run");
    await page.waitForSelector("#sem-sim table");
    assert.equal(await page.locator("#sem-rerank-out *").count(), 0);
  } finally {
    await page.close();
  }
});
