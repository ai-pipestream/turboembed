// Conveniences, in a real browser against real servers: typed input kept
// across a reload with a reset per tab, Copy and Download CSV on results,
// upload progress with a Stop, and a line per tab on what it does and
// needs. The uploads and the unload run through demos of this file's own
// (stack.ts) that forward them; nothing is stubbed.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { chromium, type Browser, type Page } from "playwright";
import { Stack } from "./stack.js";

const BASE = process.env.TURBO_E2E_URL ?? "http://127.0.0.1:8080";
const DEVICE = process.env.TURBO_E2E_DEVICE ?? "0";
const BERT = `bundle=testdata/tiny-bert-bundle,device=${DEVICE},sessions=1`;
const TABS = ["embed", "semantic", "analyze", "search", "vocabulary", "models", "server"];

async function opened(browser: Browser, url: string, tab: string): Promise<Page> {
  const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 }, acceptDownloads: true })).newPage();
  await page.goto(`${url}/#${tab}`);
  await page.waitForSelector("#server.status.up");
  return page;
}

/** A CSV file's rows, read with quoted fields. */
function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') (cell += '"'), i++;
      else if (c === '"') quoted = false;
      else cell += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") row.push(cell), (cell = "");
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(cell), rows.push(row), (row = []), (cell = "");
    } else cell += c;
  }
  if (cell || row.length) row.push(cell), rows.push(row);
  return rows;
}

test("typed input is back after a reload, and Reset to sample puts the sample back", async () => {
  const browser = await chromium.launch();
  try {
    const page = await opened(browser, BASE, "embed");
    await page.waitForSelector("#embed-models label.model");
    const sample = await page.inputValue("#embed-texts");
    const typed = "Grüße aus Köln\n世界こんにちは";
    await page.fill("#embed-texts", typed);
    await page.fill("#embed-max", "16");
    await page.fill("#token", "not-kept-anywhere-1234");
    await page.click("nav.tabs button[data-tab=semantic]");
    await page.fill("#sem-query", "Who designed the engine?");
    await page.click("nav.tabs button[data-tab=analyze]");
    await page.fill("#analyze-text", "One short document.");

    await page.reload();
    await page.waitForSelector("#server.status.up");
    assert.equal(await page.inputValue("#analyze-text"), "One short document.");
    await page.click("nav.tabs button[data-tab=embed]");
    assert.equal(await page.inputValue("#embed-texts"), typed);
    assert.equal(await page.inputValue("#embed-max"), "16");
    await page.click("nav.tabs button[data-tab=semantic]");
    assert.equal(await page.inputValue("#sem-query"), "Who designed the engine?");
    // The bearer token is never kept in localStorage.
    const kept = await page.evaluate(() => Object.keys(localStorage).map((k) => `${k}=${localStorage.getItem(k)}`));
    assert.equal(kept.some((k) => k.includes("not-kept-anywhere-1234")), false);

    // Reset puts Embed back to its sample, for good, and leaves other tabs alone.
    await page.click("nav.tabs button[data-tab=embed]");
    await page.click("#embed-reset");
    assert.equal(await page.inputValue("#embed-texts"), sample);
    assert.equal(await page.inputValue("#embed-max"), "");
    await page.reload();
    await page.waitForSelector("#server.status.up");
    assert.equal(await page.inputValue("#embed-texts"), sample);
    assert.equal(await page.inputValue("#sem-query"), "Who designed the engine?");
    // Models and Server keep nothing and have no reset.
    for (const tab of ["models", "server"]) assert.equal(await page.locator(`#${tab}-reset`).count(), 0, tab);
  } finally {
    await browser.close();
  }
});

test("results offer Copy and Download CSV with the rows they show", async () => {
  const browser = await chromium.launch();
  try {
    const page = await opened(browser, BASE, "embed");
    await page.context().grantPermissions(["clipboard-read", "clipboard-write"], { origin: BASE });
    await page.waitForSelector("#embed-models label.model");
    const texts = (await page.inputValue("#embed-texts")).split("\n").filter((t) => t.trim());
    await page.click("#embed-run");
    await page.waitForSelector("#embed-results .card table.sim");
    const models = await page.locator("#embed-results .cards .card").count();
    // Each model's matrix and vectors, and the side-by-side table, can be taken.
    assert.equal(await page.locator("#embed-results .take").count(), 2 * models + 1);

    const first = page.locator("#embed-results .cards .card").first().locator(".take").first();
    const [file] = await Promise.all([page.waitForEvent("download"), first.locator("button", { hasText: "Download CSV" }).click()]);
    assert.match(file.suggestedFilename(), /\.csv$/);
    const rows = parseCsv(readFileSync(await file.path()).toString());
    assert.equal(rows.length, texts.length + 1);
    for (const r of rows) assert.equal(r.length, texts.length + 1);
    // The cosines in the file are the ones in the table.
    const shown = await page.locator("#embed-results .cards .card").first().locator("table.sim tr").nth(1).locator("td.cell").allTextContents();
    assert.deepEqual(rows[1].slice(1).map((x) => Number(x).toFixed(3)), shown);

    await first.locator("button", { hasText: "Copy" }).click();
    await first.locator("[role=status]", { hasText: /^Copied \d+ rows\.$/ }).waitFor();
    const copied = (await page.evaluate(() => navigator.clipboard.readText())).trimEnd().split("\n");
    assert.equal(copied.length, texts.length + 1);
    assert.equal(copied[0].split("\t").length, texts.length + 1);

    // The vectors file has one column per dimension, after the text.
    const vectors = page.locator("#embed-results .cards .card").first().locator("details");
    await vectors.locator("summary").click();
    const [vfile] = await Promise.all([page.waitForEvent("download"), vectors.locator(".take button", { hasText: "Download CSV" }).click()]);
    const vrows = parseCsv(readFileSync(await vfile.path()).toString());
    assert.equal(vrows.length, texts.length + 1);
    assert.equal(new Set(vrows.map((r) => r.length)).size, 1);
    assert.equal(vrows[0].length > 2, true);

    await page.click("nav.tabs button[data-tab=semantic]");
    await page.click("#sem-run");
    await page.waitForSelector("#sem-sim .take");
    const [nfile] = await Promise.all([page.waitForEvent("download"), page.locator("#sem-sim .take").first().locator("button", { hasText: "Download CSV" }).click()]);
    const near = parseCsv(readFileSync(await nfile.path()).toString());
    assert.deepEqual(near[0], ["rank", "passage number", "passage", "cosine"]);
    assert.equal(near.length, 1 + Number(await page.inputValue("#sem-k")));
  } finally {
    await browser.close();
  }
});

/** Holds the page's uploads to `bytes` a second, so progress can be seen. */
async function slowUploads(page: Page, bytes: number): Promise<void> {
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Network.enable");
  await cdp.send("Network.emulateNetworkConditions", { offline: false, latency: 0, downloadThroughput: -1, uploadThroughput: bytes });
}

/** About `mib` MiB of plain words as a text file. */
function words(mib: number): Uint8Array {
  const line = "the analytical engine weaves algebraic patterns as the loom weaves flowers and leaves\n";
  return Buffer.from(line.repeat(Math.ceil((mib * 1024 * 1024) / line.length)));
}

test("an upload shows its progress and ends with the server's answer", async () => {
  const stack = await Stack.start({ admin: true, allow: ["/turbo.vocabulary.v1.TurboVocabularyService/LearnVocabulary"] });
  const browser = await chromium.launch();
  try {
    const page = await opened(browser, stack.url, "vocabulary");
    await page.waitForSelector("#vocab-learn:not(.off)");
    await page.fill("#vocab-docs", "");
    await page.setInputFiles("#vocab-learn-files", { name: "words.txt", mimeType: "text/plain", buffer: words(3) });
    await slowUploads(page, 512 * 1024);
    await page.click("#vocab-learn button[type=submit]");
    const bar = page.locator("#vocab-note progress");
    await bar.waitFor();
    const seen: number[] = [];
    await page.waitForFunction(() => {
      const p = document.querySelector("#vocab-note progress") as HTMLProgressElement | null;
      return p && p.value > 0 && p.value < p.max;
    });
    for (let i = 0; i < 2; i++) {
      seen.push(await bar.evaluate((p) => (p as HTMLProgressElement).value / (p as HTMLProgressElement).max));
      const line = (await page.textContent("#vocab-note span")) ?? "";
      if (line.startsWith("Uploading")) assert.match(line, /^Uploading 1 document: [\d.]+ [KM]iB of 3\.0 MiB \(\d+%\)$/);
      await page.waitForTimeout(1_000);
    }
    assert.equal(seen[1] > seen[0], true, `the bar moves: ${seen.join(" then ")}`);
    await page.waitForSelector("#vocab-note:has-text('Learned ')", { timeout: 60_000 });
    assert.equal(await page.locator("#vocab-note progress").count(), 0);
  } finally {
    await browser.close();
    await stack.close();
  }
});

test("Stop ends an upload to the index and says so", async () => {
  const stack = await Stack.start({ allow: ["/turbo.search.v1.TurboSearchService/CreateIndex", "/turbo.search.v1.TurboSearchService/IndexDocuments"] });
  const browser = await chromium.launch();
  try {
    const page = await opened(browser, stack.url, "search");
    await page.waitForSelector("#search-create:not(.off)");
    // The bundle list is read after the status: Create asks for a bundle until it is filled.
    await page.waitForSelector("#search-create-bundle option[value=tiny-bert-bundle]", { state: "attached" });
    await page.click("#search-create button[type=submit]");
    await page.waitForSelector("#search-create-note:has-text('Created demo.')");
    // The new index is picked once the list has been read again.
    await page.waitForFunction(() => (document.querySelector("#search-index") as HTMLSelectElement).value === "demo");
    assert.equal(await page.isDisabled("#search-add-stop"), true);
    await page.fill("#search-docs", "");
    await page.setInputFiles("#search-files", { name: "words.txt", mimeType: "text/plain", buffer: words(3) });
    await slowUploads(page, 128 * 1024);
    await page.click("#search-add");
    await page.waitForSelector("#search-add-note progress");
    assert.equal(await page.isDisabled("#search-add-stop"), false);
    assert.equal(await page.isDisabled("#search-add"), true);
    await page.click("#search-add-stop");
    await page.waitForSelector("#search-add-note:has-text('Cancelled.')");
    assert.equal(await page.isDisabled("#search-add"), false);
    assert.equal(await page.isDisabled("#search-add-stop"), true);
    // Nothing reached the index.
    await page.click("#search-refresh");
    await page.waitForSelector("#search-refresh:not([aria-busy])");
    const docs = await page.locator("#search-indexes tr", { hasText: "demo" }).locator("td").nth(5).textContent();
    assert.equal(docs, "0");
  } finally {
    await browser.close();
    await stack.close();
  }
});

test("every tab says what it does, and what it needs when nothing it can use is served", async () => {
  const browser = await chromium.launch();
  try {
    const page = await opened(browser, BASE, "embed");
    for (const tab of TABS) {
      await page.click(`nav.tabs button[data-tab=${tab}]`);
      const about = page.locator(`#about-${tab}`);
      assert.equal(await about.isVisible(), true, tab);
      assert.match((await about.textContent()) ?? "", /^[A-Z].{20,}\.$/, tab);
    }
    // An embedding model is served, so no tab asks for one.
    for (const tab of ["embed", "semantic", "search"]) assert.equal(await page.isHidden(`#needs-${tab}`), true, tab);
  } finally {
    await browser.close();
  }
  // With the only model unloaded, the tabs that need one say so.
  const stack = await Stack.start({ models: [BERT], admin: true, allow: ["/turbo.result.v1.AnalysisService/UnloadBundle"] });
  const other = await chromium.launch();
  try {
    const page = await opened(other, stack.url, "server");
    const row = page.locator("#server-caps tr", { hasText: "tiny-bert-bundle" });
    await row.locator(".confirm button:not([disabled])", { hasText: "Unload" }).click();
    await row.locator(".confirm button", { hasText: "Yes" }).click();
    await page.waitForSelector("#server-load-note:has-text('Unloaded tiny-bert-bundle.')");
    const needs: Record<string, string> = {
      embed: "Needs: an embedding model.",
      semantic: "Needs: an embedding model, a reranker or a QA model.",
      search: "Needs: an embedding model to create an index.",
    };
    for (const [tab, start] of Object.entries(needs)) {
      await page.click(`nav.tabs button[data-tab=${tab}]`);
      await page.waitForSelector(`#needs-${tab}:not([hidden])`);
      assert.equal(((await page.textContent(`#needs-${tab}`)) ?? "").startsWith(start), true, tab);
    }
    // Analyze runs its default profile with no model, and asks for none.
    assert.equal(await page.locator("#needs-analyze").count(), 0);
  } finally {
    await other.close();
    await stack.close();
  }
});
