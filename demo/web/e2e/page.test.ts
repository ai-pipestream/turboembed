// The page in a real browser against a real turbo-kserve-demo in front
// of a real turbo-kserve (scripts/e2e-web.sh starts both): every tab at
// a desktop and a phone width, checked for the layout faults a person
// would see, with a screenshot of each tab. Nothing is stubbed.

import assert from "node:assert/strict";
import { test } from "node:test";
import { chromium, type Browser, type Page } from "playwright";

const BASE = process.env.TURBO_E2E_URL ?? "http://127.0.0.1:8080";
const SHOTS = process.env.TURBO_E2E_SHOTS ?? "";
const SCALE = Number(process.env.TURBO_E2E_SCALE ?? "1");
const TABS = ["embed", "semantic", "analyze", "search", "vocabulary", "models", "server"];
const WIDTHS: [string, number, number][] = [
  ["desktop", 1280, 900],
  ["phone", 390, 844],
];

/** The layout faults in the tab now shown, each as one line. */
function faults(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const found: string[] = [];
    const name = (e: Element) => `${e.tagName.toLowerCase()}${e.id ? `#${e.id}` : ""}`;
    const shown = (e: Element) => {
      const r = e.getBoundingClientRect();
      return r.width > 0 && r.height > 0;
    };
    const root = document.documentElement;
    if (root.scrollWidth > root.clientWidth) found.push(`the page scrolls sideways: ${root.scrollWidth} px in ${root.clientWidth} px`);
    for (const box of document.querySelectorAll(".box")) {
      if (!shown(box)) continue;
      const b = box.getBoundingClientRect();
      for (const c of box.querySelectorAll("input, select, textarea, button")) {
        if (!shown(c)) continue;
        const r = c.getBoundingClientRect();
        if (r.left < b.left - 0.5 || r.right > b.right + 0.5) found.push(`${name(c)} leaves its box by ${Math.round(Math.max(b.left - r.left, r.right - b.right))} px`);
      }
    }
    for (const c of document.querySelectorAll("input[type=checkbox]")) {
      if (shown(c) && c.getBoundingClientRect().width > 32) found.push(`${name(c)} is ${Math.round(c.getBoundingClientRect().width)} px wide`);
    }
    for (const c of document.querySelectorAll(".card")) {
      if (shown(c) && c.childElementCount === 0 && !(c.textContent ?? "").trim()) found.push(`${name(c)} is an empty card`);
    }
    for (const t of document.querySelectorAll("table")) {
      if (shown(t) && t.querySelectorAll("td").length === 0) found.push(`a table in ${name(t.closest("[id]") ?? t)} has headings and no rows`);
    }
    return found;
  });
}

async function open(browser: Browser, width: number, height: number): Promise<Page> {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: SCALE });
  const page = await context.newPage();
  await page.goto(`${BASE}/#embed`);
  await page.waitForSelector("#embed-models label.model");
  return page;
}

/** Shows `tab`, does what a visitor does first there, and waits for the answers. */
async function visit(page: Page, tab: string): Promise<void> {
  await page.click(`nav.tabs button[data-tab=${tab}]`);
  if (tab === "embed") {
    await page.click("#embed-run");
    await page.waitForSelector("#embed-results .card table.sim");
  } else if (tab === "semantic") {
    await page.click("#sem-run");
    await page.waitForSelector("#sem-sim table");
  } else if (tab === "analyze") {
    await page.click("#analyze-run");
    await page.waitForSelector("#analyze-window .doc");
  } else if (tab === "search") {
    await page.waitForSelector("#search-indexes > *");
    await page.waitForSelector("#coll-list > *");
  } else if (tab === "vocabulary") {
    await page.waitForSelector("#vocab-lists > *");
  } else if (tab === "models") {
    await page.waitForSelector("#models-static > *");
    await page.click("#models-catalog-load");
    await page.waitForSelector("#models-filter");
  } else {
    await page.waitForSelector("#server-caps table");
  }
}

for (const [label, width, height] of WIDTHS) {
  test(`every tab lays out cleanly at ${width} px`, async () => {
    const browser = await chromium.launch();
    try {
      const page = await open(browser, width, height);
      const errors: string[] = [];
      page.on("pageerror", (e) => errors.push(e.message));
      const found: string[] = [];
      for (const [i, tab] of TABS.entries()) {
        await visit(page, tab);
        for (const f of await faults(page)) found.push(`${tab}: ${f}`);
        if (SHOTS) await page.screenshot({ path: `${SHOTS}/${label}-${i + 1}-${tab}.png`, fullPage: true });
      }
      assert.deepEqual(found, []);
      assert.deepEqual(errors, []);
    } finally {
      await browser.close();
    }
  });
}

test("the embed tab answers with a card per model and the tabs take the arrow keys", async () => {
  const browser = await chromium.launch();
  try {
    const page = await open(browser, 1280, 900);
    // The header says the server is up once it answers.
    await page.waitForSelector("#server.status.up", { timeout: 5000 });
    const models = await page.locator("#embed-models label.model").count();
    await page.click("#embed-run");
    await page.waitForSelector("#embed-results .cards .card table.sim");
    assert.equal(await page.locator("#embed-results .cards .card").count(), models);
    assert.equal(await page.locator("#embed-results .error").count(), 0);
    await page.focus("nav.tabs button[data-tab=embed]");
    await page.keyboard.press("ArrowRight");
    assert.equal(await page.getAttribute("nav.tabs button[data-tab=semantic]", "aria-selected"), "true");
    assert.equal(await page.isVisible("#tab-semantic"), true);
    await page.keyboard.press("ArrowLeft");
    await page.keyboard.press("ArrowLeft");
    assert.equal(await page.getAttribute("nav.tabs button[data-tab=server]", "aria-selected"), "true");
  } finally {
    await browser.close();
  }
});

test("the catalog narrows by words typed and by kind", async () => {
  const browser = await chromium.launch();
  try {
    const page = await open(browser, 1280, 900);
    await page.click("nav.tabs button[data-tab=models]");
    await page.click("#models-catalog-load");
    await page.waitForSelector("#models-filter");
    const shown = () => page.locator(".catalog .entry:not([hidden]) .id").allTextContents();
    const all = await shown();
    assert.equal(all.length > 10, true);
    assert.equal(await page.textContent("#models-count"), `${all.length} entries`);
    // No kind is shown by its enum name.
    assert.equal((await page.locator(".catalog").innerText()).includes("KIND_"), false);

    await page.fill("#models-filter", "BGE small");
    const bge = await shown();
    assert.equal(bge.length > 0 && bge.length < all.length, true);
    assert.deepEqual(bge.filter((id) => !(id.includes("bge") && id.includes("small"))), []);
    assert.equal(await page.textContent("#models-count"), `${bge.length} of ${all.length} entries`);

    await page.fill("#models-filter", "");
    const chip = page.locator(".chips button[data-kind=KIND_DATASET]");
    const label = (await chip.textContent()) ?? "";
    await chip.click();
    assert.equal(await chip.getAttribute("aria-pressed"), "true");
    assert.equal(label, `dataset ${(await shown()).length}`);
    const kindsShown = await page.locator(".catalog .entry:not([hidden]) .kind").allTextContents();
    assert.deepEqual(kindsShown.filter((k) => k !== "dataset"), []);

    await page.fill("#models-filter", "no such entry anywhere");
    assert.deepEqual(await shown(), []);
    assert.equal(await page.isVisible("text=No entry matches."), true);
  } finally {
    await browser.close();
  }
});
