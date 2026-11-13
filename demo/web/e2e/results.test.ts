// How results read, in a real browser against real servers: similarity
// colours scaled to the matrix's own range with each row's best match,
// annotated text coloured by layer with a legend and a panel for the
// mark picked, layer chips told apart by instance, search hits with
// their matched words marked, and results brought into view on a phone.
// The search runs on an index made through a demo of this file's own
// (stack.ts) that forwards the writes; nothing is stubbed.

import assert from "node:assert/strict";
import { test } from "node:test";
import { chromium, type Browser, type Page } from "playwright";
import { Stack } from "./stack.js";

const BASE = process.env.TURBO_E2E_URL ?? "http://127.0.0.1:8080";

async function opened(browser: Browser, url: string, tab: string, width = 1280, height = 900): Promise<Page> {
  const page = await (await browser.newContext({ viewport: { width, height } })).newPage();
  await page.goto(`${url}/#${tab}`);
  await page.waitForSelector("#server.status.up");
  return page;
}

/** Each similarity table under `selector`: its cosines, the alpha of each cell's colour, and the best-marked column per row. */
function simTables(page: Page, selector: string): Promise<{ values: number[][]; alpha: number[][]; best: number[][] }[]> {
  return page.locator(`${selector} table.sim:not(.cmp)`).evaluateAll((tables) =>
    tables.map((t) => {
      const rows = [...t.querySelectorAll("tr")].slice(1).map((r) => [...r.querySelectorAll("td.cell")] as HTMLElement[]);
      const alphaOf = (c: HTMLElement) => Number(c.style.background.match(/,\s*([\d.]+)\)\s*$/)?.[1] ?? "NaN");
      return {
        values: rows.map((r) => r.map((c) => Number(c.textContent))),
        alpha: rows.map((r) => r.map(alphaOf)),
        best: rows.map((r) => r.flatMap((c, j) => (c.classList.contains("best") ? [j] : []))),
      };
    }),
  );
}

/** The least and greatest of the off-diagonal entries. */
function spread(m: number[][]): [number, number] {
  const off = m.flatMap((row, i) => row.filter((_, j) => j !== i));
  return [Math.min(...off), Math.max(...off)];
}

test("similarity colours span the matrix's own range and each row marks its best match", async () => {
  const browser = await chromium.launch();
  try {
    const page = await opened(browser, BASE, "embed");
    await page.waitForSelector("#embed-models label.model");
    await page.click("#embed-run");
    await page.waitForSelector("#embed-results .card table.sim");
    const tables = await simTables(page, "#embed-results .cards");
    assert.equal(tables.length > 0, true);
    for (const { values, alpha, best } of tables) {
      // Whatever band the cosines sit in, the colours reach from faint to strong.
      const [alo, ahi] = spread(alpha);
      assert.equal(alo < 0.1 && ahi > 0.8, true, `colour strength ${alo}..${ahi} for cosines ${spread(values).join("..")}`);
      // Each row's one best match is its greatest cosine off the diagonal.
      best.forEach((cols, i) => {
        assert.equal(cols.length, 1, `row ${i}`);
        const off = values[i].filter((_, j) => j !== i);
        assert.notEqual(cols[0], i);
        assert.equal(values[i][cols[0]], Math.max(...off));
      });
    }
    // The switch goes back to the absolute scale, where the band looks alike.
    const own = page.locator("#embed-results .cards .scale input[type=checkbox]").first();
    assert.equal(await own.isChecked(), true);
    await own.uncheck();
    const [absolute] = await simTables(page, "#embed-results .cards .card:first-child");
    const [alo, ahi] = spread(absolute.alpha);
    assert.equal(ahi - alo < spread(tables[0].alpha)[1] - spread(tables[0].alpha)[0], true);

    // Side by side: the spread is in fixed decimals and the last row's label is whole.
    const cmp = page.locator("#embed-results table.sim.cmp");
    const spreads = await cmp.locator("tr td:last-child").allTextContents();
    for (const s of spreads.filter((x) => x !== "")) assert.match(s, /^\d+\.\d{4}$/);
    const cut = await cmp.locator("th.wrap").evaluate((th) => th.scrollWidth > th.clientWidth);
    assert.equal(cut, false);
  } finally {
    await browser.close();
  }
});

test("annotated text is coloured by layer, with a legend and a panel for the mark picked", async () => {
  const browser = await chromium.launch();
  try {
    const page = await opened(browser, BASE, "analyze");
    await page.click("#analyze-run");
    await page.waitForSelector("#analyze-window .doc");
    const chip = (name: RegExp) => page.locator("#analyze-window .chips button", { hasText: name });

    // Layer chips with one name are told apart by instance and pick alone.
    const bounds = await page.locator("#analyze-window .chips button", { hasText: /^turbo:boundaries/ }).allTextContents();
    assert.deepEqual(bounds.map((b) => b.replace(/ \d+$/, "")).sort(), ["turbo:boundaries", "turbo:boundaries [lexical]"]);
    await chip(/^turbo:boundaries \[lexical\] /).click();
    await page.waitForSelector("#analyze-window .chips button.on");
    assert.deepEqual(await page.locator("#analyze-window .chips button[aria-pressed=true]").allTextContents(), [bounds.find((b) => b.includes("lexical"))]);
    await chip(/^turbo:boundaries \[lexical\] /).click();

    // One layer picked: the legend names it and the marks are its items.
    await chip(/^turbo:sentences /).click();
    await page.waitForSelector("#analyze-window .chips button.on");
    const swatches = await page.locator("#analyze-window .legend .swatch").allTextContents();
    assert.deepEqual(swatches.map((s) => s.trim()), ["turbo:sentences 3"]);
    const marks = page.locator("#analyze-window .doc mark");
    assert.equal(await marks.count(), 3);
    // Clicking a mark lists its annotations beside the text.
    await marks.first().click();
    const panel = page.locator("#analyze-window .inspect");
    await panel.locator("table").waitFor();
    const quoted = await panel.locator(".quote").textContent();
    assert.equal(quoted, await marks.first().textContent());
    assert.equal((await panel.locator("td:first-child").allTextContents()).includes("turbo:sentences"), true);
    // A mark is reached by keyboard, and focus shows it too.
    assert.equal(await marks.nth(1).getAttribute("tabindex"), "0");
    await marks.nth(1).focus();
    const second = (await marks.nth(1).textContent()) ?? "";
    await page.waitForFunction((t) => document.querySelector("#analyze-window .inspect .quote")?.textContent === t, second);
    assert.equal(await page.locator("#analyze-window .doc mark.picked").textContent(), second);

    // Two layers picked: two swatches in two colours.
    await chip(/^turbo:tokens /).click();
    await page.waitForFunction(() => document.querySelectorAll("#analyze-window .legend .swatch").length === 2);
    const colours = await page.locator("#analyze-window .legend .swatch").evaluateAll((ss) => ss.map((s) => (s as HTMLElement).style.getPropertyValue("--mark")));
    assert.equal(new Set(colours).size, 2);
    // The heat rows carry a legend.
    assert.equal(await page.locator("#analyze-window .heat-key").count(), 1);
  } finally {
    await browser.close();
  }
});

/** A demo of this file's own with an index named demo holding the Search tab's sample documents. */
async function indexed(): Promise<Stack> {
  const stack = await Stack.start({
    allow: ["/turbo.search.v1.TurboSearchService/CreateIndex", "/turbo.search.v1.TurboSearchService/IndexDocuments"],
  });
  const browser = await chromium.launch();
  try {
    const page = await opened(browser, stack.url, "search");
    await page.waitForSelector("#search-create:not(.off)");
    await page.click("#search-create button[type=submit]");
    await page.waitForSelector("#search-create-note:has-text('Created demo.')");
    await page.waitForSelector("#search-add-box:not(.off)");
    await page.click("#search-add");
    await page.waitForSelector("#search-add-note:has-text('Indexed ')");
  } catch (e) {
    await stack.close();
    throw e;
  } finally {
    await browser.close();
  }
  return stack;
}

test("search hits mark their matched words in place, with a legend", async () => {
  const stack = await indexed();
  const browser = await chromium.launch();
  try {
    const page = await opened(browser, stack.url, "search");
    await page.waitForSelector("#search-index option[value=demo]", { state: "attached" });
    await page.selectOption("#search-index", "demo");
    await page.fill("#search-query", "memory safe language");
    await page.click("#search-form button[type=submit]");
    await page.waitForSelector("#search-results .hit");
    const marked = await page.locator("#search-results .hit .doc mark").evaluateAll((ms) => ms.map((m) => (m.firstChild?.textContent ?? "").toLowerCase()));
    assert.equal(marked.length > 0, true);
    for (const word of marked) assert.equal(["memory", "safe", "language"].some((w) => word.includes(w)), true, `marked "${word}"`);
    // Each mark names its signal, and the legend lists every signal named.
    const signals = await page.locator("#search-results .hit .doc mark sup").allTextContents();
    const legend = (await page.locator("#search-results .legend .swatch").allTextContents()).map((s) => s.trim());
    for (const s of signals.flatMap((x) => x.split("+"))) assert.equal(legend.includes(s), true, `legend names ${s}`);
    if (await page.locator("#search-results h3:has-text('Document heat')").count()) {
      assert.equal(await page.locator("#search-results .heat-key").count(), 1);
    }
  } finally {
    await browser.close();
    await stack.close();
  }
});

/** Whether the top of `selector` is inside the viewport. */
function inView(page: Page, selector: string): Promise<boolean> {
  return page.locator(selector).evaluate((e) => {
    const r = e.getBoundingClientRect();
    return r.top >= -1 && r.top < innerHeight;
  });
}

test("on a phone, Run brings the results into view", async () => {
  const stack = await indexed();
  const browser = await chromium.launch();
  try {
    // Each run, what it waits for, and the results box that should come into view.
    const runs: [string, string, string, string, string][] = [
      [BASE, "embed", "#embed-run", "#embed-results .cards", "#embed-results"],
      [BASE, "semantic", "#sem-run", "#sem-sim table.sim", "#sem-sim"],
      [BASE, "analyze", "#analyze-run", "#analyze-window .doc", "#tab-analyze .results"],
      [stack.url, "search", "#search-form button[type=submit]", "#search-results .hit", "#search-results"],
    ];
    for (const [url, tab, run, result, box] of runs) {
      const page = await (await browser.newContext({ viewport: { width: 390, height: 844 }, reducedMotion: "reduce" })).newPage();
      await page.goto(`${url}/#${tab}`);
      await page.waitForSelector("#server.status.up");
      if (tab === "embed") await page.waitForSelector("#embed-models label.model");
      if (tab === "search") {
        await page.waitForSelector("#search-index option[value=demo]", { state: "attached" });
        await page.selectOption("#search-index", "demo");
      }
      assert.equal(await page.locator(result).count(), 0);
      await page.locator(run).scrollIntoViewIfNeeded();
      await page.locator(run).click();
      await page.waitForSelector(result);
      await page.waitForFunction((sel) => {
        const r = document.querySelector(sel)!.getBoundingClientRect();
        return r.top >= -1 && r.top < innerHeight;
      }, box, { timeout: 5_000 }).catch(() => undefined);
      assert.equal(await inView(page, box), true, `${tab}: the results are in view after Run`);
      await page.context().close();
    }
  } finally {
    await browser.close();
    await stack.close();
  }
});
