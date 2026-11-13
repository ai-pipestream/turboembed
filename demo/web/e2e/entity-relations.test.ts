// The relation arrows of the entity views in Analyze, over a graph with
// an entity layer and a relation layer that server/examples/
// inspect_layer_fixture.rs writes through the production export codecs.
// No test bundle carries a relation model, so the graph is opened from
// a file. A real server and a real demo; nothing is stubbed.

import assert from "node:assert/strict";
import { test } from "node:test";
import { chromium } from "playwright";

const BASE = process.env.TURBO_E2E_URL ?? "http://127.0.0.1:8080";
const OUT = "#entities-out";

test("the entity graph draws a labelled arrow per relation and names them for a pointed-at entity", async () => {
  const browser = await chromium.launch();
  try {
    const page = await (await browser.newContext({ viewport: { width: 1400, height: 1000 } })).newPage();
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`${BASE}/#analyze`);
    await page.click("#analyze-wire summary");
    assert.equal(Boolean(process.env.TURBO_E2E_RELATION_GRAPH), true, "scripts/e2e-web.sh supplies the Rust-generated graph");
    await page.setInputFiles("#wire-file", process.env.TURBO_E2E_RELATION_GRAPH!);
    await page.waitForSelector(`${OUT} svg .edge`);
    if (process.env.TURBO_E2E_SHOTS) await page.screenshot({ path: `${process.env.TURBO_E2E_SHOTS}/entity-relations.png`, fullPage: true });

    // London is named twice and is one entity; "Babbage" alone is not "Charles Babbage".
    const summary = (await page.locator(`${OUT} > p`).first().textContent()) ?? "";
    assert.equal(summary.startsWith("4 entities from 5 mentions"), true, summary);
    assert.equal(summary.includes(", 3 relations."), true, summary);
    assert.equal(await page.locator(`${OUT} svg .node`).count(), 4);

    // An arrow per relation, named, pointing from subject to object.
    assert.equal(await page.locator(`${OUT} svg .edge`).count(), 3);
    assert.deepEqual((await page.locator(`${OUT} svg .edge .edge-name`).allTextContents()).sort(), ["born in", "located in", "works for"]);
    assert.equal(await page.locator(`${OUT} svg .edge line[marker-end="url(#arrow)"]`).count(), 3);
    assert.deepEqual((await page.locator(`${OUT} svg .edge title`).allTextContents()).sort(), [
      "Ada Lovelace works for Charles Babbage, score 0.900",
      "Babbage born in London, score 0.500",
      "Charles Babbage located in London, score 0.750",
    ]);
    assert.equal(((await page.locator(`${OUT} .legend .arrow-key`).textContent()) ?? "").trim(), "a relation, from subject to object");

    // Pointing at an entity lights its arrows and names its relations.
    await page.locator(`${OUT} svg .node[aria-label^="Charles Babbage,"]`).hover();
    await page.waitForSelector(`${OUT} svg .edge.lit`);
    assert.equal(await page.locator(`${OUT} svg .edge.lit`).count(), 2);
    const info = (await page.locator(`${OUT} .info`).textContent()) ?? "";
    assert.equal(info.startsWith("Charles Babbage (person), 1 mention: Ada Lovelace works for Charles Babbage; Charles Babbage located in London."), true, info);

    // The relations as a table, after the entity table.
    await page.locator(`${OUT} details.tables summary`).click();
    const tables = page.locator(`${OUT} details.tables table`);
    assert.equal(await tables.count(), 2);
    assert.deepEqual(await tables.nth(1).locator("th").allTextContents(), ["subject", "relation", "object", "score", "found"]);
    const rows = await tables.nth(1).locator("tr").evaluateAll((all) => all.map((r) => [...r.querySelectorAll("td")].map((c) => c.textContent)).filter((r) => r.length > 0));
    assert.deepEqual(rows.sort(), [
      ["Ada Lovelace", "works for", "Charles Babbage", "0.900", "1"],
      ["Babbage", "born in", "London", "0.500", "1"],
      ["Charles Babbage", "located in", "London", "0.750", "1"],
    ]);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});
