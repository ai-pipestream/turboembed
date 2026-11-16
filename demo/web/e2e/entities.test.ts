// The entity views in Analyze: the entities as a graph and the mention
// strips, over InspectDocumentGraph's typed layers. A real server and a
// real demo (the shared ones of scripts/e2e-web.sh); nothing is stubbed.
// The entities come from the place-name step of default-v1.

import assert from "node:assert/strict";
import { test } from "node:test";
import { chromium, type Page } from "playwright";

const BASE = process.env.TURBO_E2E_URL ?? "http://127.0.0.1:8080";
const OUT = "#entities-out";

// Ten cities in 24 mentions; Paris is named most often (five times).
const TEXT = [
  "The night train leaves Paris after supper and reaches Berlin before breakfast. From Berlin the line runs east to Warsaw, where most travellers change. Paris and Berlin have been joined by rail since the 1840s.",
  "In Vienna the old station was rebuilt, and trains from Vienna now reach Budapest in under three hours. Budapest in turn sends sleepers south to Belgrade and on to Athens. Few people ride all the way from Paris to Athens, but it can still be done.",
  "London sits apart from all this. The tunnel carries trains from London to Paris, and from Paris a traveller can be in Berlin, Vienna or Rome by the next morning. Rome and Athens were once linked by ferry and rail through Brindisi.",
].join("\n\n");

/** Opens Analyze, analyzes `text` with `profile` and waits for the document. */
async function analyze(page: Page, profile: string, text: string = TEXT): Promise<void> {
  await page.goto(`${BASE}/#analyze`);
  await page.waitForSelector("#analyze-profile option", { state: "attached" });
  await page.fill("#analyze-text", text);
  await page.selectOption("#analyze-profile", profile);
  await page.click("#analyze-run");
  await page.waitForSelector("#analyze-window .doc");
}

test("the entity views draw a node and a strip per entity and say what is pointed at", async () => {
  const browser = await chromium.launch();
  try {
    const page = await (await browser.newContext({ viewport: { width: 1400, height: 1000 } })).newPage();
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await analyze(page, "default-v1");
    await page.waitForSelector(`${OUT} svg .node`);

    // One node per entity however often it is mentioned, the most mentioned the largest.
    assert.equal(((await page.locator(`${OUT} > p`).first().textContent()) ?? "").startsWith("10 entities from 24 mentions"), true);
    assert.equal(await page.locator(`${OUT} svg .node`).count(), 10);
    const radii = await page.locator(`${OUT} svg .node .mark`).evaluateAll((all) => all.map((c) => Number(c.getAttribute("r"))));
    assert.equal(radii[0], Math.max(...radii));
    assert.equal(radii[0] > radii[9], true);
    assert.equal(((await page.locator(`${OUT} .legend .swatch`).first().textContent()) ?? "").trim(), "location");
    // Entities named in one sentence are joined.
    assert.equal((await page.locator(`${OUT} svg .together`).count()) > 0, true);

    // Pointing at an entity says what it is and what it goes with, and lights its strip.
    await page.locator(`${OUT} svg .node`).first().hover();
    await page.waitForSelector(`${OUT} svg .strip.lit`);
    const info = (await page.locator(`${OUT} .info`).textContent()) ?? "";
    assert.equal(info.startsWith("Paris (location), 5 mentions"), true, info);
    assert.equal(info.includes("Berlin"), true, info);
    assert.equal(((await page.locator(`${OUT} svg .strip.lit .row-name`).textContent()) ?? "").startsWith("Paris"), true);

    // The strips: a row per entity, a mark per mention, the chunks as a band; a mark shows its sentence.
    assert.equal(await page.locator(`${OUT} svg .strip`).count(), 10);
    assert.equal(await page.locator(`${OUT} svg .mention`).count(), 24);
    assert.equal((await page.locator(`${OUT} svg .band`).count()) > 0, true);
    await page.locator(`${OUT} svg .strip`).nth(1).locator(".mention").first().hover();
    await page.waitForFunction((out) => (document.querySelector(`${out} .info`)?.textContent ?? "").startsWith("Berlin (location) at"), OUT);
    assert.equal(((await page.locator(`${OUT} .info`).textContent()) ?? "").includes("The night train leaves Paris"), true);

    // The same numbers as a table, with the copy and CSV buttons.
    await page.locator(`${OUT} details.tables summary`).click();
    assert.equal((await page.locator(`${OUT} details.tables table`).count()) > 0, true);
    assert.equal(await page.locator(`${OUT} details.tables .take`).count(), 1);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});

test("past thirty entities the thirty most mentioned are drawn, and the page says so", async () => {
  // Thirty-six cities, each named once, and Paris twice more.
  const cities = ["Paris", "Berlin", "Warsaw", "Vienna", "Budapest", "Belgrade", "Athens", "London", "Rome", "Madrid", "Lisbon", "Dublin", "Oslo", "Stockholm", "Helsinki", "Copenhagen", "Amsterdam", "Brussels", "Prague", "Bratislava", "Zagreb", "Sofia", "Bucharest", "Kyiv", "Riga", "Vilnius", "Tallinn", "Ankara", "Cairo", "Nairobi", "Tokyo", "Seoul", "Beijing", "Canberra", "Ottawa", "Lima"];
  const text = `${cities.map((c, i) => `Stop ${i + 1} on the long way round is ${c}.`).join(" ")} The journey ends in Paris, because every journey of this kind ends in Paris.`;
  const browser = await chromium.launch();
  try {
    const page = await (await browser.newContext({ viewport: { width: 1400, height: 1000 } })).newPage();
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await analyze(page, "default-v1", text);
    await page.waitForSelector(`${OUT} svg .node`);
    const summary = (await page.locator(`${OUT} > p`).first().textContent()) ?? "";
    const found = Number(summary.split(" ")[0]);
    assert.equal(found > 30, true, summary);
    assert.equal(summary.includes("The 30 most mentioned are drawn."), true, summary);
    assert.equal(await page.locator(`${OUT} svg .node`).count(), 30);
    assert.equal(await page.locator(`${OUT} svg .strip`).count(), 30);
    // The most mentioned comes first and is among those drawn.
    assert.equal(((await page.locator(`${OUT} svg .strip .row-name`).first().textContent()) ?? "").startsWith("Paris"), true);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});

test("a graph with no entity layer says how to get one", async () => {
  const browser = await chromium.launch();
  try {
    const page = await (await browser.newContext({ viewport: { width: 1400, height: 1000 } })).newPage();
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await analyze(page, "minimal-v1");
    await page.waitForSelector(`${OUT} :text("No entities in this graph")`);
    assert.equal(await page.locator(`${OUT} svg .node`).count(), 0);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});
