// The Vocabulary tab's learning (#2354) and the Search tab's collection
// form (#2353), in a real browser against a server and demo of this
// file's own (stack.ts) that forward the learning; nothing is stubbed.

import assert from "node:assert/strict";
import { test } from "node:test";
import { chromium, type Browser, type Page } from "playwright";
import { Stack } from "./stack.js";

async function opened(browser: Browser, url: string, tab: string): Promise<Page> {
  const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
  await page.goto(`${url}/#${tab}`);
  await page.waitForSelector("#server.status.up");
  return page;
}

/** Learns from the typed documents and gives the count of terms the note reports. */
async function learned(page: Page): Promise<number> {
  await page.evaluate(() => (document.getElementById("vocab-note") as HTMLElement).replaceChildren());
  await page.click("#vocab-learn button[type=submit]");
  await page.waitForFunction(() => {
    const note = document.getElementById("vocab-note") as HTMLElement;
    return !note.hidden && !note.classList.contains("busy") && (note.textContent ?? "").length > 0;
  });
  const note = (await page.textContent("#vocab-note")) ?? "";
  const terms = /^Learned \S+: (\d+) terms /.exec(note);
  assert.notEqual(terms, null, `the note reports what was learned: ${note}`);
  return Number(terms?.[1]);
}

test("drop stopwords learns without the language's stopwords", async () => {
  const stack = await Stack.start({ admin: true, allow: ["/turbo.vocabulary.v1.TurboVocabularyService/LearnVocabulary"] });
  const browser = await chromium.launch();
  try {
    const page = await opened(browser, stack.url, "vocabulary");
    await page.waitForSelector("#vocab-learn:not(.off)");
    // Six words, three of them English stopwords.
    await page.fill("#vocab-docs", "the engine and the loom\n\nthe loom weaves and the engine");
    assert.equal(await page.inputValue("#vocab-learn-language"), "en");
    const all = await learned(page);
    assert.equal(all, 5);
    await page.check("#vocab-stopwords");
    const kept = await learned(page);
    assert.equal(kept, 3, "the, and are dropped: engine, loom, weaves stay");

    // A language with no bundled list is the server's refusal, by name.
    await page.fill("#vocab-learn-language", "tlh");
    await page.click("#vocab-learn button[type=submit]");
    await page.waitForSelector("#vocab-note.error:has-text('no bundled stopword list for language')");
    // Without a language there is no list to drop: the page says so and sends nothing.
    await page.fill("#vocab-learn-language", "");
    await page.click("#vocab-learn button[type=submit]");
    await page.waitForSelector("#vocab-note.error:has-text('Dropping stopwords needs a language')");
  } finally {
    await browser.close();
    await stack.close();
  }
});

test("the collection form's fields do not run into each other", async () => {
  const stack = await Stack.start();
  const browser = await chromium.launch();
  try {
    for (const width of [1280, 1024, 700, 390]) {
      const page = await (await browser.newContext({ viewport: { width, height: 900 } })).newPage();
      await page.goto(`${stack.url}/#search`);
      await page.waitForSelector("#tab-search:not([hidden]) #coll-set");
      // Each label's own box, and the box of every control in it: a label
      // holds its control, and no two labels share any of the page.
      const boxes = await page.locator("#coll-set > label, #coll-set > button").evaluateAll((els) =>
        els.map((el) => {
          const r = el.getBoundingClientRect();
          const inner = [...el.querySelectorAll("input")].map((i) => i.getBoundingClientRect());
          const right = Math.max(r.right, ...inner.map((i) => i.right));
          const range = document.createRange();
          range.selectNodeContents(el);
          const text = range.getBoundingClientRect();
          return { name: (el.textContent ?? "").trim(), left: Math.min(r.left, text.left), right: Math.max(right, text.right), top: r.top, bottom: r.bottom };
        }),
      );
      for (const [i, a] of boxes.entries()) {
        for (const b of boxes.slice(i + 1)) {
          const apart = a.right <= b.left + 0.5 || b.right <= a.left + 0.5 || a.bottom <= b.top + 0.5 || b.bottom <= a.top + 0.5;
          assert.equal(apart, true, `at ${width} px "${a.name}" and "${b.name}" overlap: ${JSON.stringify([a, b])}`);
        }
      }
      await page.close();
    }
  } finally {
    await browser.close();
    await stack.close();
  }
});
