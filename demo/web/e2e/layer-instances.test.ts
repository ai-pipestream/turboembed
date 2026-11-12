// Real browser and server regression for #2240. No RPC interception.
import assert from "node:assert/strict";
import { test } from "node:test";
import { chromium } from "playwright";

const BASE = process.env.TURBO_E2E_URL ?? "http://127.0.0.1:8080";

test("Analyze selects default and named boundary instances independently", async () => {
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    await page.goto(`${BASE}/#analyze`);
    await page.waitForSelector("#analyze-profile option", { state: "attached" });
    await page.click("#analyze-run");
    await page.waitForSelector("#analyze-window .doc");
    const chips = page.locator("#analyze-window .chips button");
    const labels = await chips.allTextContents();
    if (process.env.TURBO_E2E_SHOTS) await page.screenshot({ path: `${process.env.TURBO_E2E_SHOTS}/layer-instances.png`, fullPage: true });
    assert.equal(labels.filter((s) => /^turbo:boundaries \d+$/.test(s)).length, 1, "default boundary has one label");
    assert.equal(labels.filter((s) => /^turbo:boundaries \[lexical\] /.test(s)).length, 1, "named boundary includes instance");
    const plain = chips.filter({ hasText: /^turbo:boundaries \d+$/ });
    const named = chips.filter({ hasText: /^turbo:boundaries \[lexical\] \d+$/ });
    const selected = () => page.locator("#analyze-window .chips button.on").allTextContents();
    for (const [button, expected] of [[plain, [(await plain.textContent())!]], [named, [(await plain.textContent())!, (await named.textContent())!]], [plain, [(await named.textContent())!]], [named, []]] as const) {
      await button.click();
      await page.waitForFunction((labels) => {
        const actual = [...document.querySelectorAll("#analyze-window .chips button.on")].map((b) => b.textContent);
        return actual.length === labels.length && labels.every((label) => actual.includes(label));
      }, expected);
      assert.deepEqual((await selected()).sort(), [...expected].sort());
    }
    assert.equal(await page.locator("#analyze-window .doc").textContent(), await page.inputValue("#analyze-text"));
  } finally {
    await browser.close();
  }
});

test("Analyze filters the named/default token graph through the real server", async () => {
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    await page.goto(`${BASE}/#analyze`);
    await page.click("#analyze-wire summary");
    assert.equal(Boolean(process.env.TURBO_E2E_LAYER_GRAPH), true, "scripts/e2e-web.sh supplies the Rust-generated graph");
    await page.setInputFiles("#wire-file", process.env.TURBO_E2E_LAYER_GRAPH!);
    await page.waitForSelector("#analyze-window .doc");
    const chips = page.locator("#analyze-window .chips button");
    const plain = chips.filter({ hasText: /^turbo:tokens 1$/ });
    const named = chips.filter({ hasText: /^turbo:tokens \[gold\] 1$/ });
    const marks = page.locator("#analyze-window .doc mark");
    assert.deepEqual(await marks.allTextContents(), ["é", "😀"]);
    for (const [button, expected] of [[plain, ["é"]], [named, ["é", "😀"]], [plain, ["😀"]], [named, ["é", "😀"]]] as const) {
      await button.click();
      await page.waitForFunction((expected) => {
        const marks = [...document.querySelectorAll("#analyze-window .doc mark")].map((m) => m.textContent);
        return marks.length === expected.length && marks.every((text, i) => text === expected[i]);
      }, expected);
      assert.deepEqual(await marks.allTextContents(), expected);
      assert.equal(await page.locator("#analyze-window .doc").textContent(), "é 😀");
    }
  } finally {
    await browser.close();
  }
});
