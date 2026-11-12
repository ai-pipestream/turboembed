// A licensed model from the catalog to a served model, in a real browser
// against the second server and demo that scripts/e2e-web.sh starts (a
// catalog of one share-alike bundle in a file:// store, a demo started
// with --user): the Models tab shows the terms still to accept, the user
// accepts the exact text shown, installs the model and its files verify,
// the Server tab hot-loads the installed directory into the running
// server, and the Semantic tab embeds with it. Nothing is stubbed.

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { chromium } from "playwright";

const BASE = process.env.TURBO_E2E_INSTALL_URL ?? "";
const INDEX = process.env.TURBO_E2E_INSTALL_INDEX ?? "";
const ID = "tiny-bert-sa";
const TERMS = "https://creativecommons.org/licenses/by-sa/4.0/legalcode";

type File = { path: string; size: number; sha256: string };

test("a share-alike model is accepted, installed, verified and hot-loaded", async () => {
  // The flow is part of the suite: without its server it fails, never skips.
  assert.notEqual(BASE, "", "TURBO_E2E_INSTALL_URL names the install demo (scripts/e2e-web.sh)");
  assert.notEqual(INDEX, "", "TURBO_E2E_INSTALL_INDEX names its catalog index");
  const files: File[] = JSON.parse(new TextDecoder().decode(readFileSync(INDEX))).artifacts[0].files;
  const browser = await chromium.launch();
  try {
    const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`${BASE}/#models`);
    await page.click("#models-catalog-load");
    const card = page.locator(".entry", { has: page.locator(".id", { hasText: new RegExp(`^${ID}$`) }) });
    await card.waitFor();
    const install = card.getByRole("button", { name: "Install" });
    assert.equal(await install.isDisabled(), true, "Install waits for the terms");
    assert.equal(await install.getAttribute("title"), "Accept the licence terms first");

    // The terms still to accept: the licence, its class and the exact text's address.
    await card.locator("summary", { hasText: "CC-BY-SA-4.0 terms to accept" }).click();
    const terms = card.locator("details", { has: page.locator("summary", { hasText: "terms to accept" }) });
    assert.match(await terms.locator("dl").innerText(), /CC-BY-SA-4\.0, CLASS_SHARE_ALIKE/);
    assert.equal(await terms.locator("a").getAttribute("href"), TERMS);
    const accept = terms.getByRole("button", { name: "Accept CC-BY-SA-4.0" });
    assert.equal(await accept.isDisabled(), true, "Accept waits for the box");
    await terms.locator("input[type=checkbox]").check();
    await accept.click();
    await page.waitForSelector("#models-note:has-text('e2e accepted CC-BY-SA-4.0')");
    const listed = await page.locator("#models-catalog").innerText();
    const accepted = listed.slice(listed.indexOf("Accepted terms"), listed.indexOf("\nCatalog"));
    assert.match(accepted, new RegExp(`CC-BY-SA-4\\.0[\\s\\S]*${ID}`), "the acceptance names the page it was made on");

    // Install, then every file on disk is the one the catalog lists.
    const again = page.locator(".entry", { has: page.locator(".id", { hasText: new RegExp(`^${ID}$`) }) });
    await again.getByRole("button", { name: "Install" }).click();
    const status = page.locator("#models-install p");
    await page.waitForFunction(() => /^Installed /.test(document.querySelector("#models-install p")?.textContent ?? ""), null, { timeout: 60_000 });
    const done = (await status.textContent()) ?? "";
    const m = done.match(new RegExp(`^Installed ${ID} 1: ${files.length} files, .* in (.+)\\.$`));
    if (!m) throw new Error(`not an install status: ${done}`);
    const dir = m[1];
    assert.equal(await page.locator("#models-install progress").evaluate((p) => (p as HTMLProgressElement).value === (p as HTMLProgressElement).max), true);
    for (const f of files) {
      const bytes = readFileSync(`${dir}/${f.path}`);
      assert.equal(bytes.length, f.size, f.path);
      assert.equal(createHash("sha256").update(bytes).digest("hex"), f.sha256, f.path);
    }
    await page.waitForSelector(`#models-catalog table >> text=${ID}`);
    await again.locator(".tag.installed").waitFor();

    // Hot-load the installed directory into the running server.
    await page.click("#tabbtn-server");
    await page.fill("#server-load-name", ID);
    await page.fill("#server-load-path", dir);
    await page.click("#server-load button[type=submit]");
    await page.waitForSelector(`#server-load-note:has-text('Loaded ${ID}')`);
    await page.waitForSelector(`#server-caps >> text=${ID}`);

    // And use it: the Semantic tab ranks passages with the new model.
    await page.click("#tabbtn-semantic");
    await page.waitForSelector(`#sem-embed option[value=${ID}]`, { state: "attached" });
    await page.selectOption("#sem-embed", ID);
    await page.click("#sem-run");
    await page.waitForSelector("#sem-sim table");
    assert.equal(await page.textContent("#sem-sim h3"), `Nearest neighbours (${ID})`);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});
