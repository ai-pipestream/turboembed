// SPDX-License-Identifier: Apache-2.0
//
// The README screenshots, written into demo/java-web-spring/docs/screenshots.
// Opt-in, because it overwrites committed files:
//
//   npm run screenshots                       # page.png, matrix.png (mock bundle)
//   SHOT_TARGET=real E2E_BASE_URL=http://127.0.0.1:8092 npm run screenshots
//                                             # page-minilm.png, against an app
//                                             # already serving a real model
import { expect, test } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { DEFAULT_TEXTS, embed, open, setTexts } from "./app";

const outDir = path.join(__dirname, "..", "..", "docs", "screenshots");
const REAL = process.env.SHOT_TARGET === "real";

// Two paraphrase pairs and one unrelated sentence, so a real model's heat map
// shows two bright blocks off the diagonal and one cold row.
const REAL_TEXTS = [
    "a brown dog runs through the grass",
    "a dog is running on the lawn",
    "the stock market closed higher",
    "equity indexes ended the day with gains",
    "she reheated last night's soup for lunch",
];

/** Screenshots go in the README, so keep them small enough to commit. */
function underLimit(file: string, limitKb = 400): void {
    const kb = fs.statSync(file).size / 1024;
    expect(kb, `${path.basename(file)} is ${kb.toFixed(0)} KB, over the ${limitKb} KB limit`).toBeLessThan(limitKb);
}

test("page.png and matrix.png on the mock bundle", async ({ page }) => {
    test.skip(REAL, "SHOT_TARGET=real captures the real-model screenshot instead");
    fs.mkdirSync(outDir, { recursive: true });
    await open(page);

    await expect(page.locator("#texts")).toHaveValue(DEFAULT_TEXTS.join("\n"));
    await embed(page);
    await expect(page.locator("#matrix tr")).toHaveCount(DEFAULT_TEXTS.length + 1);

    const pagePng = path.join(outDir, "page.png");
    await page.screenshot({ path: pagePng, fullPage: true, scale: "css" });
    const matrixPng = path.join(outDir, "matrix.png");
    await page.locator("#matrix").screenshot({ path: matrixPng, scale: "css" });

    underLimit(pagePng);
    underLimit(matrixPng);
});

test("page-minilm.png on a real model", async ({ page }) => {
    test.skip(!REAL, "set SHOT_TARGET=real and E2E_BASE_URL to an app serving a real model");
    fs.mkdirSync(outDir, { recursive: true });
    await open(page);

    const device = await page.locator("#device").innerText();
    expect(device, "this screenshot is for a real model, not the mock provider").not.toContain("turbo/mock-embedding");

    await setTexts(page, REAL_TEXTS);
    await embed(page);
    await expect(page.locator("#matrix tr")).toHaveCount(REAL_TEXTS.length + 1);

    const out = path.join(outDir, "page-minilm.png");
    await page.screenshot({ path: out, fullPage: true, scale: "css" });
    underLimit(out);
});
