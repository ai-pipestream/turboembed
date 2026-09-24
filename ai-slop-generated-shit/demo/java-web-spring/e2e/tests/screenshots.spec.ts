// SPDX-License-Identifier: Apache-2.0
//
// The README screenshots, written into demo/java-web-spring/docs/screenshots.
// Opt-in, because it overwrites committed files:
//
//   npm run screenshots                       # the mock-bundle images
//   SHOT_TARGET=real E2E_BASE_URL=http://127.0.0.1:8092 npm run screenshots
//                                             # page-minilm.png, against an app
//                                             # already serving a real model
//   SHOT_TARGET=generate E2E_BASE_URL=http://127.0.0.1:8094 npm run screenshots
//                                             # summary-qwen.png, against an app
//                                             # already serving a real generative model
import { expect, test } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { benchmarks, DEFAULT_TEXTS, embed, open, rerank, setTexts, summarize, tab, tokenize } from "./app";

const outDir = path.join(__dirname, "..", "..", "docs", "screenshots");
const TARGET = process.env.SHOT_TARGET ?? "mock";

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

test("the mock-bundle images", async ({ page }) => {
    test.skip(TARGET !== "mock", "SHOT_TARGET selects a real-model screenshot instead");
    fs.mkdirSync(outDir, { recursive: true });
    await open(page);

    await expect(page.locator("#texts")).toHaveValue(DEFAULT_TEXTS.join("\n"));
    await embed(page);
    await expect(page.locator("#matrix tr")).toHaveCount(DEFAULT_TEXTS.length + 1);

    const pagePng = path.join(outDir, "page.png");
    await page.screenshot({ path: pagePng, fullPage: true, scale: "css" });
    const matrixPng = path.join(outDir, "matrix.png");
    await page.locator("#matrix").screenshot({ path: matrixPng, scale: "css" });

    await tab(page, "rerank");
    await rerank(page);
    const rerankPng = path.join(outDir, "rerank.png");
    await page.locator("#panel-rerank").screenshot({ path: rerankPng, scale: "css" });

    await tab(page, "tokenize");
    await tokenize(page);
    const tokenizePng = path.join(outDir, "tokenize.png");
    await page.locator("#panel-tokenize").screenshot({ path: tokenizePng, scale: "css" });

    await tab(page, "devices");
    await expect(page.locator("#devices .card").first()).toBeVisible();
    const devicesPng = path.join(outDir, "devices.png");
    await page.locator("#panel-devices").screenshot({ path: devicesPng, scale: "css" });

    await benchmarks(page);
    // One comparison opened, so the image carries the summary, the cells of a
    // comparison and the first throughput table. The whole panel is every
    // committed receipt, which is far too tall for a README image.
    await page.locator('#bench-cells details[data-file="compare-cuda-rtx4080-embed-2026-09-22.json"] summary').click();
    // Page coordinates of the panel down to the end of its first throughput
    // table, read in one go so no scrolling happens between the two rectangles.
    const clip = await page.evaluate(() => {
        const panel = document.getElementById("panel-benchmarks")!.getBoundingClientRect();
        const table = document.querySelector("#bench-throughput table")!.getBoundingClientRect();
        return {
            x: panel.x + window.scrollX,
            y: panel.y + window.scrollY,
            width: panel.width,
            height: table.bottom - panel.top,
        };
    });
    const benchmarksPng = path.join(outDir, "benchmarks.png");
    await page.screenshot({ path: benchmarksPng, fullPage: true, scale: "css", clip });

    for (const file of [pagePng, matrixPng, rerankPng, tokenizePng, devicesPng]) underLimit(file);
    // The benchmarks panel is a table of every committed receipt, so it is the
    // one image with a larger budget.
    underLimit(benchmarksPng, 512);
});

test("page-minilm.png on a real model", async ({ page }) => {
    test.skip(TARGET !== "real", "set SHOT_TARGET=real and E2E_BASE_URL to an app serving a real model");
    fs.mkdirSync(outDir, { recursive: true });
    await open(page);

    const device = await page.locator("#device").innerText();
    expect(device, "this screenshot is for a real model, not the mock provider").not.toContain("turbo/mock-embedding");

    await setTexts(page, REAL_TEXTS);
    await embed(page);
    await expect(page.locator("#matrix tr")).toHaveCount(REAL_TEXTS.length + 1);
    console.log(`device: ${device}`);
    console.log(`status: ${(await page.locator("#status").innerText()).trim()}`);

    const out = path.join(outDir, "page-minilm.png");
    await page.screenshot({ path: out, fullPage: true, scale: "css" });
    underLimit(out);
});

test("summary-qwen.png on a real generative model", async ({ page }) => {
    test.skip(TARGET !== "generate", "set SHOT_TARGET=generate and E2E_BASE_URL to an app serving a real generative model");
    test.setTimeout(180_000);
    fs.mkdirSync(outDir, { recursive: true });
    await open(page);
    await tab(page, "summarize");

    const generator = await page.locator("#generator").innerText();
    expect(generator, "this screenshot is for a real generative model, not the mock one").not.toContain("turbo/mock-generative");
    expect(generator, "no generative model is configured on this server").not.toContain("no generative bundle");

    // The document the page ships with, summarized as a visitor would see it.
    await summarize(page, 150_000);
    await expect(page.locator("#summary")).toBeVisible();

    const summary = ((await page.locator("#summary").textContent()) ?? "").trim();
    expect(summary.length, "the summary box is empty").toBeGreaterThan(0);
    console.log(`generator: ${generator}`);
    console.log(`status:    ${(await page.locator("#gen-status").innerText()).trim()}`);
    console.log(`summary:   ${summary}`);

    const out = path.join(outDir, "summary-qwen.png");
    await page.locator("#summarize-section").screenshot({ path: out, scale: "css" });
    underLimit(out);
});
