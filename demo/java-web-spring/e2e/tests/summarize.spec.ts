// SPDX-License-Identifier: Apache-2.0
//
// The Summarize section against the committed mock generative bundle
// (testdata/bundles/mock/generative): deterministic "tokNNN" pieces, finish
// LENGTH at maxNewTokens, no hardware.
import { expect, test } from "@playwright/test";
import { SHORT_DOCUMENT, open, setDocument, summarize, summarizeExpectingError, watchPageErrors } from "./app";

test("the generator line names the mock generative model", async ({ page, request }) => {
    const errors = watchPageErrors(page);
    await open(page);

    const info = await (await request.get("/api/info")).json();
    expect(info.generate, "the suite expects a generative bundle: --turbo.generate-bundle=<dir>").not.toBeNull();
    expect(info.generate.modelId).toBe("turbo/mock-generative");
    expect(info.generate.providerId).toBe("mock");

    await expect(page.locator("#generator")).toHaveText(
        `${info.generate.modelId} (max_seq ${info.generate.maxSeq}) on ${info.generate.deviceName} — ` +
            `${info.generate.providerId}:${info.generate.ordinal}`,
    );
    await expect(page.locator("#summarize"), "Summarize is disabled although a generative model is loaded").toBeEnabled();
    await expect(page.locator("#summary")).toBeHidden();
    expect(errors).toEqual([]);
});

test("Summarize streams the generated text and ends with a finish reason", async ({ page }) => {
    const errors = watchPageErrors(page);
    await open(page);

    await setDocument(page, SHORT_DOCUMENT);
    await summarize(page);

    const status = (await page.locator("#gen-status").innerText()).trim();
    expect(status, "the status line does not report a completed run").toMatch(
        /^\d+ tokens in [\d.]+ s \([\d.]+ tok\/s\), finish LENGTH, prompt \d+ tokens$/,
    );

    await expect(page.locator("#summary")).toBeVisible();
    await expect(page.locator("#gen-error")).toBeHidden();
    const summary = (await page.locator("#summary").textContent()) ?? "";
    expect(summary.length, "the summary box is empty after a completed run").toBeGreaterThan(0);
    // The mock generative model emits one "tokNNN" piece per step.
    expect(summary.trim()).toMatch(/^tok\d+( tok\d+)*$/);

    // Every streamed piece reached the page: the status counts the same tokens.
    const generated = Number(status.split(" ")[0]);
    expect(summary.trim().split(/\s+/)).toHaveLength(generated);
    // The page asks for 160 new tokens and the mock stops there, hence LENGTH.
    expect(generated).toBe(160);
    expect(errors).toEqual([]);
});

test("the same document summarizes to the same text", async ({ page }) => {
    const errors = watchPageErrors(page);
    await open(page);

    await setDocument(page, SHORT_DOCUMENT);
    await summarize(page);
    const first = (await page.locator("#summary").textContent()) ?? "";

    await summarize(page);
    const second = (await page.locator("#summary").textContent()) ?? "";

    expect(first.length).toBeGreaterThan(0);
    expect(second, "the mock generative model is not deterministic").toBe(first);
    expect(errors).toEqual([]);
});

test("an empty document shows the server's refusal and leaves the page working", async ({ page }) => {
    const errors = watchPageErrors(page);
    await open(page);

    await setDocument(page, "");
    expect(await summarizeExpectingError(page)).toBe("no text");
    await expect(page.locator("#gen-status")).toHaveText("");
    expect(errors, "the page threw while handling the refusal").toEqual([]);

    // The page recovers: a real document still streams and the error box clears.
    await setDocument(page, SHORT_DOCUMENT);
    await summarize(page);
    await expect(page.locator("#gen-error")).toBeHidden();
    expect(errors).toEqual([]);
});

test("a prompt over the generative model's max_seq shows the library's refusal", async ({ page, request }) => {
    const errors = watchPageErrors(page);
    await open(page);

    const info = await (await request.get("/api/info")).json();
    // The page ships a long default document; the mock generative bundle's
    // max_seq is small, so this is the mid-stream "error" event path.
    const refusal = await summarizeExpectingError(page);
    expect(refusal, "the refusal does not name the capacity limit").toContain("TURBO_E_CAPACITY");
    expect(refusal).toContain(`max_seq is ${info.generate.maxSeq}`);
    await expect(page.locator("#gen-status")).toHaveText("");
    expect(errors, "the page threw while handling the refusal").toEqual([]);
});
