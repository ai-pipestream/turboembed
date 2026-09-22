// SPDX-License-Identifier: Apache-2.0
//
// The Summarize panel against the committed mock generative bundle
// (testdata/bundles/mock/generative): deterministic "tokNNN" pieces, finish
// LENGTH at max_tokens, no hardware.
import { expect, test } from "@playwright/test";
import { SHORT_DOCUMENT, open, setDocument, summarize, summarizeExpectingError, tab, watchPageErrors } from "./app";

test("the generator line names the mock generative model", async ({ page, request }) => {
    const errors = watchPageErrors(page);
    await open(page);
    await tab(page, "summarize");

    const models = await (await request.get("/api/v1/models")).json();
    const generator = models.find((m: any) => m.task === "GENERATE");
    expect(generator, "the suite expects a generative bundle").toBeTruthy();
    expect(generator.model_id).toBe("turbo/mock-generative");

    await expect(page.locator("#generator")).toHaveText(
        `${generator.model_id} (max_seq ${generator.max_seq}) on ${generator.device.name} ` +
            `(${generator.device.provider_id}:${generator.device.ordinal})`,
    );
    await expect(page.locator("#summarize"), "Summarize is disabled although a generative model is loaded").toBeEnabled();
    await expect(page.locator("#summary")).toBeHidden();
    await expect(page.locator("#stop")).toBeHidden();
    expect(errors).toEqual([]);
});

test("Summarize streams the generated text and ends with a finish reason", async ({ page }) => {
    const errors = watchPageErrors(page);
    await open(page);
    await tab(page, "summarize");

    await setDocument(page, SHORT_DOCUMENT);
    await summarize(page);

    const status = (await page.locator("#gen-status").innerText()).trim();
    expect(status, "the status line does not report a completed run").toMatch(
        /^\d+ tokens in [\d.]+ s \([\d.]+ tok\/s\), finish LENGTH, prompt \d+ tokens, on Mock accelerator \(mock:\d+\)$/,
    );

    await expect(page.locator("#summary")).toBeVisible();
    await expect(page.locator("#gen-error")).toBeHidden();
    await expect(page.locator("#stop")).toBeHidden();
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
    await tab(page, "summarize");

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
    await tab(page, "summarize");

    await setDocument(page, "");
    expect(await summarizeExpectingError(page)).toBe("messages[].content must not be blank");
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
    await tab(page, "summarize");

    const models = await (await request.get("/api/v1/models")).json();
    const generator = models.find((m: any) => m.task === "GENERATE");
    // A document longer than the generative model's context: the refusal
    // arrives mid-stream as an "error" event, not as an HTTP status.
    await setDocument(page, `${SHORT_DOCUMENT} `.repeat(generator.max_seq));
    const refusal = await summarizeExpectingError(page);
    expect(refusal, "the refusal does not name the capacity limit").toContain("TURBO_E_CAPACITY");
    expect(refusal).toContain(`max_seq is ${generator.max_seq}`);
    await expect(page.locator("#gen-status")).toHaveText("");
    expect(errors, "the page threw while handling the refusal").toEqual([]);
});

test("Stop is offered while a run is in flight and leaves the panel usable", async ({ page }) => {
    const errors = watchPageErrors(page);
    await open(page);
    await tab(page, "summarize");
    await expect(page.locator("#stop"), "Stop is offered only during a run").toBeHidden();

    await setDocument(page, SHORT_DOCUMENT);
    await page.locator("#summarize").click();
    // The mock model produces its 160 tokens faster than a click, so this may
    // cut the stream or arrive after it finished. Both outcomes are checked
    // below; the cancellation path itself is covered deterministically by
    // GenerateApiTest.aSinkThatStopsCancelsTheGenerationOnTheDevice.
    await page.locator("#stop").click({ timeout: 2_000 }).catch(() => undefined);

    await expect
        .poll(async () => (await page.locator("#gen-status").innerText()).trim(), { timeout: 30_000 })
        .toMatch(/^(stopped after \d+ tokens|\d+ tokens in .*finish LENGTH.*)$/);
    await expect(page.locator("#gen-error"), "stopping is not a failure").toBeHidden();
    await expect(page.locator("#summarize")).toBeEnabled();
    await expect(page.locator("#stop")).toBeHidden();

    // Whichever way it ended, the slot went back and the next run completes.
    await summarize(page);
    await expect(page.locator("#gen-status")).toContainText("finish LENGTH");
    expect(errors).toEqual([]);
});
