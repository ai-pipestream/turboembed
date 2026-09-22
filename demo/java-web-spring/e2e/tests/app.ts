// SPDX-License-Identifier: Apache-2.0
//
// Helpers for driving the demo page. Every wait here surfaces the server's own
// refusal message when a step does not do what the test expected, so a failure
// names the cause instead of timing out on a selector.
import { expect, type Page } from "@playwright/test";

/** The sentences index.html ships in the textarea. */
export const DEFAULT_TEXTS = [
    "a brown dog runs through the grass",
    "a dog is running on the lawn",
    "the stock market closed higher",
];

/** Uncaught page errors, collected so a test can insist the page never crashed. */
export function watchPageErrors(page: Page): string[] {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e.stack ?? e)));
    page.on("console", (m) => {
        // A 4xx from /api/embed is logged by the browser itself; the tests that
        // provoke one assert on the error box instead, so it is not a crash.
        if (m.type() === "error" && !m.text().startsWith("Failed to load resource")) {
            errors.push(`console.error: ${m.text()}`);
        }
    });
    return errors;
}

/** Open the page and wait for the device line the app fills from GET /api/info. */
export async function open(page: Page): Promise<void> {
    await page.goto("/");
    await expect(page.locator("#device")).not.toHaveText("loading device…");
}

/** Replace the textarea contents with one sentence per line. */
export async function setTexts(page: Page, texts: string[]): Promise<void> {
    await page.locator("#texts").fill(texts.join("\n"));
}

/**
 * Click Embed and wait for the result table. If the app shows the error box
 * instead, the assertion fails with the server's message verbatim.
 */
export async function embed(page: Page): Promise<void> {
    await page.locator("#embed").click();
    await expect
        .poll(
            async () => {
                if (await page.locator("#error").isVisible()) {
                    return `server refused: ${(await page.locator("#error").innerText()).trim()}`;
                }
                return (await page.locator("#result").isVisible()) ? "ok" : "pending";
            },
            { timeout: 15_000, message: "the embed request never produced a result table" },
        )
        .toBe("ok");
    await expect(page.locator("#embed")).toBeEnabled();
}

/** Click Embed expecting a refusal, and return the text of the error box. */
export async function embedExpectingError(page: Page): Promise<string> {
    await page.locator("#embed").click();
    const error = page.locator("#error");
    await expect(error, "the app showed no error box").toBeVisible({ timeout: 15_000 });
    await expect(page.locator("#embed"), "the Embed button stayed disabled after the refusal").toBeEnabled();
    return (await error.innerText()).trim();
}

/** The similarity cells as the page rendered them, one array per row. */
export async function matrixCells(page: Page): Promise<string[][]> {
    return page.locator("#matrix tr").evaluateAll((rows) =>
        rows
            .map((row) => Array.from(row.querySelectorAll("td.cell"), (cell) => (cell.textContent ?? "").trim()))
            .filter((cells) => cells.length > 0),
    );
}

/** The row labels ("1. a brown dog…") down the left of the table. */
export async function matrixRowLabels(page: Page): Promise<string[]> {
    return page.locator("#matrix tr th.text").allInnerTexts();
}

/** A short document that fits any generative bundle's context with room for the summary. */
export const SHORT_DOCUMENT =
    "The provider reports its capabilities honestly instead of claiming full acceleration.";

/** Replace the summarize textarea. */
export async function setDocument(page: Page, text: string): Promise<void> {
    await page.locator("#document").fill(text);
}

/**
 * Click Summarize and wait for the stream to finish. If the run fails instead,
 * the assertion carries the message the server put in the error box.
 */
export async function summarize(page: Page, timeout = 60_000): Promise<void> {
    await page.locator("#summarize").click();
    await expect
        .poll(
            async () => {
                if (await page.locator("#gen-error").isVisible()) {
                    return `server refused: ${(await page.locator("#gen-error").innerText()).trim()}`;
                }
                const status = (await page.locator("#gen-status").innerText()).trim();
                return status.includes("finish ") ? "done" : `pending: ${status}`;
            },
            { timeout, message: "the summary stream never reached a done event" },
        )
        .toBe("done");
    await expect(page.locator("#summarize")).toBeEnabled();
}

/** Click Summarize expecting a refusal, and return the text of the error box. */
export async function summarizeExpectingError(page: Page, timeout = 60_000): Promise<string> {
    await page.locator("#summarize").click();
    const error = page.locator("#gen-error");
    await expect(error, "the app showed no summarize error box").toBeVisible({ timeout });
    await expect(page.locator("#summarize"), "the Summarize button stayed disabled after the refusal").toBeEnabled();
    return (await error.innerText()).trim();
}
