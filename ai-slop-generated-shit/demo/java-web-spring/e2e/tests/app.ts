// SPDX-License-Identifier: Apache-2.0
//
// Helpers for driving the demo page. Every wait here surfaces the server's own
// refusal message when a step does not do what the test expected, so a failure
// names the cause instead of timing out on a selector.
import { expect, type Page } from "@playwright/test";

/** The sentences index.html ships in the embed textarea. */
export const DEFAULT_TEXTS = [
    "a brown dog runs through the grass",
    "a dog is running on the lawn",
    "the stock market closed higher",
];

/** The documents index.html ships in the rerank textarea. */
export const DEFAULT_DOCUMENTS = [
    "the accelerator sustains two and a half teraflops",
    "the card draws three hundred watts under load",
    "the recipe needs two eggs and a cup of flour",
    "the library never copies a tensor to the host",
];

/** The query index.html ships with. */
export const DEFAULT_QUERY = "how fast is the accelerator";

/** The text index.html ships in the tokenize textarea. */
export const DEFAULT_TOKENIZE_TEXT = "a brown dog runs through the grass";

/** A short document that fits any generative bundle's context with room for the summary. */
export const SHORT_DOCUMENT =
    "The provider reports its capabilities honestly instead of claiming full acceleration.";

/** Uncaught page errors, collected so a test can insist the page never crashed. */
export function watchPageErrors(page: Page): string[] {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e.stack ?? e)));
    page.on("console", (m) => {
        // A 4xx from the API is logged by the browser itself; the tests that
        // provoke one assert on the error box instead, so it is not a crash.
        if (m.type() === "error" && !m.text().startsWith("Failed to load resource")) {
            errors.push(`console.error: ${m.text()}`);
        }
    });
    return errors;
}

/** Open the page and wait for the device line the app fills from GET /api/v1/models. */
export async function open(page: Page): Promise<void> {
    await page.goto("/");
    await expect(page.locator("#device")).not.toHaveText("loading device…");
}

/** Click a tab and wait for its panel. */
export async function tab(
    page: Page,
    name: "embed" | "rerank" | "tokenize" | "summarize" | "devices" | "benchmarks",
): Promise<void> {
    await page.locator(`#tab-${name}`).click();
    await expect(page.locator(`#tab-${name}`)).toHaveAttribute("aria-selected", "true");
    await expect(page.locator(`#panel-${name}`)).toBeVisible();
}

/** Open the Benchmarks panel and wait for the receipts to be drawn. */
export async function benchmarks(page: Page): Promise<void> {
    await tab(page, "benchmarks");
    await expect(page.locator("#bench-error"), "the benchmarks panel showed the server's refusal").toBeHidden();
    await expect(page.locator("#bench-table tbody tr").first()).toBeVisible();
}

/**
 * Click {@code button} and wait for either the result element or the error box.
 * Returns "ok" when the result appeared, or the server's message when it did not.
 */
async function run(page: Page, button: string, result: string, errorBox: string, timeout = 20_000): Promise<string> {
    await page.locator(button).click();
    await expect
        .poll(
            async () => {
                if (await page.locator(errorBox).isVisible()) {
                    return `refused: ${(await page.locator(errorBox).innerText()).trim()}`;
                }
                return (await page.locator(result).isVisible()) ? "ok" : "pending";
            },
            { timeout, message: `${button} never produced ${result} or ${errorBox}` },
        )
        .not.toBe("pending");
    await expect(page.locator(button)).toBeEnabled();
    if (await page.locator(errorBox).isVisible()) {
        return (await page.locator(errorBox).innerText()).trim();
    }
    return "ok";
}

/** Replace the embed textarea with one sentence per line. */
export async function setTexts(page: Page, texts: string[]): Promise<void> {
    await page.locator("#texts").fill(texts.join("\n"));
}

/** Click Embed and insist on a result table. */
export async function embed(page: Page): Promise<void> {
    const outcome = await run(page, "#embed", "#result", "#error");
    expect(outcome, "the embed the test expected to succeed was refused").toBe("ok");
}

/** Click Embed expecting a refusal, and return the text of the error box. */
export async function embedExpectingError(page: Page): Promise<string> {
    const outcome = await run(page, "#embed", "#result", "#error");
    expect(outcome, "the embed the test expected to fail succeeded").not.toBe("ok");
    return outcome;
}

/** Click Rerank and insist on a ranking table. */
export async function rerank(page: Page): Promise<void> {
    const outcome = await run(page, "#rerank", "#rerank-result", "#rerank-error");
    expect(outcome, "the rerank the test expected to succeed was refused").toBe("ok");
}

/** Click Rerank expecting a refusal, and return the text of the error box. */
export async function rerankExpectingError(page: Page): Promise<string> {
    const outcome = await run(page, "#rerank", "#rerank-result", "#rerank-error");
    expect(outcome, "the rerank the test expected to fail succeeded").not.toBe("ok");
    return outcome;
}

/** Click Tokenize and insist on token chips. */
export async function tokenize(page: Page): Promise<void> {
    const outcome = await run(page, "#tokenize", "#tok-result", "#tok-error");
    expect(outcome, "the tokenize the test expected to succeed was refused").toBe("ok");
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

/** The ranking rows the page rendered: rank, score and document. */
export async function rankingRows(page: Page): Promise<{ rank: string; score: string; document: string }[]> {
    return page.locator("#ranking tbody tr").evaluateAll((rows) =>
        rows.map((row) => {
            const cells = Array.from(row.querySelectorAll("td"), (cell) => (cell.textContent ?? "").trim());
            return { rank: cells[0], score: cells[1], document: cells[2] };
        }),
    );
}

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
