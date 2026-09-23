// SPDX-License-Identifier: Apache-2.0
//
// The Tokenize panel. The mock bundles declare a "mock" tokenizer with no
// tokenizer.json, so the suite starts the app with the committed MiniLM
// tokenizer bundle as the embedding model's tokenizer; the ids below are that
// WordPiece vocabulary's.
import { expect, test } from "@playwright/test";
import { DEFAULT_TOKENIZE_TEXT, open, tab, tokenize, watchPageErrors } from "./app";

test("the tokenizer line names the bundle the tokenizer comes from", async ({ page, request }) => {
    const errors = watchPageErrors(page);
    await open(page);
    await tab(page, "tokenize");

    const models = await (await request.get("/api/v1/models")).json();
    const embedder = models.find((m: any) => m.task === "EMBED");
    await expect(page.locator("#tokenizer")).toHaveText(`tokenizer of ${embedder.tokenizer_bundle}`);
    expect(embedder.tokenizer_bundle, "the suite expects --turbo.tokenizer-bundle").toContain("minilm-tokenizer");
    expect(errors).toEqual([]);
});

test("tokenizing the default text shows the pieces with their ids", async ({ page }) => {
    const errors = watchPageErrors(page);
    await open(page);
    await tab(page, "tokenize");

    await expect(page.locator("#tok-text")).toHaveValue(DEFAULT_TOKENIZE_TEXT);
    await tokenize(page);

    await expect(page.locator("#tok-status")).toHaveText(
        /^1 text, \d+ tokens in [\d.]+ ms with the wordpiece tokenizer \(vocab \d+, max_seq \d+\), [\d.]+ ms round trip$/,
    );

    const chips = await page.locator("#tokens .tok").allInnerTexts();
    expect(chips.length, "no token chips were rendered").toBeGreaterThan(2);
    // Special tokens are on by default and are marked as such.
    expect(chips[0]).toContain("[CLS]");
    expect(chips.at(-1)).toContain("[SEP]");
    expect(await page.locator("#tokens .tok.special").count()).toBe(2);
    // The card header repeats the text and the live token count.
    await expect(page.locator("#tokens .src")).toHaveText(`1. ${DEFAULT_TOKENIZE_TEXT} (${chips.length} tokens)`);
    expect(errors).toEqual([]);
});

test("turning the special tokens off drops them from the ids", async ({ page }) => {
    const errors = watchPageErrors(page);
    await open(page);
    await tab(page, "tokenize");

    await tokenize(page);
    const withSpecials = await page.locator("#tokens .tok").count();

    await page.locator("#tok-specials").uncheck();
    await tokenize(page);
    const without = await page.locator("#tokens .tok").count();

    expect(without, "add_special_tokens=false did not change the ids").toBe(withSpecials - 2);
    expect(await page.locator("#tokens .tok.special").count()).toBe(0);
    expect(errors).toEqual([]);
});

test("a second text is tokenized as its own row", async ({ page }) => {
    const errors = watchPageErrors(page);
    await open(page);
    await tab(page, "tokenize");

    await page.locator("#tok-text").fill("a brown dog\nthe stock market closed higher");
    await tokenize(page);
    await expect(page.locator("#tokens .tokrow")).toHaveCount(2);
    await expect(page.locator("#tok-status")).toContainText("2 texts");
    expect(errors).toEqual([]);
});
