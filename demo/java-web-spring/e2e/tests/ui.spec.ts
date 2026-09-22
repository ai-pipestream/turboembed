// SPDX-License-Identifier: Apache-2.0
//
// The page against the committed mock bundle: deterministic 8-dim vectors, no
// hardware. Every check is a hard assertion; a refusal from the server is
// reported with its own message rather than swallowed.
import { expect, test } from "@playwright/test";
import { DEFAULT_TEXTS, embed, embedExpectingError, matrixCells, matrixRowLabels, open, setTexts, watchPageErrors } from "./app";

test("the device line names the mock provider and the mock model", async ({ page, request }) => {
    const errors = watchPageErrors(page);
    await open(page);

    const info = await (await request.get("/api/info")).json();
    expect(info.providerId, "the suite expects the mock provider").toBe("mock");
    expect(info.modelId).toBe("turbo/mock-embedding");

    const device = (await page.locator("#device").innerText()).trim();
    expect(device).toBe(
        `${info.modelId} (dim ${info.dim}, max_seq ${info.maxSeq}) on ${info.deviceName} — ` +
            `${info.providerId}:${info.ordinal} ${info.deviceKind}, runtime ${info.runtimeVersion}` +
            (info.fullyAccelerated ? "" : " — host stages: tokenize"),
    );
    expect(device).toContain("mock");
    expect(errors).toEqual([]);
});

test("embedding the three default sentences renders a 3x3 matrix", async ({ page }) => {
    const errors = watchPageErrors(page);
    await open(page);

    await expect(page.locator("#texts")).toHaveValue(DEFAULT_TEXTS.join("\n"));
    await expect(page.locator("#result")).toBeHidden();
    await embed(page);

    await expect(page.locator("#status")).toHaveText(/^3 × \d+ in [\d.]+ ms$/);

    const cells = await matrixCells(page);
    expect(cells).toHaveLength(3);
    for (const row of cells) expect(row).toHaveLength(3);
    for (let i = 0; i < 3; i++) {
        expect(cells[i][i], `the diagonal at ${i + 1},${i + 1} is not 1.000`).toBe("1.000");
        for (let j = 0; j < 3; j++) {
            const v = Number(cells[i][j]);
            expect(Number.isNaN(v), `cell ${i + 1},${j + 1} is not a number: ${cells[i][j]}`).toBe(false);
            expect(v).toBeGreaterThanOrEqual(-1);
            expect(v).toBeLessThanOrEqual(1);
            expect(cells[i][j], `cell ${i + 1},${j + 1} is not to three decimals`).toMatch(/^-?\d\.\d{3}$/);
            expect(cells[i][j], "the matrix is not symmetric").toBe(cells[j][i]);
        }
    }

    expect(await matrixRowLabels(page)).toEqual(DEFAULT_TEXTS.map((t, i) => `${i + 1}. ${t}`));

    // Header row: a blank corner plus one column number per sentence.
    expect(await page.locator("#matrix tr").first().locator("th").allInnerTexts()).toEqual(["", "1", "2", "3"]);

    const vectors = ((await page.locator("#vectors").textContent()) ?? "").trim().split("\n");
    expect(vectors).toHaveLength(3);
    for (const line of vectors) expect(line).toMatch(/^\d+: \[-?\d\.\d{4}(, -?\d\.\d{4})*(, …)?\]$/);
    expect(errors).toEqual([]);
});

test("the same sentence twice gives identical rows", async ({ page }) => {
    const errors = watchPageErrors(page);
    await open(page);

    const [a, , b] = DEFAULT_TEXTS;
    await setTexts(page, [a, b, a, b]);
    await embed(page);

    const cells = await matrixCells(page);
    expect(cells).toHaveLength(4);
    expect(cells[0], "row 1 and row 3 are the same sentence but differ").toEqual(cells[2]);
    expect(cells[1], "row 2 and row 4 are the same sentence but differ").toEqual(cells[3]);
    expect(cells[0][2], "a sentence against its own duplicate is not 1.000").toBe("1.000");
    expect(cells[1][3], "a sentence against its own duplicate is not 1.000").toBe("1.000");

    const vectors = ((await page.locator("#vectors").textContent()) ?? "").trim().split("\n");
    expect(vectors[0].replace(/^1:/, ""), "the vectors for the duplicated sentence differ").toBe(vectors[2].replace(/^3:/, ""));
    expect(vectors[1].replace(/^2:/, ""), "the vectors for the duplicated sentence differ").toBe(vectors[3].replace(/^4:/, ""));
    expect(errors).toEqual([]);
});

test("an empty textarea shows the server's refusal and leaves the page working", async ({ page }) => {
    const errors = watchPageErrors(page);
    await open(page);

    await setTexts(page, []);
    expect(await embedExpectingError(page)).toBe("no texts");
    await expect(page.locator("#result")).toBeHidden();
    await expect(page.locator("#status")).toHaveText("");
    expect(errors, "the page threw while handling the refusal").toEqual([]);

    // The page recovers: the defaults still embed and the error box goes away.
    await setTexts(page, DEFAULT_TEXTS);
    await embed(page);
    await expect(page.locator("#error")).toBeHidden();
    expect(errors).toEqual([]);
});

test("more sentences than the server's batch shows the batch refusal", async ({ page, request }) => {
    const errors = watchPageErrors(page);
    await open(page);

    const info = await (await request.get("/api/info")).json();
    expect(info.maxBatch, "the server reports no batch limit").toBeGreaterThan(0);
    expect(17, "this test needs more lines than the server's batch").toBeGreaterThan(info.maxBatch);

    await setTexts(page, Array.from({ length: 17 }, (_, i) => `sentence number ${i + 1}`));
    expect(await embedExpectingError(page)).toBe(`request has 17 texts but the server's batch is ${info.maxBatch}`);
    await expect(page.locator("#result")).toBeHidden();
    expect(errors, "the page threw while handling the refusal").toEqual([]);
});
