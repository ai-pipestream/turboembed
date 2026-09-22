// SPDX-License-Identifier: Apache-2.0
//
// The Embed panel and the tab bar, against the committed mock bundle:
// deterministic 8-dim vectors, no hardware. Every check is a hard assertion; a
// refusal from the server is reported with its own message rather than
// swallowed.
import { expect, test } from "@playwright/test";
import { DEFAULT_TEXTS, embed, embedExpectingError, matrixCells, matrixRowLabels, open, setTexts, tab, watchPageErrors } from "./app";

test("the device line names the mock provider and the mock model", async ({ page, request }) => {
    const errors = watchPageErrors(page);
    await open(page);

    const models = await (await request.get("/api/v1/models")).json();
    const embedder = models.find((m: any) => m.task === "EMBED");
    expect(embedder, "the suite expects an embedding bundle").toBeTruthy();
    expect(embedder.model_id).toBe("turbo/mock-embedding");

    const device = (await page.locator("#device").innerText()).trim();
    expect(device).toBe(
        `${embedder.model_id} (dim ${embedder.dim}, max_seq ${embedder.max_seq}, batch ${embedder.max_batch})` +
            ` on ${embedder.device.name} (${embedder.device.provider_id}:${embedder.device.ordinal}` +
            ` ${embedder.device.kind}), runtime ${embedder.device.runtime_version}` +
            (embedder.fully_accelerated ? "" : "; host stages: tokenize, encode, pool, normalize"),
    );
    expect(errors).toEqual([]);
});

test("every tab opens its panel, by click and by arrow key", async ({ page }) => {
    const errors = watchPageErrors(page);
    await open(page);

    const names = ["embed", "rerank", "tokenize", "summarize", "devices"] as const;
    for (const name of names) {
        await tab(page, name);
        for (const other of names) {
            if (other === name) await expect(page.locator(`#panel-${other}`)).toBeVisible();
            else await expect(page.locator(`#panel-${other}`), `panel ${other} with ${name} selected`).toBeHidden();
        }
    }

    // Roving tabindex: the selected tab is the only one in the tab order.
    await tab(page, "embed");
    await expect(page.locator("#tab-embed")).toHaveAttribute("tabindex", "0");
    await expect(page.locator("#tab-rerank")).toHaveAttribute("tabindex", "-1");

    // The arrow keys move the selection and the focus together.
    await page.locator("#tab-embed").focus();
    await page.keyboard.press("ArrowRight");
    await expect(page.locator("#tab-rerank")).toHaveAttribute("aria-selected", "true");
    await expect(page.locator("#panel-rerank")).toBeVisible();
    await page.keyboard.press("End");
    await expect(page.locator("#tab-devices")).toHaveAttribute("aria-selected", "true");
    await page.keyboard.press("Home");
    await expect(page.locator("#tab-embed")).toHaveAttribute("aria-selected", "true");
    expect(errors).toEqual([]);
});

test("embedding the three default sentences renders a 3x3 matrix", async ({ page }) => {
    const errors = watchPageErrors(page);
    await open(page);

    await expect(page.locator("#texts")).toHaveValue(DEFAULT_TEXTS.join("\n"));
    await expect(page.locator("#result")).toBeHidden();
    await embed(page);

    // The status line carries the batch, the dimension, the device time, the
    // device itself, the output placement and the round trip.
    await expect(page.locator("#status")).toHaveText(
        /^3 × \d+ in [\d.]+ ms on Mock accelerator \(mock:\d+\), placement HOST, [\d.]+ ms round trip$/,
    );

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
    expect(await page.locator("#matrix tr").first().locator("th").allInnerTexts()).toEqual(["", "1", "2", "3"]);

    const vectors = ((await page.locator("#vectors").textContent()) ?? "").trim().split("\n");
    expect(vectors).toHaveLength(3);
    for (const line of vectors) expect(line).toMatch(/^\d+: \[-?\d\.\d{4}(, -?\d\.\d{4})*(, …)?\]$/);
    expect(errors).toEqual([]);
});

test("ctrl+enter in the textarea runs the embed", async ({ page }) => {
    const errors = watchPageErrors(page);
    await open(page);

    await page.locator("#texts").focus();
    await page.keyboard.press("Control+Enter");
    await expect(page.locator("#result")).toBeVisible({ timeout: 20_000 });
    await expect(page.locator("#status")).toContainText("round trip");
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
    expect(await embedExpectingError(page)).toBe("texts must not be empty");
    await expect(page.locator("#result")).toBeHidden();
    await expect(page.locator("#status")).toHaveText("");
    expect(errors, "the page threw while handling the refusal").toEqual([]);

    // The page recovers: the defaults still embed and the error box goes away.
    await setTexts(page, DEFAULT_TEXTS);
    await embed(page);
    await expect(page.locator("#error")).toBeHidden();
    expect(errors).toEqual([]);
});

test("more sentences than the model's batch shows the batch refusal", async ({ page, request }) => {
    const errors = watchPageErrors(page);
    await open(page);

    const models = await (await request.get("/api/v1/models")).json();
    const embedder = models.find((m: any) => m.task === "EMBED");
    expect(embedder.max_batch, "the server reports no batch limit").toBeGreaterThan(0);
    const lines = embedder.max_batch + 1;

    await setTexts(page, Array.from({ length: lines }, (_, i) => `sentence number ${i + 1}`));
    expect(await embedExpectingError(page)).toBe(
        `request has ${lines} texts but model \`${embedder.name}\` has a batch of ${embedder.max_batch}`,
    );
    await expect(page.locator("#result")).toBeHidden();
    expect(errors, "the page threw while handling the refusal").toEqual([]);
});
