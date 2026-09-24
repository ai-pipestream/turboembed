// SPDX-License-Identifier: Apache-2.0
//
// The Rerank panel against the committed mock reranker, whose score is the
// share of a document's tokens that the query also carries, so the ranking is
// deterministic and checkable without a real cross-encoder.
import { expect, test } from "@playwright/test";
import { DEFAULT_DOCUMENTS, DEFAULT_QUERY, open, rankingRows, rerank, rerankExpectingError, tab, watchPageErrors } from "./app";

test("the reranker line names the mock reranker and its device", async ({ page, request }) => {
    const errors = watchPageErrors(page);
    await open(page);
    await tab(page, "rerank");

    const models = await (await request.get("/api/v1/models")).json();
    const reranker = models.find((m: any) => m.task === "RERANK");
    expect(reranker, "the suite expects a reranker bundle").toBeTruthy();
    await expect(page.locator("#reranker")).toHaveText(
        `${reranker.model_id} (max_seq ${reranker.max_seq}, batch ${reranker.max_batch})` +
            ` on ${reranker.device.name} (${reranker.device.provider_id}:${reranker.device.ordinal})`,
    );
    await expect(page.locator("#rerank")).toBeEnabled();
    expect(errors).toEqual([]);
});

test("reranking the defaults ranks the documents and names the device", async ({ page }) => {
    const errors = watchPageErrors(page);
    await open(page);
    await tab(page, "rerank");

    await expect(page.locator("#query")).toHaveValue(DEFAULT_QUERY);
    await expect(page.locator("#documents")).toHaveValue(DEFAULT_DOCUMENTS.join("\n"));
    await rerank(page);

    await expect(page.locator("#rerank-status")).toHaveText(
        /^4 documents in [\d.]+ ms on Mock accelerator \(mock:\d+\), placement HOST, [\d.]+ ms round trip$/,
    );

    const rows = await rankingRows(page);
    expect(rows).toHaveLength(4);
    expect(rows.map((r) => r.rank)).toEqual(["1", "2", "3", "4"]);
    for (const row of rows) expect(row.score, `not a four-decimal score: ${row.score}`).toMatch(/^-?\d\.\d{4}$/);
    // Ranked best first, and every document shown exactly once.
    const scores = rows.map((r) => Number(r.score));
    for (let i = 1; i < scores.length; i++) expect(scores[i - 1]).toBeGreaterThanOrEqual(scores[i]);
    expect(rows.map((r) => r.document).sort()).toEqual([...DEFAULT_DOCUMENTS].sort());
    // The recipe shares no content word with the query about the accelerator.
    expect(rows.at(-1)!.document).toBe("the recipe needs two eggs and a cup of flour");
    expect(errors).toEqual([]);
});

test("unchecking the ranking leaves the documents in input order", async ({ page }) => {
    const errors = watchPageErrors(page);
    await open(page);
    await tab(page, "rerank");

    await page.locator("#rerank-sorted").uncheck();
    await rerank(page);
    const rows = await rankingRows(page);
    expect(rows.map((r) => r.document)).toEqual(DEFAULT_DOCUMENTS);
    expect(rows.map((r) => r.rank)).toEqual(["1", "2", "3", "4"]);
    expect(errors).toEqual([]);
});

test("an empty query shows the server's refusal and leaves the panel working", async ({ page }) => {
    const errors = watchPageErrors(page);
    await open(page);
    await tab(page, "rerank");

    await page.locator("#query").fill("   ");
    expect(await rerankExpectingError(page)).toBe("query must not be blank");
    await expect(page.locator("#rerank-status")).toHaveText("");
    expect(errors, "the page threw while handling the refusal").toEqual([]);

    await page.locator("#query").fill(DEFAULT_QUERY);
    await rerank(page);
    await expect(page.locator("#rerank-error")).toBeHidden();
    expect(errors).toEqual([]);
});

test("more documents than the model's batch shows the batch refusal", async ({ page, request }) => {
    const errors = watchPageErrors(page);
    await open(page);
    await tab(page, "rerank");

    const models = await (await request.get("/api/v1/models")).json();
    const reranker = models.find((m: any) => m.task === "RERANK");
    const lines = reranker.max_batch + 1;
    await page.locator("#documents").fill(Array.from({ length: lines }, (_, i) => `document ${i + 1}`).join("\n"));
    expect(await rerankExpectingError(page)).toBe(
        `request has ${lines} documents but model \`${reranker.name}\` has a batch of ${reranker.max_batch}`,
    );
    expect(errors, "the page threw while handling the refusal").toEqual([]);
});
