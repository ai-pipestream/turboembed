// SPDX-License-Identifier: Apache-2.0
//
// The Benchmarks panel: the committed receipts the page draws from
// GET /api/v1/benchmarks. Every expectation is taken from the endpoint itself,
// so a receipt added to testdata/receipts/turbo/bench does not break the suite.
import { expect, test } from "@playwright/test";
import { benchmarks, open, watchPageErrors } from "./app";

/** The comparison the CUDA provider's SUPPORTED verdict rests on. */
const CUDA = "compare-cuda-krick-embed-2026-09-22.json";
/** The comparison with cells under the floor, which is why the CPU cell is EXPERIMENTAL. */
const OPENVINO_CPU = "compare-openvino-krick-cpu-embed-2026-09-22.json";

async function report(request: any) {
    const response = await request.get("/api/v1/benchmarks");
    expect(response.status(), "GET /api/v1/benchmarks was refused").toBe(200);
    return response.json();
}

test("the benchmarks panel renders every committed comparison", async ({ page, request }) => {
    const errors = watchPageErrors(page);
    const data = await report(request);
    expect(data.comparisons.length, "the committed tree carries compare receipts").toBeGreaterThan(0);

    await open(page);
    await benchmarks(page);

    await expect(page.locator("#bench-dir")).toContainText("testdata/receipts/turbo/bench");
    // The floor rule is stated once, at the top of the panel.
    await expect(page.locator("#bench-rule")).toHaveText(
        `SUPPORTED: every cell at ${[...new Set(data.comparisons.map((c: any) => c.floor))]
            .sort((a: any, b: any) => a - b)
            .join(" or ")} of the runtime alone or better.`,
    );

    // One summary row and one expandable table per compare receipt.
    await expect(page.locator("#bench-table tbody tr")).toHaveCount(data.comparisons.length);
    await expect(page.locator("#bench-cells details")).toHaveCount(data.comparisons.length);
    for (const c of data.comparisons) {
        const row = page.locator(`#bench-table tbody tr[data-file="${c.file}"]`);
        await expect(row, `no summary row for ${c.file}`).toHaveCount(1);
        await expect(row).toContainText(c.device);
        await expect(row).toContainText(c.task);
        await expect(row).toContainText(c.provider);
        await expect(row).toContainText(c.native_provider);
        await expect(row).toContainText(c.verdict);
        // Every number links back to the receipt it came from.
        await expect(row).toContainText(c.file);
        const details = page.locator(`#bench-cells details[data-file="${c.file}"]`);
        await expect(details, `no cell table for ${c.file}`).toHaveCount(1);
        await expect(details.locator("caption")).toHaveText(`receipt ${c.file}`);
    }
    expect(errors).toEqual([]);
});

test("the benchmarks panel marks the cells under the floor", async ({ page, request }) => {
    const errors = watchPageErrors(page);
    const data = await report(request);
    const cpu = data.comparisons.find((c: any) => c.file === OPENVINO_CPU);
    expect(cpu, `the suite expects ${OPENVINO_CPU}`).toBeTruthy();
    const under = cpu.cells.filter((cell: any) => !cell.within_floor);
    expect(under.length, "this comparison is the one with cells under the floor").toBeGreaterThan(0);
    expect(cpu.verdict).toBe("EXPERIMENTAL");

    await open(page);
    await benchmarks(page);

    // The summary row says how many cells fell short, and of what floor.
    const row = page.locator(`#bench-table tbody tr[data-file="${OPENVINO_CPU}"]`);
    await expect(row).toContainText(`${under.length} of ${cpu.cells.length} cells under ${cpu.floor}`);

    const details = page.locator(`#bench-cells details[data-file="${OPENVINO_CPU}"]`);
    await details.locator("summary").click();
    await expect(details.locator("tbody tr")).toHaveCount(cpu.cells.length);
    await expect(details.locator("tbody tr.under")).toHaveCount(under.length);
    for (const cell of under) {
        const marked = details.locator("tbody tr.under").filter({ hasText: cell.cell });
        await expect(marked, `${cell.cell} is under the floor but is not marked`).toHaveCount(1);
        await expect(marked).toContainText(`under ${cpu.floor}`);
    }
    // A comparison every cell of which is at the floor marks nothing.
    const cuda = page.locator(`#bench-cells details[data-file="${CUDA}"]`);
    await cuda.locator("summary").click();
    await expect(cuda.locator("tbody tr.under")).toHaveCount(0);
    expect(errors).toEqual([]);
});

test("a comparison expands to every cell with its libturbo and native figures", async ({ page, request }) => {
    const errors = watchPageErrors(page);
    const data = await report(request);
    const cuda = data.comparisons.find((c: any) => c.file === CUDA);
    expect(cuda, `the suite expects ${CUDA}`).toBeTruthy();

    await open(page);
    await benchmarks(page);

    const details = page.locator(`#bench-cells details[data-file="${CUDA}"]`);
    await expect(details.locator("tbody"), "a comparison starts closed").toBeHidden();
    // The line the visitor reads before opening it.
    const ratios = cuda.cells.map((c: any) => c.ratio);
    const worst = Math.min(...ratios).toFixed(2);
    const best = Math.max(...ratios).toFixed(2);
    await expect(details.locator("summary")).toHaveText(
        `${cuda.device}, ${cuda.task}: libturbo at ${worst}x to ${best}x of ${cuda.native_provider} (${cuda.verdict})`,
    );

    await details.locator("summary").click();
    await expect(details.locator("tbody tr")).toHaveCount(cuda.cells.length);
    await expect(details).toContainText(cuda.runtime.slice(0, 40));
    await expect(details).toContainText(cuda.native_runtime.slice(0, 40));
    await expect(details).toContainText(cuda.model_id);

    // One row per cell, with both sides and the ratio drawn as a bar.
    for (const cell of cuda.cells) {
        const row = details.locator("tbody tr").filter({ hasText: cell.cell }).first();
        await expect(row).toContainText(cell.measure);
        await expect(row).toContainText(`${cell.ratio.toFixed(2)}x`);
        await expect(row.locator(".bar i")).toHaveCount(1);
    }
    // The fastest cell's bar reaches further to the right than the slowest one's.
    const widths = await details.locator("tbody tr .bar i").evaluateAll((bars) =>
        bars.map((bar) => (bar as HTMLElement).getBoundingClientRect().width),
    );
    expect(Math.max(...widths)).toBeGreaterThan(Math.min(...widths));
    expect(errors).toEqual([]);
});

test("the throughput table carries the cuda 32x32 cell", async ({ page, request }) => {
    const errors = watchPageErrors(page);
    const data = await report(request);
    const run = data.turbo.find((r: any) => r.file === "cuda-krick-embed-2026-09-22.json");
    expect(run, "the suite expects the committed CUDA embedding receipt").toBeTruthy();
    const cell = run.embed.find((c: any) => c.batch === 32 && c.seq === 32);
    expect(cell, "that receipt carries a 32x32 cell").toBeTruthy();

    await open(page);
    await benchmarks(page);

    const card = page.locator(
        `#bench-throughput .bench-device[data-device="${run.device}"][data-provider="${run.provider}"]`,
    );
    await expect(card, `no throughput card for ${run.device}`).toHaveCount(1);
    await expect(card.locator("h4")).toContainText(run.provider);
    await expect(card).toContainText(run.file);

    const row = card.locator(`table[data-file="${run.file}"] tr[data-cell="32x32"]`);
    await expect(row, "the 32x32 cell is not in the throughput table").toHaveCount(1);
    const grouped = (value: number) => Math.round(value).toLocaleString("en-US");
    await expect(row).toContainText(grouped(cell.text.rows_per_s));
    await expect(row).toContainText(grouped(cell.text.tokens_per_s));
    await expect(row).toContainText(grouped(cell.prepared_tokens.rows_per_s));
    expect(errors).toEqual([]);
});
