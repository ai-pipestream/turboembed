// SPDX-License-Identifier: Apache-2.0
//
// The Devices panel: the survey the page draws from GET /api/v1/devices and
// GET /api/v1/models.
import { expect, test } from "@playwright/test";
import { open, tab, watchPageErrors } from "./app";

test("the devices panel lists every device with its feature bits", async ({ page, request }) => {
    const errors = watchPageErrors(page);
    await open(page);
    await tab(page, "devices");

    const devices = await (await request.get("/api/v1/devices")).json();
    await expect(page.locator("#devices .card")).toHaveCount(devices.length);
    await expect(page.locator("#devices-error")).toBeHidden();

    const headings = await page.locator("#devices .card h4").allInnerTexts();
    for (const d of devices) {
        expect(headings).toContain(`${d.index}. ${d.name} (${d.provider_id}:${d.ordinal} ${d.kind})`);
    }

    // The mock accelerator advertises the option bits it honors, and not the
    // pooling override, which is why an embed with pooling=CLS is refused.
    const accelerator = devices.find((d: any) => d.name === "Mock accelerator");
    expect(accelerator.features).toContain("DETERMINISTIC");
    expect(accelerator.features).not.toContain("OPT_POOLING_OVERRIDE");
    const card = page.locator("#devices .card").filter({ hasText: "Mock accelerator" }).first();
    const chips = await card.locator(".chip").allInnerTexts();
    expect(chips.sort()).toEqual([...accelerator.features].sort());

    // Only the offered cells are drawn; the mock offers no CHUNK.
    const offered = accelerator.capabilities.filter((c: any) => c.status !== "UNSUPPORTED");
    expect(offered.length).toBeGreaterThan(0);
    await expect(card.locator(".cap")).toHaveCount(offered.length);
    const cells = await card.locator(".cap").allInnerTexts();
    expect(cells.join(" ")).toContain("EMBED / TEXT");
    expect(cells.join(" ")).not.toContain("CHUNK");
    expect(errors).toEqual([]);
});

test("the devices panel lists every loaded model and its contract", async ({ page, request }) => {
    const errors = watchPageErrors(page);
    await open(page);
    await tab(page, "devices");

    const models = await (await request.get("/api/v1/models")).json();
    await expect(page.locator("#models .card")).toHaveCount(models.length);
    const headings = await page.locator("#models .card h4").allInnerTexts();
    for (const m of models) expect(headings).toContain(`${m.name}: ${m.model_id}`);

    // The classifier card carries the bundle's label set as chips.
    const classifier = models.find((m: any) => m.task === "CLASSIFY");
    expect(classifier, "the suite expects a classifier bundle").toBeTruthy();
    const card = page.locator("#models .card").filter({ hasText: classifier.model_id }).first();
    const chips = await card.locator(".chip").allInnerTexts();
    for (const label of classifier.labels) expect(chips).toContain(label);

    // The embedding card carries the bundle's prompt prefixes and pooling.
    const embedder = models.find((m: any) => m.task === "EMBED");
    const embedCard = page.locator("#models .card").filter({ hasText: embedder.model_id }).first();
    const embedChips = await embedCard.locator(".chip").allInnerTexts();
    expect(embedChips).toContain(`pooling ${embedder.pooling}`);
    expect(embedChips).toContain(`normalize ${embedder.normalize}`);
    expect(embedChips).toContain(`query prefix "${embedder.prefix_query}"`);
    expect(errors).toEqual([]);
});

test("the header links reach the API explorer and the protocol root", async ({ page, request }) => {
    const errors = watchPageErrors(page);
    await open(page);

    await expect(page.locator("#swagger-link")).toHaveAttribute("href", "/swagger-ui.html");
    expect((await request.get("/swagger-ui/index.html")).status()).toBe(200);
    expect((await request.get("/v3/api-docs")).status()).toBe(200);
    expect((await request.get("/v2")).status()).toBe(200);
    expect(errors).toEqual([]);
});
