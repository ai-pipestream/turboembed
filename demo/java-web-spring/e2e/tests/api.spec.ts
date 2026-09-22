// SPDX-License-Identifier: Apache-2.0
//
// The JSON API the page talks to, exercised directly: GET /api/info and
// POST /api/embed, including the two refusals the app maps to 400.
import { expect, test, type APIResponse } from "@playwright/test";
import { DEFAULT_TEXTS } from "./app";

/** Parse a response, and fail with the body when the status is not the expected one. */
async function body(res: APIResponse, expected: number): Promise<any> {
    const text = await res.text();
    expect(res.status(), `${res.url()} answered ${res.status()}: ${text}`).toBe(expected);
    return JSON.parse(text);
}

test("GET /api/info describes the device and the loaded model", async ({ request }) => {
    const info = await body(await request.get("/api/info"), 200);

    expect(Object.keys(info).sort()).toEqual(
        [
            "deviceKind",
            "deviceName",
            "dim",
            "fullyAccelerated",
            "maxBatch",
            "maxSeq",
            "modelId",
            "ordinal",
            "providerId",
            "runtimeVersion",
            "stages",
        ].sort(),
    );
    expect(info.providerId).toBe("mock");
    expect(info.modelId).toBe("turbo/mock-embedding");
    expect(info.deviceName).toBe("Mock accelerator");
    expect(info.deviceKind).toBe("ACCEL");
    expect(info.runtimeVersion).toBe("mock");
    expect(info.dim).toBe(8);
    expect(info.maxSeq).toBeGreaterThan(0);
    expect(info.ordinal).toBeGreaterThanOrEqual(0);
    expect(info.maxBatch).toBeGreaterThan(0);
    expect(typeof info.fullyAccelerated).toBe("boolean");
    // Stage placement is a per-stage device code, rendered as a Java array literal.
    expect(info.stages).toMatch(/^\[\d+(, \d+)*\]$/);
});

test("POST /api/embed returns vectors, a cosine matrix and the texts it used", async ({ request }) => {
    const info = await body(await request.get("/api/info"), 200);
    const res = await body(await request.post("/api/embed", { data: { texts: DEFAULT_TEXTS } }), 200);

    expect(res.texts).toEqual(DEFAULT_TEXTS);
    expect(res.dim).toBe(info.dim);
    expect(res.vectors).toHaveLength(DEFAULT_TEXTS.length);
    expect(res.similarity).toHaveLength(DEFAULT_TEXTS.length);

    for (const v of res.vectors) {
        expect(v).toHaveLength(info.dim);
        for (const x of v) expect(typeof x).toBe("number");
        // The mock bundle's contract is l2-normalized, so every vector is a unit vector.
        const norm = Math.sqrt(v.reduce((s: number, x: number) => s + x * x, 0));
        expect(norm).toBeCloseTo(1, 4);
    }

    for (let i = 0; i < res.similarity.length; i++) {
        expect(res.similarity[i]).toHaveLength(DEFAULT_TEXTS.length);
        expect(res.similarity[i][i]).toBeCloseTo(1, 6);
        for (let j = 0; j < res.similarity.length; j++) {
            expect(res.similarity[i][j]).toBeGreaterThanOrEqual(-1.0000001);
            expect(res.similarity[i][j]).toBeLessThanOrEqual(1.0000001);
            expect(res.similarity[i][j]).toBeCloseTo(res.similarity[j][i], 12);
        }
    }
});

test("POST /api/embed is deterministic and ignores blank lines", async ({ request }) => {
    const first = await body(await request.post("/api/embed", { data: { texts: DEFAULT_TEXTS } }), 200);
    const padded = ["", ` ${DEFAULT_TEXTS[0]} `, DEFAULT_TEXTS[1], "   ", DEFAULT_TEXTS[2]];
    const second = await body(await request.post("/api/embed", { data: { texts: padded } }), 200);

    expect(second.texts, "blank lines are not dropped and text is not trimmed").toEqual(DEFAULT_TEXTS);
    expect(second.vectors, "the same texts gave different vectors").toEqual(first.vectors);
    expect(second.similarity).toEqual(first.similarity);
});

test("POST /api/embed with no texts is refused with 400 and a message", async ({ request }) => {
    expect(await body(await request.post("/api/embed", { data: { texts: [] } }), 400)).toEqual({ error: "no texts" });
    expect(await body(await request.post("/api/embed", { data: {} }), 400)).toEqual({ error: "no texts" });
    expect(await body(await request.post("/api/embed", { data: { texts: ["  ", ""] } }), 400)).toEqual({ error: "no texts" });
});

test("POST /api/embed above the server's batch is refused with 400 and a message", async ({ request }) => {
    const info = await body(await request.get("/api/info"), 200);
    const texts = Array.from({ length: info.maxBatch + 1 }, (_, i) => `sentence number ${i + 1}`);

    expect(await body(await request.post("/api/embed", { data: { texts } }), 400)).toEqual({
        error: `request has ${texts.length} texts but the server's batch is ${info.maxBatch}`,
    });
    // The batch itself is accepted.
    const ok = await body(await request.post("/api/embed", { data: { texts: texts.slice(0, info.maxBatch) } }), 200);
    expect(ok.vectors).toHaveLength(info.maxBatch);
});
