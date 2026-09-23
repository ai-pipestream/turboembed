// SPDX-License-Identifier: Apache-2.0
//
// The /api/v1 surface, exercised directly rather than through the page:
// health, the device survey, the model contracts, embed, similarity, rerank,
// classify, token-classify, tokenize, detokenize and generation, with the
// refusals each one is documented to produce.
import { expect, test, type APIResponse } from "@playwright/test";
import { DEFAULT_TEXTS, SHORT_DOCUMENT } from "./app";

/** Parse a response, and fail with the body when the status is not the expected one. */
async function body(res: APIResponse, expected: number): Promise<any> {
    const text = await res.text();
    expect(res.status(), `${res.url()} answered ${res.status()}: ${text}`).toBe(expected);
    return text ? JSON.parse(text) : null;
}

test("GET /api/v1/health names every loaded model", async ({ request }) => {
    const health = await body(await request.get("/api/v1/health"), 200);
    expect(health.status).toBe("ok");
    expect(health.abi_version).toBeGreaterThan(0);
    expect(health.device_count).toBeGreaterThan(0);
    expect(health.models).toEqual(["mock-embedding", "mock-generative", "rerank", "classify", "ner"]);
});

test("GET /api/v1/devices is the whole survey", async ({ request }) => {
    const devices = await body(await request.get("/api/v1/devices"), 200);
    expect(devices.length).toBeGreaterThan(0);

    const accelerator = devices.find((d: any) => d.name === "Mock accelerator");
    expect(accelerator, "the suite expects the mock provider").toBeTruthy();
    expect(accelerator.kind).toBe("ACCEL");
    expect(accelerator.provider_id).toBe("mock");
    expect(accelerator.runtime_version).toBe("mock");
    expect(accelerator.models).toContain("mock-embedding");
    expect(accelerator.features).toContain("DETERMINISTIC");
    expect(accelerator.features).toContain("OPT_TOP_N");
    expect(accelerator.features).not.toContain("OPT_POOLING_OVERRIDE");

    // Every task crossed with every modality, offered or not.
    expect(accelerator.capabilities).toHaveLength(32);
    const cell = (task: string, modality: string) =>
        accelerator.capabilities.find((c: any) => c.task === task && c.modality === modality);
    expect(cell("EMBED", "TEXT").status).toBe("SUPPORTED");
    expect(cell("EMBED", "TEXT").dtype).toBe("F32");
    expect(cell("EMBED", "TEXT").deterministic).toBe(true);
    expect(cell("EMBED", "AUDIO").status).toBe("UNSUPPORTED");
    expect(cell("CHUNK", "TEXT").status).toBe("UNSUPPORTED");
});

test("GET /api/v1/models carries the whole bundle contract", async ({ request }) => {
    const models = await body(await request.get("/api/v1/models"), 200);
    const embedder = models.find((m: any) => m.name === "mock-embedding");

    expect(Object.keys(embedder).sort()).toEqual(
        [
            "bundle", "device", "dim", "dtype", "fully_accelerated", "kind", "labels", "max_batch", "max_seq",
            "modality", "model_id", "name", "normalize", "pooling", "prefix_document", "prefix_query", "revision",
            "stage_placement", "task", "tokenizer_bundle", "tokenizer_sha256", "vocab_size",
        ].sort(),
    );
    expect(embedder.task).toBe("EMBED");
    expect(embedder.kind).toBe("EMBEDDING");
    expect(embedder.dim).toBe(8);
    expect(embedder.pooling).toBe("MEAN");
    expect(embedder.normalize).toBe("L2");
    expect(embedder.max_seq).toBe(16);
    expect(embedder.prefix_query).toBe("query:");
    expect(embedder.prefix_document).toBe("passage:");
    expect(embedder.stage_placement).toEqual({
        tokenize: "host", encode: "host", pool: "host", normalize: "host", postprocess: "unused",
    });

    // One model per name, and the single-model path agrees with the list.
    const one = await body(await request.get("/api/v1/models/mock-embedding"), 200);
    expect(one).toEqual(embedder);
    expect((await body(await request.get("/api/v1/models/nope"), 404)).error).toContain("no model named `nope`");
});

test("POST /api/v1/embed returns vectors, the device and the timings", async ({ request }) => {
    const res = await body(await request.post("/api/v1/embed", { data: { texts: DEFAULT_TEXTS } }), 200);
    expect(res.model).toBe("mock-embedding");
    expect(res.count).toBe(3);
    expect(res.dim).toBe(8);
    expect(res.placement).toBe("HOST");
    expect(res.device.provider_id).toBe("mock");
    expect(res.timings.total_ms).toBeGreaterThanOrEqual(0);
    for (const v of res.vectors) {
        expect(v).toHaveLength(8);
        // The mock bundle's contract is l2-normalized, so every vector is a unit vector.
        expect(Math.sqrt(v.reduce((s: number, x: number) => s + x * x, 0))).toBeCloseTo(1, 4);
    }
});

test("POST /api/v1/similarity is symmetric with a unit diagonal", async ({ request }) => {
    const res = await body(await request.post("/api/v1/similarity", { data: { texts: DEFAULT_TEXTS } }), 200);
    expect(res.similarity).toHaveLength(3);
    for (let i = 0; i < 3; i++) {
        expect(res.similarity[i]).toHaveLength(3);
        expect(res.similarity[i][i]).toBeCloseTo(1, 6);
        for (let j = 0; j < 3; j++) {
            expect(res.similarity[i][j]).toBeGreaterThanOrEqual(-1.0000001);
            expect(res.similarity[i][j]).toBeLessThanOrEqual(1.0000001);
            expect(res.similarity[i][j]).toBeCloseTo(res.similarity[j][i], 12);
        }
    }
    const again = await body(await request.post("/api/v1/similarity", { data: { texts: DEFAULT_TEXTS } }), 200);
    expect(again.vectors, "the same texts gave different vectors").toEqual(res.vectors);
});

test("an option the device does not implement is 501 with the code and the field", async ({ request }) => {
    const refusal = await body(
        await request.post("/api/v1/embed", { data: { texts: ["a"], options: { pooling: "CLS" } } }), 501);
    expect(refusal.status).toBe("TURBO_E_UNSUPPORTED_OPTION");
    expect(refusal.field).toBe(6);
    expect(refusal.error).toContain("pooling");
    expect(refusal.path).toBe("/api/v1/embed");
});

test("a request beyond the model's contract is 422 and a malformed one is 400", async ({ request }) => {
    const capacity = await body(
        await request.post("/api/v1/embed", { data: { texts: ["a"], options: { max_tokens: 99 } } }), 422);
    expect(capacity.status).toBe("TURBO_E_CAPACITY");
    expect(capacity.field).toBe(3);
    expect(capacity.error).toContain("max_seq 16");

    const enumeration = await body(
        await request.post("/api/v1/embed", { data: { texts: ["a"], options: { pooling: "banana" } } }), 400);
    expect(enumeration.error).toContain("pooling");
    expect(enumeration.error).toContain("MODEL, MEAN, CLS, LAST");
    expect(enumeration.status).toBeUndefined();

    expect((await body(await request.post("/api/v1/embed", { data: { texts: [] } }), 400)).error)
        .toBe("texts must not be empty");
    const batch = await body(
        await request.post("/api/v1/embed", { data: { texts: ["1", "2", "3", "4", "5", "6", "7", "8", "9"] } }), 400);
    expect(batch.error).toContain("has a batch of");
});

test("a model of another task is 409 and an unknown name is 404", async ({ request }) => {
    const wrongTask = await body(await request.post("/api/v1/embed", { data: { model: "rerank", texts: ["a"] } }), 409);
    expect(wrongTask.error).toContain("performs RERANK");
    const unknown = await body(await request.post("/api/v1/embed", { data: { model: "nope", texts: ["a"] } }), 404);
    expect(unknown.error).toContain("no model named `nope`");
});

test("POST /api/v1/rerank scores documents and ranks them on request", async ({ request }) => {
    const data = {
        query: "the accelerator is fast",
        documents: ["the accelerator is fast", "a pot of soup on the stove", "the accelerator is quiet"],
        options: { return_sorted: true },
    };
    const res = await body(await request.post("/api/v1/rerank", { data }), 200);
    expect(res.model).toBe("rerank");
    expect(res.results).toHaveLength(3);
    expect(res.sorted).toHaveLength(3);
    expect(res.placement).toBe("HOST");
    for (const [i, hit] of res.results.entries()) {
        expect(hit.index).toBe(i);
        expect(hit.document).toBe(data.documents[i]);
        expect(typeof hit.score).toBe("number");
        expect(hit.rank).toBe(res.sorted.indexOf(i));
    }
    // The soup shares no content word with the query, so it ranks last.
    expect(res.sorted.at(-1)).toBe(1);

    // top_n narrows the ranking without dropping the scores.
    const top = await body(
        await request.post("/api/v1/rerank", { data: { ...data, options: { top_n: 2, return_sorted: true } } }), 200);
    expect(top.results).toHaveLength(3);
    expect(top.sorted).toHaveLength(2);

    expect((await body(await request.post("/api/v1/rerank", { data: { query: " ", documents: ["a"] } }), 400)).error)
        .toBe("query must not be blank");
    expect((await body(await request.post("/api/v1/rerank",
        { data: { query: "q", documents: ["a"], options: { top_n: 9 } } }), 400)).error).toContain("top_n is 9");
});

test("POST /api/v1/classify and /token-classify use the bundle's labels", async ({ request }) => {
    const classified = await body(
        await request.post("/api/v1/classify", { data: { texts: ["the service was excellent"] } }), 200);
    expect(classified.model).toBe("classify");
    expect(classified.labels).toEqual(["negative", "neutral", "positive"]);
    expect(classified.results[0].scores).toHaveLength(3);
    expect(classified.results[0].top).toBe(classified.results[0].scores[0].label);
    const scores = classified.results[0].scores.map((s: any) => s.score);
    expect([...scores].sort((a: number, b: number) => b - a)).toEqual(scores);

    const text = "Ada Lovelace worked in London";
    const spans = await body(await request.post("/api/v1/token-classify", { data: { texts: [text] } }), 200);
    expect(spans.model).toBe("ner");
    expect(spans.labels).toEqual(["O", "PER", "LOC"]);
    expect(spans.score_shape).toEqual([1, 16, 3]);
    expect(spans.spans.length).toBeGreaterThan(0);
    const bytes = new TextEncoder().encode(text);
    for (const span of spans.spans) {
        expect(span.row).toBe(0);
        expect(spans.labels).toContain(span.label);
        expect(new TextDecoder().decode(bytes.slice(span.byte_start, span.byte_end))).toBe(span.text);
    }

    const wrongTask = await body(
        await request.post("/api/v1/token-classify", { data: { model: "classify", texts: ["a"] } }), 409);
    expect(wrongTask.error).toContain("performs CLASSIFY");
});

test("POST /api/v1/tokenize and /detokenize go through the bundle's tokenizer", async ({ request }) => {
    const res = await body(await request.post("/api/v1/tokenize", { data: { texts: ["a brown dog"] } }), 200);
    expect(res.model).toBe("mock-embedding");
    expect(res.tokenizer_bundle).toContain("minilm-tokenizer");
    expect(res.tokenizer.kind).toBe("wordpiece");
    expect(res.results[0].ids).toEqual([101, 1037, 2829, 3899, 102]);
    expect(res.results[0].tokens).toEqual(["[CLS]", "a", "brown", "dog", "[SEP]"]);
    expect(res.results[0].mask).toEqual([1, 1, 1, 1, 1]);
    expect(res.results[0].count).toBe(5);

    const bare = await body(
        await request.post("/api/v1/tokenize", { data: { texts: ["a brown dog"], add_special_tokens: false } }), 200);
    expect(bare.results[0].ids).toEqual([1037, 2829, 3899]);

    const back = await body(await request.post("/api/v1/detokenize", { data: { ids: [res.results[0].ids] } }), 200);
    expect(back.texts).toEqual(["a brown dog"]);

    // A bundle with no tokenizer.json fails with the library's own refusal.
    const refusal = await body(
        await request.post("/api/v1/tokenize", { data: { model: "rerank", texts: ["a"] } }), 500);
    expect(refusal.status).toBe("TURBO_E_BUNDLE_INVALID");
    expect(refusal.error).toContain("no `tokenizer.json` file entry");
});

test("POST /api/v1/generate returns the whole text with its usage", async ({ request }) => {
    const res = await body(
        await request.post("/api/v1/generate", { data: { prompt: SHORT_DOCUMENT, options: { max_tokens: 6 } } }), 200);
    expect(res.model).toBe("mock-generative");
    expect(res.finish_reason).toBe("LENGTH");
    expect(res.usage.generated_tokens).toBe(6);
    expect(res.usage.prompt_tokens).toBeGreaterThan(0);
    expect(res.usage.total_tokens).toBe(res.usage.prompt_tokens + res.usage.generated_tokens);
    expect(res.text.trim()).toMatch(/^tok\d+( tok\d+)*$/);
    expect(res.timings.tokens_per_second).toBeGreaterThan(0);

    expect((await body(await request.post("/api/v1/generate", { data: {} }), 400)).error)
        .toBe("give prompt (a single user turn) or messages (a chat)");
    expect((await body(await request.post("/api/v1/generate",
        { data: { prompt: "a", messages: [{ role: "user", content: "b" }] } }), 400)).error)
        .toBe("give prompt or messages, not both");
});

/** Server-sent event frames, in order: "event:" names the event, "data:" carries JSON. */
function parseSse(raw: string): { event: string; data: any }[] {
    return raw
        .split("\n\n")
        .filter((frame) => frame.trim().length > 0)
        .map((frame) => {
            let event = "message";
            let data = "";
            for (const line of frame.split("\n")) {
                if (line.startsWith("event:")) event = line.slice(6).trim();
                else if (line.startsWith("data:")) data += line.slice(5).trim();
            }
            expect(data, `a ${event} frame carried no data: ${frame}`).not.toBe("");
            return { event, data: JSON.parse(data) };
        });
}

test("POST /api/v1/generate/stream sends every chunk before one done event", async ({ request }) => {
    const res = await request.post("/api/v1/generate/stream",
        { data: { prompt: SHORT_DOCUMENT, options: { max_tokens: 12 } } });
    const raw = await res.text();
    expect(res.status(), `the stream answered ${res.status()}: ${raw}`).toBe(200);
    expect(res.headers()["content-type"]).toContain("text/event-stream");

    const frames = parseSse(raw);
    const error = frames.find((f) => f.event === "error");
    expect(error, `the stream carried an error event: ${JSON.stringify(error?.data)}`).toBeUndefined();

    expect(frames.at(-1)!.event).toBe("done");
    const chunks = frames.slice(0, -1);
    expect(chunks).toHaveLength(12);
    for (const [i, frame] of chunks.entries()) {
        expect(frame.event, `frame ${i} is not a chunk`).toBe("chunk");
        expect(frame.data.text.length, `chunk ${i} carried empty text`).toBeGreaterThan(0);
        expect(frame.data.tokens.length).toBe(1);
        expect(frame.data.generated).toBe(i + 1);
    }

    const done = frames.at(-1)!.data;
    expect(done.finish_reason).toBe("LENGTH");
    expect(done.generated_tokens).toBe(12);
    expect(done.prompt_tokens).toBeGreaterThan(0);
    expect(done.total_tokens).toBe(done.prompt_tokens + done.generated_tokens);
    expect(done.text).toBe(chunks.map((c) => c.data.text).join(""));
    expect(done.device.provider_id).toBe("mock");

    const again = await (await request.post("/api/v1/generate/stream",
        { data: { prompt: SHORT_DOCUMENT, options: { max_tokens: 12 } } })).text();
    expect(parseSse(again).at(-1)!.data.text, "the same prompt gave different tokens").toBe(done.text);
});

test("a stream that fails midway ends with an error event carrying the code", async ({ request }) => {
    const res = await request.post("/api/v1/generate/stream",
        { data: { prompt: "word ".repeat(600), options: { max_tokens: 2 } } });
    const frames = parseSse(await res.text());
    expect(res.status(), "the refusal arrives after the headers, so the status stays 200").toBe(200);
    expect(frames).toHaveLength(1);
    expect(frames[0].event).toBe("error");
    expect(frames[0].data.status).toBe("TURBO_E_CAPACITY");
    expect(frames[0].data.error).toContain("max_seq is 512");
});
