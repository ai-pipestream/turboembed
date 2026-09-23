// SPDX-License-Identifier: Apache-2.0
//
// The KServe Open Inference Protocol version 2 HTTP/REST surface, exercised as
// a protocol client would: the six endpoints of the protocol's HTTP/REST
// section, the inference mapping for each model kind, and the {"error": "..."}
// refusals with 400, 404 and 500.
import { expect, test, type APIResponse } from "@playwright/test";

async function body(res: APIResponse, expected: number): Promise<any> {
    const text = await res.text();
    expect(res.status(), `${res.url()} answered ${res.status()}: ${text}`).toBe(expected);
    return text ? JSON.parse(text) : null;
}

test("server metadata, server live and server ready", async ({ request }) => {
    const meta = await body(await request.get("/v2"), 200);
    expect(Object.keys(meta).sort()).toEqual(["extensions", "name", "version"]);
    expect(meta.name).toBe("turbo");
    expect(meta.extensions).toEqual([]);

    expect(await body(await request.get("/v2/health/live"), 200)).toEqual({ live: true, ready: true });
    expect(await body(await request.get("/v2/health/ready"), 200)).toEqual({ live: true, ready: true });
});

test("model metadata declares the tensors of every loaded kind", async ({ request }) => {
    const embed = await body(await request.get("/v2/models/mock-embedding"), 200);
    expect(embed).toEqual({
        name: "mock-embedding",
        versions: ["1"],
        platform: "mock",
        inputs: [{ name: "text", datatype: "BYTES", shape: [-1] }],
        outputs: [{ name: "embeddings", datatype: "FP32", shape: [-1, 8] }],
    });
    // The version-qualified path is the same document.
    expect(await body(await request.get("/v2/models/mock-embedding/versions/1"), 200)).toEqual(embed);

    const rerank = await body(await request.get("/v2/models/rerank"), 200);
    expect(rerank.inputs.map((t: any) => t.name)).toEqual(["query", "documents"]);
    expect(rerank.outputs.map((t: any) => t.name)).toEqual(["scores", "sorted"]);
    expect(rerank.outputs[1].datatype).toBe("INT32");

    const classify = await body(await request.get("/v2/models/classify"), 200);
    expect(classify.outputs[0].shape).toEqual([-1, 3]);
    expect(classify.outputs[1]).toEqual({ name: "labels", datatype: "BYTES", shape: [3] });

    const ner = await body(await request.get("/v2/models/ner"), 200);
    expect(ner.outputs[0].shape).toEqual([-1, 16, 3]);

    const generate = await body(await request.get("/v2/models/mock-generative"), 200);
    expect(generate.inputs.map((t: any) => t.name)).toEqual(["prompt", "messages"]);
    expect(generate.outputs).toEqual([{ name: "text", datatype: "BYTES", shape: [1] }]);

    expect(await body(await request.get("/v2/models/mock-embedding/ready"), 200))
        .toEqual({ name: "mock-embedding", ready: true });
    expect(await body(await request.get("/v2/models/mock-embedding/versions/1/ready"), 200))
        .toEqual({ name: "mock-embedding", ready: true });
});

test("embed inference returns one flattened FP32 tensor", async ({ request }) => {
    const res = await body(await request.post("/v2/models/mock-embedding/infer", {
        data: {
            id: "req-1",
            inputs: [{ name: "text", shape: [2], datatype: "BYTES", data: ["a brown dog", "the stock market"] }],
        },
    }), 200);
    expect(res.model_name).toBe("mock-embedding");
    expect(res.model_version).toBe("1");
    expect(res.id).toBe("req-1");
    expect(res.parameters.placement).toBe("HOST");
    expect(res.outputs).toHaveLength(1);
    expect(res.outputs[0].name).toBe("embeddings");
    expect(res.outputs[0].datatype).toBe("FP32");
    expect(res.outputs[0].shape).toEqual([2, 8]);
    expect(res.outputs[0].data).toHaveLength(16);

    // The same vectors the /api/v1 surface returns, flattened row-major.
    const rich = await body(await request.post("/api/v1/embed",
        { data: { texts: ["a brown dog", "the stock market"] } }), 200);
    expect(res.outputs[0].data).toEqual([...rich.vectors[0], ...rich.vectors[1]]);
});

test("rerank inference returns scores and the ranking", async ({ request }) => {
    const res = await body(await request.post("/v2/models/rerank/infer", {
        data: {
            inputs: [
                { name: "query", shape: [1], datatype: "BYTES", data: ["the accelerator is fast"] },
                { name: "documents", shape: [2], datatype: "BYTES", data: ["the accelerator is fast", "a pot of soup"] },
            ],
            parameters: { return_sorted: true },
        },
    }), 200);
    expect(res.outputs.map((o: any) => o.name)).toEqual(["scores", "sorted"]);
    expect(res.outputs[0].datatype).toBe("FP32");
    expect(res.outputs[0].data).toHaveLength(2);
    expect(res.outputs[1].datatype).toBe("INT32");
    expect(res.outputs[1].data).toEqual([0, 1]);
});

test("classify and token-classify inference return scores with the label set", async ({ request }) => {
    const classified = await body(await request.post("/v2/models/classify/infer", {
        data: { inputs: [{ name: "text", shape: [1], datatype: "BYTES", data: ["excellent"] }] },
    }), 200);
    expect(classified.outputs[0].shape).toEqual([1, 3]);
    expect(classified.outputs[0].data).toHaveLength(3);
    expect(classified.outputs[1].data).toEqual(["negative", "neutral", "positive"]);

    const ner = await body(await request.post("/v2/models/ner/infer", {
        data: { inputs: [{ name: "text", shape: [1], datatype: "BYTES", data: ["Ada in London"] }] },
    }), 200);
    expect(ner.outputs[0].shape).toEqual([1, 16, 3]);
    expect(ner.outputs[0].data).toHaveLength(48);

    // outputs narrows the response to the tensors the client asked for.
    const narrowed = await body(await request.post("/v2/models/classify/infer", {
        data: {
            inputs: [{ name: "text", shape: [1], datatype: "BYTES", data: ["excellent"] }],
            outputs: [{ name: "labels" }],
        },
    }), 200);
    expect(narrowed.outputs).toHaveLength(1);
    expect(narrowed.outputs[0].name).toBe("labels");
});

test("generate inference takes a prompt or a chat", async ({ request }) => {
    const fromPrompt = await body(await request.post("/v2/models/mock-generative/infer", {
        data: {
            inputs: [{ name: "prompt", shape: [1], datatype: "BYTES", data: ["summarize this"] }],
            parameters: { max_tokens: 4 },
        },
    }), 200);
    expect(fromPrompt.outputs[0]).toMatchObject({ name: "text", datatype: "BYTES", shape: [1] });
    expect(fromPrompt.outputs[0].data[0]).toMatch(/tok\d+/);
    expect(fromPrompt.parameters.finish_reason).toBe("LENGTH");
    expect(fromPrompt.parameters.generated_tokens).toBe(4);
    expect(fromPrompt.parameters.prompt_tokens).toBeGreaterThan(0);

    const fromChat = await body(await request.post("/v2/models/mock-generative/infer", {
        data: {
            inputs: [{
                name: "messages",
                shape: [2],
                datatype: "BYTES",
                data: [
                    JSON.stringify({ role: "system", content: "be terse" }),
                    JSON.stringify({ role: "user", content: "hello" }),
                ],
            }],
            parameters: { max_tokens: 2 },
        },
    }), 200);
    expect(fromChat.parameters.generated_tokens).toBe(2);
    expect(fromChat.parameters.prompt_tokens).toBeGreaterThan(fromPrompt.parameters.prompt_tokens);
});

test("protocol errors are an error string with 400, 404 or 500", async ({ request }) => {
    const missing = await body(await request.get("/v2/models/does-not-exist"), 404);
    expect(Object.keys(missing)).toEqual(["error"]);
    expect(missing.error).toContain("no model named `does-not-exist`");

    expect((await body(await request.get("/v2/models/mock-embedding/versions/7"), 404)).error)
        .toContain("has no version `7`");
    expect((await request.get("/v2/models/does-not-exist/ready")).status()).toBe(404);

    const infer = (data: any) => request.post("/v2/models/mock-embedding/infer", { data });
    expect((await body(await infer({ inputs: [] }), 400)).error).toContain("at least one entry in `inputs`");
    expect((await body(await infer(
        { inputs: [{ name: "text", shape: [1], datatype: "FP32", data: [1] }] }), 400)).error)
        .toBe("input `text` must be datatype BYTES, not FP32");
    expect((await body(await infer(
        { inputs: [{ name: "wrong", shape: [1], datatype: "BYTES", data: ["a"] }] }), 400)).error)
        .toContain("needs an input named `text`");
    expect((await body(await infer(
        { inputs: [{ name: "text", shape: [4], datatype: "BYTES", data: ["a"] }] }), 400)).error)
        .toContain("declares shape");
    expect((await body(await infer({
        inputs: [{ name: "text", shape: [1], datatype: "BYTES", data: ["a"] }],
        outputs: [{ name: "nope" }],
    }), 400)).error).toContain("produces no output named `nope`");

    // A libturbo refusal keeps its status name and field index inside the
    // message, because the protocol has no field for either.
    const unsupported = await body(await infer({
        inputs: [{ name: "text", shape: [1], datatype: "BYTES", data: ["a"] }],
        parameters: { pooling: "CLS" },
    }), 400);
    expect(Object.keys(unsupported)).toEqual(["error"]);
    expect(unsupported.error).toContain("TURBO_E_UNSUPPORTED_OPTION");
    expect(unsupported.error).toContain("field 6");

    expect((await body(await infer({
        inputs: [{ name: "text", shape: [1], datatype: "BYTES", data: ["a"] }],
        parameters: { pooling: "banana" },
    }), 400)).error).toContain("must be one of [MODEL, MEAN, CLS, LAST]");

    // A bundle with no tokenizer.json is the server's configuration, not the
    // client's request, so it is 500 on both surfaces.
    expect((await body(await request.post("/api/v1/tokenize", { data: { model: "rerank", texts: ["a"] } }), 500))
        .status).toBe("TURBO_E_BUNDLE_INVALID");
});
