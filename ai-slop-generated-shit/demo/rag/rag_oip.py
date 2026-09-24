#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = ["kserve>=0.15"]
# ///
"""The same retrieval-augmented answer through KServe's own client.

    uv run demo/rag/rag_oip.py                       # REST, http://127.0.0.1:8000
    uv run demo/rag/rag_oip.py --grpc 127.0.0.1:8001 # gRPC

kserve.InferenceRESTClient and kserve.InferenceGRPCClient speak the Open
Inference Protocol v2 as any KServe predictor serves it; nothing here is
specific to Inferstream except the tensor names, which server/README.md
lists per model kind. The answer is one ModelInfer on the generative
model (the protocol has no stream); the streamed form is the
turbo.inferstream.InferstreamExtension/ModelStreamInfer rpc, which
grpcurl reaches through reflection.
"""

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# kserve parses sys.argv for its model server when imported, and its option
# names overlap this script's, so it is imported with an empty argv.
_argv, sys.argv = sys.argv, sys.argv[:1]
from kserve import InferenceGRPCClient, InferenceRESTClient, InferInput, InferRequest, RESTConfig  # noqa: E402

sys.argv = _argv


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)))


def bytes_input(name, values):
    """A BYTES tensor of strings, the way the KServe client encodes one."""
    tensor = InferInput(name=name, shape=[len(values)], datatype="BYTES")
    tensor.set_data_from_numpy(np.array(values, dtype=object), binary_data=False)
    return tensor


def plain(value):
    """A response parameter as a Python value (the gRPC client returns the proto)."""
    for field in ("int64_param", "string_param", "bool_param", "double_param"):
        if hasattr(value, "HasField") and value.HasField(field):
            return getattr(value, field)
    return value


def output(response, name):
    for o in response.outputs:
        if o.name == name:
            return o
    raise SystemExit(f"no `{name}` output in {[o.name for o in response.outputs]}")


async def run(args):
    passages = json.loads(Path(args.passages).read_text())
    texts = [f"{x['title']}: {x['text']}" for x in passages]
    if args.grpc:
        client = InferenceGRPCClient(args.grpc)
        base = None
    else:
        client = InferenceRESTClient(RESTConfig(protocol="v2"))
        base = args.rest

    async def infer(model, inputs, parameters=None):
        req = InferRequest(model_name=model, infer_inputs=inputs, parameters=parameters)
        if base is None:
            return await client.infer(req)
        return await client.infer(base, req, model_name=model)

    # 1. Embeddings: `text` BYTES [n] in, `embeddings` FP32 [n, dim] out.
    t0 = time.perf_counter()
    corpus = await infer(args.embed_model, [bytes_input("text", texts)])
    query = await infer(args.embed_model, [bytes_input("text", [args.question])])
    t1 = time.perf_counter()
    emb = output(corpus, "embeddings")
    n, dim = emb.shape
    vectors = emb.as_numpy().reshape(n, dim).tolist()
    qv = output(query, "embeddings").as_numpy().reshape(-1).tolist()
    ranked = sorted(range(len(texts)), key=lambda i: -cosine(qv, vectors[i]))[: args.candidates]
    print(f"embed   {n} passages + 1 question in {1000 * (t1 - t0):.1f} ms (dim {dim})")

    # 2. Rerank: `query` BYTES [1] and `documents` BYTES [n] in; `scores`
    #    FP32 [n] and `sorted` INT32 [k] (best first) out.
    t2 = time.perf_counter()
    rr = await infer(
        args.rerank_model,
        [bytes_input("query", [args.question]), bytes_input("documents", [texts[i] for i in ranked])],
        {"top_n": args.cite},
    )
    t3 = time.perf_counter()
    scores = output(rr, "scores").as_numpy().reshape(-1).tolist()
    order = output(rr, "sorted").as_numpy().reshape(-1).tolist()
    chosen = [(ranked[j], float(scores[j])) for j in order]
    print(f"rerank  {len(ranked)} candidates in {1000 * (t3 - t2):.1f} ms")
    for k, (i, score) in enumerate(chosen, 1):
        print(f"  [{k}] {score:.3f}  {passages[i]['title']}")

    # 3. Generate: `messages` BYTES [t], one JSON turn each; `text` BYTES [1]
    #    out, with finish_reason and the token counts in the parameters.
    context = "\n\n".join(f"[{k}] {texts[i]}" for k, (i, _) in enumerate(chosen, 1))
    turns = [
        {"role": "system", "content": "Answer from the numbered passages only, in two or three sentences, "
                                      "and cite each fact with its passage number in square brackets."},
        {"role": "user", "content": f"Passages:\n\n{context}\n\nQuestion: {args.question}"},
    ]
    t4 = time.perf_counter()
    gen = await infer(args.chat_model, [bytes_input("messages", [json.dumps(t) for t in turns])], {"max_tokens": args.max_tokens})
    t5 = time.perf_counter()
    text = output(gen, "text").as_numpy().reshape(-1)[0]
    if isinstance(text, bytes):
        text = text.decode()
    params = {k: plain(v) for k, v in (gen.parameters or {}).items()}
    print(f"\nanswer  ({args.chat_model}, one ModelInfer, {t5 - t4:.2f} s, {params})\n\n{text}")
    print("\nsources")
    for k, (i, _) in enumerate(chosen, 1):
        print(f"  [{k}] {passages[i]['title']} ({passages[i]['source']})")
    if hasattr(client, "close"):
        result = client.close()
        if asyncio.iscoroutine(result):
            await result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rest", default="http://127.0.0.1:8000", help="REST base URL")
    p.add_argument("--grpc", default=None, help="gRPC host:port; when set, the gRPC client is used")
    p.add_argument("--embed-model", default="minilm")
    p.add_argument("--rerank-model", default="rerank")
    p.add_argument("--chat-model", default="qwen")
    p.add_argument("--passages", default=str(Path(__file__).with_name("passages.json")))
    p.add_argument("--question", default="Which provider serves the Hailo-8, and how is its speed checked?")
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--cite", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=200)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
