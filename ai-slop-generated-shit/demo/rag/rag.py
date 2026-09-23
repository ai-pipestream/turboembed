#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = ["openai>=1.50"]
# ///
"""Retrieval-augmented answer over Inferstream with the OpenAI SDK.

    uv run demo/rag/rag.py --question "which provider runs on the Hailo-8?"

Three models on one server, driven through the OpenAI-shaped routes:

1. embed   POST /v1/embeddings      the passages once, then the question
2. rerank  POST /v1/rerank          the nearest passages, scored by the cross-encoder
3. answer  POST /v1/chat/completions  streamed, with the passages cited as [n]

The passages are in demo/rag/passages.json; the answer cites them by
number and the sources are printed after it. Every failure raises with the
server's status; nothing is retried or defaulted.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

from openai import OpenAI


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--embed-model", default="minilm")
    p.add_argument("--rerank-model", default="rerank")
    p.add_argument("--chat-model", default="qwen")
    p.add_argument("--passages", default=str(Path(__file__).with_name("passages.json")))
    p.add_argument("--question", default="Which provider serves the Hailo-8, and how is its speed checked?")
    p.add_argument("--candidates", type=int, default=8, help="passages kept by the embedding search")
    p.add_argument("--cite", type=int, default=3, help="passages given to the model after reranking")
    p.add_argument("--max-tokens", type=int, default=200)
    args = p.parse_args()

    passages = json.loads(Path(args.passages).read_text())
    texts = [f"{x['title']}: {x['text']}" for x in passages]
    client = OpenAI(base_url=args.base_url, api_key="none")

    # 1. Embeddings: the corpus in one call, then the question. (A bundle
    #    with query and document prefixes takes prompt_role in extra_body;
    #    MiniLM declares none and the server rejects the option.)
    t0 = time.perf_counter()
    corpus = client.embeddings.create(model=args.embed_model, input=texts)
    query = client.embeddings.create(model=args.embed_model, input=args.question)
    t1 = time.perf_counter()
    vectors = [d.embedding for d in corpus.data]
    q = query.data[0].embedding
    ranked = sorted(range(len(texts)), key=lambda i: -cosine(q, vectors[i]))[: args.candidates]
    print(f"embed   {len(texts)} passages + 1 question in {1000 * (t1 - t0):.1f} ms "
          f"(dim {len(q)}, device {corpus.model_extra.get('turbo', {}).get('device', '?')})")

    # 2. Rerank the candidates with the cross-encoder. The SDK has no rerank
    #    method; its request layer posts the route directly.
    t2 = time.perf_counter()
    rr = client.post(
        "/rerank",
        cast_to=dict,
        body={"model": args.rerank_model, "query": args.question, "documents": [texts[i] for i in ranked], "top_n": args.cite},
    )
    t3 = time.perf_counter()
    chosen = [(ranked[r["index"]], r["relevance_score"]) for r in rr["results"]]
    print(f"rerank  {len(ranked)} candidates in {1000 * (t3 - t2):.1f} ms (device {rr.get('turbo', {}).get('device', '?')})")
    for n, (i, score) in enumerate(chosen, 1):
        print(f"  [{n}] {score:.3f}  {passages[i]['title']}")

    # 3. Generate with the chosen passages, streamed.
    context = "\n\n".join(f"[{n}] {texts[i]}" for n, (i, _) in enumerate(chosen, 1))
    messages = [
        {"role": "system", "content": "Answer from the numbered passages only, in two or three sentences, "
                                      "and cite each fact with its passage number in square brackets. "
                                      "If the passages do not answer the question, say so."},
        {"role": "user", "content": f"Passages:\n\n{context}\n\nQuestion: {args.question}"},
    ]
    print(f"\nanswer  ({args.chat_model}, streamed)\n")
    t4 = time.perf_counter()
    first = None
    tokens = 0
    finish = None
    stream = client.chat.completions.create(model=args.chat_model, messages=messages, max_tokens=args.max_tokens, stream=True)
    for chunk in stream:
        choice = chunk.choices[0]
        if choice.delta.content:
            if first is None:
                first = time.perf_counter()
            tokens += 1
            sys.stdout.write(choice.delta.content)
            sys.stdout.flush()
        if choice.finish_reason:
            finish = choice.finish_reason
    t5 = time.perf_counter()
    if first is None:
        raise SystemExit("the model produced no text")
    print(f"\n\nfirst token {1000 * (first - t4):.0f} ms, {tokens} chunks in {t5 - t4:.2f} s, finish {finish}")
    print("\nsources")
    for n, (i, _) in enumerate(chosen, 1):
        print(f"  [{n}] {passages[i]['title']} ({passages[i]['source']})")


if __name__ == "__main__":
    main()
