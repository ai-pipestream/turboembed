# RAG over Inferstream

Embed, rerank, generate with citations: three models on one Inferstream
server, driven by two stock clients.

| script | client | routes |
|---|---|---|
| `rag.py` | the OpenAI Python SDK | `/v1/embeddings`, `/v1/rerank`, `/v1/chat/completions` (streamed) |
| `rag_oip.py` | KServe's `InferenceRESTClient` or `InferenceGRPCClient` | `ModelInfer` on each model, the Open Inference Protocol v2 |

Both read `passages.json` (twelve short passages about this repository),
embed them and the question, keep the nearest eight, rerank those with
the cross-encoder, and hand the best three to the generative model with
the instruction to cite passage numbers. Nothing is cached between runs;
the timings printed are the server's.

## Run

Start the server with an embedding model, a reranker and a generative
model (on `krick`, MiniLM and Qwen2.5-0.5B as GGUF on the 4080 through
`ggml`, the ms-marco reranker through `cuda`):

```sh
LD_LIBRARY_PATH=.libs/nvidia/lib TURBO_CUDA_LIB_DIR=.libs/nvidia/lib \
target/debug/inferstream \
    --provider-lib target/release/libturbo_provider_cuda.so \
    --provider-lib target/release/libturbo_provider_ggml.so \
    --model "name=minilm,bundle=$HOME/opt/bundles/minilm-gguf,provider=ggml,buckets=1x256;16x256;32x256" \
    --model "name=rerank,bundle=$HOME/opt/bundles/rerank-onnx,provider=cuda,buckets=8x256;16x256" \
    --model "name=qwen,bundle=$HOME/opt/bundles/qwen05-gguf,provider=ggml,generations=2"
```

Then, with `uv` (each script names its one dependency in its header, so
nothing is installed into the system Python):

```sh
uv run demo/rag/rag.py
uv run demo/rag/rag_oip.py                         # REST
uv run demo/rag/rag_oip.py --grpc 127.0.0.1:8001   # gRPC
uv run demo/rag/rag.py --question "what is a session bucket?"
```

The output of `rag.py` on `krick` (2026-09-22):

```
embed   12 passages + 1 question in 20.8 ms (dim 384, device NVIDIA GeForce RTX 4080 SUPER (CUDA0))
rerank  8 candidates in 67.5 ms (device NVIDIA GeForce RTX 4080 SUPER (sm_89))
  [1] 0.971  Hailo provider
  [2] 0.949  Providers
  [3] 0.317  Matched benchmarks

answer  (qwen, streamed)

The Hailo-8 is served by the Hailo provider. The speed of the Hailo-8 is
checked against a reference program that drives the vendor runtime
directly on the same tokens.

first token 61 ms, 41 chunks in 0.11 s, finish stop

sources
  [1] Hailo provider (providers/hailo/README.md)
  [2] Providers (docs/providers.md)
  [3] Matched benchmarks (reference/README.md)
```

`rag_oip.py` gives the same three passages and the same answer through
`ModelInfer`, with `finish_reason`, `prompt_tokens` and
`generated_tokens` in the response parameters. The protocol has no
streamed call; the streamed form is the extension rpc
`turbo.inferstream.InferstreamExtension/ModelStreamInfer`, which
`grpcurl` reaches through the server's reflection:

```sh
grpcurl -plaintext -d '{"model_name":"qwen","inputs":[{"name":"prompt","datatype":"BYTES","shape":[1],
  "contents":{"bytes_contents":["V2hhdCBpcyBhIGJ1bmRsZT8="]}}],"parameters":{"max_tokens":{"int64_param":32}}}' \
  localhost:8001 turbo.inferstream.InferstreamExtension/ModelStreamInfer
```

## What the scripts do not do

- Ask for a `prompt_role`: MiniLM declares no query or document prefix,
  and the server rejects the option (`TURBO_E_INVALID_ARGUMENT`, field
  4) rather than silently ignoring it. A bundle with prefixes (nomic,
  e5) takes `prompt_role` in `extra_body` on the OpenAI route and in the
  request parameters on OIP.
- Retry, truncate or default anything: an over-long passage is rejected
  with the bucket limit, a missing model is 404, and the script stops
  with the server's status.
