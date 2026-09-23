# inferstream-intel: full surface on Machine B — evidence report

> **Historical.** Embed on this pass used OVMS DAG pipelines. That client
> is **removed** — Intel Embed is in-process OpenVINO GenAI only
> (`docs/intel-genai-embed.md`). A host OVMS container may still exist;
> inferstream does not depend on it. Generation notes below (llama.cpp)
> are unchanged in intent.

Status of the `inferstream-intel` binary on the Machine B host (Intel
Arc/Battlemage GPU) after the full-surface pass: **Tokenize/Detokenize,
Embed, unary generation, and real token streaming through `ModelStreamInfer`
are all live**, each backed by a real engine — no mock in any verified path.

| Surface | RPC | Engine behind it | Status |
|---|---|---|---|
| Embed | `inferstream.v1/Embed` | OVMS DAG pipelines (`minilm_pipeline`, `mpnet_pipeline`), embedding model on the Battlemage GPU | live, goldens ≥ 0.999 cosine |
| Tokenize / Detokenize (embedders) | `inferstream.v1/Tokenize`, `/Detokenize` | local HF `tokenizer.json` (server-side, `tokenizer_dir`) | live |
| Tokenize / Detokenize (gen model) | same | llama-server `/tokenize` + `/detokenize` | live |
| Generation, unary | `inference.GRPCInferenceService/ModelInfer` | llama.cpp GGML_SYCL server (`vlm-server` container) | live |
| Generation, token streaming | `inference.GRPCInferenceService/ModelStreamInfer` | same, SSE → one `token` chunk per event | live |

## Host inventory used

* **OVMS embeddings** — container `protomolt-embedder-ovms`
  (`openvino/model_server:latest-gpu`), gRPC on the bridge IP
  `172.22.0.2:8000`. Serves the `minilm_pipeline` / `mpnet_pipeline` DAGs:
  `strings` → openvino_tokenizer (CPU) → embedding model (GPU) →
  `sentence_embedding`. Untouched by this pass.
* **llama.cpp SYCL** — container `vlm-server`
  (`ghcr.io/ggml-org/llama.cpp:server-intel`, a GGML_SYCL build, `-ngl 99`)
  serving `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M` on host port 8085.
  Used as-is in server-client mode; untouched by this pass.
* **HF tokenizers** — `tokenizer.json` for `all-MiniLM-L6-v2` and
  `all-mpnet-base-v2` copied from the HF hub cache to
  `~/ovms-models/hf_tokenizer_minilm/` and `~/ovms-models/hf_tokenizer_mpnet/`
  (next to the OVMS model dirs) and referenced via `tokenizer_dir` in
  `config/intel.toml`.

### Why not OVMS GenAI for streaming

The on-disk OVMS LLM models (`~/ovms-llm-models/OpenVINO/gemma-3-12b-it-int4-ov`
etc.) carry `graph.pbtxt` files built on `HttpLLMCalculator`
(`HTTP_REQUEST_PAYLOAD` / `HTTP_RESPONSE_PAYLOAD` streams). OVMS serves those
graphs only through the REST `/v3` OpenAI-compatible endpoints, not through
KServe gRPC `ModelInfer`/`ModelStreamInfer`, so they cannot back the OIP
streaming surface directly. The llama.cpp SYCL server was the sanctioned
alternative and delivers real per-token streaming today; an OVMS `/v3` client
backend remains possible follow-up work.

## What landed

* `crates/backend-llamacpp` — server-client mode. With `endpoint` set (or
  `INFERSTREAM_LLAMACPP_ENDPOINT` for path-less models) the backend forwards
  to a running llama-server over its native HTTP API:
  * `infer`: `POST /completion` (`stream:false`) → `BYTES` output `text`,
    plus `tokens_predicted` / `tokens_evaluated` response parameters.
  * `infer_stream`: `POST /completion` (`stream:true`), incremental SSE
    parsing → one `BYTES` `token` chunk per event, `final` bool parameter on
    the last chunk. Identical wire shape to the mock backend, so existing
    `ModelStreamInfer` clients work unchanged.
  * `tokenize` / `detokenize` via `/tokenize` (`with_pieces`) and
    `/detokenize`; `model_ready` via `/health`.
  * Generation parameters: `max_tokens` (int64 → `n_predict`, default 128),
    `temperature` (double), `top_p` (double), `seed` (int64), `stop` (string).
  * The `path`-only in-process FFI mode remains a stub, as before.
* `config/intel.toml` — `tokenizer_dir` on both OVMS pipelines; new live
  model `qwen2.5-vl-7b-sycl` (`backend = "llama-cpp"`,
  `endpoint = "http://127.0.0.1:8085"`).
* All three arch binaries pass `endpoint` through to the llama-cpp backend
  (env fallback only for models without a GGUF `path`).

## Evidence (2026-09-11, Machine B)

Default test suite: `cargo test --workspace` — all green, zero failures
(live suites stay `#[ignore]`d).

Live backend tests against the real SYCL server:

```
INFERSTREAM_LLAMACPP_ENDPOINT=http://127.0.0.1:8085 \
  cargo test -p inferstream-backend-llamacpp --test llamacpp_live -- --ignored
test bad_endpoint_reports_unavailable ... ok
test health_and_metadata ... ok
test tokenize_round_trips_through_detokenize ... ok
test unary_completion_returns_text ... ok
test streaming_emits_token_chunks_with_final_flag ... ok
```

OVMS embedding goldens (historical; the `backend-ovms` crate and these
commands are **removed** — use GenAI goldens in
`docs/intel-genai-embed.md`):

E2e through the running façade (`./target/release/inferstream-intel
--config config/intel.toml`, bearer auth, port 8461), all via `grpcurl`:

* `Tokenize` `minilm_pipeline` `"Hello world"` →
  ids `[101, 7592, 2088, 102]`, tokens `[CLS] hello world [SEP]`, offsets
  mapping back into the text; `Detokenize` of those ids → `"hello world"`.
* `Tokenize` `mpnet_pipeline` `"Hello world"` →
  `[0, 7596, 2092, 2]` = `<s> hello world </s>` (MPNet vocabulary).
* `Embed` `minilm_pipeline` `"The quick brown fox"` → 384-dim vector
  (computed on the OVMS GPU pipeline).
* `Tokenize` `qwen2.5-vl-7b-sycl` `"Hello, inferstream!"` →
  `[9707, 11, 23583, 4027, 0]` = `Hello | , | ␠infer | stream | !`
  (Qwen2.5 BPE, via llama-server).
* `ModelInfer` `qwen2.5-vl-7b-sycl`, prompt `"The capital of France is"`,
  greedy, 8 tokens → `" Paris. The capital of Germany is Berlin"`,
  `tokens_evaluated=5`, `tokens_predicted=8`.
* `ModelStreamInfer` `qwen2.5-vl-7b-sycl`, prompt `"Count from 1 to 5: "`,
  greedy, 24 tokens → **25 streamed chunks**, one `token` `BYTES` output per
  chunk, `final=true` only on the last (which also carries
  `tokens_predicted=24`); concatenated text:
  `"1, 2, 3, 4, 5.\nCount from 5 to 1: 5"`.

Streaming request shape for reference:

```bash
grpcurl -import-path crates/protocol/proto -proto open_inference_grpc.proto \
  -plaintext -H 'authorization: Bearer change-me' \
  -d '{"model_name":"qwen2.5-vl-7b-sycl","id":"s1",
       "parameters":{"max_tokens":{"int64_param":24},"temperature":{"double_param":0}},
       "inputs":[{"name":"text","datatype":"BYTES","shape":[1],
                  "contents":{"bytes_contents":["<base64 prompt>"]}}]}' \
  127.0.0.1:8461 inference.GRPCInferenceService/ModelStreamInfer
```
