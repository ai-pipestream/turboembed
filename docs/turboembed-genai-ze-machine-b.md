# TurboEmbed GenAI on turbo_buffer ZE USM — Machine B LIVE

Evidence that SOLIDIFY (4) is real on the Intel GPU host: OpenVINO
GenAI embed rents Level Zero USM from
[`include/turbo_buffer.h`](../include/turbo_buffer.h) for the
token / hidden / result slots the InferRequest API accepts. Steady-state
`turbo_buffer_alloc_counter() == 0` for those slabs. MiniLM cosine vs
committed goldens stays ≥ 0.99. No silent CPU remap.

Hostnames stay out of this file (Machine B only).

## Done criteria

| # | Gate | Proof |
|---|---|---|
| 1 | GPU tokens/results are ZE SHARED USM | `turboembed_engine_create(OPENVINO_GPU)` opens a ZE arena. Load rents i32 ids/mask/types + f32 hidden as SHARED. `turbo_buffer_ze_query` on those pointers and on `embed` result rows is SHARED, not HOST. `turbo_buffer_arena_owns` is 1. |
| 2 | Infer uses caller USM where the API allows | `ov::InferRequest.set_tensor` wraps the rented pointers (`ov::Tensor(element::i32, {n, seq}, usm)`). `TextEmbeddingPipeline.embed_documents` is **not** on the hot path — that call private-allocs `token_type_ids` and `EmbeddingResults` every embed and has no caller-buffer hook. |
| 3 | Tokenizer gap is named, not faked | `ov::genai::Tokenizer.encode` still returns an engine-owned `ov::Tensor` (API has no output buffer). We copy/cast into the rented i32 USM and never pass that encode tensor to infer. That copy is **item (5)** — not claimed fixed here. |
| 4 | allocs/forward == 0 after warmup | Load warms 32×256 token/hidden and 32×dim result. A second `embed_one("minilm", "hello world")` must see `turbo_buffer_alloc_counter() == 0`. Reintroducing a per-embed `std::vector<float>` result or `posix_memalign` for those slots fails the C++ / Rust checks. |
| 5 | Intel MiniLM receipts ≥ 0.99 cosine | `testdata/receipts/turboembed/intel-minilm.json` (GPU) and `intel-minilm-cpu.json` (CPU) vs `testdata/e2e/goldens/{intel,nvidia}/minilm.json`. |
| 6 | No silent CPU | GPU/AUTO create without ZE SHARED is `UNAVAILABLE` / `NOT_IMPLEMENTED` and names USM — it does not open a CPU arena. GPU create without the GPU plugin still fails loud. PINNED on ZE is still `NOT_IMPLEMENTED`. |

## Commands

```bash
source /work/opt/openvino_genai/setupvars.sh   # or the host OpenVINO setupvars
make test-turboembed-intel                     # C++ arena test + cargo --features genai
make turboembed-genai-arena-tests              # C++ only: GPU SHARED + CPU HOST, allocs==0
```

`TURBO_BUFFER_ZE` is compile-gated (`libze_loader` + `ze_api.h`) from
`crates/turboembed/build.rs` when `--features genai` is on. A binary
without L0 cannot pretend GPU embed rented USM.

## What is not this item

ORT CUDA / TensorRT device buffers (Machine A). Apple MLX Metal
(`libTurboEmbed.dylib` still does not link this arena). NPU create is
still fail-loud on this Battlemage host (`intel-npu.json`).
