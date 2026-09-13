# gRPC output scratch (SOLIDIFY 6)

Inferstream Embed `PACKED_BYTES`, OIP `raw_output_contents` (TurboEmbed),
and Rerank score rows rent a **pre-sized freelist slab**. After warmup
of that shape, the façade does not grow a new output buffer per
unary/batch request.

This is not “protobuf has to copy, so we are done.” Prost/tonic still
copy the slab into the HTTP/2 frame. The **payload** the RPC owns is a
rented `Vec<u8>` / `Vec<f32>` wrapped as `bytes::Bytes` (`from_owner`)
so Drop returns the slab. Sequential Embed/Rerank after warmup must
see `inferstream_protocol::output_scratch::allocs() == 0`.

## What is reused

| Path | Slab | Who writes |
|---|---|---|
| `Embed` / `EmbedStream` `PACKED_BYTES` | LE FP32 `[n * dim * 4]` | `Backend::embed_packed_into` → rented dest |
| TurboEmbed `ModelInfer` | same blob in `raw_output_contents` | `output_scratch::pack_le_f32` |
| `Rerank` scores | `f32[n_docs]` | `Backend::rerank_into` / TurboRerank `score_into` |

Typed `Embed` (`repeated float`) still materializes per-row `Vec<f32>`
for grpcurl / e2e. The cheap path is `output_format = PACKED_BYTES`.

## Mock path

Explicit mock models (`mock-embed`, word-overlap Rerank) still run
`pack_fp32` / word-overlap inside `backend-mock`. The façade copies
that blob into the **same** output scratch. Mock does not become
MiniLM and does not sit on catalog CE aliases.

## Tests

```bash
cargo test -p inferstream-protocol output_scratch
cargo test -p inferstream-server packed_bytes_and_rerank_reuse
# Berlin / façade when weights are present (unchanged skip-if-missing):
cargo test -p inferstream-backend-turborerank -- rpc_berlin
```

Machine B latency bench is **LIVE**
(`docs/solidify-bench-machine-b.md`, `make bench-machine-b-ov`).
Machine A/C benches are separate host receipts under
`testdata/receipts/bench/`. SOLIDIFY (7) **Intel** wrap / accuracy is
`docs/intel-remote-usm-machine-b.md`. The Machine C special-function
slice (Metal GELU erf) is `docs/apple-turborerank-metal-gelu-machine-c.md`.

## Apple

`InferstreamCore.OutputScratch` is the same size-class freelist.
`ExtensionService` rents it for `PACKED_BYTES`; `TurboEmbedBackend`
writes OIP `rawOutputContents` into a rented `Data`.
