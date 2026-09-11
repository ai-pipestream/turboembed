# Reference embeddings (goldens)

Fixed prompts embedded once by a known-good configuration, stored as JSON, and
compared against fresh outputs in tests. A regression in tokenization, pooling,
normalization, or the wire path shows up as a cosine-similarity drop.

## Schema

One JSON file per `(model, text, params)` tuple:

```json
{
  "model": "mock-embed",          // model name the server routes
  "backend": "mock",              // backend id that produced the golden
  "text": "hello world",          // exact input text
  "pooling": null,                // "mean" | "cls" | null (backend default)
  "normalize": false,             // whether L2 normalization was requested
  "dim": 8,                       // embedding dimension
  "l2": 2.0574267,                // L2 norm of the stored vector
  "vector": [0.886, ...]          // full vector (small dims)
}
```

For large GPU vectors (768/1024 dims) storing the full vector is still fine
(tens of KB), but the schema also tolerates a reduced form: replace `vector`
with `vector_head` / `vector_tail` (first/last N dims) plus `sha256` of the
full little-endian FP32 blob, and compare `l2` + head/tail cosine instead.

## Prompt set

| file | intent |
|---|---|
| `mock_short.json` | trivial baseline |
| `mock_medium.json` | full-sentence input |
| `mock_empty.json` | empty-string edge case |
| `mock_unicode.json` | multi-script + emoji input |
| `mock_long_truncation.json` | 600 repeated tokens — exercises `max_seq_len` truncation on real models |

## Mock goldens (always-on in CI)

The mock backend is deterministic on every platform, so its goldens run
unconditionally in `cargo test --workspace`
(`crates/server/tests/goldens.rs`, cosine ≥ 0.999 and exact-value check).

Regenerate after any intentional mock-algorithm change:

```bash
cargo run -p inferstream-server --example gen_reference_embeddings
```

## GPU goldens (krick / krick-1)

GPU goldens compare a real engine against a stored vector for the same model
and parameters. They are `#[ignore]`d and feature-gated so default CI never
needs a GPU (`crates/backend-ort/tests/gpu_goldens.rs`).

Regenerate/run on **krick** (NVIDIA, ORT CUDA EP):

```bash
scripts/fetch-runtime-libs.sh nvidia
cargo build -p inferstream-arch-nvidia --release --features ort-cuda

# 1. Generate a golden from the running engine (server on :8461):
scripts/run-nvidia.sh --config config/nvidia.toml &
grpcurl -plaintext -proto crates/protocol/proto/inferstream_extension.proto \
  -d '{"model_name":"minilm-l6-v2","texts":["hello world"],"normalize":true}' \
  127.0.0.1:8461 inferstream.v1.InferstreamService/Embed \
  > /tmp/minilm_hello.json
# then fill the schema above and save as ort_cuda_minilm_short.json

# 2. Verify against the stored golden:
INFERSTREAM_ORT_MODEL=/path/to/model.onnx \
INFERSTREAM_ORT_GOLDEN=testdata/reference_embeddings/ort_cuda_minilm_short.json \
cargo test -p inferstream-backend-ort --features cuda -- --ignored gpu_golden
```

On **krick-1** (Intel, OVMS client) the OVMS pipelines embed server-side; use
the `Embed` RPC through `inferstream-intel` against `minilm_pipeline` /
`mpnet_pipeline` and store the response in the same schema with
`"backend": "ovms"`.

Do **not** stop the OVMS or TEI services on either host to regenerate goldens
— the façade only reads from them.
