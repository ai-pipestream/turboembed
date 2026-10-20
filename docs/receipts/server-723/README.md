# Analysis results and capabilities over gRPC (#723)

Base: `f62034af` (origin/main at branch time).

## What is proven where

- Each result kind crosses the wire and back unchanged from its
  in-process layer: `SparseTerms` (#708) as `SparseTermsResult`,
  `Vectors<Subwords>` with `TokenVectorWindows` (#709) as
  `TokenVectorsResult`, `Reranked` (#707) as `RerankScoresResult`, and
  the `Answers` layer of extractive QA (#745) as `ExtractionResult`.
  Round trips run at two levels: the conversions themselves
  (`server/src/results/`, encode then decode compared field by field
  with the layer) and over a real gRPC channel against the server
  in-process on the CPU (`server/tests/results.rs`), where the RoundTrip
  rpc's answer is asserted equal to the request. Nothing is mocked.
- The schema-driven structured records of #715 have no merged layer
  yet; `ExtractionResult` carries what exists today (ranked answer
  spans, scores, probabilities, the abstention mass, and the model
  identity), and `BundleCapabilities.schema` stays unset until a task
  reads one. This is stated in the proto and the doc page, not invented.
- The capability listing matches the loaded bundle against its
  manifest.json (identity, task, roles from the prefixes, widths,
  context and batch limits), and follows `LoadBundle` and
  `UnloadBundle` while serving, in `server/tests/results.rs`.
- An over-limit request returns the named status for each listed limit:
  `context_limit` and `batch_limit` through `ModelInfer`
  (`TURBO_E_CAPACITY`, gRPC `OUT_OF_RANGE`, the message naming
  `max_seq` / `max_batch`, nothing truncated), and the result-message
  limits (`max_sparse_entries`, `max_sparse_occurrences`,
  `max_token_vector_bytes`, `max_rerank_items`, `max_answers`) through
  the RoundTrip rpcs, one refusal per limit, in
  `server/tests/results.rs`.
- Malformed input never panics: every conversion re-proves the layer's
  own `check` on rebuilt columns, and garbage, empty, and oversized
  fields are refused with `TURBO_E_INVALID_ARGUMENT` naming the column,
  in `server/src/results/` and over the wire in
  `server/tests/results.rs`.

## Speed

`speed.log`: 40 alternating rounds a side in release, one encode and one
decode timed together, two warm-up rounds. Every result kind encodes and
decodes at least as fast per byte as the current vector message (the
vendored protocol's fp32 contents) and span pairs as its int contents
carries them: 1.01x to 1.08x.

## Gates

- `fmt.log`: `cargo fmt --all --check`, exit 0.
- `clippy.log`: `cargo clippy --locked -p turbo-kserve --all-targets --
  -D warnings`, exit 0.
- `tests.log`: `cargo test --locked -p turbo-kserve`, 54 tests pass (28
  unit, 8 result-layer integration, 18 server integration), exit 0.
- `tree.log`: `cargo test --locked -p turbo-bench --test tree`, exit 0.
