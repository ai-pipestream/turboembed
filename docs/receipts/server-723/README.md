# Analysis results and capabilities over gRPC (#723)

Base: `f62034af` (origin/main at branch time). Round 3: cursor's,
fable's, codex's, and argon's second-round findings (the admin gate and
resource caps, the registration identity, and the pre-allocation column
bounds) are all fixed below, and origin/main is merged in. The Open
Inference Protocol tests, one 1,090-line file, split into
`server/tests/server.rs`, `server/tests/infer.rs`,
`server/tests/startup.rs`, and the shared `server/tests/kserve/mod.rs`
so every touched file stays under 500 lines.

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
- The rerank source key crosses whole: layer name and instance name, so
  `Key::named("shortlist-a")` and `Key::named("shortlist-b")` stay
  distinct on the wire (`server/src/results/rerank.rs`).
- The token-vector windows carry their text view: offsets of
  `ViewId(7)` decode on `ViewId(7)`, never on the source text
  (`server/src/results/tokens.rs`).
- Discontinuous answers carry their fragments beside the bounding
  envelope, through both conversions and gRPC
  (`server/src/results/extraction.rs`, `server/tests/results.rs`); the
  gap inside an answer is never answer text.
- Sparse occurrence columns cannot drop data: the last `occ_at` offset
  must be one past the last occurrence, and the zero-occurrence form
  (one all-zero offset per entry) round trips
  (`server/src/results/sparse.rs`).
- The schema-driven structured records of #715 have no merged layer
  yet; `ExtractionResult` carries what exists today (ranked answer
  spans, scores, probabilities, the abstention mass, and the model
  identity), and `BundleCapabilities.schema` stays unset until a task
  reads one. This is stated in the proto and the doc page, not invented.
- The capability listing matches the loaded bundle against its
  manifest.json (identity, task, roles from the prefixes, widths,
  context and batch limits), and follows `LoadBundle` and
  `UnloadBundle` while serving, in `server/tests/results.rs`.
- Load and unload are admin calls behind the server's `--admin` flag,
  off by default: without it both refuse with `TURBO_E_UNSUPPORTED`
  (`UNIMPLEMENTED`), tested in `server/tests/limits.rs`.
- A request is bounded before anything is allocated: `sessions` over
  `max_sessions_per_model` (32) or a registration past
  `max_loaded_models` (64) refuses with `TURBO_E_CAPACITY` naming the
  limit; both caps sit in `CapabilitiesResponse.server`. Unit tests in
  `server/src/capabilities.rs` and `server/src/state.rs`.
- One request's load, answer, and rollback carry the exact registration
  it made, as an `Arc<Served>` identity: `State::add` returns it,
  `load_one` takes it, rollback (`remove_if`) deletes the map entry
  under the write lock only when `Arc::ptr_eq` holds, and no name is
  cloned on the lookup path (`peek`/`ready` return the Arc). The
  deterministic identity test is `rollback_removes_only_the_exact_registration`
  in `server/src/state.rs`; `a_same_name_unload_and_replacement_never_loses_a_registration`
  in `server/tests/limits.rs` storms one name with valid loads, a
  failing load, and an unload, asserting the listing never shows the
  name twice and a listed name answers inference.
- Message columns refuse before allocating: `TokenVectorsResult` bounds
  `window_start` against `max_windows` and `empty` against the data's
  rows; `ExtractionResult` bounds `frag_start` against
  `max_answer_fragments`; both with `TURBO_E_CAPACITY` and unit tests.
- An over-limit request returns the named status for each listed limit:
  `context_limit` and `batch_limit` through `ModelInfer`
  (`TURBO_E_CAPACITY`, gRPC `OUT_OF_RANGE`, the message naming
  `max_seq` / `max_batch`, nothing truncated), and the result-message
  limits (`max_sparse_entries`, `max_sparse_occurrences`,
  `max_token_vector_bytes`, `max_rerank_items`, `max_answers`,
  `max_answer_fragments`, `max_windows`) through the RoundTrip rpcs, one
  refusal per limit, in `server/tests/limits.rs`.
- Malformed input never panics: every conversion re-proves the layer's
  own `check` on rebuilt columns, and garbage, empty, and oversized
  fields are refused with `TURBO_E_INVALID_ARGUMENT` naming the column,
  in `server/src/results/` and over the wire in
  `server/tests/limits.rs`.

## Speed

`speed.log`: 40 alternating rounds a side in release, one encode and one
decode timed together, two warm-up rounds, the accumulators keyed on the
implementation while the side running first still swaps every round.
Every result kind encodes and decodes at least as fast per byte as the
current vector message (the vendored protocol's fp32 contents) and span
pairs as its int contents carries them: 1.10x to 7.09x on the receipt
run.

## ModelInfer latency

The registry change returns the `Arc<Served>` from `peek`/`ready`
instead of cloning the model-name `String` on every ModelInfer lookup.
Measured on tiny-bert-bundle, release, 300 sequential one-text
ModelInfer calls after 50 warm-up calls: 41.40 ms a call before the
change (head `4261c754`), 41.22 ms after; a ~100 ns allocation is
~0.0002% of a 41 ms embed round trip, so the numbers sit inside run
noise, as expected.

## Gates

- `fmt.log`: `cargo fmt --all --check`, exit 0.
- `clippy.log`: `cargo clippy --locked -p turbo-kserve -p turbo-bench
  --all-targets -- -D warnings`, exit 0.
- `tests.log`: `cargo test --locked -p turbo-kserve`, 71 tests pass (40
  unit, 7 result-layer integration, 6 limit and concurrency
  integration, 18 server integration split across server, infer, and
  startup), exit 0.
- `tree.log`: `cargo test --locked -p turbo-bench --test tree`, exit 0.
