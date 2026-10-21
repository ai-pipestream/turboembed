# Streaming bulk document analysis over gRPC (#722)

Base: `8dc5bf2c` (origin/main at branch time), which carries PR #785
(the gRPC analysis service and capabilities). All new code lives in
`server/src/stream/` (limits and shared state, reader, scheduler,
executor, handler), the new rpc and messages in
`server/proto/turbo_analysis.proto`, and the proofs in
`server/tests/stream.rs` plus the release speed bar in
`server/tests/stream_speed.rs`; the edits to existing files are
append-only (a field in `State`, the `stream_stats` accessor, one
trait method delegating to the stream handler, the `stream` field in
`ServerLimits`). Nothing is mocked anywhere.

## The call

`AnalyzeStream` on `turbo.result.AnalysisService`: one bidirectional
stream. The first message is an `AnalyzeStreamHeader` (ordering
`ANALYZE_ORDER_INPUT` or `ANALYZE_ORDER_COMPLETION`, `max_inflight`,
`batch_linger_ms`, each capped by what `Capabilities` advertises under
`server.stream`); then `AnalyzeDocument` messages with a client
correlation `id`, the served model's name, the text, and
`AnalyzeOptions` (the fields of `turbo_embed_options`, 0 for model
defaults). Each document answers exactly once, tagged with its id:
`AnalyzeVectors` whose `data` is the row the single-document call's
`raw_output_contents` carries, or `AnalyzeFailure` with the turbo
status, field and refusal the single-document call would carry.

## Flow control and cancellation

- Per stream and server-wide caps on queued documents and queued
  bytes (`256` / `4 MiB` per stream, `4096` / `64 MiB` server-wide),
  held as semaphores admitted in one order everywhere. A send past
  them waits on backpressure, never a mid-stream error. The response
  channel is bounded by the window, so a client that does not read
  stalls the engine, then the queue, then the reader.
- The window (`max_inflight`, default and cap `64`) bounds documents
  executing or awaiting their answer; window permits travel with the
  batch outcome and free where the answer frees, on cancellation too.
- Cancelling the stream drops what was queued for it. The reader
  distinguishes a client reset from a graceful half-close: a reset
  aborts the scheduler, a half-close drains it. A batch already
  running on a session finishes (the FFI run cannot be aborted), but
  its answer goes nowhere.
- `Server::stream_stats` exposes the model-execution count and the
  queued gauges; both `server.stream` constants are public so the
  advertised caps and the enforced ones are proven the same.

## Batching

Documents waiting on the same model with the same options gather into
one batch up to the bundle's `batch_limit`, one write, one run and one
read on one held session; a batch whose write or run fails comes back
whole and is retried one document at a time, so the failure lands on
its document alone and never changes a sibling's result.

## What is proven where (server/tests/stream.rs, over a real client)

- `stream_results_match_single_document_calls_byte_for_byte_in_both_orders`:
  the harness reference texts (empty, ASCII, accents, CJK, emoji and
  tabs, long documents) answer byte for byte what `ModelInfer` answers
  for each, in both ordering modes; input order answers as sent.
- `documents_waiting_on_one_model_run_as_one_batch_with_the_same_results`:
  eight same-key documents run as one model execution (counted in
  `stream_stats().runs`) and every answer still matches its unary row.
- `a_client_that_sends_without_reading_is_held_to_the_advertised_caps`:
  three times the item cap floods the server while the test samples
  `stream_stats`; queued items and bytes never pass the advertised
  caps, and every sent document still answers after the flood.
- `cancelling_a_stream_drops_the_work_queued_for_it`: with window 1
  and a 1 s linger the first document runs alone; seven more arrive
  queued; the cancel lands long before the next dispatch, and the
  model-execution count settles at exactly the one run already made.
- `a_failed_document_returns_its_error_with_its_id_and_the_stream_continues`:
  unknown model (`TURBO_E_BUNDLE_NOT_FOUND`), unknown option value
  (`TURBO_E_INVALID_ENUM`) and an over-limit document beside a sound
  one in one batch (`TURBO_E_CAPACITY`, same code as the unary
  refusal) each answer with their own id, in input order, and the
  stream completes.
- `the_stream_advertises_and_enforces_its_caps`: the listing shows the
  same constants the code enforces; over-cap `max_inflight`, unknown
  order, and a missing or misplaced header each refuse before any work.

## Speed

`speed.log`: `512` documents per round, `6` alternating rounds (which
side runs first swaps every round), release, tiny-bert-bundle on the
CPU, `8` sessions, the unary side `8` concurrent `ModelInfer` calls.
Stream: `17800 docs/sec`; concurrent single calls: `206 docs/sec`;
`86.60x`. The bar (stream at least matches) holds by a wide margin.

## Gates

- `fmt.log`: `cargo fmt --all --check`, exit 0.
- `clippy.log`: `cargo clippy --locked -p turbo-kserve -p
  turbo-kserve-demo -p turbo-bench --all-targets -- -D warnings`,
  exit 0.
- `tests.log`: `cargo test --locked -p turbo-kserve -p
  turbo-kserve-demo`: 45 unit, 6 stream integration, 6 limit, 7
  result-layer, 7 server, 5 startup, 6 infer, 3 demo; both release
  speed bars ignored; exit 0.
- `tree.log`: `cargo test --locked -p turbo-bench --test tree`,
  exit 0.
