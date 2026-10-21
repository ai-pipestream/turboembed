# Streaming bulk document analysis over gRPC (#722)

Base: `8dc5bf2c` (origin/main at branch time), which carries PR #785
(the gRPC analysis service and capabilities). Round 2: the four P1
findings from codex, fable, and cursor are fixed below, and origin/main
is merged in (including #798's TCP_NODELAY, which un-stalls the unary
speed baseline). All streaming code lives in `server/src/stream/`; the
edits to existing files are append-only (a field in `State`, the
`stream_stats` accessor, one trait method delegating to the stream
handler, the `stream` field in `ServerLimits`, and `state` stays
crate-private). Nothing is mocked anywhere.

## The call

`AnalyzeStream` on `turbo.result.AnalysisService`: one bidirectional
stream. The first message is an `AnalyzeStreamHeader` (ordering
`ANALYZE_ORDER_INPUT` or `ANALYZE_ORDER_COMPLETION`, `max_inflight`,
`batch_linger_ms`, each capped by what `Capabilities` advertises under
`server.stream`); then `AnalyzeDocument` messages with a client
correlation `id`, the served model's name, the text, and
`AnalyzeOptions` (the fields of `turbo_embed_options`, 0 for model
defaults). Each document answers exactly once, tagged with its id:
`AnalyzeVectors` whose `data` is byte for byte the row the
single-document call's `raw_output_contents` carries, or
`AnalyzeFailure` with the turbo status, field and refusal the
single-document call would carry.

## Flow control and cancellation

- Per stream and server-wide caps on queued documents and queued bytes
  (256 / 4 MiB per stream, 4096 / 64 MiB server-wide) held as
  semaphores admitted in one order everywhere; the byte semaphores
  reserve each document's own byte count, so byte pressure blocks
  admission while item permits remain, and a document over a byte cap
  refuses at once with `TURBO_E_CAPACITY` naming the limit instead of
  waiting forever. A send past the caps waits on backpressure, never a
  mid-stream error. The response channel is bounded by the window.
- Every admitted message, a validation failure included, holds its
  queue room and window permit from admission until its answer is sent;
  input-order buffering attaches them to the buffered response, so
  failures and out-of-order results are bounded by the same caps, and
  the earliest outstanding sequence always makes progress.
- The window (`max_inflight`, default and cap 64) bounds documents
  executing or awaiting their answer; window permits travel with the
  batch outcome and free where the answer frees, on cancellation too.
- Cancelling the stream drops what was queued for it. The reader
  distinguishes a client reset from a graceful half-close: a reset
  aborts the scheduler, a half-close drains it. A batch already running
  on a session finishes (the FFI run cannot be aborted), but its answer
  goes nowhere. Waiting for a session never blocks the scheduler: a
  batch whose pool is busy waits on `stalled`, retried from a timer
  whose deadline stays put across loop iterations, while the loop keeps
  servicing completions, new work and cancellation.
- `Server::stream_stats` exposes the model-execution count and the
  admitted-not-yet-answered gauges; both `server.stream` constants are
  public so the advertised caps and the enforced ones are proven the
  same.

## Batching

Documents waiting on the same load with the same options gather into
one batch up to the bundle's `batch_limit`, one write, one run and one
read on one held session; the batching key is the load's identity
(`Arc::as_ptr` of the `Loaded`), so an unload and a reload under one
name never mix loads in one batch, and each response's model and
version stay tied to the load its document was admitted against. A
failed batch is retried one document at a time, so the failure lands on
its document alone.

## What is proven where

`server/tests/stream.rs`: byte-for-byte parity with `ModelInfer` across
the harness reference texts in both ordering modes; eight same-key
documents run as one counted model execution with unchanged results;
a non-reading client never pushes queued items or bytes past the
advertised caps and every sent document still answers; cancellation
drops the queued work by the model-execution count; failures (unknown
model, invalid option, over-limit beside a sound document in one batch)
answer in input order with the unary call's codes.

`server/tests/stream_bounds.rs`, the reviewers' probes: two 2 MiB
documents fill the stream byte cap and the third is held although item
permits remain; a document over the byte cap refuses at once naming
`max_queued_bytes_per_stream`; with a window of one and the first
document in flight, five hundred validation failures fill the queue to
its advertised cap and no further; cancelling while another caller
holds the model's only session drains the queue within milliseconds;
and an unload plus reload under one name answers the pre-swap document
from the old load while the over-length post-swap document is refused
with 771, never batched together.

## Speed

`speed.log`, post-#798 (TCP_NODELAY): 512 documents per round, 6
alternating rounds, release, tiny-bert-bundle on the CPU, 8 sessions,
the unary side 8 concurrent `ModelInfer` calls. Stream: `62818
docs/sec`; concurrent single calls: `13251 docs/sec`; `4.74x`. The bar
(stream at least matches) holds. (The round-1 figure of 86.60x measured
the unary side's Nagle stall, which #798 removed; the receipt is now
taken against the un-stalled baseline.)

## Gates

- `fmt.log`: `cargo fmt --all --check`, exit 0.
- `clippy.log`: `cargo clippy --locked -p turbo-kserve -p
  turbo-kserve-demo -p turbo-bench --all-targets -- -D warnings`,
  exit 0.
- `tests.log`: `cargo test --locked -p turbo-kserve -p
  turbo-kserve-demo`: 45 unit, 6 stream, 5 stream-bounds, 6 limit, 7
  result-layer, 7 server, 5 startup, 7 infer, 3 demo; both release
  speed bars ignored; exit 0.
- `tree.log`: `cargo test --locked -p turbo-bench --test tree`,
  exit 0.
