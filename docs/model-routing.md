# Role-aware model routing

`turbo_analysis_embed::routing` chooses among model handles the application
already loaded. It opens no files and downloads nothing. A handle can be an
index into a serving session pool or an owned runtime object. `ModelProfile`
binds it to an exact `ModelRef` and validated `RoleCapability` records:

- static vectors, dense query, dense document, sparse, token vectors or rerank;
- output width and normalization;
- maximum encoded context, including role prefixes and special tokens;
- positive timed cost per item for that role and width.

Use `ModelProfile::from_bundle` to check capabilities against an opened core
bundle. Dense widths must be the full dimension or declared `output_dims`;
token-vector widths use the projection dimension when present. Reranking has
one unnormalized logit. The core manifest declares a learned sparse output
(`embed.sparse`), but `from_bundle` does not bind the sparse roles yet, so a
sparse runtime registers its facts with `ModelProfile::new`. This explicit
constructor does not assert that a bundle or runtime has been loaded: that is
the caller's responsibility, as is the accuracy of the role descriptors.

Timings belong to the serving host, device, thread count, batch configuration
and input shape. Supply a conservative per-item cost covering the registered
context limit. Rebuild profiles when these conditions change. The router does
not promise a wall-clock deadline or extrapolate from another machine.

## Selection

`Task::Index` requires dense document output; `Task::Query` requires dense
query output. Other tasks explicitly request rerank, static, sparse index,
sparse query or token vectors. `Request` supplies positive item and encoded
token counts, a latency or finite positive throughput budget, and optional
exact width, normalization and destination `Space` constraints.

`RoutingPolicy::Fastest` chooses the smallest timed cost, then the widest
output. `Widest` chooses the widest output, then the smallest cost. Ties use
model id, revision, digest, then normalization in a fixed order. Registration
order does not matter. Width is a policy preference, not a claim of comparative
quality across unrelated models. Only registered configurations are eligible.

The total latency estimate is the checked product of per-item cost and item
count. Throughput uses its reciprocal. Overflow, unsupported context or a
budget no compatible model meets produces a typed error; limits are never
relaxed. Failure contains every registered model/role/width/context/timed cost.
Successful selection borrows the handle, `ModelRef`, capability, `Space` and
`Rationale` and allocates nothing.

## Scoring

Pass `index.space()` as `Request::space` when choosing a query encoder. This
prevents a cheap model from winning a request for another model's index.

Retain the actual producer's space beside encoded output in `QueryVector`.
`Route::top_k` checks the selected space, query producer and destination index
before scoring, including `k == 0` or an empty index. A different model id,
revision, digest, width or normalization produces `RoutingError::CrossSpace`.
The index also checks vector length and finite numeric values. Sparse, rerank,
and token-vector roles cannot use this dense scoring entry point. Existing
vectors must never be relabeled with a newly selected model's identity.

## Verification

`analysis/embed/tests/routing.rs` covers deterministic selection, exact budget
boundaries, invalid inputs, overflow, role/width checks and space errors.
`analysis/embed/tests/routing_inference.rs` runs two actual same-width bundle
identities and a reranker, and verifies that their outputs cannot cross spaces.
`analysis/embed/tests/routing_allocation.rs` checks zero successful-route
allocations. Run `routing_timing` as a release example pinned to one CPU for
12 ABBA rounds comparing routed and direct calls to the same loaded encoder.

The initial receipt is `docs/receipts/routing-721/timing.txt`: CPU 4, one CPU
thread, release, 16-token tiny BERT, 3000 requests per leg, 12 ABBA rounds.
Median paired change was +0.0890%; individual rounds ranged from -16.4398%
to +5.8721%, so this run did not establish a consistent routing slowdown.
This is an end-to-end inference comparison on one host and one registry size,
not a claim of zero CPU instructions or a guarantee for large registries.
