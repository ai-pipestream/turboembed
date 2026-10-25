# Cross-encoder reranking

`TASK_RERANK` scores a query and passage jointly with a single relevance logit.
It runs the native CPU BERT or RoBERTa encoder and sequence-classification head.
The score is a raw logit, not a probability and not an embedding similarity.
The initial supported bundles are `cross-encoder/ms-marco-MiniLM-L6-v2` and
`BAAI/bge-reranker-base`. Other backends return an unsupported-task error.

## Pair construction

The manifest's `rerank` block declares `max_seq`, `max_batch`, `query_tokens`,
and `pair_template`. `PAIR_BERT` constructs `[CLS] query [SEP] passage [SEP]`,
with passage segment IDs set to one. `PAIR_ROBERTA` constructs
`<s> query </s></s> passage </s>` with zero segment IDs.

The query is truncated on the right to its budget before allocating passage
space. Every passage token is covered by a window, and the largest window
logit is the pair's score. An empty passage still has one window. Runtime
options can lower the sequence limit, change the query budget, or request
an overlapping passage stride. A `max_tokens` of zero selects the session's
`max_seq` in `turbo_rerank_write_pairs`, where a larger value is
`TURBO_E_CAPACITY`, and the manifest's rerank `max_seq` in
`turbo_tokenizer_encode_pair`, which has no session and where a larger
value is `TURBO_E_INVALID_ARGUMENT` on field 2. A `query_tokens` of zero
selects the manifest's budget, and a `stride` of zero non-overlapping
windows. Invalid budgets or a stride exceeding passage capacity return an
error. Window storage grows with passage length;
callers should bound document sizes at ingestion.

## APIs

`turbo::rerank::Reranker::open(bundle, threads)` creates a safe CPU session.
`score(&[(query, passage)], RerankOptions::default())` returns raw scores in
input order plus the model ID, revision, manifest hash, and artifact hash.
It batches any number of pairs to the model capacity. Empty input returns
an empty score vector with the model identity.

The C API in `include/turbo/turbo.h` provides `turbo_rerank_write_pairs`, followed
by the existing session run and result-read calls. The result is F32 with
shape `[pair_count, 1]`. `turbo_tokenizer_encode_pair` exposes the first window's
joint IDs and segment IDs for inspection. `turbo_model_info` reports the task,
query budget, and pair-template constant; both previous structure sizes
remain accepted. Rerank sessions report tokenization, encoding, head, and
pooling stages separately in the stage placement array.

`turbo_analysis_embed::RerankAnnotator` reads a view, `Levels`, and a
`Shortlist<Levels>` and writes `Reranked<Levels>`. Build a shortlist with
`Shortlist::from_hits(query, source_key, hits, k, row_map)` from
`turbo-vector-index` hits. The row map connects index rows to typed level
indices. Only those candidates are scored. Discontinuous fragments are
joined with a single space. Results retain typed source indices, descending
scores, and model provenance; equal scores retain shortlist order. Use
`RerankAnnotator::on(key)` for a named Levels instance. Invalid references,
view mismatches, non-finite input hit scores, and invalid UTF-8 spans fail
before inference.

## Build and verify bundles

The two recipes in `bundle/recipes/` pin upstream commits and every downloaded
file hash. Build the reference container from `bundle/reference/Dockerfile`;
its image must match the content digest recorded in the recipe. The reference
program uses upstream Transformers FP32 CPU inference. Conversion only
rewrites safetensors alignment, retaining all tensor values. Runtime inference
does not use Transformers, ONNX, or Python.

From the repository root, with a worktree-local target directory:

```sh
export CARGO_TARGET_DIR="$PWD/target"
cargo run --locked -p turbo-bundle -- make cross-encoder/ms-marco-MiniLM-L6-v2 target/rerank-minilm/upstream target/rerank-minilm/bundle
cargo run --locked -p turbo-bundle -- make BAAI/bge-reranker-base target/rerank-bge/upstream target/rerank-bge/bundle
cargo run --release --locked -p turbo --example rerank -- target/rerank-minilm/bundle 1
cargo run --release --locked -p turbo --example rerank -- target/rerank-bge/bundle 1
```

The comparison checks all five pinned reference pairs, including empty input,
Unicode, query truncation, and a passage spanning multiple windows. Missing
classifier tensors and unsupported task values produce named loader errors.
Small real-encoder fixtures cover repeated calls, batching, error recovery,
shortlists, and model identity in regular tests.

## CPU measurements

Measured on 2026-10-20, AMD Ryzen 9 9950X3D, CPU affinity 0, one inference
thread, batch four, three warmups, and the median of fifteen timed iterations.
Both baselines and native inference used the same local FP32 models and four
short query/passage pairs. Model loading is excluded. TEI includes loopback
HTTP overhead; CrossEncoder and native calls are in-process. These are local
measurements for this workload, not claims about other hardware or batches.

| Model | Largest reference score error | Native pairs/s | CrossEncoder pairs/s | TEI pairs/s |
| --- | ---: | ---: | ---: | ---: |
| MS MARCO MiniLM L6 v2 | 0.000004292 | 588.66 | 288.85 | 216.04 |
| BGE reranker base | 0.000010490 | 35.81 | 12.56 | 28.86 |

Both score errors are below `1e-4`. Score parity uses Transformers, not TEI;
TEI's BGE scores differed slightly from that reference. CrossEncoder was
sentence-transformers 6.1.0 with torch 2.14.0+cpu. TEI was CPU 1.8, image digest
`sha256:8de25e75ce39617f17f2f6c77d60a4f75b65e779ed5005420eb1400072a15c1c`.

Reproduce native timing with `taskset -c 0` and the `rerank_speed` example.
`bundle/reference/rerank_speed.py` runs the matching CrossEncoder batch;
set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` and pin its container to CPU 0.
For TEI, run the CPU image with the same affinity and thread settings, mount
the upstream model at `/data`, select F32, and send the four passages from
`core/examples/rerank_speed.rs` to `/rerank` with `raw_scores: true`.
Compact run outputs are in `docs/receipts/rerank-707/`.
