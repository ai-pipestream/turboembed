# Cross-encoder reranking

`TASK_RERANK` scores a query and passage jointly with a single relevance logit.
It runs the native BERT or RoBERTa encoder and sequence-classification head,
on the CPU or on a device backend (see [Devices](#devices)).
The score is a raw logit, not a probability and not an embedding similarity.
The initial supported bundles are `cross-encoder/ms-marco-MiniLM-L6-v2` and
`BAAI/bge-reranker-base`. A backend with no embedding encoder returns an
unsupported-task error that names the embedding encoder.

## Pair construction

The manifest's `rerank` block declares `max_seq`, `max_batch`, `query_tokens`,
and `pair_template`. `PAIR_BERT` constructs `[CLS] query [SEP] passage [SEP]`,
with passage segment IDs set to one. `PAIR_ROBERTA` constructs
`<s> query </s></s> passage </s>` with zero segment IDs.

The query is truncated on the right to its budget before allocating passage
space. Every passage token is covered by a window, and the largest window
logit is the pair's score. An empty passage still has one window. Runtime
options can lower the sequence limit, change the query budget, or request
an overlapping passage stride. Zero selects the manifest limit or, for
stride, non-overlapping windows. Invalid budgets or a stride exceeding
passage capacity return an error. Window storage grows with passage length;
callers should bound document sizes at ingestion.

## APIs

`turbo::rerank::Reranker::open(bundle, threads)` creates a safe CPU session.
`Reranker::open_with(bundle, &RerankSessionOptions { backend, precision, cpu_threads })`
names the backend instead; `backend()` and `device_info()` say where it runs.
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

## Devices

A rerank bundle loads on any backend whose embedding encoder runs its
weights: CUDA, Metal and Level Zero as well as the CPU. The pair windows go
through the embedding encoder with CLS pooling and no normalization. When
the backend leaves the hidden states in device memory, the CLS states
(`hidden` floats a window) are copied to the host. The pooler, the
classifier and the largest window logit then run on the host in F64, as on
the CPU. The result is a host buffer, and `d2h_bytes` counts the copy.

`Reranker::open` stays on the CPU even when a device is present: every
rerank cell ties as experimental, so device order would otherwise pick a
GPU. Name a device with `open_with`. The rerank cell on a device is
EXPERIMENTAL until a compiled benchmark record backs it.

### Parity

`core/tests/cuda_rerank.rs` scores a bundle's pinned reference pairs on the
device and on the CPU, with a bound of `1e-4` against both. Set
`TURBO_TEST_RERANK_BUNDLES` to comma-separated bundle directories to run the
real bundles at EXACT and MODEL precision:

```sh
TURBO_TEST_REQUIRE_CUDA=1 TURBO_TEST_RERANK_BUNDLES=<minilm bundle>,<bge bundle> \
  cargo test --release -p turbo --features cuda --test cuda_rerank -- --test-threads=1 --nocapture
```

| Backend | Model | Precision | Largest error to the reference | Largest difference to the CPU path |
| --- | --- | --- | ---: | ---: |
| CUDA, RTX 4080 SUPER | MS MARCO MiniLM L6 v2 | EXACT, MODEL | 0.0000024 | 0.0000026 |
| CUDA, RTX 4080 SUPER | BGE reranker base | EXACT, MODEL | 0.0000075 | 0.0000048 |

EXACT and MODEL gave the same scores on both bundles. Metal and Level Zero
take the same code path and have no recorded parity yet.

### Throughput

`turbo-bench rerank` takes its pairs from a SQuAD-format file. Request `i` is
the first question of paragraph `i` with 16 contexts from paragraph `i` on,
using only contexts that fit one pair window uncut, so neither side
truncates. A pass scores all 64 requests in turn; the figure is the median
of ten passes after two warmups. TEI runs on the same GPU in a container
pinned by digest, in float32, timed on `/rerank` with `raw_scores: true` and
`truncate: false`, so its figure includes loopback HTTP.

Measured on 2026-10-24 at commit `6160b5cb`, NVIDIA RTX 4080 SUPER (CUDA
runtime 12.4, driver 615.71.09), which was also driving a desktop session,
with SQuAD 2.0 dev, precision EXACT, and TEI 89-1.8 (image digest
`sha256:1a3d7c897bdaf42cd4d1cad57b3134b91b3b5db129995d30db9fb2fc8bf8fd64`).

| Model | Library pairs/s | TEI pairs/s | Library over TEI | Largest score difference to TEI |
| --- | ---: | ---: | ---: | ---: |
| MS MARCO MiniLM L6 v2 | 4298.2 | 1463.8 | 2.94x | 0.0129 |
| BGE reranker base | 650.1 | 369.0 | 1.76x | 0.116 |

The records are in `benchmarks/rerank/`. The score differences are TEI's:
the library is within `1e-5` of the Transformers FP32 reference on both
bundles (above), and TEI's BGE scores also differ from that reference on
the CPU. Reproduce from a clean, pushed commit (the tool stops on a dirty
tree), with the upstream directory the bundle was made from as TEI's model:

```sh
cargo build --release -p turbo-bench --features cuda
target/release/turbo-bench rerank --bundle <bundle> --questions <dev-v2.0.json> \
  --device cuda --precision exact \
  --tei-image ghcr.io/huggingface/text-embeddings-inference@sha256:1a3d7c897bdaf42cd4d1cad57b3134b91b3b5db129995d30db9fb2fc8bf8fd64 \
  --tei-model <upstream dir>
```
