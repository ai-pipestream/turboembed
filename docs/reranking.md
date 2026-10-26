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
pooling stages separately in the stage placement array. The encoding stage
is where the backend ran the encoder (`TURBO_STAGE_DEVICE` on a GPU). The
classifier head and the max over windows always run on the host. On a device
the scores are read back into the session's host memory, so the result is
`TURBO_PLACE_HOST` on every backend. A buffer from `turbo_result_buffer`
exports only `TURBO_HANDLE_HOST_PTR`, which is the address
`turbo_buffer_host_ptr` gives.

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

A rerank bundle loads on the CPU, on CUDA, on Level Zero and on Metal.
Those are the backends whose rerank scores and timing are recorded below;
any other refuses a rerank bundle at `turbo_model_load` with
`TURBO_E_UNSUPPORTED_TASK`, and its rerank cell is UNSUPPORTED, until it
has been run. The pair windows go
through the embedding encoder with CLS pooling and no normalization. When
the backend leaves the hidden states in device memory, the CLS states
(`hidden` floats a window) are copied to the host. The pooler, the
classifier and the largest window logit then run on the host in F64, as on
the CPU. The result is a host buffer, and `d2h_bytes` counts the copy.

`Reranker::open` stays on the CPU even when a device is present: every
rerank cell ties as experimental, so device order would otherwise pick a
GPU. Name a device with `open_with`. The rerank cells on CUDA and Level
Zero are EXPERIMENTAL until a compiled benchmark record backs them.

### Parity

`core/tests/cuda_rerank.rs` and `core/tests/levelzero_rerank.rs` score a
bundle's pinned reference pairs on the device and on the CPU, with a bound
of `1e-4` against both. Set
`TURBO_TEST_RERANK_BUNDLES` to comma-separated bundle directories to run the
real bundles at EXACT and MODEL precision:

```sh
TURBO_TEST_REQUIRE_CUDA=1 TURBO_TEST_RERANK_BUNDLES=<minilm bundle>,<bge bundle> \
  cargo test --release -p turbo --features cuda --test cuda_rerank -- --test-threads=1 --nocapture
TURBO_TEST_REQUIRE_LEVELZERO=1 TURBO_TEST_RERANK_BUNDLES=<minilm bundle>,<bge bundle> \
  cargo test --release -p turbo --features levelzero --test levelzero_rerank -- --test-threads=1 --nocapture
TURBO_TEST_REQUIRE_METAL=1 TURBO_TEST_RERANK_BUNDLES=<minilm bundle>,<bge bundle> \
  cargo test --release -p turbo --features metal --test metal_rerank -- --nocapture
```

The Level Zero rows were run with `--features levelzero` (the own kernels)
and, separately, `--features levelzero-onednn`, on an Arc Pro B70
(compute-runtime 26.27.39122.14, loader 1.32.0, oneAPI 2025.3.3).

| Backend | Model | Precision | Largest error to the reference | Largest difference to the CPU path |
| --- | --- | --- | ---: | ---: |
| CUDA, RTX 4080 SUPER | MS MARCO MiniLM L6 v2 | EXACT, MODEL | 0.0000024 | 0.0000026 |
| CUDA, RTX 4080 SUPER | BGE reranker base | EXACT, MODEL | 0.0000075 | 0.0000048 |
| Level Zero, Arc Pro B70 | MS MARCO MiniLM L6 v2 | EXACT, MODEL | 0.0000033 | 0.0000026 |
| Level Zero, Arc Pro B70 | BGE reranker base | EXACT, MODEL | 0.0000067 | 0.0000098 |
| Level Zero with oneDNN | MS MARCO MiniLM L6 v2 | EXACT | 0.0000038 | 0.0000026 |
| Level Zero with oneDNN | MS MARCO MiniLM L6 v2 | MODEL | 0.0000033 | 0.0000031 |
| Level Zero with oneDNN | BGE reranker base | EXACT, MODEL | 0.0000083 | 0.0000091 |
| Metal, Apple M2 | MS MARCO MiniLM L6 v2 | EXACT, MODEL | 0.0000010 | 0.0000048 |
| Metal, Apple M2 | BGE reranker base | EXACT, MODEL | 0.0000086 | 0.0000086 |

EXACT and MODEL gave the same scores on both bundles on CUDA and with the
Level Zero kernels; with oneDNN they differ on MiniLM by under `1e-6`.
On Metal EXACT and MODEL gave the same scores on both bundles. On Metal
the scores are written into the session's shared output buffer, which the
host and the GPU both address, so the result's placement is SHARED and a
run uploads and reads back nothing (`core/tests/metal_rerank.rs`).

### Throughput

`turbo-bench rerank` takes its pairs from a SQuAD-format file. Request `i` is
the first question of paragraph `i` with 16 contexts from paragraph `i` on,
using only contexts that fit one pair window uncut, so neither side
truncates. A pass scores all 64 requests in turn; the figure is the median
of ten passes after two warmups. TEI runs on the same GPU in a container
pinned by digest, in float32, timed on `/rerank` with `raw_scores: true` and
`truncate: false`, so its figure includes loopback HTTP.

Measured on 2026-10-24 at commit `66bbe5e2`, NVIDIA RTX 4080 SUPER (CUDA
runtime 12.4, driver 615.71.09), which was also driving a desktop session,
with SQuAD 2.0 dev, precision EXACT, and TEI 89-1.8 (image digest
`sha256:1a3d7c897bdaf42cd4d1cad57b3134b91b3b5db129995d30db9fb2fc8bf8fd64`).

| Model | Library pairs/s | TEI pairs/s | Library over TEI | Largest score difference to TEI |
| --- | ---: | ---: | ---: | ---: |
| MS MARCO MiniLM L6 v2 | 4303.7 | 1276.1 | 3.37x | 0.0129 |
| BGE reranker base | 663.9 | 427.9 | 1.55x | 0.116 |

On Level Zero, measured on 2026-10-26 at commit `6b982643` on an Arc Pro
B70 with the own kernels, MS MARCO MiniLM L6 v2 at EXACT scored 889.6
pairs/s (a second run 876.2), against 665.4 on the host CPU, an AMD Ryzen 9
9950X, unpinned, in the same tool: 1.34x. TEI was not run, since the TEI
image above is the CUDA build. The bench's compute engine was about 29%
busy during the run (xe fdinfo), so the run is bound by the host and the
launches, not by compute; a desktop client held the render engine about
68% busy. BGE was not timed on Level Zero, and these records are not in
`benchmarks/rerank/`.

On Metal, measured on 2026-10-26 at commit `e051ae76` on an Apple M2 with
`--device metal --precision exact --no-tei`, against the M2's CPU in the
same tool (TEI was not run: the native router is not wired for rerank).
p50 / min / max ms a pass, and pairs per second at the p50:

| Model | Metal | Metal pairs/s | CPU | CPU pairs/s |
| --- | ---: | ---: | ---: | ---: |
| MS MARCO MiniLM L6 v2 | 6307.6 / 5239.3 / 6929.7 | 162.3 | 33776.3 / 32864.0 / 35300.5 | 30.3 |
| BGE reranker base | 54938.2 / 53094.1 / 61751.9 | 18.6 | 305979.2 / 258876.1 / 350386.6 | 3.3 |

Metal runs 5.4x the CPU on MiniLM and 5.6x on BGE. The M2 is a desktop in
use: the Metal records started once the five-minute load average was
under 4 and the GPU under 40% busy from the desktop's apps (16% just
before), and the CPU records ran with the one-minute load at about 20,
the bench's own threads included. The records are the four `m2.metal.*`
and `aarch64.cpu.*` files in `benchmarks/rerank/`.

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
