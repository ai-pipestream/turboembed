# TurboQuant record

TurboQuant storage in `turbo-vector-index` (`IndexBuilder::build_quantized`,
feature `turboquant`), measured against the exact index in the same crate
and against the OpenNLP Java reference
`opennlp.embeddings.index.TurboQuantIndex`. The codes and search kernels
are the `turbovec` crate's, pinned to commit
`acd0c1ee632acc2f5b6ad848443cf8e77d10ef2a` of `ai-pipestream/turbovec`
(MIT).

Machine: AMD Ryzen 9 9950X3D (x86_64, AVX-512), release build, Rust 1.98,
one thread (`RAYON_NUM_THREADS=1`; the exact index is single-threaded).
Java: Temurin 25.0.3 on the same rows and queries, read from the files
`quant_bench` dumps, three warm-up passes before timing. A query time is
the median over three passes of the queries one at a time; a batch time
is the median of five runs of all queries in one call, divided by the
number of queries.

## Public embedding set

The token embedding table of `minishlab/potion-base-8M` (256 dimensions,
29,528 rows). Every 100th row is a query and the other 29,232 are
indexed, all scaled to unit length and compared by cosine; k = 10.

```
RAYON_NUM_THREADS=1 cargo run --release -p turbo-vector-index \
    --features turboquant --example quant_bench -- \
    potion-base-8M.glove.txt 100 10 DUMP_DIR
```

| storage | bits | calibrated | bytes a row | encode rows/s | query ms | batch ms a query | recall@10 | mean error | error mean square |
|---|---|---|---|---|---|---|---|---|---|
| exact `f32` | 32 | | 1024 | | 0.396 | 0.0946 | 1 | 0 | 0 |
| TurboQuant | 2 | no | 73.6 | 595,137 | 0.0282 | 0.0180 | 0.8507 | 0.000074 | 5.167e-4 |
| TurboQuant | 3 | no | 142.2 | 372,025 | 0.0526 | 0.0304 | 0.9186 | -0.000354 | 1.411e-4 |
| TurboQuant | 4 | no | 142.2 | 336,781 | 0.1156 | 0.0340 | 0.9547 | -0.000001 | 3.692e-5 |
| TurboQuant | 2 | yes | 73.6 | 1,108,488 | 0.0288 | 0.0175 | 0.8470 | 0.000355 | 5.161e-4 |
| TurboQuant | 3 | yes | 142.2 | 1,022,611 | 0.0506 | 0.0304 | 0.9159 | -0.000171 | 1.417e-4 |
| TurboQuant | 4 | yes | 142.2 | 955,440 | 0.1152 | 0.0336 | 0.9581 | 0.000011 | 3.766e-5 |
| Java exact | 32 | | | | 2.174 | | 1 | | |
| Java TurboQuant | 2 | | | 215,526 | 5.458 | | 0.8412 | | |
| Java TurboQuant | 3 | | | 164,351 | 5.977 | | 0.9118 | | |
| Java TurboQuant | 4 | | | 140,071 | 5.699 | | 0.9598 | | |

Bytes a row are the `.tv` image (`QuantizedIndex::to_bytes`) divided by
the rows; the image stores 3-bit codes in the 4-bit layout. The errors are
over every row for the first 20 queries (584,640 scores). The first index
of a width built in a process also builds the rotation, which is why the
uncalibrated encode rates are lower.

Against the Java reference on the same rows, recall is within 0.01 and a
query is 49 to 194 times faster (one at a time). Against the exact index
a query is 3.4 to 14 times faster and a batch 2.8 to 5.4 times faster.

### Against the paper's bound

With unit rows and queries and `d` the dimension (256 here), the paper
bounds the squared error of an inner-product estimate by
`sqrt(3) pi^2 / d * 4^-b`: 4.173e-3, 1.043e-3 and 2.608e-4 at 2, 3 and 4
bits. The measured mean squares are 5.2e-4, 1.4e-4 and 3.7e-5, and the
mean errors stay near zero (turbovec stores a length a row that removes
the codes' shrinkage from the estimate). `analysis/vector-index/tests/quantized.rs`
checks the bound and the mean on 128-dimensional rows.

## A million random rows

`random:1000000:256`, every 10,000th row a query (999,900 rows, 100
queries). Random uniform rows have tiny gaps between neighbours, so this
set measures the scan more than the recall a real collection gets.

| storage | bits | calibrated | query ms | batch ms a query | recall@10 |
|---|---|---|---|---|---|
| exact `f32` | 32 | | 23.56 | 3.294 | 1 |
| TurboQuant | 2 | no | 1.50 | 0.316 | 0.360 |
| TurboQuant | 3 | no | 4.21 | 0.735 | 0.658 |
| TurboQuant | 4 | no | 1.44 | 0.275 | 0.547 |
| TurboQuant | 2 | yes | 1.64 | 0.289 | 0.363 |
| TurboQuant | 3 | yes | 4.21 | 0.722 | 0.672 |
| TurboQuant | 4 | yes | 1.35 | 0.268 | 0.514 |

From 32,768 rows turbovec searches 2- and 4-bit indexes in stages: a scan
of the sign bits picks a shortlist, and only the shortlist is scored in
full. Scores stay exact for the rows returned, but on structureless random
rows the shortlist misses neighbours, which is the 4-bit recall above; a
whole-index 4-bit scan (`TURBOVEC_4BIT_PLANES=0`) took 4.27 ms a query in a separate run, at
recall 0.818 on these rows. On embeddings with real neighbours turbovec
reports the staged and whole-index scans returning the same ids for
99.9% of queries or more.

## What is not measured here

The paper's own implementation remains unmeasured. The later FAISS
comparison is recorded below.

## Compact storage and exhaustive search (#977)

The earlier tables above are historical measurements. `to_bytes()` still
writes the pinned turbovec `.tv` format. `to_compact_bytes()` now writes a
separate `TQPACK01` image with exactly `bits * dim / 8` code bytes per row,
plus one F32 correction scale. At 256 dimensions this is 100 bytes per
3-bit row and 132 per 4-bit row, before the small header and calibration.
`from_compact_bytes(bytes, metadata)` restores the model space, metric,
configuration, calibration and original-vector scaling; generic metadata
is supplied separately in insertion order. No `.tv` reader is expected to
read this compact format.

The compact header contains the eight-byte magic, the 40-byte pinned
encoder commit, width/calibration/metric/normalization bytes, dimension
(U32), row count (U64), input scale (F64), then three U64-length-prefixed
UTF-8 model strings (id, revision, SHA-256). A U32 calibration length is
followed by shift and scale arrays, row correction scales, and packed bit
planes. All numbers are little-endian. The decoder validates lengths,
flags, dimensions, encoder revision, scale and calibration, and refuses
trailing bytes. Code value validation uses the dependency's `from_parts`.
This image does not include an integrity checksum; content-address it in
storage when corruption detection is required.

For indexes of at least 32,768 rows, 2- and 4-bit `top_k` and `top_k_batch`
now merge the top results of every shard of at most 16,384 rows. Each shard
is below the dependency's sign-shortlist threshold. The union of their
local top k contains the global top k of the estimated scores. This avoids
additional recall loss from staging; quantization error remains. Three-bit
search already scans all codes and is unchanged.

The shards and their SIMD caches are built on the first large search and
retained for later queries through `OnceLock`. This adds cold-start work
and resident code/cache copies. Compact bytes are a storage-size measure,
not a resident-memory claim. Concurrent first queries may briefly build
duplicate shard sets before one is retained. No environment variable is
changed inside the library. The dependency's streaming scan and per-range
stored-row conversion could not be used: real 32,769-row tests found
out-of-bounds panics in their planes-cache slicing. The shard path uses the
whole packed representation, which passes those tests.

The tests use actual Potion embedding-table rows 32 through 159, with
provenance and MIT notice in `analysis/vector-index/tests/data/README.md`.
They cover round-trip scores with and without calibration, preserved dot
scaling, empty indexes, malformed images, compact-width size differences,
and a 32,769-row index compared to a full sort of all quantized estimates.
The large case repeats real vectors to cross the staging threshold; it is
not a benchmark corpus or evidence of application recall.

### FAISS comparison

The reference script `analysis/vector-index/reference/faiss_compare.py`
runs FAISS `IndexFlatIP` and `IndexScalarQuantizer` SQ4/SQ8 on the exact
little-endian rows, queries and ground-truth ids dumped by `quant_bench`.
FAISS SQ has no 3-bit mode, so SQ4 is the nearest-width baseline; this is
not a same-bit comparison. See the upstream
[FAISS index descriptions](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).
The pinned container recipe is
`analysis/vector-index/reference/Dockerfile.faiss` (FAISS 1.13.2, NumPy 2.2.6).
No reference packages are added to the Rust library.

Both programs use one CPU thread, pinned to the same CPU core on an AMD
Ryzen 9 9950X3D. There are three warm-up passes, then a median across three
passes of individual queries and five full-batch passes. Timing excludes
encoding, file I/O and cold-cache initialization. It includes the Python
wrapper for FAISS, so it is not an isolated C++ kernel comparison.
The two programs run sequentially on each set. Results are one run per
configuration, without confidence intervals; no significance claim is made.

The real set is the same Potion model and split as above, fetched at
commit `bf8b056651a2c21b8d2565580b8569da283cab23`, model SHA-256
`f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2`.
The second set retains the earlier fixed million-row generator and
100-query split. Its purpose is to stress near-tie ranking, not represent
text retrieval. Both sets use 256 dimensions, cosine and k=10. FAISS is
trained only on indexed rows. All output hashes and measurements are in
`docs/receipts/turboquant-977/comparison.json`.

#### Potion: 29,232 rows, 296 queries

| Index | Calibrated | Storage bytes/row | Query ms | Batch ms/query | Recall@10 |
|---|---|---:|---:|---:|---:|
| Rust exact |  | 1024.0 | 0.2426 | 0.0833 | 1.0000 |
| TurboQuant 2-bit | no | 68.0 | 0.0291 | 0.0175 | 0.8507 |
| TurboQuant 3-bit | no | 100.0 | 0.0507 | 0.0300 | 0.9186 |
| TurboQuant 4-bit | no | 132.0 | 0.1118 | 0.0321 | 0.9547 |
| TurboQuant 2-bit | yes | 68.1 | 0.0283 | 0.0176 | 0.8470 |
| TurboQuant 3-bit | yes | 100.1 | 0.0515 | 0.0300 | 0.9159 |
| TurboQuant 4-bit | yes | 132.1 | 0.1112 | 0.0320 | 0.9581 |
| FAISS FlatIP | n/a | 1024.0 | 0.3268 | 0.2254 | 1.0000 |
| FAISS SQ4 | trained on rows | 128.1 | 0.4507 | 0.4507 | 0.9071 |
| FAISS SQ8 | trained on rows | 256.1 | 0.2676 | 0.2669 | 0.9946 |

#### Random: 999,900 rows, 100 queries

| Index | Calibrated | Storage bytes/row | Query ms | Batch ms/query | Recall@10 |
|---|---|---:|---:|---:|---:|
| Rust exact |  | 1024.0 | 23.2947 | 2.9375 | 1.0000 |
| TurboQuant 2-bit | no | 68.0 | 1.6309 | 0.7467 | 0.4060 |
| TurboQuant 3-bit | no | 100.0 | 3.8849 | 0.7320 | 0.6580 |
| TurboQuant 4-bit | no | 132.0 | 4.2474 | 1.3605 | 0.8180 |
| TurboQuant 2-bit | yes | 68.0 | 1.5904 | 0.9665 | 0.4160 |
| TurboQuant 3-bit | yes | 100.0 | 3.9040 | 0.7340 | 0.6720 |
| TurboQuant 4-bit | yes | 132.0 | 4.2792 | 1.3770 | 0.8150 |
| FAISS FlatIP | n/a | 1024.0 | 24.0686 | 8.3028 | 1.0000 |
| FAISS SQ4 | trained on rows | 128.0 | 18.4350 | 19.0286 | 0.8530 |
| FAISS SQ8 | trained on rows | 256.0 | 12.1258 | 12.2279 | 0.9880 |

FAISS SQ8 gives higher recall at higher storage cost. SQ4 has lower recall
than TurboQuant 4-bit on Potion, but higher recall on the random set
(0.853 versus 0.818). TurboQuant 4-bit no longer incurs the earlier
shortlist loss (0.547): the default now reaches the full-scan 0.818 result.
The faster staged latency from the historical table is not retained.

Reproduce the dumps with `quant_bench`: Potion uses query stride 100;
`random:1000000:256` uses stride 10000; k is 10. Pass an output directory
as argument four. The reference script accepts that directory, dimension
256, k 10, and an output JSON filename. Build its pinned Dockerfile and
mount the dumps read-only. Set `RAYON_NUM_THREADS=1` for Rust and
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` for FAISS. The script also
sets FAISS's thread count to one. Use the same CPU affinity for both runs.
