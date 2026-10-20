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

The paper's own implementation and FAISS `IndexScalarQuantizer` are not
on this machine, so that comparison is still open.
