# TurboQuant record

TurboQuant storage in `turbo-vector-index` (`IndexBuilder::build_quantized`),
measured against the exact index in the same crate and against the
OpenNLP Java reference `opennlp.embeddings.index.TurboQuantIndex`.

Machine: AMD Ryzen 9 9950X3D (x86_64, AVX-512), one thread, release
build, Rust 1.98. Java: Temurin 25.0.3, same rows and queries read from
the files `quant_bench` dumps, three warm-up passes before timing. Times
are medians over three passes of the queries, one query at a time.

## Public embedding set

The token embedding table of `minishlab/potion-base-8M` (256 dimensions,
29,528 rows). Every 100th row is a query and the other 29,232 are
indexed, all scaled to unit length and compared by cosine; k = 10.

```
cargo run --release -p turbo-vector-index --example quant_bench -- \
    potion-base-8M.glove.txt 100 10 DUMP_DIR
```

| storage | bits | bytes a row | encode rows/s | query ms | recall@10 | mean error | error mean square |
|---|---|---|---|---|---|---|---|
| exact `f32` | 32 | 1024 | | 0.456 | 1 | 0 | 0 |
| Mse | 1 | 40 | 1,228,712 | 0.239 | 0.6841 | -0.000533 | 1.692e-3 |
| Mse | 2 | 72 | 942,049 | 0.379 | 0.8412 | -0.000238 | 4.810e-4 |
| Mse | 3 | 104 | 893,490 | 0.754 | 0.9111 | -0.000064 | 1.343e-4 |
| Mse | 4 | 136 | 758,093 | 0.743 | 0.9595 | -0.000027 | 3.616e-5 |
| InnerProduct | 1 | 40 | 594,798 | 0.226 | 0.7010 | -0.000287 | 2.219e-3 |
| InnerProduct | 2 | 72 | 491,509 | 0.402 | 0.7932 | -0.000049 | 8.009e-4 |
| InnerProduct | 3 | 104 | 442,541 | 0.570 | 0.8831 | -0.000021 | 2.567e-4 |
| InnerProduct | 4 | 136 | 413,825 | 0.969 | 0.9324 | -0.000018 | 7.500e-5 |
| Java exact | 32 | | | 2.174 | 1 | | |
| Java TurboQuant | 2 | | 215,526 | 5.458 | 0.8412 | | |
| Java TurboQuant | 3 | | 164,351 | 5.977 | 0.9118 | | |
| Java TurboQuant | 4 | | 140,071 | 5.699 | 0.9598 | | |

The errors are over every row for the first 20 queries (584,640 scores).
The Java reference codes 2 to 4 bits with the grid alone, which is the
`Mse` estimator here, and its recall matches within 0.0007. The Rust
encoder is 4.4 to 5.4 times faster and a query 7.7 to 14.4 times faster.

### Against the paper's bounds

With unit rows and queries and `d` the padded dimension (256 here), the
paper bounds the squared reconstruction error of the grid by
`sqrt(3) pi / 2 * 4^-b`, of which a random unit query sees `1/d` in its
dot product, and the squared error of the inner-product estimate by
`sqrt(3) pi^2 / d * 4^-b`:

| bits | Mse measured | Mse bound | InnerProduct measured | InnerProduct bound |
|---|---|---|---|---|
| 1 | 1.692e-3 | 2.657e-3 | 2.219e-3 | 1.669e-2 |
| 2 | 4.810e-4 | 6.643e-4 | 8.009e-4 | 4.173e-3 |
| 3 | 1.343e-4 | 1.661e-4 | 2.567e-4 | 1.043e-3 |
| 4 | 3.616e-5 | 4.152e-5 | 7.500e-5 | 2.608e-4 |

The inner-product estimate's mean error stays within a few standard
errors of zero at every width; `analysis/vector-index/tests/quantized.rs` checks both bounds
and the mean on 100- and 128-dimensional rows.

## A million random rows

`random:1000000:256`, every 10,000th row a query (999,900 rows, 100
queries). Random uniform rows have small gaps between neighbours, so
recall here is a lower bound for real embeddings; the set measures the
scan.

| storage | bits | query ms | recall@10 |
|---|---|---|---|
| exact `f32` | 32 | 25.67 | 1 |
| Mse | 1 | 6.93 | 0.133 |
| Mse | 2 | 12.53 | 0.418 |
| Mse | 3 | 24.57 | 0.679 |
| Mse | 4 | 26.96 | 0.833 |
| InnerProduct | 1 | 7.40 | 0.128 |
| InnerProduct | 2 | 12.92 | 0.324 |
| InnerProduct | 3 | 19.51 | 0.549 |
| InnerProduct | 4 | 33.70 | 0.733 |

The quantized scan is a portable table lookup a byte of codes, bound by
the loads of those lookups; the exact scan is AVX-512 over memory. At 1
and 2 bits the quantized scan is faster; at 3 and 4 it is about as fast
as the exact scan while storing 7.5 to 25 times fewer bytes a row.

## What is not measured here

The paper's own implementation and FAISS `IndexScalarQuantizer` are not
on this machine, so the comparison with them that the issue asks for is
still open, as is a run on aarch64 to confirm the pinned code bytes
there (`codes_and_scores_are_pinned_to_their_bytes`).
