# Learned sparse encoding receipts (SPLADE-style, #708)

Base: `17561380` (origin/main at branch time).

## What is proven where

- The pooling and term selection are held to an independent numpy
  reference (`testdata/splade-golden/make_golden.py`) in
  `core/tests/sparse_golden.rs`; the MLM head arithmetic to a second one
  (`make_head_golden.py`) in `core/tests/mlm.rs`. No SPLADE checkpoint
  could be fetched here: `naver/splade-v3` and the other SPLADE models
  are licence-gated (HuggingFace answers 401 to an anonymous fetch), so
  the issue's sanctioned substitution is used: hand-seeded logits and a
  hand-built tiny head stand in for a checkpoint's outputs, and the two
  scripts are the reference implementation. Regenerate with
  `python3 testdata/splade-golden/make_golden.py` and
  `python3 testdata/splade-golden/make_head_golden.py`.
- The loader rules (roles named at parse, shapes and dtypes at load, the
  tied decoder, refusals naming the field) are in `core/tests/mlm.rs`
  and `bundle/tests/sparse.rs`; the seal check in the latter loads the
  head through the core, so a sparse declaration the weights cannot
  serve is refused at seal with the tensor named.
- The span rule is in `analysis/embed/tests/sparse.rs`: every input
  term's span is the exact span of its pieces (first start to last end,
  occurrences inside), a text whose pieces occur once tiles byte for
  byte with the dropped pieces' spans, expanded terms carry none, and
  the annotator's terms equal the same pieces run by hand in a session
  of their own, bit for bit.
- The chunker adapter's placement and refusals are in
  `analysis/chunker/src/terms/sparse.rs`.

## Speed

`analysis/embed/tests/speed_sparse.rs` (ignored, like the SaT speed
test) measures the sparse step on the CPU. Machine: AMD Ryzen 9
9950X3D (16 cores, 32 threads), 121 GB RAM, AVX2 and FMA (an AVX-512F
part). Model: the small BERT of testdata/tiny-bert-bundle (2 layers,
hidden 32, vocab 30522) with a hand-built MLM head, decoder tied; the
texts are the SaT speed test's 90 reference texts joined in tens (36
documents, 48,828 bytes, 11,792 pieces). Command:

```
cargo test --release -p turbo-analysis-embed --test speed_sparse -- --ignored --nocapture
```

Result (`speed.log`): 524.1 ms a pass, 68.7 texts/s, 22,501 pieces/s,
1,048,061 terms/s. The pass is one document at a time through the step
(its encoder session takes one owner), so the number is a single stream
on one core of the host, not a batched server figure.

The pooling keeps each term's peak `relu(logit)` over the text's tokens
(`log(1 + x)` is monotonic, so the max commutes with the transform):
one `log1p` per term, not per token and term. The decoder runs as a
blocked GEMM over its transposed weight with an AVX2/FMA micro-kernel,
detected at runtime the way the encoder's kernels detect theirs
(`core/src/cpu/kernels/micro.rs`); the portable path is plain SSE2
arithmetic. Both changes together took the same pass from 6.57 s
(5.5 texts/s) to 0.52 s (68.7 texts/s) on this machine.

TEI comparison: Text Embeddings Inference could not be run here (no
Docker in this environment), so the bar is recorded, not matched
against TEI's own number: 68.7 texts/s on one core for a 2-layer model
whose width-32 states make the MLM head, not the encoder, the cost. A
real SPLADE model (hidden 768) amortizes the head over 24 times the
work per token in the encoder; the receipt number for that is a
benchmark-record exercise once a sealed sparse bundle of one exists.

## Retrieval record

Not recorded: no judged retrieval harness set (queries and qrels) is
committed in this tree; testdata holds models and tokenizers, no
harness. The fusion point is in place (#695's `LevelRouter` reads the
adapter's `SparseTerms<Levels>` beside `TermStats`); recording BM25,
sparse and fused scores needs a harness set the repo does not carry.
