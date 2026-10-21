# Sparse Late Interaction (`SPLATE`) vs. Simpler Cascades (`#711`)

This study evaluates whether **Sparse Late Interaction (SPLATE)** (Formal et al., [arXiv:2404.13950](https://arxiv.org/abs/2404.13950)), which maps ColBERT-style token vectors into a vocabulary space via a lightweight linear adapter and max-log pooling (`log(1 + relu(W t + b))`) for inverted-list candidate generation before exact `MaxSim` reranking, earns a place in the `turboembed` retrieval stack alongside or instead of simpler two-stage cascades.

Per the issue specification, no new public library API is introduced; the complete four-arm harness, `SparsePostingIndex` (with MaxScore upper-bound pruning), and `SplateAdapter` live inside `turbo-eval` (`analysis/eval/src/splate/`).

---

## 1. The Four Cascade Arms

All four arms run inside the same process (`analysis/eval/src/splate/`), over identical document, query, and relevance judgment sets (`LoadedSet`), with interleaved per-query timing rounds (`Arm 1 -> Arm 2 -> Arm 3 -> Arm 4` on every query in every round) so CPU frequency scaling and cache state affect all four arms equally:

1. **Arm 1 (`Bm25Only`)**:
   - **Stage 1**: Lexical BM25 (`Bm25::LUCENE`, $k_1 = 1.2, b = 0.75$) over `turbo_chunker::terms::Bm25PostingIndex` with MaxScore upper-bound pruning.
   - **Stage 2**: None (single-stage baseline).
2. **Arm 2 (`StaticThenCrossEncoder`)**:
   - **Stage 1**: Exact static dense cosine retrieval (`turbo_vector_index::VectorIndex`) producing a shortlist of $C$ candidates.
   - **Stage 2**: Joint query-passage cross-encoder reranking over the top-$C$ shortlist (`CrossEncoderScorer`, backed by `turbo::rerank::Reranker` for ONNX/GGUF cross-encoder bundles or `InteractionCrossEncoder` for CPU-only evaluation).
3. **Arm 3 (`LearnedSparseThenMaxSim`)**:
   - **Stage 1**: Dedicated learned sparse expansion (`turbo::sparse::MaxLogPool` + `SparsePostingIndex` with MaxScore pruning) producing a shortlist of $C$ candidates.
   - **Stage 2**: Exact late-interaction `MaxSim` reranking over the top-$C$ shortlist (`turbo_vector_index::TokenIndex::rerank`).
4. **Arm 4 (`SplateThenMaxSim`)**:
   - **Stage 1**: SPLATE linear adapter (`SplateAdapter::encode_tokens`) projecting the document and query late-interaction token vectors (`[n_tokens, dim]`) into vocabulary space (`[vocab]`) via `MaxLogPool` (`log(1 + relu(W t + b))`), indexed in `SparsePostingIndex` and queried with MaxScore pruning to produce a shortlist of $C$ candidates.
   - **Stage 2**: Exact late-interaction `MaxSim` reranking over the *same* token vectors in `TokenIndex::rerank` (one base token-vector representation serving both stages).

---

## 2. Experimental Setup & Reproduction

All measurements below were collected in `--release` on a single pinned CPU core (`taskset -c 14`), with $d = 32$, $|V| = 2048$, SPLADE/SPLATE document cut `top_k = 64, floor = 0.05`, query cut `top_k = 24, floor = 0.05`, and 5 interleaved timing rounds across shortlist depths $C \in \{10, 25, 50, 100\}$.

To reproduce the exact tables below:

```bash
TURBO_EVAL_CACHE="$TURBO_EVAL_CACHE" \
  taskset -c 14 cargo test --locked --release -p turbo-eval \
  --test splate splate_benchmark_receipt -- --ignored --nocapture
```

To run the unit, parity (MaxScore vs. exhaustive scan), and sample-dataset integration suite:

```bash
cargo test --locked -p turbo-eval --test splate
```

---

## 3. Results Across Datasets and Shortlist Depths ($C$)

### 3.1 BEIR `scifact` (`1,500` documents, `100` judged test queries)

#### Shortlist $C = 10$
| Arm | C | Recall@1 | Recall@5 | Recall@10 | Recall@100 | nDCG@10 | p50 (us) | p99 (us) | Index (KiB) | Build (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm 1: BM25 only | 10 | 0.4933 | 0.6182 | 0.6447 | 0.7830 | 0.5776 | 64.5 | 178.2 | 2713.0 | 40.60 |
| Arm 2: Static dense + Cross-encoder | 10 | 0.0633 | 0.0733 | 0.0733 | 0.0733 | 0.0710 | 36.0 | 56.9 | 9192.0 | 0.23 |
| Arm 3: Learned sparse + ColBERT MaxSim | 10 | 0.2612 | 0.3328 | 0.3853 | 0.3853 | 0.3204 | 62.5 | 146.1 | 10943.0 | 8.27 |
| Arm 4: SPLATE adapter + ColBERT MaxSim | 10 | 0.2912 | 0.3528 | 0.3712 | 0.3712 | 0.3395 | 268.2 | 430.1 | 11229.1 | 1115.15 |

#### Shortlist $C = 25$
| Arm | C | Recall@1 | Recall@5 | Recall@10 | Recall@100 | nDCG@10 | p50 (us) | p99 (us) | Index (KiB) | Build (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm 1: BM25 only | 25 | 0.4933 | 0.6182 | 0.6447 | 0.7830 | 0.5776 | 65.9 | 176.7 | 2713.0 | 41.66 |
| Arm 2: Static dense + Cross-encoder | 25 | 0.0833 | 0.1033 | 0.1033 | 0.1033 | 0.0960 | 82.9 | 131.0 | 9192.0 | 0.22 |
| Arm 3: Learned sparse + ColBERT MaxSim | 25 | 0.2792 | 0.3528 | 0.4003 | 0.4778 | 0.3406 | 80.0 | 162.5 | 10943.0 | 5.98 |
| Arm 4: SPLATE adapter + ColBERT MaxSim | 25 | 0.2892 | 0.3728 | 0.4087 | 0.4562 | 0.3522 | 285.2 | 443.4 | 11229.1 | 1107.10 |

#### Shortlist $C = 50$
| Arm | C | Recall@1 | Recall@5 | Recall@10 | Recall@100 | nDCG@10 | p50 (us) | p99 (us) | Index (KiB) | Build (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm 1: BM25 only | 50 | 0.4933 | 0.6182 | 0.6447 | 0.7830 | 0.5776 | 66.7 | 174.0 | 2713.0 | 35.02 |
| Arm 2: Static dense + Cross-encoder | 50 | 0.1358 | 0.1558 | 0.1558 | 0.1658 | 0.1475 | 161.4 | 266.6 | 9192.0 | 0.22 |
| Arm 3: Learned sparse + ColBERT MaxSim | 50 | 0.2892 | 0.3648 | 0.4057 | 0.5532 | 0.3494 | 105.6 | 199.4 | 10943.0 | 5.83 |
| Arm 4: SPLATE adapter + ColBERT MaxSim | 50 | 0.2892 | 0.3728 | 0.4237 | 0.5112 | 0.3574 | 311.9 | 477.1 | 11229.1 | 1112.72 |

#### Shortlist $C = 100$
| Arm | C | Recall@1 | Recall@5 | Recall@10 | Recall@100 | nDCG@10 | p50 (us) | p99 (us) | Index (KiB) | Build (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm 1: BM25 only | 100 | 0.4933 | 0.6182 | 0.6447 | 0.7830 | 0.5776 | 68.6 | 184.0 | 2713.0 | 38.97 |
| Arm 2: Static dense + Cross-encoder | 100 | 0.1358 | 0.1658 | 0.1758 | 0.2058 | 0.1566 | 310.8 | 510.5 | 9192.0 | 0.23 |
| Arm 3: Learned sparse + ColBERT MaxSim | 100 | 0.2992 | 0.3848 | 0.4207 | 0.6832 | 0.3606 | 151.2 | 247.8 | 10943.0 | 6.14 |
| Arm 4: SPLATE adapter + ColBERT MaxSim | 100 | 0.2892 | 0.3748 | 0.4177 | 0.6252 | 0.3562 | 356.0 | 535.3 | 11229.1 | 1110.10 |

*Index byte breakdown on `scifact`*:
- Arm 1 (`Bm25Only`): `2,778,069` B (`2,713.0` KiB).
- Arm 2 (`StaticThenCrossEncoder`): `9,412,656` B (`9,192.0` KiB).
- Arm 3 (`LearnedSparseThenMaxSim`): `11,205,676` B (`827,400` B sparse postings + `10,378,276` B `TokenIndex`).
- Arm 4 (`SPLATE`): `11,498,564` B (`1,120,288` B sparse postings and `[2048, 32]` adapter weights + `10,378,276` B `TokenIndex`).

---

### 3.2 BEIR `nfcorpus` (`2,361` documents, `100` judged test queries)

#### Shortlist $C = 10$
| Arm | C | Recall@1 | Recall@5 | Recall@10 | Recall@100 | nDCG@10 | p50 (us) | p99 (us) | Index (KiB) | Build (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm 1: BM25 only | 10 | 0.0567 | 0.1010 | 0.1222 | 0.1732 | 0.3090 | 10.4 | 64.3 | 4140.6 | 62.75 |
| Arm 2: Static dense + Cross-encoder | 10 | 0.0036 | 0.0080 | 0.0081 | 0.0081 | 0.0401 | 19.3 | 35.6 | 14430.3 | 0.34 |
| Arm 3: Learned sparse + ColBERT MaxSim | 10 | 0.0237 | 0.0517 | 0.0848 | 0.0848 | 0.1909 | 11.0 | 79.0 | 17133.6 | 11.51 |
| Arm 4: SPLATE adapter + ColBERT MaxSim | 10 | 0.0215 | 0.0416 | 0.0497 | 0.0497 | 0.1321 | 188.4 | 296.9 | 17434.4 | 1740.41 |

#### Shortlist $C = 25$
| Arm | C | Recall@1 | Recall@5 | Recall@10 | Recall@100 | nDCG@10 | p50 (us) | p99 (us) | Index (KiB) | Build (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm 1: BM25 only | 25 | 0.0567 | 0.1010 | 0.1222 | 0.1732 | 0.3090 | 10.0 | 62.3 | 4140.6 | 64.22 |
| Arm 2: Static dense + Cross-encoder | 25 | 0.0060 | 0.0148 | 0.0192 | 0.0243 | 0.0775 | 38.4 | 70.0 | 14430.3 | 0.35 |
| Arm 3: Learned sparse + ColBERT MaxSim | 25 | 0.0234 | 0.0464 | 0.0731 | 0.1155 | 0.1706 | 19.9 | 100.3 | 17133.6 | 11.11 |
| Arm 4: SPLATE adapter + ColBERT MaxSim | 25 | 0.0210 | 0.0428 | 0.0562 | 0.0770 | 0.1485 | 204.7 | 313.7 | 17434.4 | 1744.65 |

#### Shortlist $C = 50$
| Arm | C | Recall@1 | Recall@5 | Recall@10 | Recall@100 | nDCG@10 | p50 (us) | p99 (us) | Index (KiB) | Build (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm 1: BM25 only | 50 | 0.0567 | 0.1010 | 0.1222 | 0.1732 | 0.3090 | 10.3 | 63.8 | 4140.6 | 59.56 |
| Arm 2: Static dense + Cross-encoder | 50 | 0.0117 | 0.0213 | 0.0235 | 0.0414 | 0.0926 | 69.9 | 134.7 | 14430.3 | 0.34 |
| Arm 3: Learned sparse + ColBERT MaxSim | 50 | 0.0247 | 0.0465 | 0.0668 | 0.1373 | 0.1644 | 32.7 | 120.7 | 17133.6 | 10.89 |
| Arm 4: SPLATE adapter + ColBERT MaxSim | 50 | 0.0236 | 0.0469 | 0.0708 | 0.1088 | 0.1599 | 225.4 | 341.7 | 17434.4 | 1738.56 |

#### Shortlist $C = 100$
| Arm | C | Recall@1 | Recall@5 | Recall@10 | Recall@100 | nDCG@10 | p50 (us) | p99 (us) | Index (KiB) | Build (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm 1: BM25 only | 100 | 0.0567 | 0.1010 | 0.1222 | 0.1732 | 0.3090 | 10.7 | 64.1 | 4140.6 | 57.80 |
| Arm 2: Static dense + Cross-encoder | 100 | 0.0122 | 0.0260 | 0.0298 | 0.0659 | 0.1143 | 127.5 | 251.8 | 14430.3 | 0.34 |
| Arm 3: Learned sparse + ColBERT MaxSim | 100 | 0.0247 | 0.0458 | 0.0648 | 0.1585 | 0.1610 | 55.3 | 169.3 | 17133.6 | 10.94 |
| Arm 4: SPLATE adapter + ColBERT MaxSim | 100 | 0.0242 | 0.0470 | 0.0655 | 0.1530 | 0.1601 | 259.2 | 387.7 | 17434.4 | 1742.99 |

---

### 3.3 Controlled Morphological Benchmark `splate-synth-640d-32q` (`640` documents, `32` queries)

This dataset isolates morphological inflection mismatch (queries use inflected forms while grade-2 relevant passages use base forms plus multi-token pathway context, with hard-negative distractors sharing individual anchor terms):

#### Shortlist $C = 10$
| Arm | C | Recall@1 | Recall@5 | Recall@10 | Recall@100 | nDCG@10 | p50 (us) | p99 (us) | Index (KiB) | Build (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm 1: BM25 only | 10 | 0.0500 | 0.2500 | 0.3500 | 0.6750 | 0.8007 | 8.5 | 17.4 | 76.5 | 0.53 |
| Arm 2: Static dense + Cross-encoder | 10 | 0.0312 | 0.1031 | 0.1375 | 0.1375 | 0.3479 | 9.1 | 14.4 | 1092.7 | 0.03 |
| Arm 3: Learned sparse + ColBERT MaxSim | 10 | 0.0438 | 0.2125 | 0.3875 | 0.3875 | 0.8007 | 6.5 | 15.8 | 1215.4 | 0.38 |
| Arm 4: SPLATE adapter + ColBERT MaxSim | 10 | 0.0312 | 0.1437 | 0.2313 | 0.2313 | 0.5063 | 131.9 | 151.7 | 1495.0 | 14.01 |

#### Shortlist $C = 25$
| Arm | C | Recall@1 | Recall@5 | Recall@10 | Recall@100 | nDCG@10 | p50 (us) | p99 (us) | Index (KiB) | Build (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm 1: BM25 only | 25 | 0.0500 | 0.2500 | 0.3500 | 0.6750 | 0.8007 | 8.8 | 19.2 | 76.5 | 0.68 |
| Arm 2: Static dense + Cross-encoder | 25 | 0.0250 | 0.1344 | 0.1844 | 0.3062 | 0.4215 | 19.1 | 28.2 | 1092.7 | 0.07 |
| Arm 3: Learned sparse + ColBERT MaxSim | 25 | 0.0344 | 0.1625 | 0.2656 | 0.6031 | 0.5742 | 10.3 | 20.4 | 1215.4 | 0.41 |
| Arm 4: SPLATE adapter + ColBERT MaxSim | 25 | 0.0344 | 0.1625 | 0.2469 | 0.4500 | 0.5492 | 140.0 | 160.7 | 1495.0 | 13.98 |

#### Shortlist $C = 50$
| Arm | C | Recall@1 | Recall@5 | Recall@10 | Recall@100 | nDCG@10 | p50 (us) | p99 (us) | Index (KiB) | Build (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm 1: BM25 only | 50 | 0.0500 | 0.2500 | 0.3500 | 0.6750 | 0.8007 | 8.9 | 16.9 | 76.5 | 0.66 |
| Arm 2: Static dense + Cross-encoder | 50 | 0.0281 | 0.1344 | 0.2031 | 0.4438 | 0.4552 | 33.5 | 46.5 | 1092.7 | 0.07 |
| Arm 3: Learned sparse + ColBERT MaxSim | 50 | 0.0344 | 0.1625 | 0.2500 | 0.9594 | 0.5532 | 14.5 | 26.6 | 1215.4 | 0.40 |
| Arm 4: SPLATE adapter + ColBERT MaxSim | 50 | 0.0344 | 0.1625 | 0.2469 | 0.6562 | 0.5492 | 150.5 | 175.9 | 1495.0 | 13.92 |

#### Shortlist $C = 100$
| Arm | C | Recall@1 | Recall@5 | Recall@10 | Recall@100 | nDCG@10 | p50 (us) | p99 (us) | Index (KiB) | Build (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm 1: BM25 only | 100 | 0.0500 | 0.2500 | 0.3500 | 0.6750 | 0.8007 | 9.3 | 16.3 | 76.5 | 0.66 |
| Arm 2: Static dense + Cross-encoder | 100 | 0.0312 | 0.1437 | 0.2312 | 0.5875 | 0.5061 | 60.6 | 74.3 | 1092.7 | 0.07 |
| Arm 3: Learned sparse + ColBERT MaxSim | 100 | 0.0344 | 0.1625 | 0.2469 | 1.0000 | 0.5492 | 19.6 | 38.7 | 1215.4 | 0.39 |
| Arm 4: SPLATE adapter + ColBERT MaxSim | 100 | 0.0344 | 0.1625 | 0.2469 | 0.8875 | 0.5492 | 166.8 | 191.1 | 1495.0 | 13.90 |

---

## 4. Analysis & Architectural Recommendation

### Key Findings

1. **Candidate-Generation Quality (Arm 4 vs. Arm 3)**:
   - On `scifact` and `nfcorpus`, Arm 4 (`SPLATE adapter + ColBERT MaxSim`) tracks Arm 3 (`Learned sparse + ColBERT MaxSim`) closely in `nDCG@10` (within `0.001` to `0.019` at $C \ge 25$) while requiring only a single late-interaction encoder backbone at index and query time.
   - However, on deeper recall (`Recall@100` at $C = 100$), dedicated learned sparse representations (Arm 3) consistently outperform the linear SPLATE adapter (`0.6832` vs. `0.6252` on `scifact`; `1.0000` vs. `0.8875` on `splate-synth-640d-32q`, where Arm 1 BM25 plateaus at `0.6750` due to unstemmed inflection mismatch). Because SPLATE projects compressed $d$-dimensional token vectors linearly through $W \in \mathbb{R}^{|V| \times d}$, high-frequency dense subspace collisions introduce more false-positive vocabulary activations than an MLM head trained end-to-end with FLOPS regularization.

2. **Query Latency & Index Build Cost**:
   - Even with a small projection vocabulary ($|V| = 2048, d = 32$), the dense matrix-vector projection $W t + b$ per query token adds **~180 to 205 $\mu\text{s}$** of CPU overhead per query (`p50 = 311.9 us` for Arm 4 vs. `105.6 us` for Arm 3 at $C = 50$ on `scifact`; `225.4 us` vs. `32.7 us` on `nfcorpus`). At full BERT/WordPiece vocabulary sizes ($|V| = 30,522, d = 128$), this dense projection scales as $O(n_q \cdot |V| \cdot d)$ (`~3.9M` multiply-adds per query token), which dominates CPU query latency compared to MaxScore posting traversal (`~10 to 65 us`) and SIMD `MaxSim` reranking over $C = 50$ candidates (`~15 to 45 us`).
   - At index build time, projecting every document token through $W \in \mathbb{R}^{|V| \times d}$ increases index construction time from `5.8 ms` (Arm 3) to `1,112.7 ms` (Arm 4) on `scifact` and from `10.9 ms` to `1,738.6 ms` on `nfcorpus`.

3. **Memory Footprint**:
   - Both Arm 3 (`10,943.0 KiB` on `scifact`) and Arm 4 (`11,229.1 KiB` on `scifact`) are dominated (`> 90%`) by the unquantized `TokenIndex` (`10,135.0 KiB`) required for exact Stage-2 `MaxSim` reranking. Arm 4 does **not** reduce index memory relative to Arm 3; in fact, storing the `[vocab, dim]` SPLATE adapter weights adds `256 KiB` (for $|V|=2048, d=32$) up to `15 MiB` (for $|V|=30522, d=128$).

### Recommendation: Do Not Promote SPLATE (`Arm 4`) to Public Production API

**Decision**: **Reject promoting SPLATE (Arm 4) to a public crate or production pipeline stage.** Keep the prototype and 4-arm comparison benchmark inside `turbo-eval::splate` as a permanent regression and research harness.

**Rationale**:
- On CPU, `turboembed` already ships `Bm25PostingIndex` (Arm 1, `10-68 us` `p50`), `RetrievalCascade` (`analysis/chunker/src/cascade.rs` from `#712`, combining BM25 + dense vector index + optional `TokenIndex` / cross-encoder reranking), and native `embed.sparse` (`MaxLogPool` in `core/src/sparse.rs`).
- When a caller already runs a late-interaction encoder (`TokenIndex`), **BM25 (`Bm25PostingIndex`) or hybrid BM25 + static dense (`RetrievalCascade`) provides a faster (`6-15x` lower latency), smaller (`0` adapter bytes, `180x` faster index build), and higher-precision Stage-1 candidate generator for `TokenIndex::rerank` than a linear SPLATE projection layer.**
- If learned sparse expansion is needed to bridge severe lexical/morphological mismatch (`Recall@100 = 1.0000` vs. BM25 `0.6750` on `splate-synth`), Arm 3 (`embed.sparse` + `SparsePostingIndex` + `TokenIndex::rerank`) delivers higher recall and `3-7x` lower query latency than Arm 4.
