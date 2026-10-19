# Language-aware terms and BM25

Base: `272b9298`.

Regression command:

```sh
cargo test -p turbo-chunker --test terms linguistic_terms_and_raw_index_can_coexist
```

Before the patch, this test fails during Analyzer construction with a
collision at `turbo:terms`. With the patch, it passes.
`normalized_terms_feed_counts_and_bm25` checks that the English profile's
`Cats` becomes the retrieval term `cats`, preserves the original text,
excludes punctuation, and ranks the cats chunk first. This fixture checks
integration, not retrieval quality.

Validation:

```sh
cargo fmt --all --check
cargo test -p turbo-chunker
cargo clippy -p turbo-chunker --all-targets -- -D warnings
```

All passed locally, CPU: 127 tests, no failures or skips. Tests cover each
column selection, missing stem/lemma errors and provenance.

Review focus: textual key migration to `turbo:indexed-terms`, token
alignment, explicit form selection, query normalization responsibility.
