# turbo-glossary

Multi-token terminology and glossary span extraction (`GlossaryAnnotator`, `GlossaryMatcher`) with longest-match-first or all-matches resolution, optional case sensitivity, and static-borrow zero-copy entry storage (`analysis/glossary/src/mapped/mod.rs`).

## Features

- **Trie-backed token phrase matching (`analysis/glossary/src/lib.rs`, `analysis/glossary/src/matcher.rs`)**: Indexes multi-token glossary entries by token sequence with configurable case folding and overlap resolution (`analysis/glossary/src/select.rs`).
- **Zero-copy memory-mapped dictionaries (`analysis/glossary/src/mapped/mod.rs`)**: Supports borrowed string slices mapped directly from disk without per-entry heap allocation.
- **Document pipeline integration (`analysis/glossary/src/annotate.rs`)**: Implements `Annotator` to read `Sentences` and `Tokens` layers and emit typed glossary match annotations with canonical concept IDs and metadata.

## Testing, Accuracy, and Speed

- **Test suite**: Verified by unit and integration tests in `analysis/glossary/src/lib.rs`, `analysis/glossary/tests/pipeline.rs`, `analysis/glossary/tests/round_trip.rs`, `analysis/glossary/tests/mapped.rs`, and `analysis/glossary/tests/mapped_alloc.rs`.
- **Accuracy and speed**: Deterministic $O(n)$ token-stream traversal over prefix tries with zero per-document heap allocations on misses.
