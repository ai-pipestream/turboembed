# turbo-glossary

Multi-token terminology and glossary span extraction (`GlossaryAnnotator`, `GlossaryMatcher`, `TextMatcher`, `TokenMatcher`, `MappedGlossaryMatcher`) with leftmost-longest overlap resolution (`select`), optional case sensitivity, and mapped double-array automaton images (`analysis/glossary/src/mapped/mod.rs`).

## Features

- **Character and token phrase matching (`TextMatcher`, `TokenMatcher`, `GlossaryMatcher`, `analysis/glossary/src/lib.rs`, `analysis/glossary/src/matcher.rs`)**: Matches glossary terms by code-point Aho-Corasick automaton (`TextMatcher`) or analyzed token sequences (`TokenMatcher`), with leftmost-longest-first overlap selection (`analysis/glossary/src/select.rs`).
- **Mapped double-array images (`MappedGlossaryWriter`, `MappedGlossaryMatcher`, `analysis/glossary/src/mapped/mod.rs`)**: Reads flat character automaton images in place from byte slices or memory-mapped files (`MappedFile` under the `mmap` feature) without per-state heap allocation.
- **Glossary readers and pipeline integration (`DelimitedReader`, `TbxReader`, `GlossaryAnnotator`, `analysis/glossary/src/annotate.rs`)**: Reads CSV/TSV and TBX 2/3 termbases into `Glossary`, and implements `Annotator` (`GlossaryAnnotator`) reading a view and `Tokens` (plus `Stems` when configured) to write the `GlossaryTerms` (`turbo:glossary-terms`) layer on `Document`.

## Testing, Accuracy, and Speed

- **Test suite**: Verified by unit and integration tests in `analysis/glossary/src/lib.rs`, `analysis/glossary/tests/snapshots.rs`, `analysis/glossary/tests/pipeline.rs`, `analysis/glossary/tests/round_trip.rs`, `analysis/glossary/tests/mapped.rs`, and `analysis/glossary/tests/mapped_alloc.rs`.
- **Accuracy and speed**: Deterministic $O(n)$ token-stream traversal over prefix tries with zero per-document heap allocations on misses.
