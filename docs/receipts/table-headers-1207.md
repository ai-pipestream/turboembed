# Table row and chunk header context verification (#1207)

Code commit: `436af8c0d034f852e1ca83694e1672a8d0ecc270`.
Parent feature commit: `8a0da1427b470ff9bf06bd43869e8f49fda245d7`
(PR #1274, #1206). That parent includes tested main
`8d8f74e264b74fc992db8c361e5a8aad122afc2a`.
The commands below ran from the clean code commit. This receipt is a
subsequent documentation commit. External review remains required.

## Production behavior

The actual pipeline uses `NewlineDetector`, `SimpleTokenizer`,
`SubwordAnnotator` with the committed hash-verified tiny-BERT bundle,
`TableGridAnnotator`, `TableBoundaryAnnotator`, the base `Chunker`, and
`TableHeaderAnnotator`. No model download or inference is required.

The Markdown tests cover both 30 logical rows including the header and
30 body rows. Each produces exactly four chunks. Every chunk starts and
ends at full logical row boundaries, carries the exact original header
cell coordinates and rendered prefix, and has the expected continuation
flag: false on the first chunk, true on the other three. Model-token
counts equal the actual subwords overlapping each emitted chunk span.

Other cases cover multiline ASCII headers, escaped pipes, multiple tables,
headerless grids, oversized rows with clause splitting enabled, conflicting
sentence/section cuts, unordered tokens, and named NFC views with named
chunk/table dependencies. Graph bytes round-trip all contexts and complete
provenance. Insertion and binary import refuse malformed chunk/table IDs,
UTF-8/outside secondary header coordinates, changed prefixes, missing
contexts, continuation flags and trailing bytes.

The header allocation regression runs the production header analyzer,
including document insertion and validation. With one table, three header
cells and four chunks, total allocated bytes were 7,220 for both 30 and
3,000 body rows. The index visits body metadata once and collects/sorts
only headers. This is an allocation regression, not a timing claim.

## Checks

Build environment: `CARGO_BUILD_JOBS=2`, `CARGO_PROFILE_DEV_DEBUG=0`,
`CARGO_INCREMENTAL=0`. Existing build directories were reused.

- `CARGO_TARGET_DIR=target/verification/build cargo test --locked -p turbo-chunker -p turbo-analysis-layers -p turbo-structure`:
  545 passed, zero failed, one existing ignored timing test
  (`entity::build::cost_tests::timing_crowded_sentence`).
- `CARGO_TARGET_DIR=target/verification/build cargo clippy --locked -p turbo-analysis-layers -p turbo-chunker -p turbo-structure --all-targets -- -D warnings`: passed.
- `cargo fmt --all -- --check` and `git diff --check`: passed.
- `CARGO_TARGET_DIR=target/verification/bench cargo test --locked -p turbo-bench --test tree`:
  six passed, zero failed or ignored.
- Largest changed Rust file: 449 lines. No external dependency or regex
  was added; the three new development dependencies are workspace crates.

This proves host analysis and graph transport. It does not claim a provider
capability, matched throughput result or a full-workspace test run. No
training, timing run, model/hash/floor change or persistent setting change
was made. Header prefixes are metadata; consumers reserve their token cost
before prepending them for embedding, as described in [tables.md](../tables.md).
