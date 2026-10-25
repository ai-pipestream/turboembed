# Tables and key/value verification

Code commit: `68135d9a361156caf86ade252490da2e8ae2a3ac`.
Main parent: `22a59907c41171f5b95aa1b3b93cc3267388543e`, merged normally
at `31435a7efbb118e9ccda35f697c45892899860fb`.

The first acceptance-test build failed with E0432 for the missing
`Tables`, `ColumnAlign`, `KeyValuePairs` and annotator APIs. After the
implementation, the clean code commit passed 232 tests in
`turbo-analysis-layers` and `turbo-structure`, with zero failures or
ignores. All six repository tree checks, formatting and strict all-target
clippy for those two packages passed. The largest changed Rust file is
210 lines. No external dependency was added; `turbo-structure` directly
uses the existing workspace Unicode helpers.

The production tests cover GFM alignment and cell boundaries, escaped
pipes and backticks, empty/padded/excess cells, block and container
boundaries, ASCII header separators, multiline UTF-8 fragments,
governing column/row headers, and multiple table IDs. Canonical source
storage and logical column order are checked separately for a wrapped
grid whose first column begins on its second physical line.

The real repository `docs/analysis.md` passes through both annotators and
the graph binary boundary. Its SHA-256 at the code commit is
`610134444bb6a4416bd1d925b7ac332b122e7f62471cae995b3edcf5f394d851`.
A named NFC view also restores with exact coordinates and provenance.
Export, binary encoding/decoding, import and re-export are equal.

Malformed binary cases reject invalid IDs, extents, flags, alignments,
secondary coordinates outside the view and coordinates inside UTF-8
characters. Direct `Document::add` checks the same secondary-coordinate
conditions. Tests exercise the existing `Layer::check_document` hook;
no core extension is needed. The explicit cell limit includes GFM's
implied empty cells and fails without returning a partial layer.

Reproduce from the repository root:

```sh
CARGO_BUILD_JOBS=2 cargo test --locked -p turbo-analysis-layers -p turbo-structure
CARGO_BUILD_JOBS=2 cargo clippy --locked -p turbo-analysis-layers -p turbo-structure --all-targets -- -D warnings
CARGO_BUILD_JOBS=2 cargo test --locked -p turbo-bench --test tree
cargo fmt --all -- --check
```

[API and representation](../tables.md). This receipt records local
correctness checks. It contains no timing, hardware-speed or hosted-CI
claim. Fresh external review is required before landing.
