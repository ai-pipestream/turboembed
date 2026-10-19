# Sentiment boundary and module receipts

Base commit: `1850c3d776dcbc6df8f8dbc7bd25d74d56551a3a`.

## Red and green

- `red.log`: on the original implementation, 10 of the initial 11 regression
  tests failed. Cases cover UTF-8 slicing, invalid thresholds, invalid head
  scores, token ranges, lexicon updates and digest precision, and overflow in
  the standalone VADER normalizer. The valid duplicate-update control passed.
- `split-parity.log`: the mechanical split passed all 66 existing library tests
  before the boundary fixes were applied.
- `red-alignment.log`: after the first fixes, four additional probes failed:
  sentence assignment outside its span, short vector layers, nested chunks
  losing their scores, and discontinuous units including excluded text.
- `green.log`: all 15 regression tests passed after the fixes.
- `tests.log`: 223 tests passed across doccat and analysis-layers, zero failed
  or ignored. Includes the 15 boundary tests and existing VADER, incivility,
  pragmatics, aspect, categorizer, feature and SVM coverage.

The checked annotators now reject unsupported discontinuous inputs explicitly.
They do not claim to score fragment sequences. The low-level, infallible
`score_sentence` API retains its caller validation preconditions.

## Required gates

All commands exited zero:

- `cargo fmt --all --check` (`fmt.log`, empty on success).
- `cargo clippy --locked -p turbo-doccat -p turbo-analysis-layers -p turbo-bench --all-targets -- -D warnings` (`clippy.log`).
- `cargo test --locked -p turbo-doccat -p turbo-analysis-layers` (`tests.log`).
- `cargo test --locked -p turbo-bench --test tree` (`tree.log`, 3 passed).

Additional check: `RUSTDOCFLAGS='-D warnings' cargo doc --locked -p turbo-doccat --no-deps`
passed (`rustdoc.log`). Build output paths in these logs are relative to the
checkout. `line-counts.txt` lists every Rust file under analysis/doccat; the
largest is 483 lines and every file is strictly below 500.
