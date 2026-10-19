# Parser and POS tagger boundary receipts (#649, PR #667)

Code head the gates ran on: `d1f61b79d8c0d83d1105564b7fcbdded3d3103b5`, which merges main at
`868d2bf3673a30060389a2957cddd7c4a87ae006`. Absolute checkout paths in the logs are replaced
with `<checkout>`.

## Red cases

The new tests in `analysis/postag/tests/neural_boundaries/` were written first and run against
the unfixed `BilstmPosModel::new` and `CrfScorer::loss_and_gradients` (`red.log`): 10 of 20
failed.

- Three CRF accumulator tests panicked with index out of bounds in `crf/mod.rs` (a short
  `d_emissions` row, a short `d_transitions` row after `infer` had already written into it, and a
  short `d_start` / `d_end`).
- `new_rejects_non_finite_lstm_weights`: NaN in the character forward `W` was accepted.
- `new_checks_the_second_word_layer_tensors`: NaN in the second backward `U` was accepted.
- `new_rejects_ragged_lstm_weight_rows`: a word forward `W` narrowed by one column in every row
  was accepted. A single ragged row was already refused, but only by the serializer that
  `new` runs to compute the model hash, not by any shape check.
- `new_rejects_duplicate_tags`, `new_rejects_duplicate_word_and_char_entries`,
  `new_rejects_duplicate_pretrained_entries`, `new_requires_unknown_at_row_zero`: each model
  was accepted by `new`, although `load` refuses the same model.

The two direct tests of the LSTM tensor check in `bilstm_model/validate.rs` cover a gate bias of
the wrong length, which the public `LstmLayer` API cannot produce (`b_mut` hands out a
fixed-length slice).

## Green

`green.log`: the same 20 tests pass after the fix. `every_model_new_accepts_round_trips_through_load`
checks that a plain, a stacked, and a pretrained model that `new` accepts also serialize and load
back with identical scores.

All gate commands exited 0, with `RUSTFLAGS=-D warnings` as CI sets it:

```
cargo fmt --all --check
cargo clippy --locked -p turbo-parser -p turbo-postag -p turbo-bench --all-targets -- -D warnings
cargo test --locked -p turbo-parser -p turbo-postag
cargo test --locked -p turbo-bench --test tree
cargo clippy --locked --workspace --all-targets
RUSTDOCFLAGS=-D warnings cargo doc --locked --no-deps -p turbo-parser -p turbo-postag
cargo test --locked -p turbo-train-components -p turbo-formats
```

`tests.log`: 135 passed, 0 failed, 1 ignored across 15 binaries. `tree.log`: 3 passed.
`dependents.log` (the two crates that build on `turbo-parser`): 379 passed, 0 failed, 1 ignored.

## Splits

Pure moves. A line-multiset comparison of each old file against its new directory, with imports
and module docs set aside, differs only in `pub(super)` on items a sibling module now calls, the
`impl FeedforwardPosModel` headers the model split adds, and rustfmt rewrapping. Every public path
is kept through re-exports.

| Old file | Lines | New files (lines) |
| --- | --- | --- |
| `analysis/parser/src/treeinsert.rs` | 810 | `treeinsert/mod.rs` (213), `context.rs` (268), `parser.rs` (301), `feature_tests.rs` (55) |
| `analysis/parser/src/chunking.rs` | 539 | `chunking/mod.rs` (12), `context.rs` (285), `parser.rs` (259) |
| `analysis/parser/src/context.rs` | 813 | `context/mod.rs` (16), `sink.rs` (165), `cons.rs` (383), `chunk.rs` (284) |
| `analysis/parser/src/head_rules.rs` | 766 | `head_rules/mod.rs` (320), `english.rs` (133), `ancora.rs` (338) |
| `analysis/parser/src/model.rs` | 533 | `model/mod.rs` (249), `chunker.rs` (116), `parser_model.rs` (188) |
| `analysis/postag/src/neural/feedforward/model.rs` | 756 | `model/mod.rs` (336), `score.rs` (179), `io.rs` (267) |
| `analysis/postag/src/neural/feedforward/trainer.rs` | 592 | `trainer/mod.rs` (402), `settings.rs` (82), `vocab.rs` (74), `adagrad.rs` (67) |

The largest `.rs` file left in `analysis/parser/src` and `analysis/postag/src` is
`postag/src/neural/lstm.rs` at 486 lines. No README or `docs/` path named a moved file.
