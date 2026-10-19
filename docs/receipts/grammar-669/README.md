# Grammar boundary and split receipts

Baseline: `e098c68fdb09cfc212354f3e1150e59ac5023e58`.

The 9 oversized grammar files now use focused modules. All 104 Rust source
files in `analysis/grammar/src/` have fewer than 500 lines; the largest is
481 lines. Public type and function paths are preserved.

## Red to green

- `red-chart.log`: 5 failing tests on the baseline. Positive unary cycles
  exceeded a bounded test counter; selection returned a span inside a UTF-8
  character; construction accepted invalid numeral, lexicon, and noninitial
  rule categories. The cycle test also covers absent optional words, indirect
  cycles, changing values, and repeatable results.
- `red-decimal.log`: 2 failing tests on the baseline arithmetic. Decimal
  rounding and trailing-zero normalization underflowed at `i32::MIN` scales.
- `red-ranking.log`: 2 failing tests with the old score/application additions
  restored in the split parser. Candidates overflowed instead of being
  excluded from ranking. The regression fixture seeds boundary counts
  directly, without constructing a huge input.
- The final suite passes all 9 regressions. Additional tests check extreme
  decimal operations, zero denominators, negative ages, ordinal/exponent
  limits, calendar dates, inverted ranges, and annotator version `2`.

The chart allows at most one occurrence of a rule along a derivation of the
same span. Rules may recur while consuming additional lexemes. This bounds
unary cycles but does not impose a global memory quota for ambiguous inputs.
Offset/count conversions cannot wrap, and unrepresentable ranking totals
exclude the candidate. `select` still requires the original source text;
incompatible UTF-8 layouts return no additional mentions.

## Exact corpus comparison

The example `analysis/grammar/examples/corpus_snapshot.rs` was run at the
baseline and the implementation commit `0fdd43c8`. For the baseline run, the
same example was copied into that revision without changing library code.
It reads all 3,785 cases from the checked-in English, Spanish and Dutch
Recognizers-Text files, including cases outside the conformance pass lists.
It writes selected byte spans, values, scores, and application counts; time
IDs are expanded into their time values.

```sh
cargo run --locked -p turbo-grammar --example corpus_snapshot -- snapshot.txt
```

The dumps compare byte for byte. Both SHA-256 values are:

```text
fcfad627baff6715dc78b79d63ec6febc5571e057ba0741a3033a7a77e716194
```

This measures parity with the baseline, not agreement with all reference
expectations. The existing unresolved-date convention, including `Feb 30`,
is preserved.

## Gates

```sh
cargo fmt --all --check
cargo clippy --locked -p turbo-grammar --all-targets -- -D warnings
cargo test --locked -p turbo-grammar
cargo test --locked -p turbo-bench --test tree
```

All pass: 210 grammar tests and 3 documentation/tree tests, with no ignored
tests. Logs use `[workspace]` in place of the checkout prefix. Build artifacts
were kept in a target directory belonging to this worktree.
