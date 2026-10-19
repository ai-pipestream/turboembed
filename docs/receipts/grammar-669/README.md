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

## Review follow-up: hot-path parity

Compared against main `0cd48603144f63be897d990428ccd4afd092c0fc` on
x86_64, AMD Ryzen 9 9950X3D. The baseline grammar sources were extracted from
that revision; both builds used the same dependency versions and default
release profile. No performance thresholds are inferred from CI.

The runtime cycle check is limited to one-child, zero-word candidates on a
possible category cycle. A conservative category graph is computed at parser
construction. Child values are resized once per completed rule match group,
then overwritten for each candidate. Ranking uses i64/u64 accumulation with
one checked conversion. Scratch children are consumed consecutively, removing
stored 32-bit offsets and their repeated checks. Successful chart insertion
still checks representability: text length does not bound replaced nodes or
the total child buffer. Selection checks source length and candidate UTF-8
endpoints rather than scanning every lexeme on each output request.

File-local helpers now use pub(super), with private imports in their parent.
Builders shared between languages use pub(in crate::lang); the English rules
are reused by Spanish and Dutch, so parent-only visibility would break those
callers. No changed language file retains pub(crate). The public API remains
unchanged. Unit enum paths and the parser-boundary documentation are corrected.

`perf_probe.rs` is the benchmark source. Copy it to
`analysis/grammar/examples/perf_probe.rs` in each revision, then build with
`cargo build --release --locked -p turbo-grammar --example perf_probe`.
Run each binary pinned to the same allowed CPU. Each process warms each
configuration, then measures 20 full parse/select passes. Number configurations
select numbers, ordinals, and percentages. Run 20 pairs, alternating which
binary runs first, and compare medians per configuration. Inputs are the
100 KiB sample and the checked-in Recognizers-Text corpora. No model downloads
or machine-specific input files are needed. Remove the temporary example
after comparison.

Two independent 20-pair runs are recorded in `followup-timings.csv`. Columns
are run, round, revision, configuration, microseconds per pass, and output
hash. All output hashes agree between revisions. Median changes, first run
then second run (negative means faster):

- sample_numbers: -2.58%, -2.42%.
- sample_units: -1.82%, -1.15%.
- sample_times: -3.10%, -2.22%.
- en_numbers: -2.50%, -2.94%.
- en_datetime: -0.74%, -0.51%.
- en_dimension: -0.59%, -1.27%.
- es_numbers: -4.15%, -3.50%.
- es_datetime: -0.04%, -0.56%.
- nl_numbers: -3.25%, -3.36%.

The slowest first-run result was effectively parity (-0.04%); all nine were
faster in the second run. These shared-machine measurements support parity,
not a universal speedup guarantee. Parser construction is outside the timed
region, matching the reusable-parser workload.

The full 3,785-case snapshots still compare byte for byte, with the same
SHA-256 printed above. All original boundary regressions remain green. Added
coverage checks same-length UTF-8 mismatches and category-cycle pruning with
mandatory words and multiple children. Final gates pass: fmt, clippy with
both turbo-grammar and turbo-bench, 212 grammar tests, and 3 tree tests.
See `followup-fmt.log`, `followup-clippy.log`, `followup-tests.log`, and
`followup-tree.log`.
