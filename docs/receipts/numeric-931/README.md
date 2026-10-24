# Numeric candidate scanning, issue #931

The original extended scanner produced the same output as main on the
existing non-Latin stress input but took substantially longer. This receipt
records the failed implementation as well as the correction. The acceptance
criterion has not been narrowed or waived.

## Revisions and protocol

- `red.json`: original head `d70008607396eedb92b8ed3d6bca5ace9d44e120`
  against main `97dee9b334a06216ff6cdb4bc7bc764f26ade70c`; 12 alternating
  pairs per corpus, 100 timed iterations per sample, CPU 12.
- `rejected-intermediate.json`: intermediate code
  `225c1553ca09ac83a45e774cfaa4f4c6c11d43bf` against main
  `654f5f85528c9ab67468d7d11113ba8f778a7a70`; 24 pairs, 1,000 iterations,
  CPU 7. Despite removing most of the original penalty, its Latin output
  remained slower, so it was rejected.
- `final.json`: corrected code `4c025715266e92d02d33113241c0f122936b9f15`
  against that same main revision; 24 pairs, 1,000 iterations, CPU 7.

All are optimized Cargo release builds, Rust 1.99.0, with two build jobs.
Compilation completed before timing. Each executable warms for 10 scans,
then measures repeated complete `NumberScanner::scan` calls and output
reset using `Instant`. Output checksums are calculated after measurement.
Each round runs main then candidate or candidate then main, alternating by
round. Every sample is retained; none was removed as an outlier. The red
harness was the earlier equivalent, unformatted 100-iteration version; the
final harness is `analysis/numeric/tests/scan_timing.rs` and its SHA-256 is
recorded in the final metadata.

The CPU was an AMD Ryzen 9 9950X3D with the powersave governor. Affinity pins
the executable's threads, but does not reserve the CPU or its SMT sibling.
The final CPU was selected from a utilization sample. Other assigned timing
sweeps were coordinated, but background builds and services remained. This
was a shared host, not an isolated-machine measurement. Final load-average
snapshots were 12.11/16.85/15.82 at start and 11.01/14.83/15.18 at finish;
each sample's snapshot is retained in JSON. Small differences are not
claimed as significant speedups or as a guarantee for another machine.

The scanner and its dependency source trees are identical between the two
pinned main revisions; the intervening main changes were elsewhere.

## Inputs and output checks

The harness uses the exact three corpora from the existing PR timing data:

- `ascii`: English/German prose and US/EU numeric golden inputs, repeated
  200 times, 112,800 bytes; 6,000 mentions per iteration on both revisions.
- `non_latin`: the PR's constructed Devanagari, Thai, Arabic, CJK punctuation
  and emoji stress text with ASCII numerals, repeated 2,500 times; 15,000
  mentions per iteration on both revisions. It is not an external corpus.
- `non_latin_nd`: the corresponding constructed text containing additional
  Nd numerals. Main emits 7,500 mentions; the candidate emits 15,000. Its
  extra work is reported separately from the existing-output comparisons.

Input SHA-256 values match in every retained run. Each sample records the
byte count, iteration count, mention count, and FNV-1a checksum over every
mention's byte span, numeral end, sign, exact decimal value, scale,
multiplier, marker, and notation. The two existing-output corpora match
both counts and checksums across revisions. These are output checksums,
not cryptographic source identities; input, harness and final executable
identities use SHA-256.

## Results

Original red non-Latin median: 87.39 ms on main versus 128.35 ms on the
candidate, with identical 15,000-mention output; median paired ratio 1.451.
The original Latin median paired ratio was 1.002.

Final samples each time 1,000 iterations:

- Latin: main median 206.731 ms, candidate 200.412 ms; median paired
  candidate/main ratio 0.9664, with 6,000 matching mentions per iteration.
- Non-Latin existing output: main median 892.576 ms, candidate 873.485 ms;
  median paired ratio 0.9894, with 15,000 matching mentions per iteration.
- Non-Latin Nd work: main median 643.433 ms, candidate 1,279.858 ms; median
  paired ratio 1.9539, producing twice the mentions. This is not a
  same-output regression calculation.

A ratio above 1 means slower. Separate medians and the median of the paired
ratios are both reported; they need not divide to the same result. The
final run shows no median slowdown on either existing-output corpus. It
supports this host's regression check, not a claim of universal speed or
statistical significance for the small differences.

## Implementation and correctness

The bounded AVX2 prefilter handles 32 candidate bytes and two bytes of
lookahead at a time; every candidate is confirmed by the existing Nd
classifier. Scalar scanning remains available. The failed portable
word-skip experiment is removed from the final code.

The default number scan specializes away unused region fallback work.
Large parse/continuation helpers stay outside the compact candidate
matcher. Text start and an actual ASCII space already establish the word
and sign boundary; other contexts still use the general Unicode boundary
logic. No number syntax, stored value, span contract or parity option was
changed.

Validation: 312 numeric tests pass, including the JVM goldens, exact
boundary/UTF-8 span regressions, exhaustive Unicode candidate comparison,
and every Nd digit across wide-block edges/tails. Both full cross-layer
fuzz invariant tests pass over all named input sets. Strict all-target
Clippy for numeric/fuzz, workspace formatting, and five tree tests pass.
The ignored release timing test was exercised separately for these records.

## Reproduction

Use fresh isolated checkouts at the pinned main and candidate revisions.
Copy `analysis/numeric/tests/scan_timing.rs` from the candidate to the same
path in the main checkout, without changing main's production sources.
In each checkout:

```sh
mkdir -p target/tmp
CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$PWD/target" TMPDIR="$PWD/target/tmp" \
  cargo test --release --locked -p turbo-numeric --test scan_timing --no-run
```

Use the two test-executable paths printed by Cargo as `MAIN_BINARY` and
`HEAD_BINARY`. Do not invoke Cargo inside the timed loop. From the candidate:

```sh
CPU=7 ROUNDS=24 TURBO_BENCH_ITERS=1000 \
MAIN_SHA=654f5f85528c9ab67468d7d11113ba8f778a7a70 \
HEAD_SHA=4c025715266e92d02d33113241c0f122936b9f15 \
  docs/receipts/numeric-931/run-pairs.sh "$MAIN_BINARY" "$HEAD_BINARY" > run.txt
```

Choose an appropriate CPU for the local host and retain its ambient-load
information. The script records binary hashes and emits each raw sample.
The final JSON is the retained equivalent run, with corpus hashes and
paired medians aggregated from those raw values. Other exploratory source
patches and timing logs remain in the implementation checkout's ignored
evidence directory; they were not substituted for the final run.
