# Shared whitespace definitions

`turbo_unicode::whitespace` owns the Unicode White_Space list and the
`WhitespaceMode::{Unicode, Legacy}` predicates. The ASCII predicate is const,
so both simple and whitespace tokenizers derive their byte tables from it.
The whitespace normalizer derives its public `tables::WHITESPACE` character
list from `UNICODE_WHITESPACE`, and its membership predicate delegates to the
Unicode mode. The grammar lexer and abbreviation inventory use the same mode.
The normalizer's UTF-8 lead-byte shortcut is tested against every non-ASCII
member of the shared list, so a future table addition cannot silently disappear.

`turbo_unicode::java::is_whitespace(char)` delegates to
`java::code_point::is_whitespace(u32)`. The unused duplicate
`turbo_jvm::chars::is_java_whitespace` was removed. Both Unicode entry points
are exhaustively checked against the original JDK 21 whitespace vectors;
`turbo-jvm` remains dependency-free. Unicode whitespace includes U+0085 and
no-break spaces. Java whitespace excludes them but includes U+001C..U+001F.
These definitions remain distinct; this consolidation does not change them.

Tests exercise normalization, grammar lexemes and original byte ranges on
text containing U+000B, U+0085, U+00A0, U+3000 and U+001C. The first four
separate words; U+001C remains text in Unicode mode. Existing normalizer
alignment tests still verify source and output mappings.

## Release timing receipt

Baseline: main `b1a8a030`. Release binaries built with the same compiler,
fixed logical CPU, alternating execution order on the same shared host.
The probes execute actual normalization (including alignment), grammar
lexing and abbreviation judging. No replacement algorithms are timed.

| Probe | Work per run | Baseline median ms | Shared tables median ms |
| --- | --- | ---: | ---: |
| Whitespace normalization | 1,000 calls, paragraph repeated 4,096 times | 256.080 | 227.210 |
| Grammar lexer | 10,000 calls, paragraph repeated 64 times | 88.968 | 87.843 |
| Abbreviation judge | 10,000 passes over all periods, paragraph repeated 64 times | 214.442 | 215.892 |

Six runs per binary, all retained:

| Probe | Baseline ms | Shared tables ms |
| --- | --- | --- |
| Normalize | 335.185, 254.424, 255.845, 256.314, 255.249, 259.417 | 262.166, 225.924, 226.395, 226.344, 229.355, 228.024 |
| Grammar | 95.890, 88.205, 87.664, 88.115, 89.993, 89.730 | 87.847, 89.178, 87.225, 87.449, 87.839, 88.803 |
| Inventory | 250.299, 213.720, 214.493, 214.390, 213.189, 218.063 | 216.050, 214.147, 214.455, 229.722, 216.267, 215.733 |

The inventory's initial +0.7% median difference prompted a longer paired
measurement, after two warm-up pairs. All twelve measured pairs are below:

| Run | Baseline ms | Shared tables ms |
| --- | ---: | ---: |
| 1 | 215.489 | 213.603 |
| 2 | 218.116 | 213.502 |
| 3 | 224.841 | 215.707 |
| 4 | 213.458 | 215.564 |
| 5 | 214.389 | 214.184 |
| 6 | 213.721 | 214.183 |
| 7 | 220.052 | 214.272 |
| 8 | 213.042 | 227.500 |
| 9 | 213.268 | 213.584 |
| 10 | 213.315 | 212.839 |
| 11 | 212.513 | 215.530 |
| 12 | 213.114 | 216.127 |

The longer-run medians are 213.590 and 214.228 ms (+0.3%); the median paired
change is +0.0555 ms (+0.0263%). Six pairs favor each binary. This is within
the observed variation, not evidence of a sustained inventory slowdown or
speed improvement. Normalization and grammar were faster in these probes;
none of these measurements establishes whole-model throughput.

Reproduce the probes with:

```sh
cargo run --locked --release -p turbo-normalize --example normalize_whitespace_timing
cargo run --locked --release -p turbo-grammar --example grammar_whitespace_timing
cargo run --locked --release -p turbo-sentdetect --example inventory_whitespace_timing
```

For a comparison, build these same probe sources against both revisions,
retain both binaries, pin execution to one logical CPU and alternate order.
The short paragraphs include ordinary words, punctuation and mixed whitespace;
lexer input also includes a URL and inventory input several abbreviation kinds.
