# Code-point lowercase

`turbo_unicode::java::to_lowercase_code_points(text, out)` appends OpenNLP
`StringUtil.toLowerCase` to an existing string. It applies the simple lowercase
mapping once per code point. `lowercase_code_points` allocates the result;
`lowercase_code_points_changes` detects whether that mapping changes the text;
`eq_lowercase_code_points` compares the mapped first argument with the second
argument as supplied, without allocating.

These functions differ from `java::to_lowercase`, which implements
`String.toLowerCase(Locale.ROOT)`: `İstanbul` becomes `istanbul`, and `ΟΔΟΣ`
becomes `οδοσ` with no final sigma. Deseret U+10400 becomes U+10428 and `ǅ`
becomes `ǆ`. ASCII runs use the same byte scanner as the full-string mapping.

The dictionary, n-gram, relation, spelling, glossary, POS, format, training and
numeric consumers share these operations. POS vocabulary symbols beginning
with `*` retain their existing exemption; suffix features still lowercase the
entire token. The UTF-16 and modified-UTF-8 writers retain their existing output
representations. Training's streaming UTF-16 hash uses the shared single-character
mapping directly rather than allocating a temporary string.

## Timing receipt

Release builds of main `b1a8a030e7098b8c189f5b29997c71b5f806d013` and the
consolidation ran on the same machine, pinned to one CPU. Each binary ran once
for warmup, followed by 8 interleaved before/after pairs, reversing execution
order on alternate pairs. The same committed example was used for both builds.
The examples exercise the actual public dictionary, character n-gram model,
and POS feature extractor with ASCII, Turkish, Greek, Deseret and titlecase words.
They assert expected lookup totals or feature suffixes and use `black_box` to
retain the measured work.

| Example | Work per run | Main median ms | Shared-helper median ms |
|---|---|---:|---:|
| `dictionary_lowercase_timing` | 3.5 million lookups, reusable scratch | 86.688 | 84.544 |
| `char_lowercase_timing` | 3.5 million model membership checks | 121.870 | 115.153 |
| `feature_lowercase_timing` | 560,000 POS feature contexts | 67.644 | 64.804 |

Individual run times in milliseconds:

- Dictionary main: 85.610, 85.650, 87.408, 86.551, 86.802, 86.573, 86.839, 87.543.
  Shared: 83.409, 84.750, 84.229, 84.228, 84.337, 87.708, 86.589, 85.444.
- N-gram main: 121.859, 121.882, 120.809, 126.880, 126.116, 121.629, 130.545, 120.461.
  Shared: 112.945, 115.750, 159.274, 114.555, 139.528, 125.077, 113.034, 112.689.
- POS main: 66.989, 69.398, 68.298, 63.035, 68.600, 66.692, 65.131, 70.117.
  Shared: 60.397, 64.038, 66.273, 68.890, 61.784, 64.715, 67.014, 64.893.

All three medians decreased in this run. Individual runs varied, particularly
for the n-gram probe; these figures do not establish an end-to-end model speedup.
Run each probe with `cargo run --locked --release -p PACKAGE --example EXAMPLE`,
using packages `turbo-maxent`, `turbo-ngram` and `turbo-postag`, respectively.
