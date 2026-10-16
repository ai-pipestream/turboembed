# Parser feature buffers and candidate cloning: issue #506

CPU: AMD Ryzen 9 9950X3D, x86_64. Date: 2026-10-16.
Base: `0dcf0828276048f19b0c90db93400517e37ce35e` (`origin/main`).

The parser build, check, attach, and chunk contexts now have reusable `Features`
entry points. Inference calls `MaxentModel::eval_features(features.iter(), ...)`;
this API accepts the model's modified UTF-8 keys directly. Public string contexts
and training events use the same feature emission order. Dictionary keys are
prepared once for lookup without temporary strings. The default chunk beam path
reads its five slots directly from `TokenTag`, without collecting sentence arrays.
The optional legacy chunk cache retains its string API and key behavior.

Candidate probabilities are checked before copying a result tree. Chunking check
features receive the candidate's first label directly, removing the temporary
check tree and its copied child array. Both returned branches retain the original
probability-addition order. Tree insertion prepares a done branch only for an
accepted attachment, or for the initial done outcome.

## Red/Green receipt

The new chunk allocation test was first run on the base implementation:

```text
cargo test -p turbo-parser --test feature_buffers --test parser_tests
chunk_beam_features_allocate_nothing_after_warmup ... FAILED
left: 113
right: 0
```

The command stopped at that failed integration-test binary. The span regression
was then run separately before changing `Parse::insert`:

```text
cargo test -p turbo-parser --test parser_tests insert_rejects_reversed_spans_without_changing_the_tree
insert_rejects_reversed_spans_without_changing_the_tree ... FAILED
assertion failed: matches!(root.insert(reversed.clone()), Err(ParseError::NotContained { .. }))
```

`Span` has public fields. A caller can provide `Span { start: 2, end: 1 }`
without passing through `Span::new`. Insertion previously accepted that range
and changed the tree. It now returns `NotContained` before mutation, including
through `TreeBuilder`'s fallback. Empty constituents on valid UTF-8 boundaries
remain accepted. The bracket-reader compatibility rules are unchanged.

With the patch:

```text
cargo test -p turbo-parser -p turbo-train-components
172 passed; 0 failed; 1 ignored
cargo fmt --all --check
PASS
cargo clippy -p turbo-parser -p turbo-train-components --all-targets -- -D warnings
PASS
```

The ignored test is the existing `the_published_ud_models_are_reproduced` test,
which downloads treebanks and trains models. Checked-in parser JVM goldens,
trained-model decoding, and training-event tests passed. No golden files changed.

`cargo test -p turbo-parser --test feature_buffers -- --nocapture` also passed.
Every tested generator made zero allocations after buffer warm-up: chunk,
chunking build with and without a dictionary, chunking check, tree-insertion
build, attach, and check. Cases include multiple punctuation marks, frontier
trimming, NUL, accented text, and supplementary Unicode characters. The fixture
produces up to 275 features in a context. Byte-for-byte buffer comparisons check
feature ordering and modified UTF-8 encoding. Additional tests compare compact
frontier selection with the existing path traversal for all three-valued head
patterns through depth six, and check equal-probability punctuation precedence.

These allocation counts cover feature population after capacity growth. Parser
state, output trees, and the optional legacy cache still allocate. Timing measurements for the training-path follow-up are recorded below.

## Logs

- `red-chunk-allocations.log`: original 113-allocation failure.
- `red-reversed-span.log`: original invalid-span acceptance failure.
- `feature-allocations.log`: buffered generator checks and allocation counts.
- `green-tests.log`: full parser and train-components test output.
- `clippy.log`: required warning-free lint check.
- `fmt.log`: root formatting check.

## Training-path follow-up

String contexts now render constituent slots, chunk slots, punctuation separators,
and repeated productions once per context call. The string sink reserves 64 bytes
per output feature to avoid growth on each formatted piece. This trades some spare
string capacity for fewer reallocations. The buffered inference sink retains direct
formatting and zero allocations after warm-up.

The reproducible harness is
`analysis/train-components/examples/parser_context_bench.rs`:

```sh
CARGO_TARGET_DIR=target cargo build --release -p turbo-train-components --example parser_context_bench
taskset -c 8 target/release/examples/parser_context_bench
```

The same harness was built on main `a7cf38a8`, the prior PR head `c02a4a41`,
and the corrected sources in this commit. Each executable uses the same checked-in
JVM model fixture under `analysis/goldens/data/parse/model-{chunking,treeinsert}`
and the 50 trees in `inputs/parse-treebank/{treebank,held-out}.txt`. Inference
uses beam 20, advance percentage 0.95, and k=1 or k=5. Event streams use the gold
trees and each model's own parses, accept only trees supported by that event
stream, then repeat them 20 times: 2,000 samples for each chunking stream and
1,100 for each tree-insertion stream. No corpus or model was downloaded.

Each workload runs one warm-up and five timed repetitions. The clock excludes
result hashing and destruction. Runs were sequential on logical CPU 8, with no
build running. Run order was main, after, before, after, main, after, main.
Numbers below are milliseconds: medians of three run medians for main and after,
and one five-repetition median for before. Raw run medians, sample counts and
output digests are in `training-timing.log`.

```text
workload                        main    before     after after/main
parse-chunking-k1            480.484   291.359   293.432     0.611
parse-chunking-k5            522.765   298.135   289.633     0.554
events-chunking-build         93.908   138.814    71.600     0.762
events-chunking-check         45.205    44.441    39.376     0.871
events-chunking-chunk         21.788    31.131    21.604     0.992
parse-treeinsert-k1          264.943   303.373   298.793     1.128
parse-treeinsert-k5          272.553   319.050   305.291     1.120
events-treeinsert-build       43.064    59.242    33.538     0.779
events-treeinsert-check       21.060    20.365    17.470     0.830
events-treeinsert-attach      58.175    80.492    43.592     0.749
```

All six training streams are at or below main's median in this fixture run.
Chunking inference retains its improvement. Tree-insertion inference remains
about 12% slower than main here, compared with 15% to 17% before this follow-up;
this change does not establish a general inference speedup for tree insertion.
These fixture measurements differ from the published-model workload used in the
original review. They are local timing evidence, not a CI performance guarantee.

Every event outcome, context and value digest, and every parse Penn-string and
probability-bit digest, matched across all three executables and every run.
The allocation test still checks the real warmed buffers independently of timing.
