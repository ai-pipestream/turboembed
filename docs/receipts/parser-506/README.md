# Parser feature buffers and candidate cloning: issue #506

Implementer: codex. Host: krick. Date: 2026-10-16.
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
state, output trees, and the optional legacy cache still allocate. No elapsed-time
speedup or hardware throughput claim is made.

## Logs

- `red-chunk-allocations.log`: original 113-allocation failure.
- `red-reversed-span.log`: original invalid-span acceptance failure.
- `feature-allocations.log`: buffered generator checks and allocation counts.
- `green-tests.log`: full parser and train-components test output.
- `clippy.log`: required warning-free lint check.
- `fmt.log`: root formatting check.

The red span log includes temporary unused-item warnings from the feature-buffer
implementation in progress. The final lint run is clean.

## Review gate

Requires two independent adversary PASS reviews before merge. Codex is the
implementer and does not count as an adversary reviewer for this PR.
