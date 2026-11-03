# turbo-namefind-dl

A transformer name finder on turboembed's encoder backends, finding names with a
token-classification model.

## Model

The model is trained in Rust by this workspace, per issue #1524: a
token-classification head with CRF decode on a frozen encoder, trained on
the permissive-tier corpora #1524 lists (MultiCoNER v2, WNUT-17, MASC
named entities; GermEval 2014 for German). Until that model lands, the
only runnable configuration is the development fixture
`dslim/bert-base-NER`, which is refused for shipping because of its
CoNLL-2003 provenance; the tests that need a bundle are gated behind
`TURBO_NER_BUNDLE` and ignored by default.

## Tests

`cargo test -p turbo-namefind-dl` runs the offline tests over the crate's
own fixtures. `tests/jvm.rs` compares an ONNX Runtime export of the
development model against `tests/data/jvm-golden.json`, a record the Java
reference under `reference/` wrote; it needs `TURBO_NER_BUNDLE` and is
ignored without it.
