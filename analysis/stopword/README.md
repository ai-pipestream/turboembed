# turbo-stopword

OpenNLP 3.x's bundled stopword lists and the lookup of `opennlp.tools.stopword`,
ported to Rust as a flag per token (`turbo:stopwords`) rather than a filter that
removes tokens. The crate documentation (`src/lib.rs`) describes the lookup and
where OpenNLP's behaviour is surprising; `data/README.md` says where each list
comes from and under what license. The layer and the language enum are in
`turbo-stopword-layers`, so a consumer of the flags does not depend on this
crate.

The lists and lookup follow the behaviour documented in `src/lib.rs`.

## Snapshots and property tests

`tests/lists.rs` checks the reviewed digest snapshot in `data/SHA256SUMS`
against each bundled list file, and that the words the crate loads match
those files. `tests/codes.rs` checks language code resolution.
`tests/properties.rs` checks that flag vectors stay aligned with the
token layer and that UTF-16 and code point offsets of every token span
round trip. `tests/pipeline.rs` covers the annotator in a pipeline.
Nothing compares against JVM output.
