# turbo-stopword

OpenNLP 3.x's bundled stopword lists and the lookup of `opennlp.tools.stopword`,
ported to Rust as a flag per token (`turbo:stopwords`) rather than a filter that
removes tokens. The crate documentation (`src/lib.rs`) describes the lookup and
where OpenNLP's behaviour is surprising; `data/README.md` says where each list
comes from and under what license. The layer and the language enum are in
`turbo-stopword-layers`, so a consumer of the flags does not depend on this
crate.

It follows OpenNLP at commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` of the
OpenNLP fork, run on JDK 21.

## Snapshots and property tests

`tests/lists.rs` pins each bundled list's size, digest and words against the
files in `data/` (and the case-fold lookups documented in `src/lib.rs`).
`tests/codes.rs` checks `StopwordLanguage::from_code`. `tests/properties.rs`
checks flag alignment, `Stopwords::check`, and the documented surprises
(`İT` is `it`, `ſo` is not `so`, decomposed accents differ, `DASS` is not
`daß`).
