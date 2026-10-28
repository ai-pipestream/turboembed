# turbo-namefind

Name finders that need no model: `DictionaryNameFinder`, `RegexNameFinder`
with the patterns of `RegexNameFinderFactory`, and `dropOverlappingSpans`,
which merges the names of several finders. The crate documentation
(`src/lib.rs`) describes the modules and where the behaviour is surprising.

## Dictionaries

`DictionaryNameFinder` reads the crate's own line format, one typed entry
per line (see `src/dictionary.rs`): an optional `case-sensitive` or
`case-insensitive` directive, then `type token ...` entries. The finder
gives every name the type of the entry it matched; its entity types are
the dictionary's, in first-appearance order.

## Patterns

The five built-in patterns (`USA_PHONE_NUM`, `EMAIL`, `URL`, `MGRS`,
`DEGREES_MIN_SEC_LAT_LON`) are matched by hand-written scanners over the
space-joined tokens. A caller's own pattern is a `turbo_matcher` token
pattern: each `Tok` is the tests one token must pass, so a match is a run
of whole tokens. The crate uses no `regex` engine.

## Tests

`cargo test -p turbo-namefind` runs the unit tests, the finder and merge
integration tests, and `tests/snapshot.rs`, which compares the finders,
the factory's type order and the merge against snapshots checked in under
`tests/snapshot-data`. The snapshots are ours: refresh them with
`TURBO_SNAPSHOT_UPDATE=1` and review the diff. The same directory holds
the fixtures: the 31 hand-made sentences, the dictionaries as OpenNLP XML
(converted to the line format by the test-only converter in
`tests/snapshot/support.rs`), and the pinned GermEval 2014 excerpt that
`tests/calibrated.rs` trains on. The generated corpus and the merge cases
are seeded, so a failure repeats.

`cargo run --locked --release -p turbo-namefind --example regex_timing -- 150`
times the five built-in patterns and one token pattern over the ASCII
sentences of the fixture.

The CRF finder (`crf.rs`) and the CRF span calibration (`calibrated.rs`)
read the CRF model package through `turbo-sequence`'s model-file type.
