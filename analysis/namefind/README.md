# turbo-namefind

Named-entity recognition (`NameFinder` trait, `NameFinderAnnotator`) combining deterministic dictionary/gazetteer (`DictionaryNameFinder`), structural pattern (`RegexNameFinder`), linear-chain CRF (`CrfNameFinder` in `analysis/namefind/src/crf.rs`), and consensus/calibrated span merging (`analysis/namefind/src/consensus.rs`, `analysis/namefind/src/calibrated.rs`).

## Features

- **Trait and pipeline annotator (`analysis/namefind/src/lib.rs`, `analysis/namefind/src/annotate.rs`)**: Defines the common `NameFinder` interface producing typed token spans (`Span`) with entity type labels and confidence scores, and wires multi-finder ensembles into `Document` (`Entities` layer).
- **Deterministic dictionary, builtin, and pattern finders (`analysis/namefind/src/dictionary.rs`, `analysis/namefind/src/builtin.rs`, `analysis/namefind/src/regex_finder.rs`)**: Fast multi-token trie matching (`DictionaryNameFinder`) and Pike-VM structural pattern matching (`analysis/namefind/src/java_regex/mod.rs`) for closed-set entities and identifiers.
- **Multi-finder span merging and calibration (`analysis/namefind/src/merge.rs`, `analysis/namefind/src/consensus.rs`, `analysis/namefind/src/calibrated.rs`)**: Resolves overlapping entity predictions across heterogeneous finders by priority, confidence, or consensus policy.

## Testing, Accuracy, and Speed

- **Test suite**: Covered by `analysis/namefind/tests/pipeline.rs`, `analysis/namefind/tests/consensus.rs`, `analysis/namefind/tests/calibrated.rs`, `analysis/namefind/tests/evidence.rs`, and `analysis/namefind/tests/resource_reuse.rs`.
- **Accuracy and speed**: See `docs/namefind-dl.md`, `docs/crf-namefinder.md`, `docs/namefinder-evaluation.md`, and `docs/namefinder-crf-evaluation.md` for CoNLL/OntoNotes F1 benchmarks and CPU throughput figures.
