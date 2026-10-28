# turbo-namefind

Named-entity recognition (`TokenNameFinder` trait, `Name`) combining deterministic dictionary (`DictionaryNameFinder`), structural pattern (`RegexNameFinder`), linear-chain CRF (`CrfNameFinder` in `analysis/namefind/src/crf.rs`), and consensus/calibrated span merging (`MergeEntities`, `ConsensusEntityMerger`, `analysis/namefind/src/consensus.rs`, `analysis/namefind/src/calibrated.rs`).

## Features

- **`TokenNameFinder` trait and `Document` annotators (`analysis/namefind/src/lib.rs`, `analysis/namefind/src/annotate.rs`)**: Defines the `TokenNameFinder` interface returning typed token spans (`Name`) with entity type labels and probability scores, with `DictionaryNameFinder` and `RegexNameFinder` implementing `Annotator` directly to emit the `Entities` layer on `Document`.
- **Deterministic dictionary, builtin, and pattern finders (`analysis/namefind/src/dictionary.rs`, `analysis/namefind/src/builtin.rs`, `analysis/namefind/src/regex_finder.rs`)**: Multi-token dictionary matching (`DictionaryNameFinder`) and direct bytecode/scanner pattern matching (`BuiltIn`, `JavaRegex`, `analysis/namefind/src/java_regex/mod.rs`) for closed-set entities and identifiers.
- **Multi-finder span merging and calibration (`analysis/namefind/src/merge.rs`, `analysis/namefind/src/consensus.rs`, `analysis/namefind/src/calibrated.rs`)**: Resolves overlapping entity predictions across finders via `MergeEntities` (`drop_overlapping_spans`), `ConsensusEntityMerger` (`EntityMergePolicy`), or calibrated scoring.

## Testing, Accuracy, and Speed

- **Test suite**: Covered by `analysis/namefind/tests/pipeline.rs`, `analysis/namefind/tests/consensus.rs`, `analysis/namefind/tests/calibrated.rs`, `analysis/namefind/tests/evidence.rs`, and `analysis/namefind/tests/resource_reuse.rs`.
- **Accuracy and speed**: See `docs/namefind-dl.md`, `docs/crf-namefinder.md`, `docs/namefinder-evaluation.md`, and `docs/namefinder-crf-evaluation.md` for CoNLL/OntoNotes F1 benchmarks and CPU throughput figures.
