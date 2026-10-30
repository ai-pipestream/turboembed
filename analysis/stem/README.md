# turbo-stem

Deterministic algorithmic stemmers (`Stemmer` trait) for English and multilingual text analysis: Porter (`PorterStemmer`), Snowball across 21 algorithms in 20 languages (`SnowballStemmer`, `SnowballAlgorithm`), and light inflectional stemmers (`analysis/stem/src/light/mod.rs`).

## Features

- **Unified `Stemmer` trait (`analysis/stem/src/lib.rs`, `analysis/stem/src/stemmer.rs`, `analysis/stem/src/term.rs`)**: UTF-16 buffer stemming via `Stemmer::stem_utf16` (writing into a caller-reusable `Vec<u16>` to avoid per-word allocations) and `&str` convenience stemming via `Stemmer::stem` (returning an owned `String`), plus `StemAnnotator` (`analysis/stem/src/annotate.rs`) and `TermAnnotator` (`analysis/stem/src/profile.rs`) for `Document` pipelines (`Stems` and `Terms` layers).
- **Porter and 21-algorithm Snowball stemmers (`analysis/stem/src/porter.rs`, `analysis/stem/src/snowball/mod.rs`)**: `PorterStemmer` and `SnowballStemmer` (`SnowballAlgorithm`) covering Arabic, Catalan, Danish, Dutch, English (`English` and `Porter`), Finnish, French, German, Greek, Hungarian, Indonesian, Irish, Italian, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, and Turkish.
- **Light morphological stemmers (`analysis/stem/src/light/mod.rs`)**: High-precision inflection strippers (`analysis/stem/src/light/english_minimal.rs`, `analysis/stem/src/light/french_light.rs`, `analysis/stem/src/light/german_light.rs`, `analysis/stem/src/light/italian_light.rs`, `analysis/stem/src/light/portuguese_light.rs`, `analysis/stem/src/light/russian_light.rs`, `analysis/stem/src/light/spanish_light.rs`) tuned for information-retrieval recall without over-stemming.

## Testing, Accuracy, and Speed

- **Test suite**: Verified against the official Snowball and Porter vocabulary fixtures in `analysis/stem/tests/snowball_vocabulary.rs`, `analysis/stem/tests/snapshots.rs`, `analysis/stem/tests/pipeline.rs`, and `analysis/stem/tests/properties.rs`.
- **Accuracy and speed**: Zero-regex slice state machines operating in a single forward/backward pass at multi-million words per second on CPU.
