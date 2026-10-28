# turbo-stem

Deterministic algorithmic stemmers (`Stemmer` trait) for English and multilingual text analysis: Porter (`PorterStemmer`), Snowball across 29 languages (`SnowballStemmer`), and light inflectional stemmers (`analysis/stem/src/light/mod.rs`).

## Features

- **Unified `Stemmer` trait (`analysis/stem/src/lib.rs`, `analysis/stem/src/stemmer.rs`, `analysis/stem/src/term.rs`)**: Borrowed-or-owned (`Cow<str>`) token stemming with zero allocation when a token equals its stem, plus `StemmerAnnotator` (`analysis/stem/src/annotate.rs`) for `Document` pipelines (`Stems` layer).
- **Porter and 29-language Snowball stemmers (`analysis/stem/src/porter.rs`, `analysis/stem/src/snowball/mod.rs`)**: Algorithmic suffix stripping for Arabic, Armenian, Basque, Catalan, Danish, Dutch, English, Esperanto, Estonian, Finnish, French, German, Greek, Hindi, Hungarian, Indonesian, Irish, Italian, Lithuanian, Nepali, Norwegian, Portuguese, Romanian, Russian, Serbian, Spanish, Swedish, Tamil, Turkish, and Yiddish.
- **Light morphological stemmers (`analysis/stem/src/light/mod.rs`)**: High-precision inflection strippers (`analysis/stem/src/light/english_minimal.rs`, `analysis/stem/src/light/french_light.rs`, `analysis/stem/src/light/german_light.rs`, `analysis/stem/src/light/italian_light.rs`, `analysis/stem/src/light/portuguese_light.rs`, `analysis/stem/src/light/russian_light.rs`, `analysis/stem/src/light/spanish_light.rs`) tuned for information-retrieval recall without over-stemming.

## Testing, Accuracy, and Speed

- **Test suite**: Verified against the official Snowball and Porter vocabulary fixtures in `analysis/stem/tests/snowball_vocabulary.rs`, `analysis/stem/tests/golden.rs`, and `analysis/stem/tests/pipeline.rs`.
- **Accuracy and speed**: Zero-regex slice state machines operating in a single forward/backward pass at multi-million words per second on CPU.
