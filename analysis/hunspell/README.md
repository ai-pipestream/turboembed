# turbo-hunspell

Pure-Rust reader and morphological stemmer for Hunspell `.aff` and `.dic` dictionaries (`Dictionary`, `Stemmer`, `HunspellAnnotator`), supporting prefix/suffix stripping, circumfix and NeedAffix flags, multi-word compounding (`COMPOUNDFLAG`, `COMPOUNDBEGIN`/`MIDDLE`/`END`, `COMPOUNDRULE`, `CHECKCOMPOUNDPATTERN`), and `REP`/`ICONV`/`OCONV` conversions.

## Features

- **Affix and dictionary parser (`Dictionary`, `analysis/hunspell/src/read/mod.rs`, `analysis/hunspell/src/read/affix_file.rs`, `analysis/hunspell/src/read/word_list.rs`)**: Parses UTF-8 and single-byte 8-bit (`ISO8859-*`, `KOI8-*`, `cp125*`) `.aff` and `.dic` files (`analysis/hunspell/src/charset/mod.rs`), supporting single-char, two-char (`FLAG long`), numeric (`FLAG num`), and UTF-8 (`FLAG UTF-8`) flag encodings.
- **Morphological analysis and stemming (`Dictionary`, `Stemmer`, `analysis/hunspell/src/dictionary/mod.rs`, `analysis/hunspell/src/stemmer/mod.rs`)**: Evaluates root words, single and twofold prefix/suffix chains (`analysis/hunspell/src/stemmer/affixes.rs`), circumfixes, and recursive compound rules (`analysis/hunspell/src/stemmer/compound.rs`) to validate word forms and return base stems or morphological tags (`analysis/hunspell/src/morphology.rs`).
- **Document pipeline integration (`HunspellAnnotator`, `analysis/hunspell/src/annotate.rs`)**: Implements `Annotator` reading `Tokens` and writing `Stems` (`turbo:stems`) on `Document`.

## Testing, Accuracy, and Speed

- **Test suite**: Covered by `analysis/hunspell/tests/snapshots.rs`, `analysis/hunspell/tests/properties.rs`, `analysis/hunspell/tests/hostile.rs`, and `analysis/hunspell/tests/pipeline.rs`.
- **Accuracy and speed**: Indexed root-table and affix-suffix tries (`analysis/hunspell/src/dictionary/index.rs`) provide sub-microsecond single-word validation without external C++ dependencies.
