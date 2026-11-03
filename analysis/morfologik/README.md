# turbo-morfologik

Zero-copy reader and traversal engine for Morfologik `FSA5` and `CFSA2` finite-state automata (`.dict`) and companion metadata (`.info`), providing morphological dictionary lookup (`MorfologikDictionary`, `MorfologikLemmatizer`, `MorfologikTagDictionary`, `MorfologikDictionaryBuilder`).

## Features

- **FSA5 and CFSA2 binary decoders (`FsaReader`, `Fsa5Reader`, `Cfsa2Reader`, `analysis/morfologik/src/fsa/mod.rs`, `analysis/morfologik/src/fsa/cfsa2.rs`, `analysis/morfologik/src/reader.rs`)**: Parses and traverses standard Morfologik finite-state automata directly from borrowed byte slices (`&[u8]`) without heap-allocating graph nodes.
- **Separator and prefix/infix/suffix decoding (`BaseFormEncoding`, `MorfologikMetadata`, `DictionaryCharset`, `analysis/morfologik/src/encoding.rs`, `analysis/morfologik/src/metadata.rs`, `analysis/morfologik/src/charset.rs`)**: Supports standard suffix, prefix, and infix run-length encoders (`fsa.dict.encoder`) and custom charset conversions.
- **Morphological lookup and lemmatization (`MorfologikDictionary`, `MorfologikLemmatizer`, `MorfologikTagDictionary`, `analysis/morfologik/src/dictionary.rs`, `analysis/morfologik/src/lemmatizer.rs`)**: Extracts `(stem, tag)` `WordData` entries for inflected surface forms and integrates with `Document` pipelines.

## Testing, Accuracy, and Speed

- **Test suite**: Verified by `analysis/morfologik/tests/morfologik.rs`, `analysis/morfologik/tests/lemmatizer.rs`, `analysis/morfologik/tests/allocations.rs`, `analysis/morfologik/tests/polimorf_whitespace.rs`, and `analysis/morfologik/tests/validation.rs`.
- **Accuracy and speed**: Zero-copy byte-slice arc traversal runs in $O(|w|)$ time per word with zero heap allocations during state transitions (`analysis/morfologik/tests/allocations.rs`).
