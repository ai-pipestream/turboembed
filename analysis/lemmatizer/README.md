# turbo-lemmatizer

Dictionary and statistical edit-script lemmatization (`DictionaryLemmatizer`, `MaxentLemmatizer`, `LemmatizerModel`) mapping `(token, pos_tag)` pairs to canonical base forms.

## Features

- **Dictionary lookup (`analysis/lemmatizer/src/dictionary.rs`)**: Supports tab-separated `(word, pos, lemma)` dictionaries (`DictionaryLemmatizer`) with case-aware lookup and zero-allocation borrowed stem return paths.
- **Statistical edit-script lemmatizer (`MaxentLemmatizer`, `LemmatizerModel`, `analysis/lemmatizer/src/maxent.rs`, `analysis/lemmatizer/src/edit_script.rs`, `analysis/lemmatizer/src/encoding.rs`)**: Encodes word-to-lemma transformations as shortest prefix/suffix edit scripts (`LemmaEncoding`) for out-of-vocabulary morphological generalization.
- **Document pipeline integration (`analysis/lemmatizer/src/lib.rs`)**: `DictionaryLemmatizer` and `MaxentLemmatizer` implement `Annotator`, reading `Tokens` and `PosTags` and writing the `Lemmas` layer onto `Document`.

## Testing, Accuracy, and Speed

- **Test suite**: Covered by `analysis/lemmatizer/tests/dictionary_allocations.rs` and `analysis/lemmatizer/src/edit_script/tests.rs`, plus case-restoration evaluation in `docs/lemma-case-evaluation.md`.
- **Accuracy and speed**: Dictionary lookup runs with zero per-token heap allocations (`analysis/lemmatizer/tests/dictionary_allocations.rs`); see `docs/lemma-case-evaluation.md` for Universal Dependencies accuracy numbers.
