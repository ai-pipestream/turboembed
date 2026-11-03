# turbo-stopword

Multilingual stopword flagging (`StopwordList`, `StopwordLanguage`, `StopwordAnnotator`, `Stopwords`) with bundled stopword tables (`analysis/stopword/src/list.rs`) and allocation-free case-insensitive lookup across eleven languages.

## Features

- **Bundled multilingual stopword tables (`StopwordList`, `StopwordLanguage`, `analysis/stopword/src/lib.rs`, `analysis/stopword/src/list.rs`, `analysis/stopword/data/`)**: Curated stopword sets for eleven languages compiled into sorted slices at build time.
- **Allocation-free case-insensitive lookup (`StopwordList::contains`)**: Lowercases each token into a stack buffer and binary-searches the sorted stopword slice with no heap allocation and no hashing.
- **Document pipeline integration (`StopwordAnnotator`, `Stopwords`, `analysis/stopword/src/annotate.rs`)**: Implements `Annotator` (`StopwordAnnotator`) reading `Tokens` (and optional `Language`) to write the `Stopwords` (`turbo:stopwords`) per-token flag layer on `Document` without removing tokens.

## Testing, Accuracy, and Speed

- **Test suite**: Covered by `analysis/stopword/src/lib.rs`, `analysis/stopword/tests/lists.rs`, `analysis/stopword/tests/codes.rs`, `analysis/stopword/tests/properties.rs`, and `analysis/stopword/tests/pipeline.rs` verifying language coverage, case folding, and pipeline annotation.
- **Accuracy and speed**: Stack-buffer lowercasing and binary search over sorted static slices per token.
