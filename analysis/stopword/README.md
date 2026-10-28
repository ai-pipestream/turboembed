# turbo-stopword

Multilingual stopword filtering (`StopwordFilter`, `StopwordList`, `StopwordAnnotator`) with bundled stopword tables (`analysis/stopword/src/list.rs`), case-insensitive ASCII/Unicode folding, and custom domain list composition.

## Features

- **Bundled multilingual stopword tables (`analysis/stopword/src/lib.rs`, `analysis/stopword/src/list.rs`, `analysis/stopword/data/`)**: Curated stopword sets indexed by language with $O(1)$ hash lookup.
- **Case-sensitive and case-insensitive matching**: Supports allocation-free case-folded membership checks and custom additions/exclusions.
- **Document pipeline integration (`analysis/stopword/src/annotate.rs`)**: Implements `Annotator` (`StopwordAnnotator`) to mark or filter stopword tokens on `Document`.

## Testing, Accuracy, and Speed

- **Test suite**: Covered by `analysis/stopword/tests/golden.rs` and `analysis/stopword/tests/pipeline.rs` verifying language coverage, case folding, and pipeline annotation.
- **Accuracy and speed**: Allocation-free hash-set lookup per token.
