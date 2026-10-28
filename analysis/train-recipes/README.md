# turbo-train-recipes

Reproducible training and evaluation recipes for Universal Dependencies (`UD`), CoNLL/OntoNotes named-entity recognition, lemmatization dictionaries, and language/script models.

## Features

- **Universal Dependencies recipes (`analysis/train-recipes/src/ud.rs`, `analysis/train-recipes/src/postag.rs`, `analysis/train-recipes/src/depparse.rs`)**: End-to-end data loaders and evaluation harnesses over CoNLL-U treebanks for POS tagging, lemmatization, and dependency parsing.
- **Named-entity and sequence recipes (`analysis/train-recipes/src/namefind.rs`, `analysis/train-recipes/src/namefind_eval/mod.rs`)**: Training and F1 evaluation harnesses for linear-chain CRF and sequence token classifiers.
- **Corpus reference and release cards (`analysis/train-recipes/src/reference.rs`, `analysis/train-recipes/src/release.rs`, `analysis/train-recipes/src/card.rs`)**: Deterministic SHA-256 dataset and artifact verification before training runs.

## Testing, Accuracy, and Speed

- **Test suite**: Covered by `analysis/train-recipes/tests/ud.rs`, `analysis/train-recipes/tests/namefind_eval.rs`, `analysis/train-recipes/tests/lemma_release.rs`, `analysis/train-recipes/tests/langdetect.rs`, and `analysis/train-recipes/tests/fineweb.rs`.
- **Accuracy and speed**: Produces the evaluation tables recorded in `docs/neural-annotators.md`, `docs/crf-namefinder.md`, `docs/dependency-parser.md`, and `docs/feedforward-dependency-parser.md`.
