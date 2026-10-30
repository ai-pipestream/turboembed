# turbo-postag

Part-of-speech tagging with the concrete `PosModel` and `PosTagger` `Document` annotator (`analysis/postag/src/lib.rs`) alongside compact neural sequence taggers (`neural::FeedforwardPosTagger`, `neural::BilstmPosTagger`) with morphological affix, character-shape, and tag-dictionary constraints.

## Features

- **`PosModel` and `PosTagger` `Document` annotator (`analysis/postag/src/lib.rs`, `analysis/postag/src/model.rs`, `analysis/postag/src/tagger.rs`)**: Concrete `PosTagger` annotator that reads the `Tokens` layer and writes the `PosTags` layer on `Document`, predicting Universal Dependencies (`UPOS`) or Penn Treebank (`XPOS`) tags and per-token probabilities over sentence token slices with `PosTagFormat` mapping and `TagDictionary` validation.
- **Neural feedforward and BiLSTM taggers (`neural::FeedforwardPosTagger`, `neural::BilstmPosTagger`, `analysis/postag/src/neural/mod.rs`, `analysis/postag/src/neural/feedforward.rs`, `analysis/postag/src/neural/bilstm_tagger.rs`)**: CPU inference and training (`neural::FeedforwardPosTrainer`, `neural::BilstmPosTrainer`) for hashed token, prefix, suffix, and character-shape embeddings with CRF and transition decoding (`analysis/postag/src/neural/crf.rs`, `analysis/postag/src/neural/lstm.rs`).

## Testing, Accuracy, and Speed

- **Test suite**: Covered by `analysis/postag/tests/neural.rs`, `analysis/postag/tests/bilstm_ewt.rs`, `analysis/postag/tests/bilstm_batched.rs`, `analysis/postag/tests/packed_lstm.rs`, and `analysis/postag/tests/accuracy.rs`.
- **Accuracy and speed**: See `docs/neural-annotators.md` for Universal Dependencies UPOS accuracy and single-core CPU tokens/s benchmarks.
