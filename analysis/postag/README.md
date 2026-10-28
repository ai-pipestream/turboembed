# turbo-postag

Part-of-speech tagging (`PosTagger` trait, `PosTaggerAnnotator`) supporting compact neural sequence taggers (`FeedForwardPosTagger`, `BiLstmPosTagger`) with morphological affix, character-shape, and tag-dictionary constraints.

## Features

- **Common `PosTagger` trait and `Document` annotator (`analysis/postag/src/lib.rs`)**: Predicts Universal Dependencies (`UPOS`) or treebank-specific (`XPOS`) tags and per-token confidence scores over sentence token slices, emitting the `PosTags` layer on `Document`.
- **Neural feedforward and BiLSTM taggers (`analysis/postag/src/neural/mod.rs`, `analysis/postag/src/neural/feedforward.rs`, `analysis/postag/src/neural/bilstm_tagger.rs`)**: Batched CPU/device inference and training for hashed token/prefix/suffix/shape embeddings with transition decoding (`analysis/postag/src/neural/lstm.rs`).

## Testing, Accuracy, and Speed

- **Test suite**: Covered by `analysis/postag/tests/neural.rs`, `analysis/postag/tests/bilstm_ewt.rs`, `analysis/postag/tests/bilstm_batched.rs`, `analysis/postag/tests/packed_lstm.rs`, and `analysis/postag/tests/accuracy.rs`.
- **Accuracy and speed**: See `docs/neural-annotators.md` for Universal Dependencies UPOS accuracy and single-core CPU tokens/s benchmarks.
