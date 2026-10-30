# Feedforward dependency parsing

`turbo-depparse` provides `FeedforwardDependencyModel`,
`FeedforwardDependencyParser`, `FeedforwardDependencyTrainer`, and
`FeedforwardContext`. Models use a shared embedding table, one hidden layer
with cube or ReLU activation, and a linear output layer with bias. Parameters
are f32; forward sums and gradients are f64. Greedy decoding selects the best
legal action. Beam decoding sums log probabilities, retaining stable outcome
order for ties. Both use the checked arc-standard transition system.

## Templates and compatibility

The assignment requests 18 word, 18 POS, and 12 dependency-label slots. The
pinned Java implementation actually uses 14 word, 14 POS, and 8 label slots.
Both layouts are explicit in `FeatureLayout`; training defaults to
`ChenManning18`. Select `OpenNlp14` and cube activation for Java parity.

Both templates start with the top three stack positions and first three
buffer positions. `OpenNlp14` adds the first minimum/maximum-index dependent
of s0 and s1, followed by each node's left-of-left and right-of-right dependent.
It retains the reference's extrema convention even if all children are on
one side. `ChenManning18` uses directional children: for s0 then s1, first
leftmost, first rightmost, second leftmost, second rightmost; then the same
four grandchild positions. A left child precedes its head and a right child
follows it. Missing positions and the artificial root have separate symbols.
Child extrema and second children are updated during attachment in O(1).

`FeedforwardContext` carries symbolic word/POS/label groups;
`model.feature_ids` maps them to the shared embedding rows. `extract_ids`
returns separate `word_ids`, `pos_ids`, and `label_ids` vectors whose `flatten`
method preserves the network input order. Words use the
reference's simple lowercase mapping plus contextual final sigma. Reserved
`*UNK*`, `*NULL*`, and `*ROOT*` symbols retain their spelling.

## Model files

`from_bytes` and `to_bytes` read and write the deterministic native
`TURBO-DEPPARSE-FF-V1` binary format (`open` and `save` operate on files).
Readers reject invalid counts, dimensions, duplicate symbols/IDs, non-finite
parameters, invalid transitions and trailing network bytes. Float arrays are
allocated after their bytes and aggregate count have been checked. Limits
remain 2 million vocabulary rows, 100,000 outcomes, embedding width 4,096,
hidden width 65,536 and 100 million float values.

The model exposes immutable weights and vocabulary. `precompute_words`
accepts word frequencies, orders by frequency with a deterministic ID tie
break, and caches hidden contributions within a caller-supplied value budget
(capped at 8 million f64 values). This cache leaves serialization and exact
scoring unchanged. Training returns a model with frequent-word contributions
precomputed after the last weight update, using a one-million-value budget
(8 MB of values). Callers can clear or replace that cache with
`precompute_words`. Loaded models do not need training-time assets.

## Training and annotation

AdaGrad training uses oracle events from projective trees (or pseudo-projectivized /
SWAP-derived trees), deterministic `BTreeMap` vocabulary ordering and seeded
random initialization/shuffling, minibatches, inverted hidden dropout, and L2 on
dense matrices. Biases and embeddings have no L2 penalty. Numerical divergence
returns an error. Settings are validated before parameter allocation.

`turbo_train_components::depparse::train_feedforward_conllu` accepts CoNLL-U
sentences and a UPOS/XPOS selection. It validates each sentence through the
existing lossless dependency carrier, including multiword-token ranges,
then trains over syntactic words. Malformed input includes its sentence
number in the error.

`DependencyAnnotator<A>::new` accepts either `DependencyParserME` or
`FeedforwardDependencyParser`. Neural provenance records a canonical model digest
and beam width in the configuration hash. Token and word anchors retain
separate types and POS dependencies.

See `analysis/depparse/src/feedforward/mod.rs` and
`analysis/train-components/src/depparse.rs`.
