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

`from_bytes` accepts legacy Java raw streams, current Apache models and Turbo
ZIP versions 1, 2 and 3. Readers reject invalid counts, dimensions, duplicate
symbols/IDs, non-finite parameters, invalid transitions and trailing network
bytes. Float arrays are allocated after their bytes and aggregate count have
been checked. Limits remain 2 million vocabulary rows, 100,000 outcomes,
embedding width 4,096, hidden width 65,536 and 100 million float values.

- Legacy fork: `from_java_bytes` / `to_java_bytes` read and write
  `ONLP-FFDP-1`, with Java modified-UTF strings and 14/14/8 cube features.
  Export refuses expanded/ReLU models and default beams other than 1.
  This export does not write the current Apache format.
- Current Apache: `from_apache_bytes` reads `FFDP` version 1, directly or
  within the `FeedforwardDependencyParser` BaseModel ZIP artifact
  `feedforward.network`. ZIP import preserves the manifest's default
  `BeamSize`; a bare network has no beam metadata and defaults to 1.
  Unsupported versions, mixed schemas, malformed UTF-8 and invalid beam
  settings fail explicitly. Native allocation limits still apply.
- Native: `to_bytes` writes Turbo ZIP version 3. It records layout,
  activation and default beam, and uses length-prefixed UTF-8 vocabulary
  strings, including keys beyond legacy `writeUTF`'s 65,535-byte limit.
  The version 1 reader remains supported with its original default beam 1.
  Both versions require unique, recognized entries and valid configuration;
  uncompressed entry sizes are capped before decompression.

These native-format changes preserve parameters and decoding metadata; they
are not a measured load-speed or inference-speed improvement. Conversion
prints input/output SHA256 values so callers can retain the import receipt.
The model's canonical identity describes the converted native artifact.

### Native command line

Build with `cargo build -p turbo-depparse --bin turbo-depparse`, or use:

```sh
cargo run -p turbo-depparse --bin turbo-depparse -- inspect feedforward apache.bin
cargo run -p turbo-depparse --bin turbo-depparse -- convert feedforward apache.bin model.turbo
cargo run -p turbo-depparse --bin turbo-depparse -- parse feedforward model.turbo input.conllu output.conllu
```

Use `linear` instead of `feedforward` for OpenNLP linear ZIP or raw maxent
models. Conversion packages raw classifiers and preserves existing linear
packages. Parsing uses existing syntactic word forms and UPOS; it performs
no tokenization, POS prediction or access to gold heads/labels. Output changes
only basic HEAD/DEPREL, preserving comments, multiword ranges, empty nodes and
other columns. Enhanced DEPS are retained unchanged, not recomputed. Word IDs
must be contiguous from 1 and every word needs UPOS. Neural parsing uses the
recorded default beam. Existing output files are never overwritten.

No Python installation is required. Native training is also available:

```sh
cargo run --release -p turbo-depparse --bin turbo-depparse -- train feedforward en train.conllu neural.properties model.turbo --projective-only
cargo run --release -p turbo-depparse --bin turbo-depparse -- train linear en train.conllu linear.properties linear.bin --projective-only
```

These commands require an explicit parameter file and `--projective-only`.
They report input sentences/words and excluded non-projective sentences/words
before training, then elapsed training seconds and corpus/parameter/model
SHA256 values. Retain this receipt with the source corpus and parameter file;
the neural container does not store training language or corpus provenance.
Training consumes the supplied file only; keep train, development and test
roles separate. An empty projective population is an error. This command does
not solve non-projective training or constitute a measured accuracy release.

The linear CLI currently supports `Algorithm=MAXENT` and requires explicit
positive `Iterations` and `Threads`, plus nonnegative `Cutoff`. Other algorithms
remain available through Rust APIs. Example small compatibility pilot:

```properties
Algorithm=MAXENT
Iterations=5
Cutoff=1
Threads=1
```

Neural parameters require positive `Epochs`. Optional keys are `EmbeddingSize`,
`HiddenSize`, `BatchSize`, `LearningRate`, `L2`, `Dropout`, `WordCutoff`, `Seed`,
`Layout` (`ChenManning18` or `OpenNlp14`) and `Activation` (`cube` or `relu`).
Omitted values use `FeedforwardSettings::default()`; the complete resolved
settings are printed. Unknown keys fail. A one-epoch pilot with the default
50-dimensional embeddings and 200 hidden units uses `Epochs=1`. Tiny fixtures
can explicitly set `EmbeddingSize=3` and `HiddenSize=5`; results from that
configuration do not predict full-width training cost or accuracy.

The model exposes immutable weights and vocabulary. `precompute_words`
accepts word frequencies, orders by frequency with a deterministic ID tie
break, and caches hidden contributions within a caller-supplied value budget
(capped at 8 million f64 values). This cache leaves serialization and exact
scoring unchanged. Training returns a model with frequent-word contributions
precomputed after the last weight update, using a one-million-value budget
(8 MB of values). Callers can clear or replace that cache with
`precompute_words`. Loaded models do not need training-time assets.

## Training and annotation

AdaGrad training uses oracle events from projective trees, Java-compatible
vocabulary iteration and seeded random initialization/shuffling, minibatches,
inverted hidden dropout, and L2 on dense matrices. Biases and embeddings have
no L2 penalty. Non-projective samples are skipped; an entirely skipped or
empty corpus is an error. Numerical divergence returns an error. Settings
are validated before parameter allocation. Expanded ReLU and cube models
are covered by training/decoding tests; Java training parity covers the
reference's 14/14/8 cube layout, with and without dropout/L2.

`turbo_train_components::depparse::train_feedforward_conllu` accepts CoNLL-U
sentences and a UPOS/XPOS selection. It validates each sentence through the
existing lossless dependency carrier, including multiword-token ranges,
then trains over syntactic words. Malformed input includes its sentence
number in the error.

`DependencyAnnotator<A>::new` accepts either `DependencyParserME` or
`FeedforwardDependencyParser`. Existing linear parser behavior and provenance
version stay unchanged. Neural provenance records a canonical model digest
and beam width in the configuration hash. Token and word anchors retain
separate types and POS dependencies.

See `analysis/depparse/src/feedforward/mod.rs`,
`analysis/train-components/src/depparse.rs`, and
`analysis/depparse/tests/feedforward/data/README.md`.
