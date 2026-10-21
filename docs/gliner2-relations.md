# GLiNER2 relation scorer

Part of #714. `turbo_relation::Gliner2Relations` runs the relation
extraction of a GLiNER2 checkpoint on the native DeBERTa-v3 encoder
(docs/deberta.md), with no Python, ONNX or external inference engine.
It is a `RelationScorer` for `ZeroShotRelationAnnotator` and also
extracts relations on its own, as the reference does.

The checkpoint is `fastino/gliner2-base-v1` at
`f9634218e53580c56edf0de97ca1a7d3f1c2354e` (Apache-2.0). The zoo lists it
as `fastino-gliner2-base-v1` and fetches it by digest; GLiREL's
checkpoints and code are CC BY-NC-SA 4.0 (docs/licences.md) and are not
used here.

## What runs

The code is in `analysis/relation/src/gliner2/`, written from GLiNER2's
inference path (`fastino-ai/GLiNER2` at
`ca4f68235b50fe4c3ecf3876851d748fe4b15c6c`):

- `words`: GLiNER2's word splitter (URLs, e-mail addresses, handles,
  hyphen- or underscore-joined words, single characters), lower-cased,
  with byte offsets, written without a regular expression engine. The
  text gets a final `.` when it lacks sentence punctuation.
- `prompt`: a schema per relation label, `( [P] <label> ( [R] head [R]
  tail ) )`, joined by `[SEP_STRUCT]`, then `[SEP_TEXT]` and the words,
  each element tokenized on its own with the checkpoint's SentencePiece
  model; no `[CLS]` or `[SEP]`.
- `heads`: the count head on each label's `[P]` vector; the
  `count_lstm_v2` query head (a GRU over position rows from each `[R]`
  vector, two post-norm transformer layers of width 128 across the two
  fields, an MLP); the `SpanMarkerV0` span head over each word's first
  subword, factored as the reference's `score_queries` factors it.
- Decoding: for each instance, the head and the tail are each the span
  of highest score (the logistic of the logit) at or above the
  threshold, ties to the earliest start and then the shortest; an
  instance with both is a relation.

As a scorer, a pair of mentions scores, under a label, the best over
the label's instances of the lower of its head span's and its tail
span's scores. A mention that is not a run of whole words no longer than
the checkpoint's `max_width` (8) scores 0. The input is limited to the
encoder's `max_sequence` (512 subwords) and fails with that limit named.

## Loading

`Gliner2Relations::load(encoder_package, checkpoint, id, revision)`
takes the encoder as `turbo-bundle deberta-convert` packages it
(`encoder.` prefix) and the checkpoint's directory for the heads,
`spm.model`, `added_tokens.json` and `config.json`; the model's SHA-256
is computed from `model.safetensors` and recorded as the relations'
provenance. Only `counting_layer: count_lstm_v2` and `token_pooling:
first` are read; any other configuration is refused by name.

## Parity

`analysis/relation/reference/gliner2/build_fixture.py` (pinned in its
`requirements.txt`) writes both fixtures; only fixture production uses
Python.

- `analysis/relation/tests/gliner2_reduced.rs`: the encoder and heads sliced to width 32,
  as the encoder fixture slices it, on fixed token sequences, against
  GLiNER2's own layer classes: count logits, counts, queries and span
  logits within 2e-5. It runs in every test run.
- `analysis/relation/tests/gliner2_real.rs`: the full checkpoint, fetched through the zoo
  and packaged on first use, against GLiNER2's `extract_relations` on
  three texts: identical input ids and instance counts, the same
  relations with the same byte offsets, confidences within 1e-4; then the
  scorer behind the annotator, with the model in the evidence. It is
  ignored by default for its 834 MB download:
  `cargo test --release -p turbo-relation --test gliner2_real -- --ignored`.

## Speed

Single-threaded on the x86_64 timing machine, release, on the three
parity texts (37 to 50 subwords), best of 7 rounds of 5. The reference
is GLiNER2 on torch 2.14.1 CPU with one thread, its encoder timed by
hooks around the encoder module.

| | reference | native |
|---|---|---|
| encoder | 194 ms | 3,776 ms |
| everything else (tokenization, heads, decoding) | 12.5 ms | 13.3 ms |

The heads are at the reference's cost; the encoder's scalar CPU path is
the gap (docs/deberta.md), and the speed bar of #714 waits on its packed
GEMM work.
