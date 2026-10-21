# GLiNER2 relation scorer

Part of #714. `turbo_relation::Gliner2Relations` runs the relation
extraction of a GLiNER2 checkpoint on the native DeBERTa-v3 encoder
(docs/deberta.md), with no Python, ONNX or external inference engine.
It is a `RelationScorer` for `ZeroShotRelationAnnotator` and also
extracts relations on its own, as the reference does.

The checkpoint is `fastino/gliner2-base-v1` at
`f9634218e53580c56edf0de97ca1a7d3f1c2354e` (Apache-2.0). The zoo lists it
as `fastino-gliner2-base-v1` and fetches it by digest; it is the
default. GLiREL's checkpoints and code are CC BY-NC-SA 4.0
(docs/licences.md). The zoo lists `jackboyla/glirel-large-v0` as the
local-only `jackboyla-glirel-large-v0`, fetched only after the user runs
`turbo-zoo accept CC-BY-NC-SA-4.0`; nothing here runs it yet.

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

Single-threaded, release, one pinned core. The reference is GLiNER2 on
torch 2.14.1 CPU with one thread.

The heads by themselves, measured the same way in both: the reference's
`count_pred`, `count_embed` and the span head's `score_queries` timed
around each call during `extract_relations`; the native span
projections, count and query heads and span logits on a stored encoder
output. Best of 7 rounds of 5, repeated three times in alternating
order. Every row is from the timing machine (x86_64, Zen 5, AVX-512, a
128 MB L3 cache):

| per text | reference heads | native, AVX-512 | native, AVX2 |
|---|---|---|---|
| the three parity texts (37 to 50 subwords) | 10.4 to 10.5 ms | 3.6 to 3.8 ms (0.34 to 0.37x) | 6.0 to 6.7 ms (0.57 to 0.63x) |
| a 42-word text, 3 labels | 18.0 to 18.1 ms | 7.5 to 7.8 ms (0.42 to 0.44x) | 13.5 to 13.6 ms (0.75x) |

The AVX2 column is the native AVX2 kernel with AVX-512 switched off,
against torch still on AVX-512. A cache this large flatters code that
rereads weights. On a second x86_64 machine with AVX-512 and a smaller
cache, timing GLiNER2's head classes directly at the same shapes, the
same heads run at 0.88 to 0.98x the reference with AVX-512 and 1.03 to
1.1x with AVX2.

How the heads get there (`kernel.rs`, `heads.rs`): the layers run over
all words or all queries at once, on weights transposed at load into
panels of 32 outputs that a tile reads in order. A tile is 32 outputs by
8 rows with AVX-512, 16 by 6 with AVX2 and FMA, plain loops otherwise.
Rows are padded with zeros to whole tiles. The last span layer folds
into every query of a relation in one pass over its weights.

The encoder is the gap: 194 ms per parity text in the reference, 3,776 ms
on the native encoder's scalar CPU path (docs/deberta.md). The speed bar
of #714 waits on its packed GEMM work (#391).
