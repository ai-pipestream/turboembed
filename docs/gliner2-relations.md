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
`turbo-zoo accept CC-BY-NC-SA-4.0`, and docs/glirel-relations.md runs it.

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

## On a device

The encoder can also be a `FAMILY_DEBERTA_V2` core bundle on the device
`EncoderOptions` names (cpu, cuda, levelzero), through the backend table
(#920): open it with `turbo_relation::encoder::DebertaEncoder::open` and
pass it to `Gliner2Relations::load_with`. Only the rows the heads read come back from the
device, as one-token ranges of the range write: each word's first
subword and the label markers. The heads stay on the host. `execution()`
names where the encoder ran. `analysis/relation/tests/device.rs` holds a device to the CPU encoder
over `testdata/tiny-deberta-bundle`, with every count equal and every
logit within 1e-4.

The real checkpoint's encoder is sealed as that core bundle by
`bundle/recipes/gliner2-base-v1-encoder.json`. The shared
`bundle/reference/gliner_encoder.py`, in the pinned reference container,
writes its upstream directory from the zoo's `fastino-gliner2-base-v1`
files: the `encoder.` tensors as a `DebertaV2Model` of the checkpoint's
`encoder_config/config.json`, with 128011 rows, and `spm.model` converted
to `tokenizer.json`, with the eleven tokens of `added_tokens.json` as
pieces at their ids. With `Z` the zoo home's `artifacts` directory and
`W` an empty one:

```
docker run --rm --network none --user $(id -u):$(id -g) -e HOME=/tmp \
    -v $PWD/bundle/reference:/scripts:ro -v $Z/fastino-gliner2-base-v1:/ckpt:ro -v $W:/w --entrypoint python \
    turbo-reference@sha256:3fe2d4ad441927475b97a400364ebcb7fca3f324dae1a0463720eff7fc0a53f4 \
    /scripts/gliner_encoder.py /ckpt/encoder_config/config.json /ckpt/spm.model /ckpt/model.safetensors /w/out \
    encoder. /ckpt/added_tokens.json
cargo run --release -p turbo-bundle -- reference bundle/recipes/gliner2-base-v1-encoder.json $W/out $W/bundle
```

`DebertaEncoder::open_bound` takes the bundle with the DeBERTa package
that `turbo-bundle deberta-convert` writes from the same checkpoint. The
bundle's weights must be the package's, tensor for tensor.
`analysis/relation/tests/device_real.rs` and
`analysis/schema-extract/tests/device_real.rs` (ignored; `GLINER2_BUNDLE`,
`GLINER2_PACKAGE`, `GLINER2_CHECKPOINT`, `TURBO_TEST_DEVICE`) hold the
device to the package's CPU encoder on the texts of each `real.json`
fixture and time the passes on both, interleaved.

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

The timings above and below are historical measurements of the earlier
head implementation. They have not been rerun for the buffer-reuse change.
Current projections (`analysis/relation/src/projection.rs`) use shared
`turbo::cpu::Matrix` panels and handle partial rows without padded input
copies. The last span layer folds into every query of a relation in one
pass over its weights.

End to end, on the same machine and core, with the same method:
`extract_relations` against `Gliner2Relations::extract` on the three
parity texts, the reference's encoder timed with hooks around its module
and ours on its own:

| per text | reference | native | ratio |
|---|---|---|---|
| end to end | 196.7 to 196.9 ms | 41.0 to 41.7 ms | 0.21x |
| encoder | 185.2 to 185.5 ms | 36.8 to 36.9 ms | 0.20x |
| everything else (tokenization, heads, decoding) | 11.4 to 11.8 ms | 4.1 to 4.8 ms | 0.36 to 0.41x, each round with its own reference |

The native encoder, which computes only the visible positions since #391
(docs/deberta.md), is about nine tenths of the native time.

## Reusing head buffers

`heads::Heads` exposes `spans_into`, `queries_of_into` and
`logits_all_into` with caller-owned `turbo_relation::Scratch` and output
storage. Retain them between calls to reuse projection capacity. A span
output can start as `heads::Spans::default()`; fill it with `spans_into`
before scoring it. The allocating methods remain available.

The ordinary relation pass reuses scratch and query storage across labels.
Its projections use `turbo::cpu::Matrix`, including partial tiles, without
padded input copies. The `reused_head_buffers` test checks warmed span and
logit execution, including empty and shorter inputs. Recurrent and attention
temporaries in query construction remain local allocations, and returned
per-label scores still own their storage. This change does not claim an
allocation-free end-to-end encoder request.
