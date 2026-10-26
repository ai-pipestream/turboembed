# GLiREL relation scorer

Part of #714. `turbo_relation::GlirelRelations` runs GLiREL's zero-shot
relation classification on the native DeBERTa-v3 encoder
(docs/deberta.md), with no Python, ONNX or external inference engine. It
is a `RelationScorer` for `ZeroShotRelationAnnotator`, and it also
predicts relations on its own, as the reference does. GLiNER2
(docs/gliner2-relations.md) is the default relation scorer; GLiREL is an
opt-in.

The checkpoint is `jackboyla/glirel-large-v0` at
`40a523e12a8432d6da364cf2a195a28755ff04d3`, CC BY-NC-SA 4.0
(docs/licences.md). Nothing in this repository ships, bundles or
redistributes it or anything converted from it. The zoo lists it as the
local-only `jackboyla-glirel-large-v0` and fetches it from its
publisher's address only after the user has run `turbo-zoo accept
CC-BY-NC-SA-4.0`. Its tokenizer is `microsoft/deberta-v3-large`'s
`spm.model` (MIT), listed as `microsoft-deberta-v3-large-tokenizer`
with the base model's `config.json`.

## From the checkpoint to a package

The checkpoint is a `torch.save` file (`pytorch_model.bin`).
`turbo-bundle glirel-convert <pytorch_model.bin> <config.json>
<spm.model> <repo> <revision> <license> <new-dir>` reads it as data and
writes a package:

- `encoder/`: the DeBERTa-v3-large encoder as a native package. Its
  configuration is the base model's `config.json` with the vocabulary
  the checkpoint resized it to (128,005 rows).
- `heads.safetensors`: every other tensor, under its checkpoint name.
- `glirel.json`: the repository, revision and licence; the SHA-256 of
  `pytorch_model.bin`, recorded as the relations' provenance; the
  digests of the heads and of the `spm.model` the package was written
  with; and the ids of `[CLS]`, `[SEP]` and `[REL]`.

The reader (`bundle/src/torch/`) takes only `torch.save`'s zip form, with
stored entries and each entry's CRC-32 checked. It reads `data.pkl` with
a stack machine that runs no code: it accepts 23 opcodes and three
globals (`collections.OrderedDict`, `torch.FloatStorage` and
`torch._utils._rebuild_tensor_v2`), and it builds only a dict of
contiguous F32 tensors from them. What it copies through the pickle's memo is charged against a budget of 65,536 values, and values nest at most 32 deep, so a pickle that doubles a tuple at every level is refused before it grows. Every offset the archive gives is checked before it is added to or read at. Anything else is refused by name: the
legacy format, another global or opcode, another dtype, or a tensor
outside its storage. `turbo-bundle torch-convert` writes the same
tensors as one safetensors file. On the real checkpoint, all 420 tensors
equal what `torch.load(weights_only=True)` gives.

GLiREL's tokenizer adds `[FLERT]` as a special token, then `[REL]`,
`[E]` and `[/E]`, after the SentencePiece pieces and `[MASK]`. The
converter checks that the checkpoint's vocabulary is exactly that long,
and takes `[REL]` from that position.

## What runs

The code is in `analysis/relation/src/glirel/`, written from GLiREL's
inference path (`jackboyla/GLiREL` at
`1f485a268c2144cdc0221bcf0b24152f93c0a155`):

- `words`: the text's words, as Python's `re` matches
  `\w+(?:[-_]\w+)*|\S`. A word character is a letter, a number or `_`.
  Marks and joiners are words of their own. Case is kept.
- `prompt`: `[CLS]`, `[REL] <label>` for each label, `[SEP]`, the words
  and `[SEP]`. Each label and each word is tokenized on its own, and
  each is read at its first subword.
- `heads`: `token_rep_layer.projection` (1024 to 768) on the rows read;
  a bidirectional LSTM (384 per direction) over the words; each label is
  the prompt MLP on the mean of its `[REL]` row and its label row; each
  entity is `SpanMarkerV1` on its first and last word; each ordered pair
  of entities is the pair MLP on the two. The pair layer's first matrix
  splits into head and tail columns, applied once per entity, and its
  last folds into each label.
- A pair's score for a label is the logistic of the dot product of the
  pair and the label. `extract` returns, for every ordered pair of
  distinct entities, each label scoring above the threshold.

As a scorer, a mention scores 0 if its span does not start where a word
starts and end where a word ends. The text is limited to GLiREL's
`max_len` (512 words) and to the encoder's `max_sequence` (512 subwords),
and fails with the limit named.

## Parity

`analysis/relation/reference/glirel/build_fixture.py`, pinned in its
`requirements.txt`, writes both fixtures. Only fixture production uses
Python. It loads the checkpoint with `weights_only=True` and never
fetches the base model's weights.

- `analysis/relation/tests/glirel_reduced.rs`, run every time, uses a
  fixture of seeded random weights. It holds no GLiREL or DeBERTa value,
  so the repository holds none of the non-commercial weights. Against
  GLiREL's own `LstmSeq2SeqEncoder`, `RelRepLayer` and prompt MLP, the
  projected rows, LSTM rows, entities, label representations and pair
  logits agree within 2e-5.
- `analysis/relation/tests/glirel_real.rs` (`#[ignore]`, release) runs
  the full checkpoint, through the zoo once the licence is accepted, or
  from `TURBO_GLIREL_CHECKPOINT`, checked against the index's digest.
  Against GLiREL's `predict_relations` on three texts, the words and
  input ids are identical, every ordered pair's logit for every label
  agrees within 1e-4, and the relations above 0.5 are the same, with
  scores within 1e-4. The word splitter is also checked against
  Python's `re` on marks, joiners, numbers and joined words.

## Speed

Single-threaded on the x86_64 timing machine (Zen 5, AVX-512), release,
one pinned core, on the three parity texts (5, 5 and 4 entities; 20, 20
and 12 ordered pairs). Best of 7 rounds of 5, three rounds alternating
between the reference and native. The reference is GLiREL's
`predict_relations` on torch 2.14.1 CPU with one thread, its encoder
timed with hooks around the module.

| per text | reference | native | ratio |
|---|---|---|---|
| end to end | 624.5 to 628.7 ms | 106.5 to 106.9 ms | 0.17x |
| of which the reference's encoder | 607.6 to 611.7 ms | | |

## Reusing head buffers

`heads::Heads` keeps its projections packed with `turbo::cpu::Matrix`.
Callers retaining head intermediates can keep a `turbo_relation::Scratch`
and output vectors, then use `project_into`, `lstm_into`, `entities_into`,
`types_into` and `logits_into`. These methods resize existing storage and
reuse its capacity. Scratch belongs to the caller, so concurrent requests
use separate scratch while sharing immutable heads. The allocating methods
remain available for callers that want an owned result immediately.

The ordinary relation pass reuses scratch across its head stages. The
`reused_head_buffers` test checks warmed LSTM, entity and pair-logit calls
with changing row counts, including empty inputs. This is an allocation
check on the stored reduced reference fixture, not a throughput claim for
a released checkpoint or an allocation-free encoder request.
