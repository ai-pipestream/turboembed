# Schema-driven extraction

Part of #715. `turbo_schema_extract` fills one extraction schema with one
GLiNER2 encoder pass on the native DeBERTa-v3 encoder (docs/deberta.md):
entity types, document classification tasks, and record types with
fields. It uses no Python, ONNX or external inference engine. The
checkpoint is `fastino/gliner2-base-v1` at
`f9634218e53580c56edf0de97ca1a7d3f1c2354e` (Apache-2.0, docs/licences.md),
which the zoo lists as `fastino-gliner2-base-v1` and fetches by digest.
The code follows GLiNER2's inference path (`fastino-ai/GLiNER2` at
`ca4f68235b50fe4c3ecf3876851d748fe4b15c6c`, Apache-2.0). The span heads
are the relation scorer's (docs/gliner2-relations.md).

## The schema

An `ExtractionSchema` holds:

- **entity types**, each with an optional description, its own threshold,
  a cardinality (`Single` keeps the best span, `Repeated` every span), and
  checks;
- **classification tasks**, each with its labels (optionally described),
  single-label (a softmax, the best label) or multi-label (a sigmoid per
  label, every label at or above the task's threshold, else the best;
  `top_k` keeps the best few);
- **record types** (`StructureSpec`), each with fields (`FieldSpec`):
  a name, an optional description and threshold, a cardinality, optional
  closed `choices`, and checks.

`ExtractionSchema::validate` refuses a schema before anything is
tokenized:

- an empty schema;
- a blank or duplicate name, label, field or choice;
- a task or record named `entities`, which is the entity types' task in the
  prompt;
- a string holding a model control token (`[P]`, `[DESCRIPTION]` and the
  rest);
- a threshold or boost that is not a finite value in [0, 1], and a `top_k`
  of 0.

Each limit in `Limits` is refused by its name: `max_entity_types`,
`max_classification_tasks`, `max_labels_per_task`, `max_structures`,
`max_fields_per_structure` and `max_text_bytes`. After the prompt is built
come `max_prompt_subwords`, `max_words` and the encoder's `max_sequence`.
`Gliner2Extractor::prompt` and `pass` refuse text over `max_words` or
`max_sequence`; the annotator reads such text in windows instead.

## Long text

`SchemaAnnotator` reads text that fits one pass in one pass, as before.
Longer text is cut into sentence-aligned windows (the document's
`Sentences` when it has them, its tokens otherwise), each the longest run
of sentences that fits `max_words` and `max_sequence`, each window after
the first starting one sentence back so a value cut by a boundary is seen
whole. `SchemaAnnotator::windows` lists them. A sentence that does not fit
alone is refused, naming it. Each window is extracted on its own, its
spans mapped back to the source view's bytes, and the results merged:

- a span value found in more than one window is one value, with the
  highest score a window gave it (and that window's logit and support);
  an entity type of `Single` cardinality keeps its best value over all
  windows;
- a record lives in one window: its fields never combine values from
  different windows. A record whose values all equal an earlier record's
  is the same record seen again in the overlap and is dropped; instances
  count the records kept, per type, in text order;
- a classification label's probability is the highest a window gave it;
  the chosen labels are every window's, best first, one for a
  single-label task and at most `top_k` for a multi-label one.

## What runs

- `prompt`: GLiNER2's input. A task per record type, then the entity types
  as one task, then a task per classification. Each task is
  `( [P] <prompt> ( <marker> <name> ... ) )`, joined by `[SEP_STRUCT]`, then
  `[SEP_TEXT]` and the text tokens. The marker is `[C]` for fields, `[E]`
  for entity types and `[L]` for labels. The prompt is the task's name and a
  ` [DESCRIPTION] <name>: <description>` for each described child. The text
  tokens start with a prefix listing each record type's closed-choice
  fields, `( <record>: <field> ( <choice> | <choice> ) , ... )`, and then
  the words of GLiNER2's word splitter, lower-cased. Each element is
  tokenized on its own, with `[DESCRIPTION]` kept as its id.
- `model`: the encoder runs once. For each span task, the count head on
  `[P]` gives the number of instances, and the count transformer, attending
  across all of the task's fields, gives one query per instance and field.
  The span head scores every span of up to 8 text tokens against each
  query. Entity types use instance 0. For each classification, GLiNER2's
  classifier (Linear, ReLU, Linear) gives one logit per `[L]`.
- `decode`: GLiNER2's decoding.
  - **Spans:** a field's candidates are its word spans with a score
    (the logistic of the logit) at or above the threshold. They are ranked
    by score and kept if they overlap no span kept before them; a
    single-valued field keeps the first.
  - **Choices:** a closed-choice field scores each choice at the first
    prefix token equal to it, ignoring case.
  - **Records:** a record instance is kept if any of its fields has a
    value.

## Deterministic layers

A field or entity type can name checks against layers already on the
document: `GlossaryTerms` (`Source::Glossary`), `Pii`, `Numbers` and
`Locations`. They run where GLiNER2 runs its validators: after the
candidates above the threshold are found, and before overlaps are
suppressed.

- `Check::Require(source)` keeps a candidate only if the source has a span
  overlapping it.
- `Check::Boost(source, b)` adds `b` to its probability, capped at 1,
  before the threshold.

The record layer stores, for each value, the rows of every source that
overlapped it, and the evidence cites them as support.

## Layers

`SchemaAnnotator` (`turbo:schema-extract`) reads the view, `Tokens`, and
any of the four deterministic layers present. It writes:

- `Entities`, over the entity types (kinds `schema-entities`), with each
  span's covering tokens and its score.
- The document's `Categories`, one category per task label as
  `task: label`. Each score is the label's probability within its task,
  whether or not the label was chosen: a single-label task's scores are
  one softmax summing to 1, a multi-label task's are independent sigmoids.
  The selected set is every task's chosen labels; the best category is the
  highest-scoring selected label.
- `Records` (`turbo:records`). Each span value has an exact byte span of
  the view's text (from its first word's start to its last word's end),
  its record, its `record.field` path, its probability and logit, and its
  supporting rows. Each record has its type and the model's instance index.
  Each closed-choice value has its record, path, choice, probability and
  logit. Values of one record may nest and overlap. `Records::check`
  verifies:
  - span bounds and lengths;
  - finite logits and probabilities in [0, 1];
  - every tag inside its set;
  - every path a field of its record's type;
  - every record with a value;
  - support runs in order, from the four sources.

The model's identity is on every layer's provenance, with a config hash:
FNV-1a over a canonical, length-prefixed encoding of the schema (every
entity type, task, label, description, threshold, `top_k`, field, choice
and check), the request threshold, every `Limits` field and the span
heads' `max_width`. Floats are hashed by their bits with `-0.0` folded into
`0.0`, so the hash does not depend on how anything is formatted.

The text read gets a final `.` when it lacks sentence punctuation, as
GLiNER2 reads it; a span reaching into that `.` is cut back to the text.

`export::register_codecs` registers the `DocumentGraph` codec of
`Records` and of `Evidence<Records>`. Import reads the payload as
untrusted:

- every count against the rows that follow;
- every tag against its set;
- every support row against the restored source layer;
- the result against `Records::check`.

## Parity

`analysis/schema-extract/reference/build_fixture.py` writes both fixtures,
with the pinned packages of
`analysis/relation/reference/gliner2/requirements.txt`. Only fixture production uses
Python.

- `analysis/schema-extract/tests/real.rs` (`#[ignore]`, release, fetches
  the checkpoint through the zoo) runs four texts, each with entities,
  classifications and records together, against GLiNER2's own `extract`.
  The input ids, each span task's instance count and each classifier logit
  agree, and so does every decoded value: entities, labels, record fields
  and choices, with byte offsets equal and confidences within 1e-4.
- `analysis/schema-extract/tests/reduced.rs` runs on the relation scorer's
  reduced encoder and heads, with the classifier sliced to width 32. Count
  queries across three and four fields, for one and three instances, and
  classifier logits agree with GLiNER2's own layer classes within 2e-5.
- `analysis/schema-extract/tests/offline.rs` and `limits.rs` run the whole
  annotator on that reduced model, with 800 word embeddings and
  `testdata/sentencepiece/unigram-identity.model`, with no checkpoint. They
  cover:
  - byte spans on character boundaries of multi-byte text, with the values
    of one record nesting and overlapping;
  - empty, whitespace and zero-width (`U+200B`, `U+FEFF`) documents;
  - every named limit and rule;
  - `Require` and `Boost` against real glossary, PII, number and location
    layers, with their rows cited in verified evidence;
  - the record layer across the document graph, byte-identical on
    re-export, with every truncation and every flipped byte of its payload
    refused or checked.
- `analysis/schema-extract/tests/scores_and_hash.rs` checks that every
  label of a task keeps its probability in `Categories`, chosen or not,
  and that the config hash changes with each limit, threshold and schema
  setting while equal settings hash equal.

## Speed

Single-threaded on the x86_64 timing machine (Zen 5, AVX-512), release,
one pinned core, on the four parity texts and their schemas (78 to 126
input tokens). Best of 7 rounds of 5, three rounds alternating between the
reference and native. The reference is GLiNER2 on torch 2.14.1 CPU with
one thread, `extract` per document or `batch_extract` over all four.

| per document | GLiNER2, one at a time | GLiNER2, batch of 4 | native | documents/s, native |
|---|---|---|---|---|
| end to end | 274.8 to 275.4 ms | 242.3 to 246.7 ms | 84.3 to 84.5 ms | 11.8 to 11.9 |

The native extractor takes one document at a time. It runs 2.9 times as
many documents per second as the reference with a batch of four (4.1
documents/s), and 3.3 times as many as one at a time (3.6).
