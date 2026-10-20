# Extractive question answering

`TASK_EXTRACTIVE_QA` runs a two-output token head over joint question/context
inputs. `turbo::qa::ExtractiveQa` loads the bundle, produces window logits, and
returns either calibrated abstention or ranked byte spans. The first execution
path is CPU F32 and is advertised as experimental until task benchmarks exist.

## Bundle contract

A QA manifest uses `task: "TASK_EXTRACTIVE_QA"` and a `qa` block with `max_seq`,
`max_batch`, `max_query_tokens` (default 64), and `doc_stride` (default 128).
`doc_stride` is context overlap, not window advance. It must be smaller than the
context capacity after the full query budget and special tokens. Small test
models must explicitly set smaller query and overlap budgets.

The encoder is BERT or RoBERTa. The manifest has no `embed`, `projection`, or
`static_embedding` block. `classifier.kind` is `CLASSIFIER_TOKEN`, with labels
`["start", "end"]` in that order. A single raw safetensors artifact carries the
encoder and `classifier_weight` / `classifier_bias` tensors with shapes
`[2, hidden]` and `[2]`. A sequence pooler is forbidden. Typical upstream tensor
names are `qa_outputs.weight` and `qa_outputs.bias`.

Each reference case has `text` (the question), `context`, and `prompt_role`.
The reference file contains `ids` and `types` for every window, `lengths`, and
flattened `logits: [sum(lengths), 2]`. Tokenizer loading verifies IDs, segment
IDs, padding, dimensions, and finite logits. Bundle verification also executes
the CPU model and compares every logit with absolute tolerance `1e-4`.
`bundle/reference/qa_reference.py` generates these tensors from Hugging Face
`AutoModelForQuestionAnswering` in the pinned reference container.

## Windows and answer decoding

BERT inputs are `[CLS] question [SEP] context [SEP]`; RoBERTa inputs are
`<s> question </s></s> context </s>`. Questions up to the query budget are
preserved; longer questions are right-truncated. Context tokens retain offsets
into the original UTF-8 text. Each token can start candidates only in its most
interior window, with earlier windows breaking ties. Endpoints may extend to
later tokens in that same window even when they have another owner.

`DecodeOptions` defaults to 30 answer tokens, five non-overlapping candidates,
and null threshold zero. Duplicate spans retain their highest score. Ties use
earlier byte starts, then shorter spans. The null score is the minimum CLS
start-plus-end score across windows. Abstention occurs when
`best_span_score - null_score <= null_threshold`, including equality.
Stable softmax is over the returned candidate spans and the null candidate;
these are relative candidate probabilities, not a measured calibration claim.
An abstained `Answers` layer is empty and records null probability one.

## Document integration

`ExtractiveQaAnnotator` reads a `View` and `Question` and writes `Answers`
(`turbo:answers`). It records the bundle identity and hashes every decoder
option. Answers retain score rank and exact input-view byte spans.
`Answers::project(document, ViewId::SOURCE)` maps them through the checked LCA
view path. A lossy view transformation can widen the source interval; it cannot
reconstruct text removed by that transformation.

## Validation evidence and limits

`core/tests/qa_decode.rs` covers null wins, equality, second-window answers,
UTF-8, extreme finite logits, malformed inputs, duplicate spans, and ownership
boundaries. `core/tests/qa_inference.rs` verifies real tiny-BERT encoder and
head logits against Transformers, complete window coverage, query truncation,
and one owner per context token. `analysis/embed/tests/qa.rs` checks annotation,
option provenance and source projection. `bundle/tests/qa.rs` checks reference
parity and rejects modified reference logits even with updated file hashes.

`testdata/tiny-qa-reference` uses an untrained head with deterministic seed 745.
It establishes arithmetic parity and SQuAD-v2 null-score mechanics. It does not
establish pretrained SQuAD-v2 accuracy, production probability calibration,
GPU support, or latency/throughput claims.
