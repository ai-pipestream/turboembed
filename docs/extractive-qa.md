# Extractive question answering

`TASK_EXTRACTIVE_QA` runs a two-output token head over joint question/context
inputs. `turbo::qa::ExtractiveQa` loads the bundle, produces window logits, and
returns either calibrated abstention or ranked byte spans. It runs on any
backend that offers range writes (`embed_write_ranges`): the CPU and CUDA are
measured below. The capability cell reports the task as experimental on every
device: SQuAD 2.0 accuracy and throughput are recorded, but no benchmark record
compiled into the build backs SUPPORTED for it yet.

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
It establishes arithmetic parity and SQuAD-v2 null-score mechanics.
`core/tests/cuda_qa.rs` runs the tiny model on CUDA, one window per run and
batched, against the reference logits (1e-4) and the CPU path (1e-5), and checks
that the CUDA capability cell reports the task as experimental. With
`TURBO_TEST_QA_BUNDLE` naming the bundle below, it runs that bundle's reference
cases on CUDA at EXACT and MODEL against the transformers logits and the CPU path.
The probabilities are relative candidate probabilities, not a calibration claim.

## The SQuAD 2.0 checkpoint

`bundle/recipes/roberta-base-squad2.json` makes a bundle of
[deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)
at `adc3b06f79f797d1c575d5479d6f5efe54a9e3b4`: RoBERTa-base, 384-token windows,
a 64-token question budget and a 128-token overlap. Its licence is CC BY 4.0, but
it was trained on SQuAD 2.0 (CC BY-SA 4.0), so the zoo lists it
(`deepset-roberta-base-squad2`) as share-alike: fetched only after the user
accepts CC BY-SA 4.0, never bundled or shipped (docs/licences.md). The
repository has no `tokenizer.json`, so the recipe's `bpe_tokenizer` has the bundle
tool write one from `vocab.json` and `merges.txt` (`bundle/src/bpe_files.rs`);
verification holds it to the token ids the upstream tokenizer gives, and the
CPU logits to the transformers reference within 1e-4.

```sh
cargo run --release -p turbo-bundle -- make bundle/recipes/roberta-base-squad2.json <upstream> <bundle>
```

## Accuracy

`turbo-eval squad` scores a bundle on a SQuAD 2.0 file the way the official
evaluation script does (`analysis/eval/src/squad.rs`): normalized exact match and
token F1, the best over a question's gold answers, and an empty prediction right
only for an unanswerable question. A question is answered when its best span's
score is above the null score by more than the threshold; the record gives the
scores at threshold 0, the decoder's default, and, as the script's
`best_exact` and `best_f1`, the best exact match and the best F1 over thresholds
on the same file, each at its own threshold. The search starts from abstaining on
every question and ends with answering every question.

On the SQuAD 2.0 dev set (`dev-v2.0.json`, 11,873 questions, SHA-256
`80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8`), exact
precision:

| | EM | F1 | HasAns EM / F1 | NoAns |
|---|---|---|---|---|
| deepset's card | 79.93 | 82.95 | | |
| CUDA, threshold 0 | 79.92 | 82.92 | 78.02 / 84.03 | 81.82 |
| CPU, threshold 0 | 79.92 | 82.92 | 78.02 / 84.03 | 81.82 |
| CUDA and CPU, best EM (threshold 1.65) | 80.65 | 83.46 | 74.98 / 80.62 | 86.29 |
| CUDA and CPU, best F1 (threshold 1.64) | 80.65 | 83.46 | 75.00 / 80.64 | 86.27 |

The CPU and CUDA scores are the same to the third decimal. The best thresholds
are chosen on the dev set itself, so those rows are an upper bound, not a
held-out number. Scoring the dev set took 22 s on an RTX 4080 SUPER and 155 s on
the CPU (a Ryzen 9 9950X3D, 32 threads), 16 questions per call. The two records
are `docs/receipts/qa-914/squad2-dev-cuda.json` and `squad2-dev-cpu.json`.

```sh
turbo-eval squad --bundle <bundle> --dev dev-v2.0.json --device cuda --out squad.json
```

## Throughput

`turbo-bench qa` times questions per second by the context's length in tokens
(the bundle's tokenizer, no special tokens), up to 256 questions of the file
per bin, 8 questions per call and as many windows per run as the session holds,
and writes a record under `benchmarks/qa/` that names the clean, pushed commit
it was built from. The records for this bundle, at exact precision:

| Context tokens | Windows a question | CUDA questions/s | CPU questions/s |
|---|---|---|---|
| 0 to 128 | 1.00 | 1,099 | 118 |
| 128 to 256 | 1.00 | 599 | 70 |
| 256 to 384 | 1.03 | 353 | 42 |
| 384 and more | 2.24 | 160 | 21 |

The CUDA cell is an RTX 4080 SUPER, the CPU cell a Ryzen 9 9950X3D on all 32
threads. The head runs on the host in F64 for both.

```sh
turbo-bench qa --bundle <bundle> --questions dev-v2.0.json --device cuda --precision exact
```

## What is still missing

The capability cell stays experimental: no benchmark record compiled into the
build backs SUPPORTED for extractive QA, and the records above are task records,
not the embedding records `turbo-bench check` reads. Metal and Level Zero offer
range writes and so open QA sessions, but neither has been run against the
reference here; they are reached only by naming them in `QaOptions`, as
`ExtractiveQa::open`, which the bundle check and the annotator use, stays on the
CPU. The threshold is not calibrated on held-out data. On a GPU every token's
final hidden state comes back to the host for the span head, about 1.2 MB a
384-token RoBERTa-base window; #1003 moves the head onto the device.
