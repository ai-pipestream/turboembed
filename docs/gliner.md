# Native GLiNER span extraction

Part of #391, #715, and #962. `turbo-gliner` adds the entity-schema portion of
GLiNER extraction using the native DeBERTa-v3 and mDeBERTa-v3 encoder. It supports the
Apache-2.0 `urchade/gliner_small-v2.1` checkpoint at
`4e091416cf7c3481db542c2a3d26156916f3a47f` (`urchade-gliner-small-v2.1` with
`microsoft-deberta-v3-small-tokenizer` in the zoo) and the multilingual
Apache-2.0 `urchade/gliner_multi-v2.1` checkpoint at
`443d26d654e0324125a96bebd8e796c14ff2efe6` (`urchade-gliner-multi-v2.1` with
`microsoft-mdeberta-v3-base-tokenizer` in the zoo). See `docs/licences.md`.
No encoder attention or matrix-multiplication loop is changed here.

This is a Rust CPU API; GLiNER and GLiNER2 heads on other devices are
#920. GLiNER2 document classification, structured records and checks
against other layers are in `turbo-schema-extract`
([schema-extract.md](schema-extract.md)), and SaT sentence segmentation is
in `turbo-sat`. The entity schema here is not a substitute for GLiNER2's
counting and structured-field heads. There is no C session task for
GLiNER.

## One pass and exact checkpoint heads

`ExtractionSchema::new` accepts ordered entity labels, a finite threshold in
[0, 1], and `Decode::Flat` or `Decode::Nested`. `Model::pack` constructs
`[CLS] <<ENT>> label_0 <<ENT>> label_1 ... <<SEP>> words ... [SEP]`.
Each label and input word is encoded independently as the reference's
pretokenized input does. The pretrained SentencePiece vocabulary is used,
with the HF fast-tokenizer behavior of joining byte-fallback runs into an
unknown token. Reserved control-token strings in caller labels or words
are rejected. A nonempty token erased by SentencePiece normalization receives
one unknown piece, preserving its source token index. Requests are not silently
truncated.

A single native encoder call produces all label and document hidden states.
The checkpoint's 768-to-512 projection is applied to first-subword and entity
marker rows. Document words pass through the trained one-layer bidirectional
LSTM. The markerV0 start/end and concatenated-span MLPs, and the prompt MLP,
use the checkpoint's ReLU and linear layers in evaluation mode. Dot products
produce logits and stable sigmoid scores. Only valid spans up to `max_width`
are scored. Every head layer is one packed product (`turbo::cpu::Matrix`) over
all of its rows: the projection over prompt and word rows together, the LSTM
input weights over all words for both directions, and out_project and label
scoring over blocks of 128 spans, `[spans, 2*hidden]` in and
`[spans, hidden] x [hidden, labels]` out. Only the LSTM recurrent step runs word
by word, in one workspace per call. Flat decoding greedily removes all overlap; nested decoding permits
containment but rejects crossing intervals and duplicate interval labels.
Threshold comparison is strict `score > threshold`, matching the reference.
Equal-score ordering is stable in start, width, then caller-label order.

Hard limits precede expensive operations: 32 nonempty unique labels, 128 UTF-8
bytes per label, 384 input tokens, 4096 bytes per token, encoder context length,
and checkpoint span width 1..=32. Each error names its limit. Tensor shapes and
finite weights are checked at load, and non-finite intermediate outputs fail
instead of becoming apparently confident predictions.

## Layers and evidence

`GlinerAnnotator` reads `View`, `Tokens`, and optional `Sentences`, and writes `Entities` and aligned
`Decisions`. Both outputs can be bound to named instances. It validates token
view identity, continuous token layout and UTF-8 slicing before inference.
Entity offsets use the first source token's start and last token's end; they
are not reconstructed from decoded subwords. Entity scores are sigmoid values;
`Decisions` stores raw logits and the threshold. Empty or token-free documents
produce empty outputs without inference; the direct packing API still requires
at least one token. Annotator version 3 records the windowing semantics below.

After the outputs are added, `GlinerAnnotator::evidence` builds
`Evidence<Entities>` for their keys. Its raw scores are logits, its producer
contains the checkpoint identity, and support references both the token range
and the aligned decision holding the threshold. Sigmoid values are not labeled
calibrated probabilities. Register `Decisions` with the graph codec registry
alongside `turbo_analysis_layers::export::codecs()` to export the entire graph.
The annotator hash includes the ordered labels, threshold, decoding mode,
checkpoint configuration and model identity.

## Long documents

The annotator packs consecutive whole sentences into windows that fit both
384 tokens and the encoder context length, including the complete label prompt
and final separator. Empty sentences consume no window space. A supplied
sentence that cannot fit is refused with its zero-based sentence index and
`max_tokens` or `max_sequence`; it is never truncated. Sentence spans must be
contiguous, in the input view, on UTF-8 boundaries, and cover every token.
When token sentence indices exist, they must agree with those spans.

Without a `Sentences` layer, windows end at token boundaries. A single token
that cannot fit is refused with its zero-based token index. Tokens erased by
normalization still occupy an unknown piece, including in later windows.
Each window is scored and decoded independently. No entity crosses a window
boundary, and entities near that boundary may differ from a hypothetical
unlimited-context prediction. A single fitting window retains the original
checkpoint behavior. The prompt is encoded once per document; tokenizer
scratch and unit buffers are reused. Packing and sentence ownership checks
are linear in the input size, apart from tokenizer work.

Decoded token indices are shifted to document-global indices; source byte
spans, logits, and evidence support remain aligned across all windows. Named
outputs and evidence graph export/import work as for a single window.
`analysis/gliner/src/window_tests.rs` verifies a 2,048-token document against
the union of independently scored windows in flat and nested modes, including
UTF-8, empty sentences, normalization-erased tokens, and boundary errors.

## End-to-end validation on multi-sentence documents

`bundle/reference/gliner/build_e2e_reference.py` builds six fixed documents
from fixed sentences: three short ones that fit one window, two longer ones of
442 and 1,107 words that the 384-word limit splits, and one of 756 words with
many pieces per word, which the 512-piece limit splits. It also builds three
timing documents of about 100, 400 and 1,600 words. The script windows each
document by the native rule (whole sentences, at most 384 words and 512 pieces
including the prompt and final separator), but with the HF tokenizer, and
scores every window with the upstream GLiNER 0.2.21 classes as
`build_reference.py` does. Labels are person, organization, location and date;
the threshold is 0.5. The output (about 2.4 MB) is not committed. Python versions are pinned in
`bundle/reference/gliner/requirements.txt`.

```
python build_e2e_reference.py <checkpoint-dir> e2e-reference.json [reps]
GLINER_PACKAGE=... GLINER_E2E_REFERENCE=e2e-reference.json cargo test --release --locked -p turbo-gliner --test e2e -- --ignored --nocapture
```

`analysis/gliner/tests/e2e.rs` checks the following. For every reference
window, the packed IDs must be equal and logits must be within 0.002. Decoded
flat and nested entities must be identical, with scores within 0.0001. The
`GlinerAnnotator` output for the whole document, with a `Sentences` layer,
must equal the union of the standalone windows. For single-window documents,
the annotator output must also equal `Model::predict` exactly.

Results for urchade/gliner_small-v2.1 (entity agreement is matched over the
union of (start, end, label); the counts are the same in both modes except
where noted):

| document | words | windows | max logit error | entities agreed (flat / nested) |
|---|---|---|---|---|
| short-3 | 34 | 1 | 4.2e-5 | 9/9 |
| short-utf8 | 42 | 1 | 5.7e-5 | 10/10 |
| short-2 | 26 | 1 | 4.2e-5 | 4/4 / 5/5 |
| long-2x | 442 | 2 | 8.4e-5 | 73/73 |
| long-6x | 1107 | 3 | 9.2e-5 | 180/180 |
| long-pieces | 756 | 3 | 7.3e-5 | 180/180 / 189/189 |

`multi_sentence_timing` times, for each timing document:
- the annotator with packed sentence windows;
- one window per sentence, which repeats the label prompt;
- encoder-only time for the packed windows as they are, and padded to 512
  with masked rows.

Pass `reps` to the script to time the Python reference end to end on one
thread. On a Ryzen 9 9950X3D, `taskset -c 6`, median of 5, runs ordered
native, Python, Python, native, with times in ms:

| words | windows | native packed | native per sentence | Python reference | encoder packed | encoder padded to 512 |
|---|---|---|---|---|---|---|
| 108 | 1 | 83.5 / 86.2 | 119.6 / 125.4 | 224.9 / 225.8 | 48.9 / 49.1 | 50.4 / 50.5 |
| 403 | 2 | 333.0 / 334.0 | 528.5 / 473.7 | 832.2 / 832.6 | 228.5 / 204.1 | 237.5 / 206.4 |
| 1606 | 5 | 1377.1 / 1350.1 | 1836.3 / 1998.4 | 3128.1 / 3078.0 | 827.2 / 793.7 | 813.5 / 806.8 |

The native encoder computes only the visible rows, so padding a window to 512
changes encoder time by under 4%; there is no separate padded execution path.
Packing sentences into shared windows is 1.3x to 1.6x faster than one window
per sentence. The native annotator is 2.3x to 2.7x faster than the Python
reference on one thread.

knowledgator/gliner-multitask-large-v0.5 (Apache-2.0) cannot be converted. It
uses `span_mode: token_level` on DeBERTa-v3-large with max_width 100, while
the converter and the heads support only the small-v2.1 markerV0 profile.

## Conversion and loading

The converter accepts either F32 safetensors (`model.safetensors`, as published
by `urchade/gliner_multi-v2.1`) or a `torch.save` zip archive
(`pytorch_model.bin`, as published by `urchade/gliner_small-v2.1`, read with
the code-free pickle reader in `bundle/src/torch/`). It requires the
small-v2.1 (`microsoft/deberta-v3-small`, 128,004 vocabulary rows) or
multi-v2.1 (`microsoft/mdeberta-v3-base`, 250,105 vocabulary rows)
markerV0/first-pooling profile, RNN and marker embeddings, and rejects
unsupported switches or mismatched `spm.model` vocabularies. Shapes and finite
values are checked for every required encoder, recurrent and head tensor. It
does not substitute random weights for missing tensors. Source revision and
licence must be supplied.

```
turbo-bundle gliner-convert config.json gliner_config.json pytorch_model.bin spm.model urchade/gliner_small-v2.1 4e091416cf7c3481db542c2a3d26156916f3a47f Apache-2.0 new-gliner-small
turbo-bundle gliner-verify new-gliner-small
turbo-bundle gliner-convert config.json gliner_config.json model.safetensors spm.model urchade/gliner_multi-v2.1 443d26d654e0324125a96bebd8e796c14ff2efe6 Apache-2.0 new-gliner-multi
turbo-bundle gliner-verify new-gliner-multi
```

The new directory contains `encoder/`, `heads.safetensors`, `spm.model` and
`gliner.json`. Existing directories are preserved. A failed write may leave an
incomplete directory, which fails loading. The package verifies SHA-256 for
heads, tokenizer and the encoder manifest; the encoder verifies its weights.
`turbo_gliner::package::Manifest::load` returns the immutable model. Its ModelRef
digest binds the whole package via the GLiNER manifest.

## Numerical and boundary proof

`bundle/reference/gliner/build_reference.py` executes the real encoder
(six-layer DeBERTa-v3-small or twelve-layer mDeBERTa-v3-base) and GLiNER
0.2.21's actual LSTM/markerV0 classes with strict state-dictionary loading. It
avoids importing the optional ONNX serving frontend. Reference package versions
are pinned in the adjacent requirements file.
`analysis/gliner/tests/fixtures/full-reference.json` records `urchade/gliner_small-v2.1`
on four English and Unicode fixtures, and
`analysis/gliner/tests/fixtures/multi-reference.json` records `urchade/gliner_multi-v2.1`
on five multilingual and Unicode fixtures (French, German, Japanese,
emoji/Gothic supplementary-plane characters, and nested English).

The independent Rust full-checkpoint tests (`full_checkpoint_matches_python`
and `multi_checkpoint_matches_python` in `analysis/gliner/tests/checkpoint.rs`)
fetch the checkpoints and tokenizers through `turbo-zoo` and package them via
`turbo-bundle` (or read `GLINER_PACKAGE` / `GLINER_MULTI_PACKAGE` when set).
They compare token IDs and positions exactly, logits within 0.002 and selected
scores within 0.0001, and require identical decoded labels and spans. The
maximum logit error is 0.00003242 on `urchade/gliner_small-v2.1` and 0.00033951
on `urchade/gliner_multi-v2.1`. An additional native annotation test checks
nested source slices with multi-byte UTF-8, named instances, threshold/logit
evidence and graph export/import:

```
cargo test --release --locked -p turbo-gliner --test checkpoint -- --ignored --nocapture
```

The normal offline test suite uses a small, real-weight head slice generated by
`bundle/reference/gliner/build_head_fixture.py`, plus limit and malformed-weight
checks. Full weights are not committed, so the full-width test is explicitly
ignored unless its artifacts are available. Neither fixture establishes broad
NER accuracy or throughput across devices. `analysis/gliner/tests/timing.rs`
(ignored, needs `GLINER_PACKAGE`) prints the median `score_packed` time at 64,
128 and 384 words with 4 and 32 labels; pin it to one core with `taskset`.
