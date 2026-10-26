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

`Model::new` takes the CPU encoder. `Model::open_on` takes the encoder as
a core bundle of `FAMILY_DEBERTA_V2` instead, through the backend table, on
the device `EncoderOptions` names: cpu, cuda or levelzero (#883). One run
a window, and only the rows the heads read come back: a one-token range at
each `<<ENT>>` marker and each word's first subword, through the range
write, unnormalized, as SaT reads its tokens. The heads, packing and
decoding are the same; the longest sequence is the session's, and
`Model::execution` names where it ran. `analysis/gliner/tests/device.rs` holds a device to
the CPU encoder on `testdata/tiny-deberta-bundle` (every logit within
1e-4, and 1e-3 of its input's largest, the packed ids and decoded entities
the same): on the Arc Pro B70 and the RTX 4080 SUPER the largest difference
is under 1e-10, at MODEL and EXACT.

### The real checkpoint on a device

`bundle/recipes/gliner-small-v2.1-encoder.json` seals `urchade/gliner_small-v2.1`'s
encoder as that core bundle. `bundle/reference/gliner_encoder.py`, in the
pinned reference container, writes it from the zoo's checkpoint and
`microsoft-deberta-v3-small-tokenizer` files (microsoft/deberta-v3-small at
`a36c739020e01763fe789b4b85e2df55d6180012`): the checkpoint's
`token_rep_layer.bert_layer.model` tensors as a `DebertaV2Model` of 128004
rows (loaded with `strict=False`, then refused on any missing key but
`position_ids` and on any unexpected one), and `spm.model` converted to `tokenizer.json` (with `[MASK]` and
GLiNER's `[FLERT]`, `<<ENT>>` and `<<SEP>>` as pieces at the checkpoint's
ids, its normalizer in SentencePiece's own order: the map, the collapse of
spaces, the strip). The conversion runs only in the container; no Python
ships. `turbo-bundle reference` then writes the reference (mean pooled, not
normalized, at 512 tokens) and seals, checking the config against the
recipe. With `Z` the zoo home's `artifacts` directory and `W` an empty one:

```
docker run --rm --network none --user $(id -u):$(id -g) -e HOME=/tmp \
    -v $PWD/bundle/reference:/scripts:ro -v $Z/microsoft-deberta-v3-small-tokenizer:/tok:ro \
    -v $Z/urchade-gliner-small-v2.1:/ckpt:ro -v $W:/w --entrypoint python \
    turbo-reference@sha256:3fe2d4ad441927475b97a400364ebcb7fca3f324dae1a0463720eff7fc0a53f4 \
    /scripts/gliner_encoder.py /tok/config.json /tok/spm.model /ckpt/pytorch_model.bin /w/out
cargo run --release -p turbo-bundle -- reference bundle/recipes/gliner-small-v2.1-encoder.json $W/out $W/bundle
```

`package::Manifest::open_on` takes the converted package with that bundle:
the package is checked as `load` checks it, and the bundle's weights must
be the package encoder's, tensor for tensor. Core conformance holds the
bundle to its reference (10 cases, within 1e-4) on the CPU and the devices.
`analysis/gliner/tests/real_device.rs` (ignored; `GLINER_PACKAGE`,
`GLINER_ENCODER_BUNDLE`, `TURBO_TEST_DEVICE`) holds the device to the
package on the CPU (packed ids identical, every logit and row within 1e-4, decoded
entities alike but for candidates within that of the threshold) and times
`score_packed` on both, interleaved.

Receipts of 2026-10-26, at commit `beb41c4369022f7a05b59060095f904fde21210f`,
release builds, the converted package against the same package over the
sealed encoder bundle. Parity is five consecutive runs at each precision on
each device:

```
GLINER_PACKAGE=<package> GLINER_ENCODER_BUNDLE=<bundle> TURBO_TEST_DEVICE=levelzero TURBO_TEST_REQUIRE_LEVELZERO=1 \
    TURBO_TEST_PRECISION=model cargo test --release --locked -p turbo-gliner --features turbo/levelzero \
    --test real_device the_real -- --ignored --nocapture
```

(`cuda`, `TURBO_TEST_REQUIRE_CUDA=1` and `--features turbo/cuda` on the
RTX 4080 SUPER; `exact` likewise.) Every run passed, and the five gave the
same figures, so the minimum and the maximum are equal:

| Device | Precision | Runs | Marker and first-subword rows, largest difference | Lowest cosine | Logits, largest difference |
| --- | --- | ---: | ---: | ---: | ---: |
| Arc Pro B70 | MODEL, EXACT | 5 each | 3.4e-5 | 1.000000000 | 5.1e-5 |
| RTX 4080 SUPER | MODEL, EXACT | 5 each | 4.8e-5 | 1.000000000 | 5.7e-5 |

The packed ids and the decoded entities were the same in every run.

Timing is `real_checkpoint_timing` (`GLINER_REPS=50`, the same features,
`real_checkpoint_timing` in place of `the_real`): `score_packed`, and the
encoder alone (`Model::states`, the rows the heads read), on the package's
CPU encoder and on the device, interleaved, 50 timed runs each after one
untimed; milliseconds, min / p50 / mean / p99. The difference between a
column's call and its encoder is the heads (projection, BiLSTM, span
scoring), which run on the host on both sides: on the device they are most
of a call past 64 words.

Arc Pro B70 (levelzero), host AMD Ryzen 9 9950X (16 cores, 32 threads). No
build or test process in the CI runner containers that mount the device,
before or after. Another client held the render engine about 55% busy and
none other the compute engines; GPU clock median 2800 MHz, at most 66 C.
The host's load average was 21 to 38, so the CPU columns, and every ratio
to them, ran under load.

| Words, labels (tokens) | CPU call | CPU encoder | levelzero call | levelzero encoder |
| --- | ---: | ---: | ---: | ---: |
| 64, 4 (75) | 31.7 / 37.7 / 38.1 / 47.4 | 15.1 / 17.8 / 18.1 / 25.8 | 30.3 / 35.7 / 35.7 / 44.2 | 30.1 / 35.6 / 35.3 / 45.0 |
| 64, 32 (131) | 39.5 / 53.3 / 54.0 / 74.6 | 19.0 / 27.8 / 29.5 / 45.7 | 26.1 / 32.8 / 34.5 / 51.3 | 13.6 / 36.5 / 34.7 / 48.6 |
| 128, 4 (139) | 66.2 / 89.3 / 89.7 / 146.9 | 18.0 / 27.9 / 28.0 / 51.5 | 47.6 / 83.9 / 81.4 / 133.5 | 3.2 / 6.0 / 8.0 / 20.8 |
| 128, 32 (195) | 66.5 / 89.3 / 93.5 / 172.5 | 22.4 / 35.7 / 35.6 / 64.0 | 55.4 / 82.3 / 83.9 / 147.1 | 4.3 / 10.8 / 10.4 / 23.2 |
| 384, 4 (395) | 243.2 / 329.0 / 328.8 / 412.2 | 72.5 / 113.7 / 119.0 / 192.4 | 188.5 / 220.6 / 232.1 / 350.8 | 9.1 / 27.3 / 30.0 / 70.4 |
| 384, 32 (451) | 211.0 / 275.3 / 296.9 / 426.2 | 56.4 / 95.4 / 110.3 / 202.0 | 172.9 / 229.7 / 230.6 / 320.3 | 21.6 / 55.1 / 51.6 / 77.9 |

At 64 words the device encoder is unexplained: its p50 is about the whole
call's (35.6 of 35.7 ms) or above it (36.5 of 32.8 ms), and slower than at
128 words. The first of these shapes was the first timed, under host load,
but that does not account for the second. These two rows are left out of
the conclusion below until they are retaken with a warm-up per shape.

RTX 4080 SUPER (cuda), host AMD Ryzen 9 9950X3D (16 cores, 32 threads). A
shared machine: a resident server process, kept running there on purpose
because it is under test, held 5.5 GB and kept the GPU about 24% busy; no
other test process and no CI build or test process ran. SM clock median
2610 MHz, at most 47 C. The host's load average was 22 to 40, so every
ratio from this table, the device over the CPU in particular, is from a
shared, loaded machine, not a quiet one.

| Words, labels (tokens) | CPU call | CPU encoder | cuda call | cuda encoder |
| --- | ---: | ---: | ---: | ---: |
| 64, 4 (75) | 33.0 / 35.2 / 37.1 / 54.7 | 13.1 / 14.7 / 15.2 / 19.9 | 21.5 / 22.4 / 23.3 / 31.7 | 1.6 / 1.7 / 1.7 / 1.9 |
| 64, 32 (131) | 38.4 / 46.5 / 49.2 / 107.5 | 18.1 / 23.0 / 23.6 / 59.8 | 22.4 / 27.2 / 27.9 / 39.2 | 2.3 / 2.6 / 2.7 / 3.4 |
| 128, 4 (139) | 59.8 / 65.2 / 66.7 / 80.8 | 17.5 / 22.4 / 22.5 / 35.0 | 42.7 / 46.1 / 46.7 / 53.5 | 2.4 / 2.7 / 2.7 / 3.4 |
| 128, 32 (195) | 67.9 / 72.8 / 73.3 / 87.5 | 24.9 / 28.5 / 29.2 / 36.7 | 43.9 / 45.7 / 46.8 / 54.8 | 2.8 / 2.8 / 3.0 / 4.0 |
| 384, 4 (395) | 171.4 / 245.8 / 287.8 / 793.3 | 49.0 / 86.1 / 108.0 / 356.6 | 127.2 / 176.9 / 200.2 / 819.2 | 4.6 / 5.0 / 5.2 / 6.7 |
| 384, 32 (451) | 197.8 / 229.4 / 240.0 / 340.9 | 61.0 / 80.2 / 83.4 / 123.8 | 130.7 / 160.2 / 158.8 / 209.1 | 4.9 / 5.2 / 5.4 / 6.5 |

From 128 words up, the encoder is a small part of a call on both devices
(on the B70, 6.0 to 55.1 ms of 82 to 230 ms at p50), and on the 4080 at
every shape (1.7 to 5.2 ms of 22 to 177 ms): there the heads on the host
bound it.

This is a Rust API. SaT ships as `turbo-sat`, and GLiNER2 document
classification, structured records and deterministic constraint layers
ship in `turbo-schema-extract` (`analysis/schema-extract`); this page
covers the entity-extraction API only. The entity schema is not a
substitute for GLiNER2's counting and structured-field heads.

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
