# Native GLiNER span extraction

Part of #391 and #715. `turbo-gliner` adds the entity-schema portion of
GLiNER extraction using the native DeBERTa-v3 encoder. It supports the
Apache-2.0 `urchade/gliner_small-v2.1` checkpoint at
`4e091416cf7c3481db542c2a3d26156916f3a47f`. See `docs/licences.md`.
No encoder attention or matrix-multiplication loop is changed here.

This is a Rust CPU API. SaT, other devices, GLiNER2 document classification,
structured records and deterministic constraint layers remain open work in
#391 and #715. The entity schema is not a substitute for GLiNER2's counting
and structured-field heads. There is no new C session task or throughput
acceptance claim for #715 in this change.

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
are scored. Flat decoding greedily removes all overlap; nested decoding permits
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

## Conversion and loading

The converter accepts F32 safetensors exported from the upstream PyTorch state
dictionary with `torch.load(..., weights_only=True)`. It requires the
small-v2.1 markerV0/first-pooling profile, RNN and marker embeddings, and rejects
unsupported switches. Shapes and finite values are checked for every required
encoder, recurrent and head tensor. It does not substitute random weights for
missing tensors. Source revision and licence must be supplied.

```
turbo-bundle gliner-convert encoder-config.json gliner_config.json checkpoint.safetensors spm.model urchade/gliner_small-v2.1 4e091416cf7c3481db542c2a3d26156916f3a47f Apache-2.0 new-gliner-package
turbo-bundle gliner-verify new-gliner-package
```

The new directory contains `encoder/`, `heads.safetensors`, `spm.model` and
`gliner.json`. Existing directories are preserved. A failed write may leave an
incomplete directory, which fails loading. The package verifies SHA-256 for
heads, tokenizer and the encoder manifest; the encoder verifies its weights.
`turbo_gliner::package::Manifest::load` returns the immutable model. Its ModelRef
digest binds the whole package via the GLiNER manifest.

## Numerical and boundary proof

`bundle/reference/gliner/build_reference.py` executes the real six-layer,
768-wide encoder and GLiNER 0.2.21's actual LSTM/markerV0 classes with strict
state-dictionary loading. It avoids importing the optional ONNX serving
frontend. Reference package versions are pinned in the adjacent requirements
file. `analysis/gliner/tests/fixtures/full-reference.json` records the immutable
checkpoint, source hashes, packed IDs, pooling positions, all valid logits,
and flat/nested predictions for four fixtures, including accented text,
unknown emoji and supplementary-plane characters.

The independent Rust full-checkpoint test compares token IDs and positions
exactly, logits within 0.002 and selected scores within 0.0001, and requires
identical decoded labels and spans. The current maximum logit error is
0.00003434. An additional native annotation test checks nested source slices
with multi-byte UTF-8, named instances, threshold/logit evidence and graph
export/import. It runs explicitly with local artifact environment variables:

```
GLINER_PACKAGE="$PWD/new-gliner-package" GLINER_REFERENCE="$PWD/reference.json" cargo test --release --locked -p turbo-gliner --test checkpoint -- --ignored --nocapture
```

The normal offline test suite uses a small, real-weight head slice generated by
`bundle/reference/gliner/build_head_fixture.py`, plus limit and malformed-weight
checks. Full weights are not committed, so the full-width test is explicitly
ignored unless its artifacts are available. Neither fixture establishes broad
NER accuracy or throughput across devices.
