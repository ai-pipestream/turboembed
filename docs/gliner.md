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
are rejected. Requests are not silently truncated.

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

`GlinerAnnotator` reads `View` and `Tokens`, and writes `Entities` and aligned
`Decisions`. Both outputs can be bound to named instances. It validates token
view identity, continuous token layout and UTF-8 slicing before inference.
Entity offsets use the first source token's start and last token's end; they
are not reconstructed from decoded subwords. Entity scores are sigmoid values;
`Decisions` stores raw logits and the threshold.

After the outputs are added, `GlinerAnnotator::evidence` builds
`Evidence<Entities>` for their keys. Its raw scores are logits, its producer
contains the checkpoint identity, and support references both the token range
and the aligned decision holding the threshold. Sigmoid values are not labeled
calibrated probabilities. Register `Decisions` with the graph codec registry
alongside `turbo_analysis_layers::export::codecs()` to export the entire graph.
The annotator hash includes the ordered labels, threshold, decoding mode,
checkpoint configuration and model identity.

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
