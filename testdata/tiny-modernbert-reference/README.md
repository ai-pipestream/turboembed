# tiny-modernbert-reference

A small ModernBERT with random weights and the reference vectors the
upstream pipeline gives for it. It is not a bundle: `turbo-bundle` writes
the manifest, from these files, on the Rust side.

- `model/`: what `bundle/reference/tiny_modernbert.py` wrote, the
  directory `reference.py` reads (the upstream layout: `config.json`,
  `model.safetensors`, the tokenizer files, and the sentence-transformers
  files for mean pooling with no normalization).
- `cases.json`: the texts, in the form the bundle tool hands the
  container (`bundle/src/reference.rs`): the cases of the MiniLM recipe
  with no prompts, `max_seq` 64, `max_batch` 8. The last two run to the
  64-token limit, far past the local window.
- `reference.safetensors`, `produced_by.json`: what `reference.py` wrote
  for those cases: `ids` [9, 64] I32 (padded with the pad id), `lengths`
  [9] I32, `embeddings` [9, 32] F32, mean pooled over every token of the
  row, not normalized; one text at a time, and checked against batches
  of 8.

## Geometry

Three layers, hidden 32, two heads of width 16, intermediate 48 (GeGLU:
`Wi` is [96, 32]; rows 0..48 are the half that goes through the
activation, rows 48..96 the half it multiplies: `Wo(gelu(Wi[0..48] x) *
Wi[48..96] x)`, gelu the erf form), vocabulary 1200, 64 positions,
LayerNorm eps 1e-5 with weight only, pre-norm with no `attn_norm` on
layer 0 and `final_norm` after the last layer. Layer 0 is global
(`global_attn_every_n_layers` 3) with RoPE theta 160000; layers 1 and 2
are local with RoPE theta 10000 and `local_attention` 8: query i sees key
j iff `abs(i - j) <= 4`. RoPE is rotate_half over the two halves of each
head. Weights: transformers' initialisation with seed 0 and
`initializer_range` 0.2 (so the attention and MLP terms are not lost
under the residual), then every norm weight drawn as `1 + 0.2 N(0, 1)`.
The checkpoint is saved as `ModernBertForMaskedLM`, as upstream's is:
the encoder under `model.`, beside a masked-LM head (`head.`,
`decoder.bias`) the encoder does not read; the decoder weight is tied to
the token embeddings and not stored.

The tokenizer is `testdata/tiny-bpe/tokenizer.json` as transformers wrote
it again (equal as JSON, not byte for byte): byte-level BPE in RoBERTa's
form, `<s> $A </s>`, pad id 1.

## Tensors (`model/model.safetensors`, all F32)

| Name | Shape |
|---|---|
| `model.embeddings.tok_embeddings.weight` | [1200, 32] |
| `model.embeddings.norm.weight` | [32] |
| `model.layers.{0,1,2}.attn.Wqkv.weight` | [96, 32] (Q rows, then K, then V) |
| `model.layers.{0,1,2}.attn.Wo.weight` | [32, 32] |
| `model.layers.{1,2}.attn_norm.weight` | [32] (absent for layer 0) |
| `model.layers.{0,1,2}.mlp_norm.weight` | [32] |
| `model.layers.{0,1,2}.mlp.Wi.weight` | [96, 32] |
| `model.layers.{0,1,2}.mlp.Wo.weight` | [32, 48] |
| `model.final_norm.weight` | [32] |
| `head.dense.weight` | [32, 32] (unused by the encoder) |
| `head.norm.weight` | [32] (unused) |
| `decoder.bias` | [1200] (unused) |

`model.safetensors` is 270120 bytes, SHA-256
`93492f80652a76ed18e2c1fc7c4cf82c8afe6f0812c757aaeb1dd8d9fdbd9f5e`;
`reference.safetensors` is 3700 bytes, SHA-256
`c152d999a370a264019629dab6aef6e8b83156cc52f2b56c62981b03c88bd055`.

## Making it again

From the repository root, with `W` an empty directory and the image built
from `bundle/reference` (`docker build -t turbo-reference bundle/reference`):

```
cp testdata/tiny-bpe/tokenizer.json testdata/tiny-modernbert-reference/cases.json $W/
docker run --rm --network none --user $(id -u):$(id -g) -e HOME=/tmp \
    -v $PWD/bundle/reference:/scripts:ro -v $W:/w --entrypoint python \
    turbo-reference@sha256:29b0fe887a20d2da5d1eb042693f8149d46c97d672b835f94f42ccfe2d14d27e \
    /scripts/tiny_modernbert.py /w/tokenizer.json /w/model
docker run --rm --network none --user $(id -u):$(id -g) \
    --mount type=bind,src=$W/model,dst=/model,readonly --mount type=bind,src=$W,dst=/work \
    turbo-reference@sha256:29b0fe887a20d2da5d1eb042693f8149d46c97d672b835f94f42ccfe2d14d27e \
    /model /work/cases.json /work/reference.safetensors /work/produced_by.json
```

The image is transformers 5.17.0, sentence-transformers 6.1.0, torch
2.14.0 (CPU); the digest above is its id under Docker's containerd image
store (its config digest is
`sha256:11747bb4dfbd7d955047408fda3971640a7451cb5b0f606daaf2a07f6a8fb7f6`).
Running both steps twice gave the same bytes. A float64 reimplementation
of the arithmetic above, outside the container's model code, matches the
reference vectors within 4e-7; the same with the `Wi` halves exchanged,
or with the window two keys wider or narrower, misses by more than 0.1.
Made for this repository, under its licence (the tokenizer was trained
here too, `testdata/README.md`).
