# tiny-nomic-reference

A small nomic-bert with random weights and the reference vectors the
upstream pipeline gives for it. It is not a bundle: `turbo-bundle` writes
the manifest, from these files, on the Rust side.

- `model/`: what `bundle/reference/tiny_nomic.py` wrote, the directory
  `reference.py` reads (the upstream layout of nomic-embed-text-v1.5:
  `config.json`, `model.safetensors`, the tokenizer files, `modules.json`,
  `1_Pooling/config.json` for mean pooling, `sentence_bert_config.json`;
  no Normalize module).
- `cases.json`: the texts, in the form the bundle tool hands the
  container (`bundle/src/reference.rs`): the cases of the MiniLM recipe,
  with the prefixes `search_query: ` and `search_document: ` on the query
  and document cases, `max_seq` 64, `max_batch` 8.
- `reference.safetensors`, `produced_by.json`: what `reference.py` wrote
  for those cases: `ids` [9, 64] I32 (padded with the pad id), `lengths`
  [9] I32, `embeddings` [9, 32] F32, mean pooled over every token of the
  row, not normalized; one text at a time, and checked against batches
  of 8.

## Geometry

Two layers, hidden 32, two heads of width 16, intermediate 48 (SwiGLU:
`fc2(silu(fc12 x) * fc11 x)`, `fc12` the gate that goes through the
activation, `fc11` the value it multiplies), vocabulary 30528 (the
tokenizer has 30522 entries; upstream pads its table the same way), two
token types (type 0 is added to every token), 64 positions, RoPE theta
1000 on every layer (rotate_half over the two halves of each head),
every layer global, post-norm (`norm1(x + attn(x))`, `norm2(x + mlp(x))`)
with LayerNorm eps 1e-12 with weight and bias, no final norm, no bias on
any projection. Weights: transformers' initialisation of its native
`NomicBertModel` with seed 0 and `initializer_range` 0.2 (so the
attention and MLP terms are not lost under the norms), then every norm
weight drawn as `1 + 0.2 N(0, 1)` and every norm bias as `0.2 N(0, 1)`.
transformers saves it under the upstream checkpoint's names (it splits
`Wqkv` and renames `fc11`, `fc12` on load, and writes them back on save).
`config.json` also carries the upstream config's own keys
(`activation_function` swiglu, `n_positions` 64, `n_inner` 48,
`layer_norm_epsilon`, `rotary_emb_base` 1000, `rotary_emb_fraction` 1,
`rotary_emb_interleaved` false, `rotary_scaling_factor` null, `prenorm`
false, `use_rms_norm` false, `qkv_proj_bias`, `mlp_fc1_bias`,
`mlp_fc2_bias` false), with this model's values.

The tokenizer is `testdata/tiny-bert-bundle/tokenizer.json`, the same
bytes: WordPiece, bert-base-uncased's vocabulary, `[CLS] $A [SEP]`.

## Tensors (`model/model.safetensors`, all F32)

| Name | Shape |
|---|---|
| `embeddings.word_embeddings.weight` | [30528, 32] |
| `embeddings.token_type_embeddings.weight` | [2, 32] |
| `emb_ln.weight`, `emb_ln.bias` | [32] |
| `encoder.layers.{0,1}.attn.Wqkv.weight` | [96, 32] (Q rows, then K, then V) |
| `encoder.layers.{0,1}.attn.out_proj.weight` | [32, 32] |
| `encoder.layers.{0,1}.norm1.weight`, `.bias` | [32] |
| `encoder.layers.{0,1}.mlp.fc12.weight` | [48, 32] (gate) |
| `encoder.layers.{0,1}.mlp.fc11.weight` | [48, 32] (value) |
| `encoder.layers.{0,1}.mlp.fc2.weight` | [32, 48] |
| `encoder.layers.{0,1}.norm2.weight`, `.bias` | [32] |

`model.safetensors` is 3980880 bytes, SHA-256
`e575c6d838ffb5d31a409aea0ca2eaee01035e523d2051d96ac1047d9c432de3`;
`reference.safetensors` is 3700 bytes, SHA-256
`8be7ad633d9036fa74ee1b4179f8fb2c3c70a63ec6c31d66a2c6c792a47d850e`.

## Making it again

From the repository root, with `W` an empty directory and the image built
from `bundle/reference` (`docker build -t turbo-reference bundle/reference`):

```
cp testdata/tiny-bert-bundle/tokenizer.json testdata/tiny-nomic-reference/cases.json $W/
docker run --rm --network none --user $(id -u):$(id -g) -e HOME=/tmp \
    -v $PWD/bundle/reference:/scripts:ro -v $W:/w --entrypoint python \
    turbo-reference@sha256:29b0fe887a20d2da5d1eb042693f8149d46c97d672b835f94f42ccfe2d14d27e \
    /scripts/tiny_nomic.py /w/tokenizer.json /w/model
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
reference vectors within 6e-7; the same with `fc11` and `fc12` exchanged
misses by more than 2. Under Apache-2.0, the licence of the MiniLM
tokenizer.
