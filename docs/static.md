# Static models

A static model has no encoder. It holds one vector per vocabulary
entry, and a text's vector is the mean of its tokens' vectors: a
tokenizer, a gather and a sum. That is Model2Vec's model, and
sentence-transformers' `StaticEmbedding`. It is many times faster than
the model it was distilled from, on any CPU, with no sequence limit
beyond the bundle's `max_seq`, and less accurate. The bundle says by how
much.

A static model is its own model, with its own `model.id`, not a
precision of the base model: its vectors live in a space of their own,
and are never compared with the base model's.

## What a run computes

The bundle carries the table, `[vocab_size, dim]`, and a weight per
entry, `[vocab_size]`, in F32, F16 or BF16, both in the same dtype. A row
is tokenized with the bundle's template, which for a static model is
`$TEXT` alone: no special tokens. Then, by `pooling`:

- `POOLING_MEAN`: each token's row times its weight, averaged over the
  tokens whose weight is not 0. A weight of 0 leaves the token out of
  the mean and out of the count: the unknown token has 0, as Model2Vec
  leaves it out.
- `POOLING_CLS`: the row's first column times its weight.
- `POOLING_LAST`: the row's last token times its weight.

A row with no token left is the zero vector, under every pooling, and
stays zero under `NORMALIZE_L2`. A text that gives no tokens (the empty
text, or only whitespace) is such a row, not an error: an encoder
refuses it with `INVALID_ARGUMENT`, a static model returns zeros. Rows
written with `turbo_embed_write_tokens` still need one live token each.

`output_dim` cuts the vector before it is normalized, as for an encoder.
The table's columns are principal components in order of the variance
they carry, so a cut keeps the most of it.

Every precision computes in F32, from the table as stored: the same
vectors at `PRECISION_MODEL`, `FASTEST` and `EXACT`.
`turbo_result_info.stage` reports `LOOKUP` on the host, `ENCODE` unused
and `POOL` fused into the lookup.

On the CPU each row is summed by one task in token order, so a batch
gives the same bits on any number of threads (docs/cpu.md). The rows of
a run are split over the session's threads once a batch holds a few
thousand tokens. `turbo_embed_write_text` tokenizes a batch of 64 texts
or more on one thread per processor; at that point the tokenizer, not
the sum, is most of a run.

## The bundle

A static bundle has a `static_embedding` block instead of
`architecture` (docs/bundle.md):

```json
"static_embedding": {
  "vocab_size": 30522,
  "distilled_from": {
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": "2",
    "manifest_sha256": "<the base bundle's manifest hash>"
  },
  "quality": {
    "texts": "quality/texts.jsonl",
    "base_top1": 0.9,
    "static_top1": 0.8,
    "similarity_spearman": 0.85
  }
}
```

Its one artifact is `FORMAT_SAFETENSORS` with the `static_embeddings`
and `static_weights` tensor roles. `embed.dim` is the table's width.

`quality` is the measured cost, on the texts the bundle carries: groups
of texts that say the same thing in other words.

- `base_top1`, `static_top1`: the share of texts whose nearest other
  text, by cosine, is of their group, with the base model and with the
  static one.
- `similarity_spearman`: Spearman's rank correlation between the two
  models' cosines over every pair of texts. 1 is the same order.

`distilled_from` names the base bundle by its manifest hash, so the
numbers are tied to the exact model they were measured against.

## Making one

`turbo-bundle distill <recipe.json> <base-bundle-dir> <bundle-dir>`
distils a static bundle from a sealed bundle of the base model. The
recipe is a manifest with `static_embedding` (without `distilled_from`
and the numbers, which the tool fills), the quality texts among its
`local` files, and a `distill` block:

```json
"distill": {
  "pca_dims": 256,
  "sif_coefficient": 0.0001,
  "skip": "\\[unused\\d+\\]",
  "dtype": "F16"
}
```

The tokenizer block must be the base bundle's with the template
`["$TEXT"]`. The steps:

1. **stage**: the base bundle's encoder, run by this library on the CPU,
   embeds every vocabulary entry alone, between the base template's
   special tokens, mean-pooled with no normalization. Entries `skip`
   matches whole are left out. The rest are centred and projected on
   the leading `pca_dims` eigenvectors of their covariance, then scaled
   by Zipf's law: the entry of rank r (in id order) is taken to occur
   with probability proportional to 1 / (r + 2), and its row is scaled
   by `sif / (sif + p)`. The weights are 1, and 0 for the unknown token
   and every entry left out. The table is rounded to `dtype` and written,
   and the quality texts are embedded by both models.
2. **reference**: Model2Vec's `StaticModel`, holding the table in F32
   with the bundle's own tokenizer file, embeds the reference cases in
   the pinned reference container (`static_reference.py`). It is checked
   against `StaticModel.encode` on the same texts.
3. **seal**: the manifest is filled and parsed by the core, and the
   library's vectors on the CPU are checked against the reference and
   against the tool's own on the quality texts.

`distill-stage` and `distill-seal` run steps 1 and 3 alone.

The table is the base model's, so it keeps the base model's licence.

## References

The speed references, on the same machine and the same table: Model2Vec's
own `StaticModel.encode`, and sentence-transformers' `StaticEmbedding`
(a PyTorch `EmbeddingBag` in mean mode). `StaticEmbedding` keeps the
unknown token in the mean; Model2Vec and this library leave it out.
