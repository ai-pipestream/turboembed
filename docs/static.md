# Static models

A static model has no encoder. It holds one vector per vocabulary
entry, and a text's vector is the mean of its tokens' vectors: a
tokenizer, a gather and a sum. That is Model2Vec's `StaticModel`, and
sentence-transformers' `StaticEmbedding`. It is many times faster than
the model it was distilled from, on any CPU, and less accurate.

A static model is its own model, with its own `model.id`, not a
precision of a base model: its vectors live in a space of their own,
and are never compared with the base model's.

The library computes what Model2Vec's `StaticModel` computes, at the
commit of its repository the reference image pins
(`bundle/reference/requirements.txt`): the same ids for every text, and
the same vectors to the bit.

## What a run computes

A text becomes ids as `StaticModel` makes them, with `max_tokens` (the
bundle's `static_embedding.max_length` under `TRUNCATE_MODEL`, 512 for
Model2Vec's models):

1. Under `TRUNCATE_RIGHT` or `TRUNCATE_MODEL`, the text is cut to
   `max_tokens` times the vocabulary's median entry length, in
   characters (code points; the median of every entry's length, rounded
   down). The cut can fall inside a word, which then gives the tokens
   its first part gives.
2. The text is tokenized with the bundle's template, which for a static
   model is `$TEXT` alone: no special tokens are added. Special tokens
   typed in the text are matched, as upstream matches them.
3. The ids are cut to `max_tokens`, on the right, or on the left under
   `TRUNCATE_LEFT`.
4. Every unknown token is dropped. So `max_tokens` counts the unknown
   tokens too: a text whose first `max_tokens` tokens are unknown gives
   no ids.

Under `TRUNCATE_NONE` steps 1 and 3 are skipped, and a text with more
ids than `max_tokens` is refused with `TURBO_E_CAPACITY`.

Then, by `pooling`:

- `POOLING_MEAN`: each token's row of the table (the row the token
  mapping names, when the bundle has one) times the token's weight (when
  it has weights), summed in token order and divided by the number of
  ids.
- `POOLING_CLS`: the row's first column alone, times its weight.
- `POOLING_LAST`: the row's last id alone, times its weight.

A text with no ids is the zero vector, under every pooling, and stays
zero under `NORMALIZE_L2`: the empty text, whitespace, or only unknown
characters. An encoder refuses such a text with `INVALID_ARGUMENT`; a
static model returns zeros. Rows written with `turbo_embed_write_tokens`
still need one live token each.

`output_dim` cuts the vector before it is normalized, as for an encoder.
A distilled table's columns are principal components in order of the
variance they carry, so a cut keeps the most of it. `NORMALIZE_L2`
divides by the norm plus 1e-32, as `StaticModel` does.

### The arithmetic, to the bit

The sum, the mean and the norm are numpy's, in the dtypes numpy uses for
the table as stored:

- An F32, F16 or BF16 table: the rows summed in F32, one token after
  another, each value times its weight first in its own rounding, then
  divided by the count in F32.
- An F64 or I8 table, or F64 weights: summed and divided in F64, then
  rounded to F32. An I8 value is the number it holds.
- An F16 table: the mean rounded to F16, as numpy stores it, and the
  normalized vector rounded to F16 again.
- The norm: the squares in F32, summed pairwise as numpy's `add.reduce`
  sums a row (eight running sums over blocks of up to 128 values, halves
  above that), then the square root.

`PRECISION_MODEL` and `EXACT` compute this and give the same bits.
`turbo_result_info.stage` reports `LOOKUP` on the host, `ENCODE` unused
and `POOL` fused into the lookup.

### FASTEST: the table in I8

`PRECISION_FASTEST` sums a copy of the table in I8, and the session
reports the compute dtype `I8`. Each row is stored as its largest
magnitude over 127, an F32 scale, and its values divided by that scale
and rounded to the nearest integer. A token adds its row's I8 values
times the scale (times its weight) to F32 sums; the mean and the norm
are as above, with no F16 rounding. The copy is a quarter of an F32
table's bytes; the model makes it for its first FASTEST session and
keeps it for the others.

The cost is measured against Model2Vec's own vectors for every text the
parity test reads (`core/tests/static_parity.rs`), and held to a cosine
of at least 0.999. The vectors are not bit-for-bit those of another
processor, since the AVX2 sums fuse the multiply and the add.
`TURBO_CPU_STATIC_TABLE` forces either table (docs/cpu.md).

On the CPU each row is summed by one task in token order, so a batch
gives the same bits on any number of threads (docs/cpu.md). The rows of
a run are split over the session's threads once a batch holds a few
thousand tokens. `turbo_embed_write_text` tokenizes a batch on up to
one thread per processor, one thread for each 4 KiB of text.

## Beyond the reference

What the library guarantees that `StaticModel` at that commit does not,
or does only in part:

- A vector never depends on the other texts of its batch. Older
  Model2Vec padded a batch with the tokenizer file's padding and averaged
  the pad token in, and promoted a whole batch to F64 when one of its
  texts had no tokens; the library does neither, and neither does the
  pinned commit.
- A text's work is bounded by `max_tokens`: under `TRUNCATE_MODEL` the
  text is cut before it is tokenized, so a megabyte of text costs what
  its first few thousand characters cost.
- `TRUNCATE_LEFT` keeps a text's last tokens, and `TRUNCATE_NONE`
  refuses a text over `max_tokens` instead of cutting it.
- `CLS` and `LAST` pooling, and `output_dim`, which suits a table
  trained at several widths (Matryoshka).
- An id past the table, or a mapping value past its rows, is refused
  when the bundle is loaded, not when a text reaches it.
- Every file of the bundle is hashed in its manifest and checked when it
  is loaded, and the tokenizer is checked against the reference's ids.
- `PRECISION_FASTEST` sums the table in I8, a quarter of an F32 table's
  memory traffic, at a cost measured against Model2Vec's own vectors.
- The table is mapped from its file and read in place in its stored
  dtype: an F16 table stays F16 in memory, converted eight values at a
  time as it is summed.
- A model is fetched at a pinned commit with every file's SHA-256
  checked, only when a command names it, and its provenance is written
  beside the files.

Older Model2Vec differs from the pinned commit, and so from the
library, in ways a caller comparing against it may see: 0.9.0 cuts to
`max_length` after it drops the unknown token rather than before, keeps
the unknown token of a Unigram tokenizer, applies the tokenizer file's
own truncation even with no `max_length`, and normalizes an F16 table in
F16.

## Model2Vec's models

`turbo-bundle` makes a bundle from each of Model2Vec's potion models by
name:

```
turbo-bundle catalogue
turbo-bundle make minishlab/potion-base-8M <upstream-dir> <bundle-dir>
```

Each name is a recipe in `bundle/recipes/potion/`, pinned to a commit of
the model's repository with the SHA-256 of each file it fetches: the
tokenizer, the table (`model.safetensors`, unchanged), `config.json` and
the model card. Nothing is downloaded until `fetch` or `make` names the
model, and files already in `<upstream-dir>` with the pinned hashes are
used as they are. No model's weights are in this repository or in its
release archives.

`fetch` prints the model's licence, and any terms the model carries
beyond it, before it downloads: every potion model is MIT, and
`potion-retrieval-32M` was fine-tuned on MS MARCO, whose terms allow
non-commercial use only. A model with such terms is fetched only with
`--accept-terms`, given once they have been read:

```
turbo-bundle fetch --accept-terms minishlab/potion-retrieval-32M <upstream-dir>
```

Beside the files, `fetch` writes `turbo-fetch.json`: the repository,
the commit, the licence, the terms and whether they were accepted, and
the SHA-256 of every file.

The reference step builds `StaticModel` from the upstream files in the
pinned reference image, and checks that `StaticModel.from_pretrained`
gives the same vectors. The bundle's `max_length` and normalization must
be the ones the model's `config.json` gives (512 when it gives none).

| Model | Tokenizer | Table |
|---|---|---|
| potion-base-2M, 4M, 8M | WordPiece, 29,528 entries | F32, 64, 128 and 256 wide |
| potion-base-32M, potion-retrieval-32M | WordPiece, 63,091 entries | F32, 512 wide |
| potion-science-32M | WordPiece, 124,428 entries | F32, 256 wide |
| potion-code-16M | WordPiece, 61,826 entries, `[PAD]` matched normalized | F32 behind a token mapping, F64 weights |
| potion-code-16M-v2 | WordPiece, 63,457 entries, `[PAD]` matched normalized | F16, 256 wide |
| potion-multilingual-128M | Unigram, 500,353 entries | F32, 256 wide |

`core/tests/static_parity.rs` holds such a bundle to the model's own
ids and vectors, written by `bundle/reference/static_golden.py` in the
reference image for a texts file and a list of edge cases, at the
model's `max_length` and at none. `<reference-image>` is the
image built from `bundle/reference/Dockerfile`, by its tag or its ID;
the recipes pin its config digest, which `docker run` does not take as
a reference:

```
docker run --rm --network none --entrypoint python \
  --mount type=bind,src=<upstream-dir>,dst=/model,readonly \
  --mount type=bind,src=<golden-dir>,dst=/golden \
  <reference-image> /static_golden.py /model /golden/texts.jsonl /golden/out
TURBO_PARITY_BUNDLE=<bundle-dir> TURBO_PARITY_GOLDEN=<golden-dir>/out \
  cargo test --release -p turbo --test static_parity
```

## The bundle

A static bundle has a `static_embedding` block instead of
`architecture` (docs/bundle.md):

```json
"static_embedding": {
  "vocab_size": 30522,
  "max_length": 512,
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

The table is mapped from its file where the host can map files (Linux
and macOS), not copied: the model's pages are the page cache's, shared
by every process that loads the same bundle. Every byte is still hashed
against the manifest when the model is loaded, so the files must not
change while a model is loaded from them.

Its one artifact is `FORMAT_SAFETENSORS` with the `static_embeddings`
tensor role, and `static_weights` and `static_mapping` when the table
has them (with `rows`, the table's height). `embed.dim` is the table's
width.

`distilled_from` and `quality` are there for a table this tool
distilled, and absent for a model made elsewhere. `quality` is the
measured cost, on the texts the bundle carries: groups of texts that say
the same thing in other words.

- `base_top1`, `static_top1`: the share of texts whose nearest other
  text, by cosine, is of their group, with the base model and with the
  static one.
- `similarity_spearman`: Spearman's rank correlation between the two
  models' cosines over every pair of texts. 1 is the same order.

`distilled_from` names the base bundle by its manifest hash, so the
numbers are tied to the exact model they were measured against.

## Distilling one

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
   matches whole are left out, and so are the special tokens a text
   never gives (every one but the padding and the unknown token), as
   Model2Vec leaves them out. The rest are centred and projected on the
   leading `pca_dims` eigenvectors of their covariance, then scaled by
   Zipf's law: the entry of rank r (in id order) is taken to occur with
   probability proportional to 1 / (r + 2), and its row is scaled by
   `sif / (sif + p)`. An entry left out keeps a row of zeros. The table
   is rounded once to `dtype` and written alone, with no weights, and
   the quality texts are embedded by both models.
2. **reference**: Model2Vec's `StaticModel`, built from the table as
   stored and the bundle's tokenizer file, gives the reference ids and
   vectors in the pinned reference container (`static_reference.py`).
3. **seal**: the manifest is filled and parsed by the core, and the
   library's vectors on the CPU must be the reference's to the bit, and
   close to the tool's own on the quality texts.

`distill-stage` and `distill-seal` run steps 1 and 3 alone.

The table is the base model's, so it keeps the base model's licence.

## References

The speed references, on the same machine and the same table: Model2Vec's
own `StaticModel.encode` at the pinned commit, model2vec-rs (Model2Vec's
Rust port), and sentence-transformers' `StaticEmbedding` (a PyTorch
`EmbeddingBag` in mean mode). `StaticEmbedding` keeps the unknown token
in the mean and neither cuts the text nor its ids; Model2Vec,
model2vec-rs and this library drop it and cut both.

`core/tests/static_speed.rs` times a bundle in texts a second, the way
each reference is timed beside it: a JSON array of texts embedded in
order, batch by batch, at `max_length` with normalization, one warm pass
and then at least two seconds three times, the best of the three, at
`PRECISION_MODEL` and `FASTEST`, for each batch size up to the bundle's
`max_batch`:

```
TURBO_SPEED_BUNDLE=<bundle-dir> TURBO_SPEED_TEXTS=<texts.json> TURBO_SPEED_BATCHES=1,32,256,1024 \
  cargo test --release -p turbo --test static_speed -- --nocapture
```

The potion bundles take batches of up to 1024 texts, Model2Vec's default
batch.
