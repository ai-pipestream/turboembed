# Static models

A static model has no encoder. It holds one vector per vocabulary
entry, and a text's vector is the mean of its tokens' vectors: a
tokenizer, a gather and a sum. It is many times faster than the model
it was distilled from, on any CPU, and less accurate.

A static model is its own model, with its own `model.id`, not a
precision of a base model: its vectors live in a space of their own,
and are never compared with the base model's.

This is the library's own implementation, in Rust, with no Python
anywhere in it or in making its bundles. It reads the files Model2Vec
writes (a safetensors table, with token weights and a token mapping
when the model has them, and a `tokenizer.json`), and this page is its
specification. Where Model2Vec's `StaticModel` is correct, the library
gives the same ids and the same vectors to the bit; where it is not,
the library differs on purpose ([Fixed in the library](#fixed-in-the-library)).

## What a run computes

A text becomes ids by these steps, with `max_tokens` (the bundle's
`static_embedding.max_length` under `TRUNCATE_MODEL`, 512 for the
potion models):

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

`output_dim` cuts the vector before it is normalized, as for an
encoder: a cut is served at the width the bundle declares or at the
table's own width, and no other (`UNSUPPORTED_OPTION`, naming the
declared widths; docs/bundle.md). A distilled table's columns are
principal components in order of the variance they carry, so a cut
keeps the most of it; a table made for several widths declares them in
the recipe's `matryoshka` field, as a dense model does.
`NORMALIZE_L2` divides by the norm plus 1e-32, so the zero vector
stays zero.

### The arithmetic, to the bit

The sum, the mean and the norm run in a fixed order and in fixed dtypes,
chosen by the table as stored, so a vector's bits depend on nothing but
its text. They are the order and the dtypes numpy uses, so a correct
numpy implementation gives the same bits:

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

The cost is measured against the exact vectors for every text the
parity test reads (`core/tests/static_parity.rs`), and held to a cosine
of at least 0.999. The vectors are not bit-for-bit those of another
processor, since the AVX2 sums fuse the multiply and the add.
`TURBO_CPU_STATIC_TABLE` forces either table (docs/cpu.md).

On the CPU each row is summed by one task in token order, so a batch
gives the same bits on any number of threads (docs/cpu.md). The rows of
a run are split over the session's threads once a batch holds a few
thousand tokens. `turbo_embed_write_text` tokenizes a batch of more
than 1 KiB of text in tasks of about 1 KiB on the process's tokenizing
threads, one per processor, which wait between batches.

## Fixed in the library

Where the library deliberately differs from Model2Vec's `StaticModel`
(0.9.0, its latest release, and its repository's main branch), because
Model2Vec has a bug there or no guarantee:

- **A vector never depends on the other texts of its batch.** The
  tokenizer file's own padding and truncation are never applied, so no
  pad token is averaged in; Model2Vec 0.9.0 applies them, and the
  potion code models' files pad and truncate at 512. A text with no
  tokens is the zero vector in F32 and changes nothing else in its
  batch; Model2Vec 0.9.0 returns it in F64 and turns the whole batch to
  F64.
- **A Unigram tokenizer's unknown token is dropped**, as a WordPiece
  one is; Model2Vec 0.9.0 averages it in.
- **Truncation counts the tokens the text gives**, before the unknown
  ones are dropped, the same for every tokenizer; Model2Vec 0.9.0 cuts
  after dropping them, and applies the file's own truncation even with
  no `max_length`.
- **A long text's work is bounded.** Under `TRUNCATE_MODEL` the text is
  cut before it is tokenized, so a megabyte of text costs what its first
  few thousand characters cost.
- **Normalization runs in F32** for an F16 table, the result rounded to
  F16 once; Model2Vec 0.9.0 normalizes in F16.

And what the library adds:

- `TRUNCATE_LEFT` keeps a text's last tokens, and `TRUNCATE_NONE`
  refuses a text over `max_tokens` instead of cutting it.
- `CLS` and `LAST` pooling, and `output_dim`, which suits a table
  trained at several widths (Matryoshka).
- An id past the table, or a mapping value past its rows, is refused
  when the bundle is loaded, not when a text reaches it.
- Every file of the bundle is hashed in its manifest and checked when it
  is loaded, and the tokenizer is checked against the reference's ids.
- `PRECISION_FASTEST` sums the table in I8, a quarter of an F32 table's
  memory traffic, at a stated cost.
- The table is mapped from its file and read in place in its stored
  dtype: an F16 table stays F16 in memory, converted eight values at a
  time as it is summed.
- A model is fetched at a pinned commit with every file's SHA-256
  checked, only when a command names it, and its provenance is written
  beside the files.

## The potion models

`turbo-bundle` makes a bundle from each of the potion models Minish
publishes in Model2Vec's format, by name:

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

`make` checks the model's `config.json` against the recipe (its
`normalize`, and its `max_length`, 512 when it gives none), copies the
table and the tokenizer into the bundle unchanged, and writes the
reference itself: each case's ids from the bundle's tokenizer and its
vector from the table as stored, by the rules above, in plain code
apart from the library's. The library must then give every case to the
bit, alone and in a batch. No container and no Python run.

| Model | Tokenizer | Table |
|---|---|---|
| potion-base-2M, 4M, 8M | WordPiece, 29,528 entries | F32, 64, 128 and 256 wide |
| potion-base-32M, potion-retrieval-32M | WordPiece, 63,091 entries | F32, 512 wide |
| potion-science-32M | WordPiece, 124,428 entries | F32, 256 wide |
| potion-code-16M | WordPiece, 61,826 entries, `[PAD]` matched normalized | F32 behind a token mapping, F64 weights |
| potion-code-16M-v2 | WordPiece, 63,457 entries, `[PAD]` matched normalized | F16, 256 wide |
| potion-multilingual-128M | Unigram, 500,353 entries | F32, 256 wide |

### Measured against Model2Vec

Model2Vec is a yardstick for tests, never part of making a bundle.
`bundle/model2vec-golden/` builds an image with Model2Vec at a pinned
commit of its repository, whose `golden.py` writes a model's ids and
vectors as `StaticModel` gives them, for a texts file and a list of
edge cases, at the model's `max_length` and at none.
`core/tests/static_parity.rs` holds a bundle to them: every text's ids
exactly and every vector to the bit, and FASTEST to its cosine floor.

```
docker build -t turbo-model2vec-golden bundle/model2vec-golden
docker run --rm --network none \
  --mount type=bind,src=<upstream-dir>,dst=/model,readonly \
  --mount type=bind,src=<golden-dir>,dst=/golden \
  turbo-model2vec-golden /model /golden/texts.jsonl /golden/out
TURBO_PARITY_BUNDLE=<bundle-dir> TURBO_PARITY_GOLDEN=<golden-dir>/out \
  cargo test --release -p turbo --test static_parity
```

All nine potion models agree with `StaticModel` at main's commit
3ef2bf2 on every text of such a run, ids and vectors.

CI runs this check on every pull request (the `static-parity` lane of
`.github/workflows/ci.yml`): it makes the potion-base-2M bundle from the
recipe's pinned files, builds the golden image (its base pinned by
digest), writes the goldens for `testdata/tokenizer-texts.jsonl` with no
network, and runs `static_parity` against them. The model files are
cached under a key made from the recipe, so they are fetched again only
when its digests change, and the image's layers are cached by content.

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

`skip` accepts only `PREFIX\d+SUFFIX`: literal edges and one non-empty
run of ASCII digits. Escape ASCII punctuation with a backslash (for
example `\[` or `\.`). Other regex constructs are errors that name the
pattern. Matching covers the entire token; non-ASCII digits do not match.

The tokenizer block must be the base bundle's with the template
`["$TEXT"]`. The steps:

1. **stage**: the base bundle's encoder, run by this library on the CPU,
   embeds every vocabulary entry alone, between the base template's
   special tokens, mean-pooled with no normalization. Entries `skip`
   matches whole are left out, and so are the special tokens a text
   never gives (every one but the padding and the unknown token). The rest are centred and projected on the
   leading `pca_dims` eigenvectors of their covariance, then scaled by
   Zipf's law: the entry of rank r (in id order) is taken to occur with
   probability proportional to 1 / (r + 2), and its row is scaled by
   `sif / (sif + p)`. An entry left out keeps a row of zeros. The table
   is rounded once to `dtype` and written alone, with no weights, and
   the quality texts are embedded by both models.
2. **reference**: the tool writes the reference ids and vectors from
   the table as stored, as `make` does.
3. **seal**: the manifest is filled and parsed by the core, and the
   library's vectors on the CPU must be the reference's to the bit, and
   close to the tool's own F64 arithmetic on the quality texts.

Every step runs in this tool and the library: no container, no Python.

The table is the base model's, so it keeps the base model's licence.

## Speed

`turbo-bench static` times a bundle on the CPU and each program of
`bench/static-peers` beside it, on the same machine and the same texts,
and writes the result under `benchmarks/static/`. These records are not
compiled into the library and decide no capability cell: they compare
the library with other programs and say nothing about a device's support.
Each text set is embedded in order, batch by batch, at `max_length` with
L2 normalization, one warm pass and then at least two seconds three
times, the best of the three. Each peer also writes its vectors for the
first set, which the record compares with the library's.

```
bench/static-peers/peers.sh setup          # MODELS_DIR, GOLDEN_DIR as its README says
cargo build --release -p turbo-bench
target/release/turbo-bench static --bundle <bundle-dir> \
  --texts golden=<W>/texts/golden.json,t60=<W>/texts/t60.json,w256=<W>/texts/w256.json \
  --precision model --peers <W>
target/release/turbo-bench static --bundle <bundle-dir> --texts ... --precision fastest --no-peers
```

The committed records were made on 2026-10-24 on an AMD Ryzen 9 9950X
(16 cores, 32 threads), Linux: MODEL with every peer at commit
`35a2d4f7`, FASTEST with the library alone at `d95afdbd`. A vLLM engine
held one core (CPU 0) at 100% throughout, which takes from the batches
that use every thread more than from batch 1. The text sets come from
the golden texts of `testdata/tokenizer-texts.jsonl` (88 texts with the
edge cases), by `bench/static-peers/texts.py`: `golden`, `t60` (2000
texts of 45 words) and `w256` (1000 chunks of 256 words).
potion-retrieval-32M is not recorded: fetching it needs its MS MARCO
terms accepted, which has not been done.

Texts a second on `t60`, one text a call:

| Model | turbo MODEL | turbo FASTEST | Model2Vec | model2vec-rs | StaticEmbedding | statembed | statembed+rayon | go-potion | model2vec-zig |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| potion-base-2M | 418k | 413k | 18k | 31k | 7.5k | 218k | 216k | 248k | 148k |
| potion-base-4M | 404k | 385k | 18k | 31k | 7.1k | 215k | 213k | 244k | 141k |
| potion-base-8M | 374k | 362k | 18k | 31k | 7.4k | 190k | 195k | 210k | 116k |
| potion-base-32M | 320k | 325k | 17k | 29k | 7.2k | 166k | 159k | 198k | 90k |
| potion-science-32M | 370k | 364k | 16k | 30k | 7.2k | 192k | 194k | 222k | 117k |
| potion-code-16M | 262k | 333k | 16k | 29k | 7.1k | 194k | 192k | 131k | - |
| potion-code-16M-v2 | 344k | 354k | 14k | 30k | 7.2k | 187k | 191k | 244k | - |
| potion-multilingual-128M | 173k | 172k | 12k | 19k | 6.3k | 111k | 109k | 2.0k | - |

On `t60`, 1024 texts a call:

| Model | turbo MODEL | turbo FASTEST | Model2Vec | model2vec-rs | StaticEmbedding | statembed | statembed+rayon | go-potion | model2vec-zig |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| potion-base-2M | 2711k | 2652k | 133k | 318k | 96k | 889k | 847k | 915k | 844k |
| potion-base-4M | 2602k | 2621k | 124k | 310k | 95k | 762k | 829k | 912k | 830k |
| potion-base-8M | 2483k | 2479k | 91k | 314k | 91k | 569k | 810k | 873k | 783k |
| potion-base-32M | 2059k | 2197k | 66k | 289k | 83k | 373k | 774k | 874k | 639k |
| potion-science-32M | 2443k | 2324k | 79k | 288k | 84k | 559k | 799k | 1194k | 728k |
| potion-code-16M | 2021k | 2370k | 59k | 158k | 87k | 561k | 802k | 760k | - |
| potion-code-16M-v2 | 2132k | 2445k | 47k | 153k | 92k | 562k | 702k | 2101k | - |
| potion-multilingual-128M | 1716k | 1746k | 87k | 248k | 90k | 404k | 535k | 27k | - |

On `w256`, 1024 texts a call:

| Model | turbo MODEL | turbo FASTEST | Model2Vec | model2vec-rs | StaticEmbedding | statembed | statembed+rayon | go-potion | model2vec-zig |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| potion-base-2M | 766k | 749k | 30k | 84k | 25k | 267k | 354k | 224k | 235k |
| potion-base-4M | 747k | 741k | 27k | 83k | 25k | 218k | 345k | 226k | 226k |
| potion-base-8M | 714k | 706k | 23k | 82k | 24k | 152k | 328k | 225k | 206k |
| potion-base-32M | 631k | 640k | 15k | 74k | 23k | 91k | 292k | 233k | 165k |
| potion-science-32M | 694k | 699k | 18k | 80k | 24k | 149k | 319k | 292k | 199k |
| potion-code-16M | 542k | 672k | 14k | 55k | 23k | 152k | 266k | 200k | - |
| potion-code-16M-v2 | 611k | 693k | 9.0k | 61k | 24k | 150k | 285k | 484k | - |
| potion-multilingual-128M | 416k | 418k | 17k | 58k | 23k | 110k | 188k | 2.9k | - |

model2vec-zig reads WordPiece tables in F32 or I8 only, so it is not run
on the code and multilingual models, and its batches above 1 split the
texts over one thread per CPU. Each record has every set at batches 1,
32, 256 and 1024.

Each peer's vectors for the 88 golden texts against the library's (which
are Model2Vec's to the bit): rows equal to the bit, and the lowest
cosine. The lowest cosine is one row's; a record keeps the counts, not
the rows, and `peers.sh accuracy` prints the rows under 0.999 with their
texts.

| Model | Model2Vec | model2vec-rs | StaticEmbedding | statembed | statembed+rayon | go-potion | model2vec-zig |
|---|---|---|---|---|---|---|---|
| potion-base-2M | 88/88, 1.000000 | 46/88, 1.000000 | 48/88, 0.742468 | 47/88, 0.896444 | 47/88, 0.896444 | 16/88, 0.896444 | 13/88, 0.140055 |
| potion-base-4M | 88/88, 1.000000 | 43/88, 1.000000 | 52/88, 0.742119 | 45/88, 0.880880 | 45/88, 0.880880 | 70/88, 0.880880 | 13/88, 0.055388 |
| potion-base-8M | 88/88, 1.000000 | 33/88, 1.000000 | 47/88, 0.709418 | 32/88, 0.877154 | 32/88, 0.877154 | 70/88, 0.877154 | 13/88, 0.089679 |
| potion-base-32M | 88/88, 1.000000 | 27/88, 1.000000 | 46/88, 0.589825 | 28/88, 0.896862 | 28/88, 0.896862 | 66/88, 0.896862 | 13/88, 0.010651 |
| potion-science-32M | 88/88, 1.000000 | 30/88, 1.000000 | 47/88, 0.861295 | 31/88, 0.892953 | 31/88, 0.892953 | 69/88, 0.892953 | 13/88, 0.199497 |
| potion-code-16M | 88/88, 1.000000 | 1/88, 0.059055 | 8/88, 0.364622 | 14/88, 0.540645 | 14/88, 0.540645 | 13/88, 0.059713 | not run |
| potion-code-16M-v2 | 88/88, 1.000000 | 1/88, 0.902368 | 0/88, 0.677058, 8 refused | 14/88, 0.400679 | 14/88, 0.400679 | 13/88, 0.400679 | not run |
| potion-multilingual-128M | 88/88, 1.000000 | 22/88, 1.000000 | 59/88, -0.059411 | 21/88, -0.059411 | 21/88, -0.059411 | 65/88, -0.059411 | not run |

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
