# turbo-trainer-recipes

The corpora `turbo-trainer` heads train on, the scoring every run
reports, and the `turbo-train-head` command (#1518, stage R1b).

| Corpus id | Data | Task | Pinned by | Splits |
|---|---|---|---|---|
| `masc-pos` | MASC 3.0.0, Penn tags (CC BY 3.0 US) | tags | corpus digest of the extracted release (`turbo_train_recipes::masc::RELEASE`) | test: the recipes' held-out documents (every tenth, Twitter genre left out), so numbers compare with the MASC tagger's gate; dev: every tenth training-side document |
| `multiconer2-en` | MultiCoNER v2 English (CC BY-SA 4.0: heads trained on it are share-alike, download only after the user accepts, never in a default bundle; docs/licences.md) | BIO, 33 types in 6 groups | size and SHA-256 of each file | the release's own train, dev and test |

A directory that is not the recorded release is refused unless the run
passes `--unrecorded-corpus`, and its receipt then says
`"recorded": false`. Neither corpus is checked in.

Each corpus records its licence's SPDX id, and the head a run writes
carries that licence's class by the model zoo's rules as
`license_class` (`CorpusRecord::class`): a MASC head is
`CLASS_PERMISSIVE`, a MultiCoNER head `CLASS_SHARE_ALIKE`, so a bundle
holding it is share-alike whatever its encoder (`Manifest::license_tier`)
and is fetched only after the user accepts. A corpus licence the zoo does
not know or refuses makes no head.

Every run scores the test split against the majority baseline (each
word's commonest training label, the commonest label overall for an
unseen word). Tags score by word accuracy, also over unseen words;
BIO by exact span F1, per type and per coarse group, spans read the
conlleval way. A trained head is compared with the baseline by a paired
bootstrap over sentences (1,000 resamples, seeded).

```text
turbo-train-head check    --corpus <id> --dir <corpus dir>
turbo-train-head baseline --corpus <id> --dir <corpus dir> --out <new dir>
turbo-train-head train    --corpus <id> --dir <corpus dir> --bundle <encoder bundle> --out <new dir>
    [--device cpu|cuda|levelzero|metal] [--precision model|exact] [--pooling first|mean]
    [--crf] [--bilstm <hidden>x<layers>] [--seed N] [--epochs N] [--batch N] [--lr X]
    [--patience N] [--max-train N] [--commit <sha>]
turbo-train-head cache    --corpus <id> --dir <corpus dir> --bundle <encoder bundle> --out <new dir>
    --layers N [--device ..] [--precision ..] [--pooling first|mean] [--max-batch N]
    [--splits train,dev,test] [--commit <sha>]
turbo-train-head fit-cache --corpus <id> --dir <corpus dir> --cache <cache dir> --out <new dir>
    (--layer N | --mix [--from-layer N]) [--crf] [--bilstm ..] [--hash-embed ..] [training settings]
```

`cache` (#1616) reads every split once and writes each word's features
at the encoder's last N layers, so heads on any one layer, or on a
learned mix of them, train without the encoder: `cache.json`
(`cache::CacheRecord`: command, commit, corpus record, encoder, pooling,
the layers counted from 1, and per split its sentences, words and files)
and `<split>.layer<NN>.f32`, [words, hidden] little-endian f32 in corpus
order. `cache::read_layer` reads one split at one layer back, checked
against the corpus's sentences and the file size. The layers come from
the core's per-layer taps (`turbo_embed_write_layer_ranges`): CPU, BERT,
RoBERTa, ModernBERT and nomic encoders. A run holds a window's words at
every layer, so `--max-batch` is raised to N when lower. A cache is
[words, hidden, layers] f32: MultiCoNER v2 English's train and dev
(some 250,000 words) at all 22 layers of ModernBERT-base is about 17 GB,
and its 249,980-sentence test split many times that, so `--splits`
names the splits to read (all three by default).

`fit-cache` trains a head on a cache's train split and scores it on its
dev split, without the encoder: on one layer (`--layer N`; on the last
layer it is the run `train` makes, epoch for epoch), or on a learned mix
of the cached layers from `--from-layer` on (`--mix`: ELMo's scalar
mix, `turbo_trainer::mix`, a softmax weight a layer and a gain, trained
with the head). It writes `head.safetensors`, `head.json` and
`fit.json` (`fit_cache::FitRecord`: the cache's record, the layers, the
mix's weights and gain, the head, the settings, and the loss and dev
score each epoch). Comparing the best single layer's dev score with the
mix's is #1616's step 2; the winner is then scored on test through the
encoder.

A head with a BiLSTM adapter waits 15 epochs without progress before
it stops (`--patience`, 6 for other heads): its dev score creeps up for
many epochs, and each better epoch is saved as it comes.

`train` reads per-word features from the encoder through
`turbo_tagger::WordEncoder`, the path the annotators use, trains with
`turbo_trainer::fit`, predicts the test split through the core's
`TokenTagger`, and writes `head.safetensors` (the head's F32 tensors
under `heads.<name>.`), `head.json` (its `HeadSpec`) and `receipt.json`
(`receipt::Receipt`: command, commit, host, corpus record with digests
and counts, encoder model and manifest digest, backend and device, head,
settings, per-epoch loss and dev score, test scores of model and
baseline, the bootstrap, timings, and the outputs' digests).

Majority baselines on the test splits (CPU, 2026-10-28, at the recorded
releases): MASC Penn POS accuracy 0.8964 (85,292 words), MultiCoNER v2
English span F1 0.1404 (377,805 gold spans).

```text
cargo test -p turbo-trainer-recipes
TURBO_TEST_MASC_DIR=<MASC-3.0.0> TURBO_TEST_MULTICONER_DIR=<multiconer2023> \
    cargo test --release -p turbo-trainer-recipes
```

Not yet here: bundle output (a sealed bundle carrying the head),
a core head that reads a mix of layers (only if the mix wins, #1616),
WNUT-17 and MASC named entities.
