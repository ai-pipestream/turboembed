# turbo-trainer-recipes

The corpora `turbo-trainer` heads train on, the scoring every run
reports, and the `turbo-train-head` command (#1518, stage R1b).

| Corpus id | Data | Task | Pinned by | Splits |
|---|---|---|---|---|
| `masc-pos` | MASC 3.0.0, Penn tags (CC BY 3.0 US) | tags | corpus digest of the extracted release (`turbo_train_recipes::masc::RELEASE`) | test: the recipes' held-out documents (every tenth, Twitter genre left out), so numbers compare with the MASC tagger's gate; dev: every tenth training-side document |
| `wnut17` | WNUT-17 (CC BY 4.0, default-tier training under #1524) | BIO, six types kept separate from MASC | size and SHA-256 of each file | publisher train/dev/test, canonical `emerging.test.annotated`; [recipe and limits](wnut17.md) |
| `germeval2014-outer`, `germeval2014-inner` | GermEval 2014 corrected 2022-04-28 (CC BY 4.0) | separate outer and inner BIO heads, 12 types each | size and SHA-256 of each file | publisher train/dev/test; [recipe and limits](germeval.md) |
| `multiconer2-en` | MultiCoNER v2 English (CC BY-SA 4.0: heads trained on it are share-alike, download only after the user accepts, never in a default bundle; docs/licences.md) | BIO, 33 types in 6 groups | size and SHA-256 of each file | the release's own train, dev and test |
| `re-docred` | Re-DocRED at commit `ccfb54f5` (CC BY-SA 4.0, share-alike, as MultiCoNER: Wikipedia documents, MIT annotations; docs/licences.md) | relations between a document's entities, `relation` on `turboembed:relations` | size and SHA-256 of each file and the repository's `LICENSE` | the release's own train, dev and test; counts are of documents |
| `ud-en-ewt-upos`, `ud-en-ewt-xpos`, `ud-en-ewt-feats` | UD English EWT r2.18 (commit `b7711cce`), one corpus a column, read as syntactic words (CC BY-SA 4.0: share-alike, as MultiCoNER) | tags, heads `upos` on `turboembed:pos`, `xpos`, `feats` | size and SHA-256 of each file (`ud::EN_EWT`) | the release's own train, dev and test; the test file is the one the spaCy, Stanza and Trankit peer records score |
| `ud-<lang>-<bank>-upos`, `-xpos`, `-feats` for `es-ancora`, `ca-ancora`, `fi-ftb`, `it-markit`, `pt-porttinari`, `de-gsd`, `fr-gsd`, `nl-alpino`, `ru-gsd`, `pt-gsd`, and in `ud_banks_sa` and `ud_banks_sa2` `af-afribooms`, `hy-bsut`, `hr-set`, `da-ddt`, `ka-glc`, `is-icepahc`, `id-gsd`, `ga-idt`, `ko-kaist`, `lv-lvtb`, `no-bokmaal`, `fa-perdt`, `ro-rrt`, `sr-set`, `sk-snk`, `sl-ssj`, `sv-talbanken`, `tr-boun` | UD r2.16 (`ud_banks`): the first five CC BY 4.0, a default-tier head (FTB under its CC BY 4.0 option; Porttinari's text rights not settled, so no head of it ships), the rest CC BY-SA 4.0 (Irish IDT CC BY-SA 3.0), share-alike | tags, as EWT | size and SHA-256 of each file, and the tag's commit | the release's own train, dev and test |

A directory that is not the recorded release is refused unless the run
passes `--unrecorded-corpus`, and its receipt then says
`"recorded": false`. No corpus is checked in. UD EWT is fetched at its
release tag:

```text
for f in en_ewt-ud-train.conllu en_ewt-ud-dev.conllu en_ewt-ud-test.conllu; do
  curl -sSO https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.18/$f
done
```

The r2.16 treebanks the same way, each from its repository at `r2.16`
(`UD_Spanish-AnCora`, `UD_Catalan-AnCora`, `UD_Finnish-FTB`,
`UD_Italian-MarkIT`, `UD_Portuguese-Porttinari`, `UD_German-GSD`,
`UD_French-GSD`, `UD_Dutch-Alpino`, `UD_Russian-GSD`, `UD_Portuguese-GSD`,
`UD_Afrikaans-AfriBooms`, `UD_Armenian-BSUT`, `UD_Croatian-SET`,
`UD_Danish-DDT`, `UD_Georgian-GLC`, `UD_Icelandic-IcePaHC`,
`UD_Indonesian-GSD`, `UD_Irish-IDT`, `UD_Korean-Kaist`, `UD_Latvian-LVTB`,
`UD_Norwegian-Bokmaal`, `UD_Persian-PerDT`, `UD_Romanian-RRT`,
`UD_Serbian-SET`, `UD_Slovak-SNK`, `UD_Slovenian-SSJ`,
`UD_Swedish-Talbanken`, `UD_Turkish-BOUN`),
the `<code>_<bank>-ud-{train,dev,test}.conllu` files the repository holds:

```text
for f in de_gsd-ud-train.conllu de_gsd-ud-dev.conllu de_gsd-ud-test.conllu; do
  curl -sSO https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/r2.16/$f
done
```

A UD treebank's multiword token lines (`2-3 al`) and empty nodes
(`8.1`) are not words: heads train on the syntactic words, where UD's
tags, features, lemmas and dependencies attach (`ud::UdSentence`).

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
turbo-train-head train-lemma --corpus ud-en-ewt --dir <treebank dir> --bundle <encoder bundle> --out <new dir>
    [as train, without --crf and --hash-embed] [--min-tree-freq N]
turbo-train-head train-parse --corpus ud-en-ewt --dir <treebank dir> --bundle <encoder bundle> --out <new dir>
    [as train, without --crf and --hash-embed] [--arc N] [--rel N]
turbo-train-head train-split --corpus ud-en-ewt --dir <treebank dir> --bundle <encoder bundle> --out <new dir>
    [as train, without --crf and --hash-embed] [--min-tree-freq N]
turbo-train-head export   --corpus <id> --dir <corpus dir> --out <test words.conllu>
turbo-train-head peer     --corpus <id> --dir <corpus dir> --peer <name and version> --predictions <peer.conllu>
    --out <new dir> [--column upos|xpos|feats|lemma|deps|bio] [--map FROM=TO,..] [--commit <sha>]
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

`train-lemma` trains the edit-tree lemmatizer (`lemma` on
`turboembed:lemmas`) over the trees that come up at least
`--min-tree-freq` times in training (3, spaCy's default);
`train-parse` trains the biaffine parser (`deps` on
`turboembed:dependencies`, MST decoding, arc and relation widths 256
and 64 by default) over the training relations. Both take the
treebank's id, read its syntactic words, and write the same
`head.safetensors`, `head.json` and receipt as `train`. They score as
word labels: lemma accuracy for the lemmatizer, LAS for the parser
(each word's label is `<head> <relation>`), with the parser's UAS
beside it (`eval.uas`). Their baselines: each word's commonest
training lemma, the word itself when unseen; and each word's
commonest training attachment, the signed distance to its head with
the relation, the commonest overall when unseen.

`train-split` trains the token split head (`words` on
`turboembed:words`, #1698) over the treebank's surface tokens, where
the other UD commands read syntactic words: each multiword token's
label is one edit tree a word, mined over the tokens that come up at
least `--min-tree-freq` times, beside `_`, which keeps a token whole.
It writes the same three files and scores as token labels, against
each token's commonest training label (`_` when unseen). A treebank
whose training split has no split that frequent is refused: English
EWT's are its contractions, and Spanish or Catalan AnCora and Italian
MarkIT hold the fused forms the head is for.

`export` writes a corpus's test words, and nothing else, as CoNLL-U
for a peer system (spaCy, Stanza) to tag as given; `peer` reads the
peer's CoNLL-U back and scores the column named exactly as the
matching head is scored, with the same baseline and bootstrap, so the
receipts stand side by side. `upos`, `xpos` and `feats` read a tag
corpus (`ud-es-ancora-feats`); `lemma` and `deps` read the treebank
(`ud-es-ancora`), against the labels `train-lemma` and `train-parse`
score, LAS with UAS beside it. A treebank id exports the same
syntactic words as its tag corpora. The LAS here is strict: a relation
counts only with its subtype (`nsubj:pass`), as `train-parse` scores
it, where CoNLL 2018 and published Stanza tables count the universal
part alone. The words are syntactic words, multiword tokens already
split, so a peer runs pretokenized with no multiword expansion (Stanza
`tokenize_pretokenized=True` and no `mwt` processor), or `peer` refuses
its file for words other than the split's.

`train-relation` trains the relation classifier head (`relation` on
`turboembed:relations`, #1734) on `re-docred`: each document is one
sequence, its entities typed by their first mention, each entity's
mentions pooled before the pair is scored, and every label above
`none`, the pair's own threshold, kept, so a pair holds several. Its
labels are `none` and the property ids training gives; `--mlp` and
`--type-dim` set the projection and entity type widths, and `--lr`
defaults to 3e-4, below the other heads' 2e-3, at which this head's
loss climbs again after its second epoch. It writes the
same three files and scores exact (head, tail, label) triples on the
test split, micro and per label, against the majority baseline, which
labels every pair `none` and so finds nothing, with a paired bootstrap
over documents:

```text
turbo-train-head train-relation --corpus re-docred --dir <Re-DocRED checkout> --bundle <encoder bundle> --out <new dir>
```

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
English span F1 0.1404 (377,805 gold spans). UD EWT r2.18 test (25,094
words): UPOS 0.8617, XPOS 0.8381, features 0.8292, lemmas 0.9503,
LAS 0.3322, UAS 0.3707.

```text
cargo test -p turbo-trainer-recipes
TURBO_TEST_MASC_DIR=<MASC-3.0.0> TURBO_TEST_MULTICONER_DIR=<multiconer2023> \
    TURBO_TEST_UD_EWT_DIR=<UD_English-EWT> \
    cargo test --release -p turbo-trainer-recipes
```

Not yet here: bundle output (a sealed bundle carrying the head),
a core head that reads a mix of layers (only if the mix wins, #1616),
full NER training and peer records (WNUT-17, GermEval and MASC), and a lemmatizer that reads the
tagger's UPOS (`pos_input`).
