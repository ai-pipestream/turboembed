# turbo-trainer-recipes

The corpora `turbo-trainer` heads train on, the scoring every run
reports, and the `turbo-train-head` command (#1518, stage R1b).

| Corpus id | Data | Task | Pinned by | Splits |
|---|---|---|---|---|
| `masc-pos` | MASC 3.0.0, Penn tags (CC BY 3.0 US) | tags | corpus digest of the extracted release (`turbo_train_recipes::masc::RELEASE`) | test: the recipes' held-out documents (every tenth, Twitter genre left out), so numbers compare with the MASC tagger's gate; dev: every tenth training-side document |
| `multiconer2-en` | MultiCoNER v2 English (CC BY-SA 4.0: heads trained on it are share-alike, download only after the user accepts, never in a default bundle; docs/licences.md) | BIO, 33 types in 6 groups | size and SHA-256 of each file | the release's own train, dev and test |
| `ud-en-ewt-upos`, `ud-en-ewt-xpos`, `ud-en-ewt-feats` | UD English EWT r2.18 (commit `b7711cce`), one corpus a column, read as syntactic words (CC BY-SA 4.0: share-alike, as MultiCoNER) | tags, heads `upos` on `turboembed:pos`, `xpos`, `feats` | size and SHA-256 of each file (`ud::EN_EWT`) | the release's own train, dev and test; the test file is the one the spaCy, Stanza and Trankit peer records score |
| `ud-<lang>-<bank>-upos`, `-xpos`, `-feats` for `es-ancora`, `ca-ancora`, `fi-ftb`, `it-markit`, `pt-porttinari`, `de-gsd`, `fr-gsd`, `nl-alpino`, `ru-gsd`, `pt-gsd` | UD r2.16 (`ud_banks`): the first five CC BY 4.0, a default-tier head (FTB under its CC BY 4.0 option; Porttinari's text rights not settled, so no head of it ships), the last five CC BY-SA 4.0, share-alike | tags, as EWT | size and SHA-256 of each file, and the tag's commit | the release's own train, dev and test |

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
`UD_French-GSD`, `UD_Dutch-Alpino`, `UD_Russian-GSD`, `UD_Portuguese-GSD`),
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
turbo-train-head train-lemma --corpus ud-en-ewt --dir <treebank dir> --bundle <encoder bundle> --out <new dir>
    [as train, without --crf and --hash-embed] [--min-tree-freq N]
turbo-train-head train-parse --corpus ud-en-ewt --dir <treebank dir> --bundle <encoder bundle> --out <new dir>
    [as train, without --crf and --hash-embed] [--arc N] [--rel N]
```

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

Not yet here: bundle output (a sealed bundle carrying the head), the
learned mix of the last layers, WNUT-17 and MASC named entities, and
a lemmatizer that reads the tagger's UPOS (`pos_input`).
