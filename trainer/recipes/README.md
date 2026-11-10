# turbo-trainer-recipes

The corpora `turbo-trainer` heads train on, the scoring every run
reports, and the `turbo-train-head` command (#1518, stage R1b).

| Corpus id | Data | Task | Pinned by | Splits |
|---|---|---|---|---|
| `masc-pos` | MASC 3.0.0, Penn tags (CC BY 3.0 US) | tags | corpus digest of the extracted release (`turbo_train_recipes::masc::RELEASE`) | test: the recipes' held-out documents (every tenth, Twitter genre left out), so numbers compare with the MASC tagger's gate; dev: every tenth training-side document |
| `masc-ne` | MASC 3.0.0 named entities (CC BY 3.0 US) | BIO, person/location/organization/date | same recorded corpus digest as MASC POS | same held-out documents and Twitter exclusion; training-only dev; [recipe and refusals](masc-ne.md) |
| `wnut17` | WNUT-17 (CC BY 4.0, default-tier training under #1524) | BIO, six types kept separate from MASC | size and SHA-256 of each file | publisher train/dev/test, canonical `emerging.test.annotated`; [recipe and limits](wnut17.md) |
| `germeval2014-outer`, `germeval2014-inner` | GermEval 2014 corrected 2022-04-28 (CC BY 4.0) | separate outer and inner BIO heads, 12 types each | size and SHA-256 of each file | publisher train/dev/test; [recipe and limits](germeval.md) |
| `multiconer2-en` | MultiCoNER v2 English (CC BY-SA 4.0: heads trained on it are share-alike, download only after the user accepts, never in a default bundle; docs/licences.md) | BIO, 33 types in 6 groups | size and SHA-256 of each file | the release's own train, dev and test |
| `litbank` | LitBank (`dbamman/litbank` at `3e50db0f`, CC BY 4.0 over public-domain Project Gutenberg works) | coref (`coref` on `turboembed:coref`, `HEAD_COREF_RANKER`) | size and SHA-256 of all 100 `coref/conll/*.conll` files (`litbank::pins`) | sorted 80/10/10 work split (`turbo_train_recipes::litbank::SPLIT`) |
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
    --layers N [--task tag|lemma|parse|split] [--device ..] [--precision ..] [--pooling first|mean]
    [--max-batch N] [--splits train,dev,test] [--commit <sha>]
turbo-train-head fit-cache --corpus <id> --dir <corpus dir> --cache <cache dir> --out <new dir>
    (--layer N | --mix [--from-layer N]) [--task tag|lemma|parse|split] [--crf] [--bilstm ..]
    [--hash-embed ..] [--min-tree-freq N] [--seen-lookup K] [--arc N] [--rel N] [training settings]
turbo-train-head sweep    --corpus <id> --dir <corpus dir> --cache <cache dir> --out <new dir>
    [--task tag|lemma|parse|split] [--from-layer N] [--peer-receipt <receipt.json>]
    [--bilstm ..] [--min-tree-freq N] [--seen-lookup K] [--arc N] [--rel N] [training settings]
turbo-train-head train-lemma --corpus ud-en-ewt --dir <treebank dir> (--bundle <encoder bundle> | --cache <cache dir> (--layer N | --mix [--from-layer N])) --out <new dir>
    [as train, without --crf and --hash-embed] [--min-tree-freq N] [--seen-lookup K]
turbo-train-head train-parse --corpus ud-en-ewt --dir <treebank dir> (--bundle <encoder bundle> | --cache <cache dir> (--layer N | --mix [--from-layer N])) --out <new dir>
    [as train, without --crf and --hash-embed] [--arc N] [--rel N]
turbo-train-head train-split --corpus ud-en-ewt --dir <treebank dir> (--bundle <encoder bundle> | --cache <cache dir> (--layer N | --mix [--from-layer N])) --out <new dir>
    [as train, without --crf and --hash-embed] [--min-tree-freq N]
turbo-train-head train-open-span --corpus <id> --dir <corpus dir> --bundle <encoder bundle> --out <new dir>
    [--cache <cache dir>] [--layer N] [--max-words N] [--width-dim N]
    [--hold-out <type> ...] [--balance] [training settings]
turbo-train-head train-coref --corpus litbank --dir <litbank dir> --bundle <encoder bundle> --out <new dir>
    [as train, without --crf and --hash-embed] [--mlp N] [--max-antecedents N] [--max-words N] [--width-dim N]
turbo-train-head assemble --bundle <encoder bundle> --run <run dir> [--run <run dir> ...] --out <new dir>
    [--zoo-id <id> [--component <COMPONENT_*>] [--language <lang>] [--version <ver>]
     [--license-terms <https://...>] [--url-prefix <https://...>] [--origin <https://...>]
     [--zoo-out <entry.json>]]
turbo-train-head export   --corpus <id> --dir <corpus dir> --out <test words.conllu>
turbo-train-head peer     --corpus <id> --dir <corpus dir> --peer <name and version> --predictions <peer.conllu>
    --out <new dir> [--column upos|xpos|feats|lemma|deps|bio] [--map FROM=TO,..] [--commit <sha>]
turbo-train-head train-sentences --corpus <masc-pos|ud-*> --dir <corpus dir> --bundle <encoder bundle> --out <new dir>
    [as train]
turbo-train-head summarize <dir> ... [--peer <dir> ...] [--format md|json] [--out <new dir>] [--check]
```

`assemble` (#1726) verifies one or more training run directories against
an encoder bundle (checking `receipt.json` schema, encoder model,
revision, manifest digest and hidden width, corpus record and licence
class, output file sizes and SHA-256 digests, `head.json` equality, and
exact F32 tensor names and shapes loaded through the core head
constructors), merges the head tensors into `weights/model.safetensors`,
writes `heads[]` and `head_provenance[]` into `manifest.json`, verifies
the assembled bundle with `Bundle::open`, and optionally writes a
validated `turbo-zoo` artifact entry (`Kind::Head` for one head or
`Kind::Model` for a multi-task bundle) whose `class` is computed from the
encoder and every head's `CorpusRecord`.

`summarize` (#1830) reads the `receipt.json` of several run directories
plus peer receipts, groups them by corpus and task, and prints one table
per corpus with the head variants as rows and the baseline and peer rows
beneath them: model and baseline scores, the paired-bootstrap delta and
interval, epochs run and kept, wall time, and the output digests.
`--out` writes the tables as `summary.md` with the numbers as
`summary.json`, with any `host` field scrubbed (#1804), so a run
directory is self-describing before anything is copied. `--check`
fails when a receipt's `commit` differs from the others or a digest in
`summary.json` or a `receipt.json` no longer matches the file on disk,
which guards the copy to the models folder. `summary.json` records each
row's `dir` and `receipt` (and the `command`) relative to the `--out`
directory, and `--check` reads them relative to the summary's own
directory, so a run directory copied with its summary is checked as
copied, even once the original is gone; a summary that records an
absolute path is refused. Rows of one corpus table must share a metric.

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

`fit-cache` (and alias `train-cached`, `#1616`, `#1822`), or `--cache` on
`train`, `train-lemma`, `train-parse`, and `train-split`, trains a head
on a cache's train split and scores it on its dev split (and test split
when `"test"` is cached), without the encoder: on one layer (`--layer N`;
on the last layer it is the run `train` makes, epoch for epoch, and on a
lower layer `k` it records `layer_mix: { layers: L - k + 1, layer: k }` in
`head.json`), or on a learned mix of the cached layers from `--from-layer`
on (`--mix`: ELMo's scalar mix, `turbo_trainer::mix`, `fit_lemmatizer_mix`,
and `fit_parser_mix`, a softmax weight a layer and a gain, trained with
the head). One syntactic-word cache (`cache --corpus ud-<lang>-<bank>`)
feeds `upos`, `xpos`, `feats`, `lemma` and `parse`; `train-split` uses a
surface-token cache (`cache --corpus ud-<lang>-<bank> --task split`). It
writes `head.safetensors`, `head.json`, `fit.json` (`fit_cache::FitRecord`),
and (when `"test"` is cached) `receipt.json`. `sweep` runs every layer
from `--from-layer` to the top cached layer plus `--mix`, writes each
run's `fit.json` and `receipt.json`, and writes `sweep.json` and
`sweep.md` with paired bootstrap against the last layer and an optional
`--peer-receipt`. The peer must identify its prediction file and match the
same gold test corpus, release hashes, task and score column; incompatible
receipts are refused before training. `--max-train N` selects the original
first N training sentences before unsupported-label filtering, including
when training from a cache. Later sentences never backfill the prefix.

`train-lemma` trains the edit-tree lemmatizer (`lemma` on
`turboembed:lemmas`) over the trees that come up at least
`--min-tree-freq` times in training (3, spaCy's default), and with
`--seen-lookup K` records an unambiguous seen-form lookup table
(`seen_lookup` in `head.json`, forms seen at least `K` times with a
single training lemma) applied in front of the edit-tree classifier at
decode during dev and test evaluation and at runtime;
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
label is one edit tree a word, kept over the tokens that come up at
least `--min-tree-freq` times, beside `_`, which keeps a token whole.
A word that sits inside its token may take a cut (`(t 2 ())` keeps the
last two characters) or a tree that names what it drops
(`(m 5 0 (s 'hacer' '') ())`); each training token takes, of the
labels that give its words, the one that gives the most distinct
training tokens theirs, so a clitic's label serves every verb and a
contraction's names its letters (#1798). It writes the same three
files and scores each token's words being right, against each token's
commonest training words (the token itself when unseen). A treebank
whose training split has no split that frequent is refused: English
EWT's are its contractions, and Spanish or Catalan AnCora and Italian
MarkIT hold the fused forms the head is for.

`train-sentences` trains the sentence boundary head (`sentences` on
`turboembed:sentences`, #1725) on multi-sentence MASC (`masc-pos`) or
UD documents (`ud-es-ancora`, `ud-ca-ancora`, `ud-fi-ftb`,
`ud-it-markit`, `ud-en-ewt`, and the other pinned UD treebanks),
keeping original MASC inter-sentence whitespace and UD `# newdoc` /
`# newpar` / `SpacesAfter` / `SpaceAfter=No` joins intact. It scores
the test split by exact sentence span precision, recall and F1, with
paired bootstrap comparisons against both the newline baseline
(`eval.baseline` and `eval.comparison`) and the rule/abbreviation
sentencer in `analysis/sentdetect` (`eval.rules`).

`train-open-span` (#1735) trains the open-label span matching head
(`open_ner` on `turboembed:entities`, `HEAD_OPEN_SPAN`) on any BIO
corpus (`wnut17`, `germeval2014-outer`, `germeval2014-inner`,
`multiconer2-en`) or UD treebank (`ud-<lang>-<bank>`, extracting
contiguous `PROPN` runs as `"proper noun"` spans). Entity type names
are encoded by the same frozen encoder (`turbo::head::open_span::label_words`
and `pool_label_words`), centered against the empirical training word
and label means (`center.weight`), and scored against candidate spans
up to `--max-words` (8 by default). Passing `--hold-out <type>` drops
that entity type from training and dev and evaluates test span F1 on
all types, seen types, and held-out (zero-shot unseen) types alongside
the majority baseline and the case-insensitive label-name dictionary
matcher. Comma-separated corpus ids and directories train one shared head,
with per-dataset and combined metrics plus paired bootstrap against both
baselines. Each corpus supervises only its declared entity types, so a type
absent from one corpus is not learned as outside there; the shared label
center covers all seen types. `--balance` repeats each smaller training
corpus and its feature rows to the largest split, without changing dev or
test. Single-corpus cache input is supported; mixed-corpus cache input is
refused.

`eval-gliner` evaluates a converted pretrained GLiNER checkpoint without
training a `HEAD_OPEN_SPAN` model. It uses the same corpora, `--hold-out`
schema, all-type and seen-type scores, per-dataset metrics, baselines and
paired bootstraps, and writes word-aligned `predictions.bio` beside the
receipt. Schemas over GLiNER's 32-label prompt limit are scored in label
batches and decoded together, including candidates from every batch.
`--threads N` evaluates contiguous partitions on independent model sessions
and restores their original order. `--split dev` selects the publisher dev
split for threshold tuning; the default is `test`. Use `--package <converted GLiNER>`
with `--encoder-bundle <sealed encoder>` and `--device` for backend execution,
or the package alone for CPU execution. `--guidelines <json>` supplies a
label-to-prompt map for aliases and annotation guidelines. Gold labels and
scoring stay unchanged. Long prompts are batched per sentence to fit the
encoder, and the receipt's peer name includes the full schema digest.
`--calibrate` is valid only with `--split dev`: it makes two coordinate-ascent
passes over a fixed 0.05 threshold grid. A label changes only when its
exact-span F1 improves in the joint flat decode without reducing overall F1.
Ties prefer overall F1, then the value closest to `--threshold`, then the
higher value. It writes `thresholds.json`
containing every label, its threshold, and the full schema digest. Pass that
file to a later test run with `--thresholds <json>`; evaluation refuses a file
whose version, labels, schema, or thresholds do not match. Every run writes
this artifact, including uniform-threshold runs, and its SHA-256 is part of
the peer identity. Calibration never reads test labels. The per-label filter
runs before flat overlap resolution across all labels. New artifacts also carry
`held_thresholds`, fitted independently on the held-out schema's dev
candidates and used only for held-out decoding. Older artifacts without that
map use the all-label thresholds for held-out decoding. Candidate tensors are
reused for calibrated all-label and held-out predictions; only seen labels are
scored again.

`train-gliner` fine-tunes the two-layer markerV0 `out_project` MLP in Rust on
frozen encoder, prompt, recurrent and boundary features. Each training span
keeps its gold labels and at most four highest-logit negative labels; gold
spans are accompanied by up to two hard-negative spans per positive span,
with a minimum of four. Positive BCE terms have weight 4. `--balance` repeats
smaller corpus training splits to the largest, and `--hold-out` removes those
types from every training batch. Each corpus supplies only its declared seen
labels, so absent types are not treated as negative evidence. The command
selects the zero-shot initialization or any strictly better epoch by dev F1,
never evaluates test, and writes the best complete package under `model/`
plus dev predictions and a receipt. Use `eval-gliner --split dev --calibrate`
on that package before one frozen test evaluation. `--max-train N` is a
per-corpus smoke-test or ablation limit, not balanced sampling. `--threads N`
computes at most N sentences per batch concurrently and reduces their
gradients in original sentence order; 0 uses the whole batch. Set
`TURBO_CPU_THREADS=1` when using multiple sentence workers so encoder calls
do not create nested thread pools.

`train-coref` trains the antecedent-ranking coreference head (`coref` on
`turboembed:coref`, #1527) on LitBank's 100 literary works (80 train,
10 dev, 10 test), checkpointing `head.safetensors` and `head.json` by
dev antecedent accuracy and scoring the test split by micro-averaged
MUC, B-cubed, CEAF-e and their mean CoNLL F1 beside the deterministic
singleton baseline (every mention in its own cluster) with a seeded
paired bootstrap over documents.

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
its file for words other than the split's. For `litbank`, `export`
writes the ten test works as one CoNLL coreference file each under the
output directory (the gold words and sentence framing, the coreference
column blank) and `peer --column coref` reads a peer's prediction file
of one document a work, matched by document id, scored with the CoNLL
coreference metrics against the same works beside the singleton
baseline (#1825).

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

`baseline --corpus re-docred --dependency-path` writes the deterministic
dependency-path rule row (#1825): the documents' own sentences parsed by
our dependency head (`--parser`, a `train-parse` run directory, over
`--bundle`), an ordered entity pair labelled by the most frequent
training relation on the shortest path between the two entities' first
mentions' head words, no relation when the path is one training never
gave (`redocred_path`). The receipt's `baseline` is the rule's score,
its `encoder` and `head` name the parser, and `outputs` digests the
rule's predictions:

```text
turbo-train-head baseline --corpus re-docred --dir <Re-DocRED checkout> \
  --dependency-path --parser <train-parse run dir> --bundle <encoder bundle> --out <new dir>
```

`--decode relation` trains the single-label variant instead (#1826):
`DECODE_RELATION`, a softmax cross-entropy over the labels and `none`
for each pair, and at most one label a pair kept. `--decode relations`
is the default. `--against <head dir>` reads a head an earlier
`train-relation` wrote (its `head.json`, `head.safetensors` and
`receipt.json`) before training starts. It refuses the head unless it
reads the features this run computes: the same labels, `--pooling` and
entity type table, on the same encoder (model, revision, manifest
digest, width and layers; backend, device and precision may differ).
It then scores that head on the same test documents and records it in
the receipt's `eval.against`: its metrics, the new head's paired
bootstrap over documents against it, and the three files with their
digests. Every run prints the per-label
table of the ten labels with the most gold test relations.

```text
turbo-train-head train-relation --corpus re-docred --dir <Re-DocRED checkout> --bundle <encoder bundle> --out <new dir> --decode relation --against <threshold head dir>
```

A head with a BiLSTM adapter waits 15 epochs without progress before
it stops (`--patience`, 6 for other heads): its dev score creeps up for
many epochs, and each better epoch is saved as it comes.

`train` reads per-word features from the encoder through
`turbo_tagger::WordEncoder`, the path the annotators use, trains with
`turbo_trainer::fit`, predicts the test split through the core's
`TokenTagger`, and writes `head.safetensors` (the head's F32 tensors
under `heads.<name>.`), `head.json` (its `HeadSpec`) and `receipt.json`
(`receipt::Receipt`: command, commit, corpus record with digests and
counts, encoder model and manifest digest, backend and device, head,
settings, per-epoch loss and dev score with `best_epoch` counted from 1
as in the training log, test scores of model and baseline, the
bootstrap, timings, and the outputs' digests).

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

Not yet here: a core head that reads a mix of layers (only if the mix
wins, #1616), full NER training and peer records (WNUT-17, GermEval and
MASC), and a lemmatizer that reads the tagger's UPOS (`pos_input`).

Native NER bundle prediction and paired peer comparison: [ner-release.md](ner-release.md).
