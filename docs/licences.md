# Licences of datasets and checkpoints

The datasets and published checkpoints the analysis add-ons may use, with
their licences, their class under the policy on #326 and where the terms
are stated. The classes and what the zoo does with each are in
docs/zoo.md. The zoo's built-in index (`analysis/zoo/index/zoo.json`)
lists the checkpoints of the first table below; the refused checkpoints,
and the local-only ones the index does not list, are only here.
Restricted but public data and non-commercial data, fetched only from
their original sources and never shipped by this project, have sections
of their own at the end.

A checkpoint's class is the most restrictive of its own licence and the
licences of the data it was trained on. The OpenNLP UD models are
published under Apache-2.0, but each is trained on one Universal
Dependencies treebank, and the treebank decides the class.

"Checked" says how the row was verified:

- **here**: the licence was read at the source linked, on 2026-10-14,
  for this table. For a UD treebank that is the `License:` line of its
  README at the release tag the models were trained on (r2.16).
- **plan**: the licence was verified on 2026-10-14 for the retraining
  plan (its section 6, "Candidate corpora per model") against the
  dataset's own site, licence file or card. The link is the dataset's
  home page as this table names it; it could not be opened from the
  machine this table was written on, and should be opened before the
  dataset is used.

## OpenNLP UD models 1.3 (fetchable)

Published at <https://archive.apache.org/dist/opennlp/models/ud-models-1.3/>
under Apache-2.0 (its README), trained by OpenNLP 2.5.4 on UD 2.16.
Each treebank has four models: `sentence`, `tokens`, `pos` and `lemmas`,
named `opennlp-<lang>-ud-<treebank>-<model>-1.3-2.5.4.bin`; that name
without `.bin` is the zoo id. Every file's size, SHA-256 and SHA-512 in
the index were computed from the file fetched on 2026-10-14 and matched
against the `.sha256` and `.sha512` files published beside it.

| Language | Treebank | Treebank licence | Class | Checked |
|---|---|---|---|---|
| Catalan (`ca`) | [AnCora](https://github.com/UniversalDependencies/UD_Catalan-AnCora/blob/r2.16/README.md) | CC BY 4.0 | permissive | here |
| Afrikaans (`af`) | [AfriBooms](https://github.com/UniversalDependencies/UD_Afrikaans-AfriBooms/blob/r2.16/README.txt) | CC BY-SA 4.0 | share-alike | here |
| Armenian (`hy`) | [BSUT](https://github.com/UniversalDependencies/UD_Armenian-BSUT/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Croatian (`hr`) | [SET](https://github.com/UniversalDependencies/UD_Croatian-SET/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Danish (`da`) | [DDT](https://github.com/UniversalDependencies/UD_Danish-DDT/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Dutch (`nl`) | [Alpino](https://github.com/UniversalDependencies/UD_Dutch-Alpino/blob/r2.16/README.txt) | CC BY-SA 4.0 | share-alike | here |
| English (`en`) | [EWT](https://github.com/UniversalDependencies/UD_English-EWT/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Finnish (`fi`) | [TDT](https://github.com/UniversalDependencies/UD_Finnish-TDT/blob/r2.16/README.txt) | CC BY-SA 4.0 | share-alike | here |
| French (`fr`) | [GSD](https://github.com/UniversalDependencies/UD_French-GSD/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Georgian (`ka`) | [GLC](https://github.com/UniversalDependencies/UD_Georgian-GLC/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| German (`de`) | [GSD](https://github.com/UniversalDependencies/UD_German-GSD/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Icelandic (`is`) | [IcePaHC](https://github.com/UniversalDependencies/UD_Icelandic-IcePaHC/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Indonesian (`id`) | [GSD](https://github.com/UniversalDependencies/UD_Indonesian-GSD/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Irish (`ga`) | [IDT](https://github.com/UniversalDependencies/UD_Irish-IDT/blob/r2.16/README.txt) | CC BY-SA 3.0 | share-alike | here |
| Kazakh (`kk`) | [KTB](https://github.com/UniversalDependencies/UD_Kazakh-KTB/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Korean (`ko`) | [Kaist](https://github.com/UniversalDependencies/UD_Korean-Kaist/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Latvian (`lv`) | [LVTB](https://github.com/UniversalDependencies/UD_Latvian-LVTB/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Norwegian Bokmål (`no`) | [Bokmaal](https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Persian (`fa`) | [PerDT](https://github.com/UniversalDependencies/UD_Persian-PerDT/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Portuguese (`pt`) | [GSD](https://github.com/UniversalDependencies/UD_Portuguese-GSD/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Romanian (`ro`) | [RRT](https://github.com/UniversalDependencies/UD_Romanian-RRT/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Russian (`ru`) | [GSD](https://github.com/UniversalDependencies/UD_Russian-GSD/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Serbian (`sr`) | [SET](https://github.com/UniversalDependencies/UD_Serbian-SET/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Slovak (`sk`) | [SNK](https://github.com/UniversalDependencies/UD_Slovak-SNK/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Slovenian (`sl`) | [SSJ](https://github.com/UniversalDependencies/UD_Slovenian-SSJ/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Spanish (`es`) | [GSD](https://github.com/UniversalDependencies/UD_Spanish-GSD/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Swedish (`sv`) | [Talbanken](https://github.com/UniversalDependencies/UD_Swedish-Talbanken/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |
| Turkish (`tr`) | [BOUN](https://github.com/UniversalDependencies/UD_Turkish-BOUN/blob/r2.16/README.md) | CC BY-SA 4.0 | share-alike | here |

The Spanish and Portuguese models are trained on GSD, not on the CC BY
AnCora and Porttinari treebanks, so they are share-alike; only Catalan
uses AnCora.

The tests run eight of these share-alike models: the English, German and
Portuguese sentence models and the English, German, Spanish, French and
Portuguese tokenizer models. Under the 2026-10-14 decision share-alike
files are not checked in, so the tests fetch them: `cargo run -p
turbo-goldens-tool -- fetch-models <dir>` downloads them and checks their
SHA-512 (`analysis/goldens-tool/src/recipe.rs`), the tests read them from
that directory, and CI fetches them and requires them. The READMEs in
`analysis/sentdetect/tests/models/` and `analysis/tokenize/tests/models/`
list them. The small fixture models trained for this project
(`analysis/goldens/data/fixtures`, listed in `analysis/goldens/data/README.md`)
stay checked in.

## OpenNLP models (refused)

| Model | Trained on | Licence of the data | Reason | Checked |
|---|---|---|---|---|
| `langdetect-183.bin` | Leipzig corpora ([model README](https://archive.apache.org/dist/opennlp/models/langdetect/1.8.3/README.txt), [corpora](https://wortschatz.uni-leipzig.de/en/download)) | conflicting: one source says CC BY, another CC BY-NC | terms not confirmed (`LicenseRef-Unconfirmed`) | plan; README here |
| 1.5 `en-chunker` | CoNLL-2000 (Wall Street Journal, LDC) ([1.5 models](http://opennlp.sourceforge.net/models-1.5/)) | LDC | proprietary | plan |
| 1.5 `en-pos-maxent` | not published (the OpenNLP 1.5.1 test plan heads its regression-training section "(private) English data", which may not describe these models) | none stated | proprietary: refused for shipping, bundling and training; allowed only as a reference that is scored (below) | plan |
| 1.5 `en-pos-perceptron` | not published (as `en-pos-maxent`) | none stated | proprietary | plan |
| 1.5 English NER, sentence and token models | no corpus named | none stated | no licence | plan |

`en-pos-maxent.bin` is used only as a reference whose output is scored
or compared. Nothing ships, bundles, mirrors or trains on it, the zoo
does not list it, and CI does not fetch it. It is downloaded by hand
from <https://opennlp.sourceforge.net/models-1.5/en-pos-maxent.bin> into
a directory the user names. Its SHA-256 is
`645a094f45a866687a617385233fd23ae8b0f5fa8b1b76996781a50c17bdcf3d`. The
JVM goldens (`PosGolden`, `BeamGolden`) record its tags for the test
sentences and that digest; turbo-sequence's tests check the digest and
compare what turbo-sequence makes of the model with the goldens; and the
turbo-train-recipes tagger gate checks its size and SHA-256 (5,696,197
bytes, `postag::EN_POS_MAXENT`), scores it on MASC's held-out documents
and compares that score with the trained model's.

The goldens tool fetches `langdetect-183.bin` to make the language
detector's goldens (`analysis/goldens-tool/src/recipe.rs`); the zoo does
not list it.

## Datasets

| Dataset | Use | Licence | Class | Notes | Checked |
|---|---|---|---|---|---|
| [UD_Spanish-AnCora r2.16](https://github.com/UniversalDependencies/UD_Spanish-AnCora/blob/r2.16/README.md) | sentence, token, POS, lemma | CC BY 4.0 | permissive | | here |
| [UD_Catalan-AnCora r2.16](https://github.com/UniversalDependencies/UD_Catalan-AnCora/blob/r2.16/README.md) | sentence, token, POS, lemma | CC BY 4.0 | permissive | | here |
| [UD_Portuguese-Porttinari r2.16](https://github.com/UniversalDependencies/UD_Portuguese-Porttinari/blob/r2.16/README.md) | sentence, token, POS, lemma | CC BY 4.0 | permissive | | here |
| [UD_Portuguese-DanteStocks r2.16](https://github.com/UniversalDependencies/UD_Portuguese-DanteStocks/blob/r2.16/README.md) | sentence, token, POS, lemma | CC BY 4.0 | permissive | | here |
| [UD_Finnish-FTB r2.16](https://github.com/UniversalDependencies/UD_Finnish-FTB/blob/r2.16/README.md) | sentence, token, POS, lemma | CC BY 4.0 | permissive | | here |
| [UD_Italian-MarkIT r2.16](https://github.com/UniversalDependencies/UD_Italian-MarkIT/blob/r2.16/README.md) | sentence, token, POS, lemma | CC BY 4.0 | permissive | | here |
| [MASC](https://anc.org/data/masc/) | English sentences, tokens, Penn POS, noun and verb chunks, named entities | CC BY 3.0 US | permissive | 500K words, manually validated; the recipes in `analysis/train-recipes` train on it, from the 3.0.0 archive anc.org links to, recorded by size and digest (`masc::RELEASE`). Its Twitter genre (`data/written/twitter`, two documents, 24,180 words, from 2008, source given as "unknown" in their headers) has no terms of its own: the 3.0.0 release's README has no licence text, and anc.org states CC BY 3.0 US for MASC with no exception. Whether Twitter's terms reach the tweets' text is not settled here; the recipes do not leave the genre out | here |
| [Open English WordNet](https://github.com/globalwordnet/english-wordnet/blob/main/LICENSE.md) | English lemmas | CC BY 4.0 | permissive | derived from Princeton WordNet; attribute both | here |
| [Princeton WordNet](https://wordnet.princeton.edu/license-and-commercial-use) | English lemmas | WordNet licence (BSD-like) | permissive | | plan |
| [PoliMorf / Morfologik](https://github.com/morfologik/polimorf) | Polish lemmas | BSD-2-Clause | permissive | | plan |
| [GermEval 2014](https://sites.google.com/site/germeval2014ner/) | German NER | CC BY 4.0 | permissive | | plan |
| [MultiCoNER v2](https://registry.opendata.aws/multiconer/) | NER, 12 languages | CC BY 4.0 | permissive | | plan |
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | language detection | ODC-By 1.0 | permissive | also subject to Common Crawl's terms of use | plan |
| [Tatoeba](https://tatoeba.org/en/downloads) | language detection | CC BY 2.0 FR | permissive | by default; some sentences carry other licences, check per item | plan |
| [WNUT-17](https://noisy-text.github.io/2017/emerging-rare-entities.html) | NER | CC BY 4.0 | permissive, pending | its text comes from Twitter, Reddit and YouTube; platform terms need a look before it is used | plan |
| Other UD treebanks (the plan counts 280 of 359 in UD 2.18; not recounted here), including English EWT, German GSD and HDT, French GSD, Dutch Alpino | sentence, token, POS, lemma | CC BY-SA (3.0 or 4.0) | share-alike | check each treebank's README at the release used | plan; the ones above here |
| [UniMorph](https://unimorph.github.io/) | lemmas | mostly CC BY-SA 3.0, some LGPL-LR | share-alike | check per language | plan |
| [Few-NERD](https://huggingface.co/datasets/DFKI-SLT/few-nerd) | NER | CC BY-SA 4.0 | share-alike | | plan |
| [Universal NER](https://www.universalner.org/) | NER | CC BY-SA 4.0 | share-alike | except its Tagalog-Ugnayan subset, which is non-commercial (local-only) | plan |
| [DaNE](https://github.com/alexandrainst/danlp) | Danish NER | CC BY-SA 4.0 | share-alike | | plan |
| [Wikipedia](https://dumps.wikimedia.org/) | language detection | CC BY-SA 4.0 | share-alike | | plan |
| [DBpedia-14](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | document categorizer | CC BY-SA 3.0 and GFDL | share-alike | | plan |
| [UD_English-GUM r2.16](https://github.com/UniversalDependencies/UD_English-GUM/blob/r2.16/README.md) | | CC BY-NC-SA 4.0 | local-only | non-commercial | here |
| [UD_English-LinES r2.16](https://github.com/UniversalDependencies/UD_English-LinES/blob/r2.16/README.txt) | | CC BY-NC-SA 4.0 | local-only | non-commercial | here |
| [UD_Swedish-LinES r2.16](https://github.com/UniversalDependencies/UD_Swedish-LinES/blob/r2.16/README.txt) | | CC BY-NC-SA 4.0 | local-only | non-commercial | here |
| [UD_Italian-ISDT r2.16](https://github.com/UniversalDependencies/UD_Italian-ISDT/blob/r2.16/README.md) | | CC BY-NC-SA 3.0 | local-only | non-commercial | here |
| [UD_Italian-ParTUT r2.16](https://github.com/UniversalDependencies/UD_Italian-ParTUT/blob/r2.16/README.md) | | CC BY-NC-SA 4.0 | local-only | non-commercial | here |
| [UD_Czech-PDT r2.16](https://github.com/UniversalDependencies/UD_Czech-PDT/blob/r2.16/README.md) | | CC BY-NC-SA 4.0 | local-only | non-commercial (the plan counts about 62 non-commercial UD treebanks; not recounted here) | here |
| [UD_English-GUMReddit r2.16](https://github.com/UniversalDependencies/UD_English-GUMReddit/blob/r2.16/README.md) | | CC BY 4.0 annotations | refused | the text is rebuilt from Reddit, under Reddit's terms | here (annotations); plan (text) |
| [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) | NER | Reuters RCV1, research only | refused | | plan |
| [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) | NER | LDC, non-commercial | refused | not a CC licence: an LDC user agreement, outside the 2026-10-14 decision on non-commercial data | plan |
| [MultiNERD](https://github.com/Babelscape/multinerd) | NER | CC BY-NC-SA 4.0 | local-only | non-commercial | plan |
| [WikiNEuRal](https://github.com/Babelscape/wikineural) | NER | CC BY-NC-SA 4.0 | local-only | non-commercial | plan |
| [CoNLL-2002](https://www.clips.uantwerpen.be/conll2002/ner/) | NER | none stated | refused | no licence | plan |
| [NorNE](https://github.com/ltgoslo/norne) | NER | annotations CC0, text restricted | refused | | plan |
| [OpenLID, GlotLID corpora](https://github.com/cisnlp/GlotLID) | language detection | mixed, parts non-commercial | refused | as a whole | plan |
| [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) | language detection | its card allows evaluation only | refused for training | evaluation runs may use it; numbers recorded, files not kept | plan |
| [Leipzig corpora](https://wortschatz.uni-leipzig.de/en/download) | language detection | conflicting (CC BY or CC BY-NC) | refused | until the terms are confirmed; under the 2026-10-14 decision either reading is fetchable (CC BY permissive, CC BY-NC local-only), and an entry would take the stricter one, CC-BY-NC-4.0 (local-only); no entry is made | plan |
| [AG News](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) | document categorizer | non-commercial | refused | not a CC licence: the host's own non-commercial terms, outside the 2026-10-14 decision on non-commercial data | plan |
| [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) | document categorizer | none stated | refused | no licence | plan |
| [Europarl](https://www.statmt.org/europarl/) | | none stated; its makers know of no restrictions | refused | no licence, unless decided otherwise | plan |

## Unverified

Not classified until someone reads their terms at the source; nothing
here is used.

- GMB: texts are public domain; the annotation licence is unverified.
- WikiANN: listed as ODC-By by one source and "unknown" on its card.
- Sentiment (#326, #318): SST, SemEval 2014 to 2016 ABSA and the ASTE
  derivatives, Jigsaw Toxic Comment, Civil Comments, iSarcasm, SARC,
  GoEmotions, and public sentiment and toxicity checkpoints.

## Restricted but public data (local only)

Data anyone can download from its publisher, under the publisher's own
terms rather than an open licence. Under Kristian's 2026-10-12 comment
on #326 it is `CLASS_LOCAL_ONLY` (docs/zoo.md, "Restricted but public
data"): fetched only from the original host after the user accepts that
host's terms, used for local training and evaluation only, and never
carried, mirrored or checked in. Any model trained on it is local-only
too and is never shipped; evaluation numbers on it may be recorded, the
files not.

An index entry for one of these names `LicenseRef-RestrictedPublic`,
the host's terms page as `license_terms`, and the publisher's own
location on its host as `origin`, which every file must lie under. The
built-in index has none of them: the host and terms page of each must
be read at the source first.

| Dataset or checkpoint | Use | Terms as #326 records them | Class once entered | Checked |
|---|---|---|---|---|
| IMDB (Large Movie Review Dataset) | polarity, evaluation | no stated licence | local-only | #326 only; host and terms page not opened |
| Sentiment140 | polarity | Twitter's terms | local-only | #326 only; host and terms page not opened |
| SemEval Twitter sentiment sets | polarity | Twitter's terms | local-only | #326 only; host and terms page not opened |
| `lvwerra/distilbert-imdb` | the checkpoint OpenNLP's DL doccat eval downloads | a model trained on IMDB | local-only, by the class rule | #326 only; card not opened |

Research-only data stays refused (the tables above) unless an entry
here places it in this class, with the reason.

## Non-commercial data (local only)

Data under CC BY-NC or CC BY-NC-SA. Under Kristian's 2026-10-14 decision
it is `CLASS_LOCAL_ONLY` (docs/zoo.md, "Non-commercial data"), the way
OpenNLP handles data it cannot distribute: our code reads it from files
the user supplies, with nothing to accept; the zoo fetches it only from
its original source, after the user accepts the licence; and neither the
data nor any model trained on it is shipped, bundled, mirrored or
checked in. By accepting, the user agrees to treat a model they train
on it as non-commercial and keep it on their machine. These rules are
this project's policy; the licences themselves allow sharing the data
and adapted material for non-commercial purposes.

Data that is also no-derivatives (CC BY-NC-ND) stays refused, as CC
BY-ND data does. That is this project's default (2026-10-14), not what
the licences require: they allow producing adapted material and forbid
only sharing it, so they would allow a model trained on the data and
kept on the user's machine.

The non-commercial datasets are in the datasets table above, marked
local-only. The built-in index lists none of them, and none of the
models below. An entry for one records the CC licence (for a model,
under `trained_on`) with the address where the publisher states it, and
the publisher's own location as `origin`, which every file must lie
under: for a UD treebank its repository path on
`raw.githubusercontent.com`
(`https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/`),
not the whole host, and for a model the model publisher's.

The OpenNLP UD 1.3 models trained on non-commercial treebanks are
local-only by the class rule (Apache-2.0 models, non-commercial data):

| Model | Trained on | Licence of the data | Checked |
|---|---|---|---|
| UD 1.3 `bg-ud-btb-*` | [UD_Bulgarian-BTB r2.16](https://github.com/UniversalDependencies/UD_Bulgarian-BTB/blob/r2.16/README.txt) | CC BY-NC-SA 3.0 | here |
| UD 1.3 `cs-ud-pdtc-*` | [UD_Czech-PDTC r2.16](https://github.com/UniversalDependencies/UD_Czech-PDTC/blob/r2.16/README.md) | CC BY-NC-SA 4.0 | here |
| UD 1.3 `el-ud-gdt-*` | [UD_Greek-GDT r2.16](https://github.com/UniversalDependencies/UD_Greek-GDT/blob/r2.16/README.md) | CC BY-NC-SA 3.0 | here |
| UD 1.3 `et-ud-edt-*` | [UD_Estonian-EDT r2.16](https://github.com/UniversalDependencies/UD_Estonian-EDT/blob/r2.16/README.md) | CC BY-NC-SA 4.0 | here |
| UD 1.3 `eu-ud-bdt-*` | [UD_Basque-BDT r2.16](https://github.com/UniversalDependencies/UD_Basque-BDT/blob/r2.16/README.md) | CC BY-NC-SA 3.0 | here |
| UD 1.3 `it-ud-vit-*` | [UD_Italian-VIT r2.16](https://github.com/UniversalDependencies/UD_Italian-VIT/blob/r2.16/README.md) | CC BY-NC-SA 3.0 | here |
| UD 1.3 `pl-ud-pdb-*` | [UD_Polish-PDB r2.16](https://github.com/UniversalDependencies/UD_Polish-PDB/blob/r2.16/README.md) | CC BY-NC-SA 4.0 | here |
| UD 1.3 `uk-ud-iu-*` | [UD_Ukrainian-IU r2.16](https://github.com/UniversalDependencies/UD_Ukrainian-IU/blob/r2.16/README.md) | CC BY-NC-SA 4.0 | here |
