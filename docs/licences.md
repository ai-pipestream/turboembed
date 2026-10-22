# Licences of datasets and checkpoints

The datasets and published checkpoints the analysis add-ons may use, with
their licences, their class under the policy on #326 and where the terms
are stated. The classes and what the zoo does with each are in
docs/zoo.md. The zoo's built-in index (`analysis/zoo/index/zoo.json`)
lists the checkpoints of the first two tables below; the refused
checkpoints, and the local-only ones the index does not list, are only
here.
Restricted but public data and non-commercial data, requested only at
addresses under their original sources and never shipped by this
project, have sections of their own at the end.

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
SHA-512 (`analysis/goldens-tool/src/recipe/mod.rs`), the tests read them from
that directory, and CI fetches them and requires them. The READMEs in
`analysis/sentdetect/tests/models/` and `analysis/tokenize/tests/models/`
list them. The small fixture models trained for this project
(`analysis/goldens/data/fixtures`, listed in `analysis/goldens/data/README.md`)
stay checked in.

## Models trained by this project (fetchable once released)

Trained by the recipes in `analysis/train-recipes` only on permissive
data, each gated against the OpenNLP model for the same task, under
Apache-2.0. The index lists each at its address in a release of this
repository: `models-masc-3.0.0-1` for the MASC models, which
`.github/workflows/models.yml` creates, `models-ud-2.16-1` for the UD
models, which `.github/workflows/models-ud.yml` creates, and
`models-tatoeba-2026-10-10-1` for the language detector, which
`.github/workflows/models-tatoeba.yml` creates, and
`models-tatoeba-fineweb-2026-10-10-1` for the web language detector,
which `.github/workflows/models-fineweb.yml` creates, each with every
model's card and gate records, when it is run by hand on main.
Until it has been, the release's addresses answer 404 and fetching its
models through the zoo fails. Every file's size, SHA-256 and SHA-512 in
the index are the ones its card records, and a workflow publishes
nothing unless the models it trains have them. The attribution of the
data (MASC's, the treebank's, Tatoeba's, or for the web language
detector Tatoeba's, FineWeb-2's and FineWeb's under ODC-By 1.0) is in
each model's card
(the `attribution` field, published as `<model id>.card.json` beside the
model) and in the release notes; the zoo's entries have no field for
it. The language detectors name the sentences they were trained on by
id in `training-sentences.txt`, which the zoo lists as each model's
second file, and the web language detector names its documents
(repository, file, row and FineWeb id) in `training-documents.txt`, its
third. The Tatoeba export it is trained on is kept, unchanged, as release
`data-tatoeba-2026-10-10`, with the CC BY 2.0 FR and CC0 1.0 notices in
its release notes. The Portuguese Porttinari models and the Italian
MarkIT models are in no release (the first for the rights to the text,
the second for failing their gates).

| Model | Trained on | Licence of the data | Class | Checked |
|---|---|---|---|---|
| `turbo-en-masc-3.0.0-sentence-1` | [MASC](https://anc.org/data/masc/) 3.0.0, its Twitter genre left out | CC BY 3.0 US | permissive | as MASC's row under Datasets |
| `turbo-en-masc-3.0.0-tokens-1` | [MASC](https://anc.org/data/masc/) 3.0.0, its Twitter genre left out | CC BY 3.0 US | permissive | as MASC's row under Datasets |
| `turbo-en-masc-3.0.0-pos-1` | [MASC](https://anc.org/data/masc/) 3.0.0, its Twitter genre left out | CC BY 3.0 US | permissive | as MASC's row under Datasets |
| `turbo-en-masc-3.0.0-chunks-1` | [MASC](https://anc.org/data/masc/) 3.0.0, its Twitter genre left out | CC BY 3.0 US | permissive | as MASC's row under Datasets |
| `turbo-es-ud-ancora-2.16-sentence-1` | [UD_Spanish-AnCora r2.16](https://github.com/UniversalDependencies/UD_Spanish-AnCora/blob/r2.16/README.md) | CC BY 4.0 | permissive | as UD_Spanish-AnCora's row under Datasets |
| `turbo-es-ud-ancora-2.16-tokens-1` | [UD_Spanish-AnCora r2.16](https://github.com/UniversalDependencies/UD_Spanish-AnCora/blob/r2.16/README.md) | CC BY 4.0 | permissive | as UD_Spanish-AnCora's row under Datasets |
| `turbo-es-ud-ancora-2.16-pos-1` | [UD_Spanish-AnCora r2.16](https://github.com/UniversalDependencies/UD_Spanish-AnCora/blob/r2.16/README.md) | CC BY 4.0 | permissive | as UD_Spanish-AnCora's row under Datasets |
| `turbo-es-ud-ancora-2.16-lemmas-1` | [UD_Spanish-AnCora r2.16](https://github.com/UniversalDependencies/UD_Spanish-AnCora/blob/r2.16/README.md) | CC BY 4.0 | permissive | as UD_Spanish-AnCora's row under Datasets |
| `turbo-ca-ud-ancora-2.16-sentence-1` | [UD_Catalan-AnCora r2.16](https://github.com/UniversalDependencies/UD_Catalan-AnCora/blob/r2.16/README.md) | CC BY 4.0 | permissive | as UD_Catalan-AnCora's row under Datasets |
| `turbo-ca-ud-ancora-2.16-tokens-1` | [UD_Catalan-AnCora r2.16](https://github.com/UniversalDependencies/UD_Catalan-AnCora/blob/r2.16/README.md) | CC BY 4.0 | permissive | as UD_Catalan-AnCora's row under Datasets |
| `turbo-ca-ud-ancora-2.16-pos-1` | [UD_Catalan-AnCora r2.16](https://github.com/UniversalDependencies/UD_Catalan-AnCora/blob/r2.16/README.md) | CC BY 4.0 | permissive | as UD_Catalan-AnCora's row under Datasets |
| `turbo-ca-ud-ancora-2.16-lemmas-1` | [UD_Catalan-AnCora r2.16](https://github.com/UniversalDependencies/UD_Catalan-AnCora/blob/r2.16/README.md) | CC BY 4.0 | permissive | as UD_Catalan-AnCora's row under Datasets |
| `turbo-fi-ud-ftb-2.16-pos-1` | [UD_Finnish-FTB r2.16](https://github.com/UniversalDependencies/UD_Finnish-FTB/blob/r2.16/README.md) | CC BY 4.0 | permissive | as UD_Finnish-FTB's row under Datasets |
| `turbo-fi-ud-ftb-2.16-lemmas-1` | [UD_Finnish-FTB r2.16](https://github.com/UniversalDependencies/UD_Finnish-FTB/blob/r2.16/README.md) | CC BY 4.0 | permissive | as UD_Finnish-FTB's row under Datasets |
| `turbo-mul-tatoeba-langdetect-1` | the CC BY 2.0 FR sentences of [Tatoeba](https://tatoeba.org)'s export of 2026-10-10, in the 71 languages with at least 1,500 of them after the licence check and de-duplication | CC BY 2.0 FR | permissive | as Tatoeba's row under Datasets |
| `turbo-mul-tatoeba-fineweb-langdetect-1` | the same Tatoeba sentences, and windows of [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) and [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) documents at the revisions `fineweb-files.txt` records, in 100 languages | CC BY 2.0 FR; ODC-By 1.0 | permissive | as Tatoeba's, FineWeb-2's and FineWeb's rows under Datasets |

## OpenNLP models (refused)

| Model | Trained on | Licence of the data | Reason | Checked |
|---|---|---|---|---|
| `langdetect-183.bin` | Leipzig corpora ([model README](https://archive.apache.org/dist/opennlp/models/langdetect/1.8.3/README.txt), [corpora](https://wortschatz.uni-leipzig.de/en/download)) | conflicting: one source says CC BY, another CC BY-NC | terms not confirmed (`LicenseRef-Unconfirmed`) | plan; README here |
| 1.5 `en-chunker` | CoNLL-2000 (Wall Street Journal, LDC) ([1.5 models](http://opennlp.sourceforge.net/models-1.5/)) | LDC | proprietary: refused for shipping, bundling and training; allowed only as a reference that is scored (below) | plan |
| 1.5 `en-pos-maxent` | not published (the OpenNLP 1.5.1 test plan heads its regression-training section "(private) English data", which may not describe these models) | none stated | proprietary: refused for shipping, bundling and training; allowed only as a reference that is scored (below) | plan |
| 1.5 `en-pos-perceptron` | not published (as `en-pos-maxent`) | none stated | proprietary | plan |
| 1.5 English NER, sentence and token models | no corpus named | none stated | no licence | plan |

`en-pos-maxent.bin` and `en-chunker.bin` are used only as references
whose output is scored or compared. Nothing ships, bundles, mirrors or
trains on them, and the zoo does not list them. The `ci` workflow
(`ci.yml`) does not fetch them. The `models` workflow
(`.github/workflows/models.yml`), which runs the recipes' gates before
it publishes their models, downloads them from
<https://opennlp.sourceforge.net/models-1.5/>, checks them against the
SHA-256s `analysis/train-recipes/masc-references.txt` lists, and keeps
them in the repository's GitHub Actions cache. GitHub restores that
cache only in this repository's workflow runs, and it cannot be
downloaded. Anywhere else they are downloaded by hand from that address
into a directory the user names. The SHA-256 of `en-pos-maxent.bin` is
`645a094f45a866687a617385233fd23ae8b0f5fa8b1b76996781a50c17bdcf3d`, and
that of `en-chunker.bin` is
`7861a0c2f134d9c12a022a1ba501e88bc7039f6db72b4140e1bafd1fb5ef76cc`
(`chunk::EN_CHUNKER`, 2,560,304 bytes); the turbo-train-recipes chunker
gate scores it on MASC's held-out documents as the tagger gate scores
`en-pos-maxent.bin`. The
JVM goldens (`PosGolden`, `BeamGolden`) record its tags for the test
sentences and that digest; turbo-sequence's tests check the digest and
compare what turbo-sequence makes of the model with the goldens; and the
turbo-train-recipes tagger gate checks its size and SHA-256 (5,696,197
bytes, `postag::EN_POS_MAXENT`), scores it on MASC's held-out documents
and compares that score with the trained model's.

The goldens tool fetches `langdetect-183.bin` to make the language
detector's goldens (`analysis/goldens-tool/src/recipe/mod.rs`); the zoo does
not list it. The language detector recipe and the web language detector
recipe (`analysis/train-recipes`, #429) score it as their gates'
reference, the web one in both of its gates: each reads the file from a
directory the user names and checks its size and SHA-256
(`langdetect::LANGDETECT_183`), and ships, bundles and trains on nothing
of it.

## Neural annotator checkpoints

The SaT and GLiNER checkpoints docs/neural-annotators.md considers, and
the GLiNER2 and GLiREL checkpoints the relation scorer considers
(docs/gliner2-relations.md). Two are in the zoo's index:
`fastino/gliner2-base-v1` (`fastino-gliner2-base-v1`, permissive) and
`jackboyla/glirel-large-v0` (`jackboyla-glirel-large-v0`, local-only,
fetched from its publisher only after the user accepts CC BY-NC-SA 4.0).
The others were not fetched, so no entry has the SHA-512s an entry
needs. "Checked" is **here**: the
`license` of the model card's metadata at the revision named, read on
2026-10-15 from the hub's model API, and on 2026-10-21 for the GLiNER2
and GLiREL rows.

| Checkpoint | Revision | Licence | Base and data | Class | Checked |
|---|---|---|---|---|---|
| [segment-any-text/sat-1l-sm](https://huggingface.co/segment-any-text/sat-1l-sm) | `00a783439c201b20ee91ee1624dc817f1bf0e578` | MIT | XLM-R (MIT); trained on mC4, then a mixture that includes Universal Dependencies training sets | local-only unless the mixture left out UD's non-commercial treebanks (English GUM and others, under Datasets); to be confirmed against wtpsplit's training configuration | here |
| [segment-any-text/sat-12l-sm](https://huggingface.co/segment-any-text/sat-12l-sm) | `d70c72a9331b2d5a9e82baad00c64964a23a09bb` | MIT | as `sat-1l-sm` | as `sat-1l-sm` | here |
| [segment-any-text/sat-3l](https://huggingface.co/segment-any-text/sat-3l) | `a3b7ba6b61881d619989757aea971cde7d3d6a98` | MIT | XLM-R (MIT); mC4 (ODC-BY) | permissive; needs limited lookahead, which the core lacks | here |
| [urchade/gliner_small-v2.1](https://huggingface.co/urchade/gliner_small-v2.1) | `4e091416cf7c3481db542c2a3d26156916f3a47f` | Apache-2.0 | deberta-v3-small (MIT); `urchade/pile-mistral-v0.1` (Apache-2.0, labels Mistral wrote for text from The Pile) | permissive | here |
| [urchade/gliner_multi-v2.1](https://huggingface.co/urchade/gliner_multi-v2.1) | `443d26d654e0324125a96bebd8e796c14ff2efe6` | Apache-2.0 | mdeberta-v3-base (MIT); as `gliner_small-v2.1` | permissive | here |
| [urchade/gliner_small](https://huggingface.co/urchade/gliner_small) (v1) | `0f0f4e7d3f10e48844110162d6b5c6072ddd5a4e` | CC BY-NC 4.0 | | local-only; not used | here |
| [fastino/gliner2-base-v1](https://huggingface.co/fastino/gliner2-base-v1) | `f9634218e53580c56edf0de97ca1a7d3f1c2354e` | Apache-2.0 | deberta-v3-base (MIT); training data not named on the card | permissive; in the index; the relation scorer and schema extraction (docs/schema-extract.md) | here |
| [jackboyla/glirel-large-v0](https://huggingface.co/jackboyla/glirel-large-v0) | `40a523e12a8432d6da364cf2a195a28755ff04d3` | CC BY-NC-SA 4.0 (the card and the GLiREL repository) | deberta-v3-large (MIT); training data not named on the card | local-only; in the index, fetched after acceptance | here |

## Datasets

| Dataset | Use | Licence | Class | Notes | Checked |
|---|---|---|---|---|---|
| [UD_Spanish-AnCora r2.16](https://github.com/UniversalDependencies/UD_Spanish-AnCora/blob/r2.16/README.md) | sentence, token, POS, lemma | CC BY 4.0 | permissive | the recipes in `analysis/train-recipes` train on it (#415); `LICENSE.txt` and the README's metadata say CC BY 4.0, while a README paragraph still says "The GNU license is inherited from the original dataset"; the README's changelog entry "2021-11-15 v2.9" says "The license changed to CC BY 4.0 (https://doi.org/10.5281/zenodo.4762030)." | here |
| [UD_Catalan-AnCora r2.16](https://github.com/UniversalDependencies/UD_Catalan-AnCora/blob/r2.16/README.md) | sentence, token, POS, lemma | CC BY 4.0 | permissive | the recipes train on it (#415); the same README paragraph and changelog entry as Spanish AnCora | here |
| [UD_Portuguese-Porttinari r2.16](https://github.com/UniversalDependencies/UD_Portuguese-Porttinari/blob/r2.16/README.md) | sentence, token, POS, lemma | CC BY 4.0 | permissive, recipe only | the CC BY 4.0 grant is the treebank authors', for their annotations; its README says "The texts in the treebank are from Folha de São Paulo newspaper, which are publicly available at Kaggle website." No licence from Folha for the articles' text is shown (the Kaggle dataset's own terms were not read), so the rights to the text are not settled: the recipes stay, so a user can train the models locally (#415), and no model trained on it is shipped (`treebank::TextRights::NotSettled`; `UdModelCard::artifact` refuses one whatever its gate says) | here |
| [UD_Finnish-FTB r2.16](https://github.com/UniversalDependencies/UD_Finnish-FTB/blob/r2.16/README.md) | POS, lemma | CC BY 4.0 | permissive | `LICENSE.txt` says "FTB1U is Copyright 2014 FIN-CLARIN" and offers LGPLv3 or later or, at the user's option, CC BY 4.0; the recipes take CC BY 4.0 (#415), and a model's attribution keeps the copyright notice. Its text is grammar examples from VISK with a space between every two tokens, so it has no sentence or token recipe | here |
| [UD_Italian-MarkIT r2.16](https://github.com/UniversalDependencies/UD_Italian-MarkIT/blob/r2.16/README.md) | POS, lemma | CC BY 4.0 | permissive | the recipes train on it (#415), gated against OpenNLP's Italian UD models, which are local-only (below) and only scored. Its text has a space between almost every two tokens (its files have 9 `SpaceAfter=No`, 2 in `train` and 7 in `dev`), so it has no sentence or token recipe | here |
| [MASC](https://anc.org/data/masc/) | English sentences, tokens, Penn POS, noun and verb chunks, named entities | CC BY 3.0 US | permissive | 500K words, manually validated; the recipes in `analysis/train-recipes` train on it, from the 3.0.0 archive published as a release asset of github.com/nancyide/anc-website (tag `datasets-2026-09-25`), recorded by size and digest (`masc::RELEASE`); that anc.org's download page links to it was reported but not checked, as anc.org could not be reached from the machine the recipes were run on. Its Twitter genre (`data/written/twitter`, two documents, 24,180 words, from 2008, source given as "unknown" in their headers) has no terms of its own: the 3.0.0 release's README has no licence text, and anc.org states CC BY 3.0 US for MASC with no exception. Whether Twitter's terms reach the tweets' text is not settled here, and the policy refuses platform-terms data, so the recipes leave the genre out of training and of the held-out set (`masc::Exclusion::TwitterGenre`), as Kristian Rickert decided on 2026-10-15 | here |
| [Open English WordNet](https://github.com/globalwordnet/english-wordnet/blob/main/LICENSE.md) | English lemmas | CC BY 4.0 | permissive | derived from Princeton WordNet; attribute both | here |
| [Princeton WordNet](https://wordnet.princeton.edu/license-and-commercial-use) | English lemmas | WordNet licence (BSD-like) | permissive | | plan |
| [PoliMorf / Morfologik](https://github.com/morfologik/polimorf) | Polish lemmas | BSD-2-Clause | permissive | | plan |
| [GermEval 2014](https://sites.google.com/site/germeval2014ner/data) | German NER | CC BY 4.0 | permissive | the data page says "This data set is distributed under the CC-BY license" and links CC BY 4.0. Two inputs of the `ner` goldens hold its sentences, each crediting D. Benikova, C. Biemann and M. Reznicek (NoSta-D Named Entity Annotation for German: Guidelines and Dataset, LREC 2014) in its comment lines: `germeval-opennlp.txt`, the 4 sentences of OpenNLP's test sample, and `germeval-excerpt.txt`, 105 sentences of `NER-de-train.tsv`, whose line ranges and SHA-512 are in `analysis/goldens-tool/src/recipe/excerpts.rs`. The full train, dev and test files are not checked in | here |
| [MultiCoNER v2](https://registry.opendata.aws/multiconer/) | NER, 12 languages | CC BY 4.0 | permissive | | plan |
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | language detection | ODC-By 1.0 | permissive | also subject to Common Crawl's terms of use; read by the web language detector recipe on GitHub's runners, as Hugging Face could not be reached from the machine it was written on; the windows trained on are listed by document id in `training-documents.txt` | here |
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | language detection (English) | ODC-By 1.0 | permissive | as FineWeb-2's row | here |
| [Tatoeba](https://tatoeba.org/en/downloads) | language detection | CC BY 2.0 FR | permissive | by default; a sentence is CC BY 2.0 FR, CC0 1.0 or has no licence (tatoeba2 at commit 0fbb4506: `docs/database/tables/sentences.sql`, `src/Lib/Licenses.php`). `sentences_detailed.csv` holds the sentences with a licence and `sentences_CC0.csv` the CC0 1.0 ones, so each sentence's licence is read from the two, which relies on Tatoeba having exactly these two sentence licences (recording a run re-checks it); the language detector recipe (`analysis/train-recipes`, #429) keeps the CC BY 2.0 FR ones only and leaves out the CC0 ones, as #429 asks. CC BY 2.0 FR asks for the authors to be credited, and Tatoeba's terms accept the addresses of the sentences used: the model ships with `training-sentences.txt`, the ids of the sentences it was trained on, each at `https://tatoeba.org/sentences/show/<id>`. The zoo names the licence `LicenseRef-CC-BY-2.0-FR` (SPDX has no identifier for it). The export of 2026-10-10 was read on GitHub's runners (downloads.tatoeba.org refused the machine the recipe was written on); on 2026-10-15 the three tatoeba2 files had not changed since 2025-10-28 at the latest, so the export was written with the two sentence licences. It is kept unchanged as release `data-tatoeba-2026-10-10` of this repository, as Tatoeba replaces its export each week | here, the licence rules; the export of 2026-10-10 (`analysis/train-recipes` README, "Full run") |
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
| [UD_Portuguese-DANTEStocks r2.16](https://github.com/UniversalDependencies/UD_Portuguese-DANTEStocks/blob/r2.16/README.md) | | CC BY 4.0 | refused | its text is tweets (the README: "a collection of Brazilian Portuguese tweets"), and the policy refuses platform-terms data, as it does MASC's Twitter genre, so no recipe trains on it | here |
| [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) | NER | Reuters RCV1, research only | refused | | plan |
| [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) | NER | LDC, non-commercial | refused | not a CC licence: an LDC user agreement, outside the 2026-10-14 decision on non-commercial data | plan |
| [MultiNERD](https://github.com/Babelscape/multinerd) | NER | CC BY-NC-SA 4.0 | local-only | non-commercial | plan |
| [WikiNEuRal](https://github.com/Babelscape/wikineural) | NER | CC BY-NC-SA 4.0 | local-only | non-commercial | plan |
| [CoNLL-2002](https://www.clips.uantwerpen.be/conll2002/ner/) | NER | none stated | refused | no licence | plan |
| [NorNE](https://github.com/ltgoslo/norne) | NER | annotations CC0, text restricted | refused | | plan |
| [OpenLID, GlotLID corpora](https://github.com/cisnlp/GlotLID) | language detection | mixed, parts non-commercial | refused | as a whole | plan |
| [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) | language detection | its card allows evaluation only | refused for training | evaluation runs may use it; numbers recorded, files not kept | plan |
| [Leipzig corpora](https://wortschatz.uni-leipzig.de/en/download) | language detection | conflicting (CC BY or CC BY-NC) | refused | stays refused as unconfirmed terms (`LicenseRef-Unconfirmed`) until the terms are confirmed; confirmed as CC BY it would be permissive, as CC BY-NC local-only | plan |
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
data"): requested only at addresses on the original host after the user
accepts that host's terms, used for local training and evaluation only,
and never carried, mirrored or checked in. Any model trained on it is
local-only too and is never shipped; evaluation numbers on it may be
recorded, the files not.

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
the user supplies, with nothing to accept; the zoo requests it only at
addresses under its original source, after the user accepts the
licence; and neither the data nor any model trained on it is shipped,
bundled, mirrored or checked in. By accepting, the user agrees to treat
a model they train on it as non-commercial and keep it on their machine.
These rules are this project's policy; the licences themselves allow
sharing the data and adapted material for non-commercial purposes.

Data that is also no-derivatives (CC BY-NC-ND 4.0) stays refused, as
CC BY-ND 4.0 data does. That is this project's default (2026-10-14), not
what the licences require: version 4.0 of each allows producing adapted
material (CC BY-NC-ND 4.0 for non-commercial purposes) and forbids only
sharing it, so they would allow a model trained on the data and kept on
the user's machine. Earlier versions differ: CC BY-ND 3.0 grants no
right to make adaptations (its section 3).

The non-commercial datasets are in the datasets table above, marked
local-only. The built-in index lists none of them, and none of the
models below. An entry for one records the CC licence (for a model,
under `trained_on`) with the address where the publisher states it, and
the publisher's own location as `origin`, which every file must lie
under: for a UD treebank its repository path on
`raw.githubusercontent.com`
(`https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/`),
not the whole host, and for a model the model publisher's.

The 32 OpenNLP UD 1.3 models trained on the eight non-commercial
treebanks below, four per treebank (sentence, tokens, POS and lemmas),
are local-only by the class rule (Apache-2.0 models, non-commercial
data):

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

`opennlp-it-ud-vit-pos-1.3-2.5.4.bin` and
`opennlp-it-ud-vit-lemmas-1.3-2.5.4.bin` are used only as the references
of the Italian UD recipes' gates in `turbo-train-recipes` (#415). They
are read from a directory the user names, checked against the size and
SHA-256 the recipes record (`ud::IT_UD_VIT_POS`, `ud::IT_UD_VIT_LEMMAS`),
and scored on UD_Italian-MarkIT's `test` file beside the model trained on
MarkIT. Nothing ships, bundles, mirrors or trains on them, the zoo does
not list them, and CI does not fetch them.

## GLiNER small-v2.1 numerical fixtures

`urchade/gliner_small-v2.1` revision
`4e091416cf7c3481db542c2a3d26156916f3a47f` declares Apache-2.0 in its
[model card](https://huggingface.co/urchade/gliner_small-v2.1/blob/4e091416cf7c3481db542c2a3d26156916f3a47f/README.md).
The underlying `microsoft/deberta-v3-small` encoder and SentencePiece tokenizer
declare MIT in their [pinned model card](https://huggingface.co/microsoft/deberta-v3-small/blob/a36c739020e01763fe789b4b85e2df55d6180012/README.md). GLiNER 0.2.21 reference code is Apache-2.0.
The head fixture in `analysis/gliner/tests/fixtures` contains modified, reduced
slices of those trained weights, not a usable extraction checkpoint. Its
provenance lists the slicing rules. Full-width reference outputs and source
SHA-256 values are in `analysis/gliner/tests/fixtures/full-reference.json`.
No full model is added to the zoo by this change. The exact checkpoint above
was checked; this is not a licence decision for other GLiNER checkpoints.
