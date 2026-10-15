# turbo-train-recipes

Recipes that train models only on permissive data, and the
accuracy gate each model passes before it ships (#401). docs/analysis.md
("Models trained on permissive data" and "Model gates") describes both.

| Module | What it holds |
|---|---|
| `masc` | MASC 3.0.0 (CC BY 3.0 US): the release record, its zoo dataset entry and fetch, the digest of an extracted copy, the fixed split, and the Twitter genre left out of it |
| `segmentation` | The English sentence detector and tokenizer recipes: training on the split's training documents, the held-out samples, scoring a model with OpenNLP's evaluators, the gate against the OpenNLP English UD model |
| `postag` | The English part-of-speech tagger recipe, the same way: training on MASC's Penn tags, the held-out samples, scoring with `POSEvaluator`, the gate against OpenNLP's English Penn tagger (`en-pos-maxent.bin`, OpenNLP 1.5) |
| `card` | The model card, the model's directory (model, `card.json`, `gate.json`) and its permissive zoo entry, made only for a card whose gate against its recipe's reference (by id and SHA-256) passed on the recorded release |
| `tatoeba` | Tatoeba's sentence exports: the strict reader of `sentences_detailed.csv` and `sentences_CC0.csv`, each sentence's licence from the two files, and the export's digest |
| `langdetect` | The language detector recipe: the licence check, the table of Tatoeba's and `langdetect-183.bin`'s language codes, the split, training, the gate against `langdetect-183.bin`, its card and the model's directory |

```
cargo run --release -p turbo-train-recipes -- masc --corpus <masc-dir> \
    --references <dir> --out <dir> [--unrecorded-corpus] [--recipe <model id>]...
cargo run --release -p turbo-train-recipes -- fetch-masc --home <zoo-home>
```

`--references` is a directory holding the reference models under their
published file names: `opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin` and
`opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin`, as `cargo run -p
turbo-goldens-tool -- fetch-models <dir>` writes them and checked against
the zoo's SHA-256, and `en-pos-maxent.bin`, downloaded by hand from
<https://opennlp.sourceforge.net/models-1.5/en-pos-maxent.bin> and checked
against the size and SHA-256 the recipe records (5,696,197 bytes,
`645a094f45a866687a617385233fd23ae8b0f5fa8b1b76996781a50c17bdcf3d`).
`--recipe` runs only the recipes named by the trained model's id; without
it every recipe runs. The recipes run in the order of the table below,
whatever the order they are named in. Before anything is trained, every
reference the run needs is read and checked, every model directory it
would write is checked not to exist, and every recipe's training and
held-out samples are read and checked as training and the gate check
them (the skipped sentences of each side, and the held-out tokens whose
tag the tagger's reference cannot give): a run with a reference missing
or wrong, a model directory already there, or a recipe whose data fails
those checks names every such problem and writes nothing. The exit
status is 1 when a gate fails and 2 on an error.

## The recipes

| Model | Reference | Data | Measure |
|---|---|---|---|
| `turbo-en-masc-3.0.0-sentence-1` | `opennlp-en-ud-ewt-sentence-1.3-2.5.4` | MASC sentence samples of 10 sentences | F-measure of sentence spans |
| `turbo-en-masc-3.0.0-tokens-1` | `opennlp-en-ud-ewt-tokens-1.3-2.5.4` | MASC Penn tokens, one sample per sentence | F-measure of token spans |
| `turbo-en-masc-3.0.0-pos-1` | `opennlp-en-pos-maxent-1.5` (`en-pos-maxent.bin`) | MASC Penn tokens and their `msd` tags, one sample per sentence | Word accuracy, one unit per sentence |

- Split: the documents in the reader's order (each directory's headers
  by name); positions 9, 19, 29 and so on are held out, one in ten. No
  randomness, so no seed. A document the reader skips (it logs the
  error and leaves the document out, as OpenNLP's `MascDocumentStream`
  does) would move every document after it to the next position, so a
  corpus with one is refused (`Error::SkippedDocuments`); MASC 3.0.0
  has none. MASC's Twitter genre is left out of both sides
  (`masc::EXCLUSIONS`, `Exclusion::TwitterGenre`), as Kristian Rickert
  decided on 2026-10-15: a document whose header's `textClass` names
  the category `TW` of MASC's taxonomy (`catRef="WR TW"`, "tweets"),
  which in MASC 3.0.0 are the two documents of `data/written/twitter`
  and no other. The positions are counted over every document read, so
  leaving the genre out moves no other document from one side to the
  other. The card lists the documents left out
  (`corpus.excluded_documents`), and the gate record's held-out set is
  named with the exclusion.
- Parameters: those the UD 1.3 models were trained with
  (`ud::training_parameters`: GIS, 300 iterations, cutoff 5, 4 threads),
  the factory defaults for English (for the tagger: the default feature
  generator descriptor, no tag dictionary, beam size 3), OpenNLP 2.5.4's
  whitespace, and a fixed manifest `Timestamp` (2026-10-14T00:00:00Z), so
  a run gives the same model bytes. The sentence detector's and the
  tokenizer's references, UD 1.3 models, were trained with the same
  settings, so they differ from the trained models in their data alone.
  The tagger's reference was not: its manifest records OpenNLP 1.5.0, 100
  iterations and cutoff 5, and it holds a tag dictionary (`tags.tagdict`)
  and no feature generator descriptor, so its gate compares data and
  settings together.
- Samples: what OpenNLP's MASC streams read, except that the token
  stream skips a sentence without a sample (no tokens, or an empty
  token) alone, where OpenNLP's stops reading the corpus, and a sentence
  whose tokens are out of text order alone, which the port's
  `TokSpanEventStream` refuses (`segmentation::read_token_samples`); and
  the part-of-speech stream skips a sentence with a token without a tag
  alone, where OpenNLP's throws and ends the training run; a sentence of
  no tokens gives the tagger no event and no held-out unit, and is
  skipped too. MASC 3.0.0 has one sentence whose tokens are out of text
  order, in `data/spoken/face-to-face/Bed012`: its Penn layer has a
  token `non-deterministic` over three quarks and a token `-` over the
  second, inside it. OpenNLP's `TokSpanEventStream` would make events of
  it, among them a split inside `non-deterministic` after `non-`, where
  it also gave a no-split. Held-out sentences are skipped the same way.
  The card records how many training sentences
  (`training.skipped_sentences`) and how many held-out sentences
  (`held_out.skipped_sentences`) were skipped, and a recipe fails when
  its reader skips more than one in a thousand of either
  (`MAX_SKIPPED_SENTENCES`), so a held-out document whose tags are
  missing cannot drop out of the gate. In the recorded release (corpus
  digest `445449f5...`, the Twitter genre left out), the token recipe
  skips that one training sentence of 24,958 and the tagger none; of
  the 6,385 held-out sentences the token and tagger recipes each skip
  one, a sentence of no tokens; the sentence detector skips none.
- The tagger's gate: MASC's tags are Penn tags, so the reference is
  OpenNLP's English maxent tagger of the 1.5 models, `en-pos-maxent.bin`.
  OpenNLP's only English taggers trained on Penn tags are that one and
  the 1.5 perceptron tagger; the maxent one is trained with the
  algorithm the recipe uses. The reference is asked for Penn tags
  (`PosTagFormat::Penn`), its own outcomes, which come out unconverted,
  and the trained model for its outcomes as they are, MASC's tags; the
  card records both formats. The trained model's tag is compared with
  the held-out tag as written. The reference's tag is compared through a
  mapping of its tags (`postag::REFERENCE_MAPPINGS`), where MASC 3.0.0
  follows another convention than the reference: brackets, which MASC
  tags `(` and `)` and the reference `-LRB-` and `-RRB-`; hyphens,
  which MASC tags `-` or `:` and the reference `:` (MASC tags most en
  and em dashes `NN`, which the mapping does not touch); ellipses, `...`
  in MASC and `:` in the reference; the two semicolons MASC tags `;` (it
  tags the other 725 `:`, as the reference does); and opening quotation
  marks, which MASC tags `''` as it tags closing ones (``` `` ``` only
  for a backtick), and the reference ``` `` ```. For those held-out tags
  the reference's tag of the other convention counts as right too; the
  trained model is not given that. The mapping is by tag, not by token,
  so it is lenient: the reference's ``` `` ``` counts right on every
  held-out `''`, also on a closing quotation mark (34 of the held-out
  `"` tokens it credits close a quotation and 30 are neither opening
  nor closing plainly). It only ever favours the reference, and those
  64 tokens are at most 0.08 points. A held-out token whose tag the
  reference cannot give, as written or through the mapping, would count
  against it whatever it did, so it is left out of both models' scores;
  the card lists those tokens by tag (`held_out.tokens.left_out`), and
  the recipe fails when they are more than one in a thousand of the
  held-out tokens (`MAX_TOKENS_LEFT_OUT`). The card also lists the
  reference's ten most frequent mismatches on the scored tokens
  (`held_out.tokens.reference_mismatches`), so a gap of convention the
  mapping does not cover, or an effect of the reference's tag
  dictionary, shows. The gate record (`gate.json`) holds the skipped
  held-out sentences and the scoring too (the tokens, those left out and
  the mapping), and its samples digest covers which tokens were scored
  and the mapping as well as the tokens and their tags. The reference's
  training data is not published, and the zoo does not list it: the
  recipe records its
  address, size and SHA-256 (`postag::EN_POS_MAXENT`), a run reads it
  from `--references` and checks it, and it is only scored. It is never
  shipped or bundled, and the trained model is trained on MASC alone.
  The English UD tagger is not the reference: asked for Penn tags,
  `POSTagFormatMapper` turns each of its UD outcomes into one Penn tag
  (`VERB` into `VB`, `NOUN` into `NN`, `PUNCT` into `.`), so it can never
  give `VBD`, `NNS`, `,` or `TO`, and a gate against it would pass any
  tagger.
- Licence: the models are Apache-2.0, trained on CC BY 3.0 US data, so
  their zoo class is permissive; the card carries the attribution CC BY
  asks for, which ships with the model: MASC 3.0.0, by the American
  National Corpus project (anc.org), its address and its licence.
  MASC's Twitter genre (two documents, 24,180 words, tweets from 2008
  whose headers give their source as unknown) is neither trained on nor
  scored, as Kristian Rickert decided on 2026-10-15: the licence policy
  refuses data under a platform's terms, and neither the release nor
  anc.org says whether Twitter's terms reach the tweets' text
  (docs/licences.md, "Datasets").
  Whether the ANC asks for a particular citation is to be confirmed on
  their site, which could not be reached from the machine the recipes
  were written and run on.

## The release record

MASC is fetched through the zoo by digest and read from its extracted
directory. It is not checked in: CC BY would allow it with attribution,
but the policy prefers fetching by digest. `masc::RELEASE` records the
release:

| Field | Value |
|---|---|
| Archive | `MASC-3.0.0.zip`, zip |
| Address | <https://github.com/nancyide/anc-website/releases/download/datasets-2026-09-25/MASC-3.0.0.zip> |
| Size | 36,358,521 bytes |
| SHA-256 | `0b300d96016233c7c79b43559b477a6bb0ed16bb416d5c517eccb108a6228493` |
| SHA-512 | `a88795055fab5c600f26215cbe6143c2325627adaf9ae80d06c7e815a76834debd4b7728a745aa67608ee1d028b35ca139b2b90760e69d0e7227af83bf05d6a9` |
| Corpus directory | `MASC-3.0.0`, 4,425 files |
| Corpus digest | `445449f535cac62e74b2712ad975d7b727e72b8276720dd27492266565ae0afd` |

The archive is a release asset of the GitHub repository
`github.com/nancyide/anc-website` (tag `datasets-2026-09-25`), not a
file on anc.org itself. That anc.org's MASC download page links to it
was reported but not checked: anc.org could not be reached from the
machine the recipes were run on. What was checked there, on the
downloaded archive: its size, SHA-256 and SHA-512 above; it holds one
directory, `MASC-3.0.0`, with 4,425 regular files and no links; its
README dates the release 2012-12-15; and the corpus digest above. The
corpus digest (`masc::corpus_digest`) is the SHA-256 of a `sha256sum`
manifest of every file under that directory, so this gives the same
digest:

```
cd MASC-3.0.0 && find . -type f -printf '%P\n' | LC_ALL=C sort |
  while IFS= read -r f; do printf '%s  %s\n' "$(sha256sum -- "$f" | cut -d' ' -f1)" "$f"; done |
  sha256sum
```

`fetch-masc --home <zoo-home>` fetches the archive into
`<zoo-home>/artifacts/masc-3.0.0/` and checks its size and digests; the
zoo does not extract archives, so it is unzipped by hand, and `masc
--corpus` names the `MASC-3.0.0` directory it extracts to. `masc`
refuses a corpus directory whose digest is not the recorded one unless
given `--unrecorded-corpus`, and then writes cards with
`release_recorded: false`, which are not a release of the models (the
tests run the binary on their own corpora that way).

## Gates on the recorded release

`masc` run on the recorded release (corpus digest
`445449f535cac62e74b2712ad975d7b727e72b8276720dd27492266565ae0afd`: 351
training documents and 39 held out, the two of the Twitter genre left
out) with the three references from `--references`. Every gate passed.
The scores are in points, rounded to two places from the gate records'
values (`result.candidate.score`, `result.reference.score` and
`result.difference.mean`, `low` and `high` in `gate.json`; the card
repeats all of them but the difference's mean); the difference is the
paired bootstrap's mean with its 95% interval (10,000 resamples, seed
154), and a gate passes when the interval does not start below -0.5.

| Model | Units | Measure | Model's score | Reference's score | Difference (95% interval) | Gate |
|---|---|---|---|---|---|---|
| `turbo-en-masc-3.0.0-sentence-1` | 639 samples of 10 sentences | F-measure | 79.94 | 77.67 | +2.28 (+1.66 to +2.92) | pass |
| `turbo-en-masc-3.0.0-tokens-1` | 6,384 sentences | F-measure | 98.83 | 94.61 | +4.21 (+3.97 to +4.47) | pass |
| `turbo-en-masc-3.0.0-pos-1` | 6,384 sentences, 85,286 tokens scored | accuracy | 95.16 | 91.70 | +3.46 (+3.21 to +3.71) | pass |

Both models of a gate are scored on the same held-out samples, so the
held-out sentences skipped are skipped for both.

| Model | Training samples | Training sentences skipped | Events | Held-out sentences skipped | Held-out tokens left out |
|---|---|---|---|---|---|
| `turbo-en-masc-3.0.0-sentence-1` | 2,496 | 0 | 25,490 | 0 | |
| `turbo-en-masc-3.0.0-tokens-1` | 24,957 | 1 | 1,408,312 | 1 | |
| `turbo-en-masc-3.0.0-pos-1` | 24,958 | 0 | 431,637 | 1 | 6 of 85,292: `"` (2), `PRN` (1), ``UH` `` (3) |

The trained models are not checked in or published. A second run into
another directory wrote the same bytes:

| Model | Size | SHA-256 |
|---|---|---|
| `turbo-en-masc-3.0.0-sentence-1.bin` | 44,601 bytes | `d03253a5559a230d1f42e1cf881119204302fbb7fefb613ff1af11bcbd3c261e` |
| `turbo-en-masc-3.0.0-tokens-1.bin` | 916,027 bytes | `2c5fe73121c41300e6eae4a3c4d7a2247ade20a1e182d8e36d7781124aa9b7db` |
| `turbo-en-masc-3.0.0-pos-1.bin` | 3,671,103 bytes | `391d7f79c7246e9e56ed9d8849fde129c5bed8683fa33af31e532cb0c37f96ea` |

The mapping of ``` `` ``` to MASC's `''` makes the tagger's reference
right on 336 more held-out tokens, 0.39 points: of the 644 held-out `"`
tokens MASC tags `''`, it tags 336 ``` `` ``` (272 that open a
quotation, 34 that close one and 30 neither plainly). The mapping is
lenient, by tag and not by token: a rule that knew which marks open
would not credit the 64 that do not, 0.08 points that only ever favour
the reference. Its ten most
frequent mismatches on the scored tokens, as the card lists them
(held-out tag, then the reference's): `NN` as `JJ` 446, `NN` as `NNP`
333, `NN` as `VB` 263, `JJ` as `NN` 230, `VBP` as `VB` 225, `NNP` as
`NN` 187, `NN` as `RB` 178, `IN` as `RB` 169, `NN` as `.` 146 and `NN`
as `VBP` 137.

Many of the `NN` rows are punctuation, which MASC tags `NN`: over the
split's documents it tags `[` (301) and `]` (312) `NN` every time, and
mostly `NN` the curly quotation marks (`“` 993 of 1,010 times, `”`
1,031 of 1,047, `‘` all 100, `’` 1,259 of 1,264), en dashes (91 of
100), em dashes (114 of 120) and `*` (all 307). On the held-out set
these are 1,309 `NN` tokens. The reference is wrong on 947 of them,
mostly because it does not know punctuation outside ASCII (it tags `”`
`.` 100 times and `''` 118 times), and the trained model, which learned
MASC's tags, on 7, so about 1.10 points of the 3.46-point lead come
from these tokens. Of them, only `”` as `''` is a Penn convention a rule
that looked at the token could credit; the recipe adds no mapping for
them. Scored without the 1,309 tokens, the gate still passes: 95.09
against 92.70, a lead of +2.39 (95% interval +2.18 to +2.63). The `NN`
rows also hold words that are not nouns, such as `pray`, `make`,
`right` and `fine`, many of them from a few held-out documents
(`data/written/fiction/Nathans_Bylichka`,
`data/written/movie-script/pirates`, `data/written/ficlets/1402` and
`data/spoken/debate-transcript/3rd_Bush-Kerry`). The `VBP` rows are
verbs such as `do`, `think` and `have`, and the `IN` rows words such as
`so`, `around`, `on`, `out` and `up`.

## The language detector recipe

| Model | Reference | Data | Measure |
|---|---|---|---|
| `turbo-mul-tatoeba-langdetect-1` | `opennlp-langdetect-183` (`langdetect-183.bin`) | Tatoeba's CC BY 2.0 FR sentences of the languages both models cover, one sample per sentence | accuracy, one unit per held-out sentence |

```
cargo run --release -p turbo-train-recipes -- tatoeba --export <dir> \
    --references <dir> --out <dir> [--unrecorded-export]
```

`--export` names a directory holding `sentences_detailed.csv` and
`sentences_CC0.csv`, extracted from Tatoeba's weekly exports
(`tatoeba::DETAILED_ARCHIVE_URL` and `tatoeba::CC0_ARCHIVE_URL`).
`--references` names a directory holding `langdetect-183.bin` (`cargo run
-p turbo-goldens-tool -- fetch-models <dir>` writes it), checked against
the size and SHA-256 the recipe records (`langdetect::LANGDETECT_183`:
10,568,188 bytes,
`2ddf585fac2e02a9dcfb9a4a9cc9417562eaac351be2efb506a2eaa87f19e9d4`). The
model directory, the reference and the export are checked before
anything is trained. The exit status is 1 when the gate fails
and 2 on an error.

- Data and licence check. Tatoeba's `sentences.license` column is
  `enum('CC BY 2.0 FR', 'CC0 1.0') DEFAULT NULL`, and both export
  queries keep a sentence only when its `license` is not empty, which
  leaves out `NULL` too (tatoeba2 at commit 0fbb4506:
  `docs/database/tables/sentences.sql`, `src/Lib/Licenses.php` and
  `docs/database/scripts/weekly_exports.sql`). `sentences_detailed.csv`
  holds every sentence with a licence that is not marked incorrect;
  `sentences_CC0.csv` holds the CC0 1.0 ones. So a sentence of the
  detailed file is CC0 1.0 when the CC0 file lists its id and CC BY 2.0
  FR otherwise. This relies on Tatoeba having exactly these two
  sentence licences: a third would be read as CC BY 2.0 FR, and nothing
  in the files would show it, so recording a run re-checks those three
  files ("Full run" below). The reader (`tatoeba`) reads the MySQL
  `OUTFILE` layout strictly (tab between fields, a backslash before a
  backslash, tab or newline, `\0` for NUL, `\N` for NULL) and refuses
  any other escape, a wrong number of fields and a repeated id.
- The two files are not one snapshot: each is written by its own
  statement on the live database, the CC0 file 14 statements after the
  detailed one, so a sentence added, edited or deleted in between can be
  in one file only, or in both with other values. A sentence is
  classified by its id alone. The CC0 rows whose language or text
  differ from the detailed file's are counted (`cc0_differing` in the
  card) and still make the sentence CC0; the CC0 ids the detailed file
  does not hold are counted (`cc0_not_in_detailed`) and otherwise
  ignored. A misread licence is never one the recipe may not use: a
  sentence read as CC0 is left out, and one read as CC BY 2.0 FR is
  under it or under CC0, which allows any use. The files carry no export
  date and no header, so the reader cannot tell the files of two
  different weeks apart and does not refuse them; the archives' tar
  headers give each file's time, which recording a run checks.
- The licence check keeps CC BY 2.0 FR only
  (`langdetect::ACCEPTED_LICENCES`), as #429 asks; the CC0 sentences,
  which the policy would also allow, are left out and counted per
  language (`licence_left_out` in the card). CC BY 2.0 FR is permissive
  (`LicenseRef-CC-BY-2.0-FR` in the zoo, as SPDX has no identifier for
  it) and asks for the authors to be credited; Tatoeba's terms accept
  the addresses of the sentences used for that. The model's directory
  holds `training-sentences.txt` (`langdetect::SENTENCES_FILE`): the id
  of each sentence trained on, one per line in ascending order, each
  sentence shown with its author at `https://tatoeba.org/sentences/show/<id>`.
  The card lists the file (`training_sentences`: the count, size and
  digests) and its `attribution` points at it, and the zoo entry ships it
  beside the model. No username is stored.
- The export's record. The exports are replaced each week at the same
  addresses, so an export is recorded by its digest
  (`Export::digest`: the SHA-256 of a `sha256sum` manifest of the two
  files), not by address. `langdetect::EXPORT` records none (see
  "Full run" below), so `tatoeba` refuses every export unless given
  `--unrecorded-export`, and then writes a card with `recorded: false`,
  which is not a release of the model.
- Languages. `langdetect::LANGUAGES` is the explicit table of the 100
  Tatoeba languages the recipe reads and the codes of
  `langdetect-183.bin` (ISO 639-3) counted right for each. 93 use the
  same code on both sides. Seven pairs are a macrolanguage and one of
  its individual languages: Tatoeba's `est`, `lvs` and `pes` are
  counted right as `est` or `ekk`, `lvs` or `lav`, and `pes` or `fas`
  (the reference has both codes of each pair; the largest confusion in
  its own evaluation report is `fas` taken for `pes`, 1,205 of 3,982
  samples), and Tatoeba's
  `zsm`, `npi`, `mlg` and `swh` as the reference's `msa`, `nep`, `plt`
  and `swa`. The reference's 103 codes are each in exactly one pair; a
  test checks the table against the model's outcomes. The Tatoeba codes
  are those of Tatoeba's language list; tatoeba.org could not be reached
  from the machine the recipe was written on, so they are not checked
  against an export. A code the export does not use leaves its
  language with no sentences, listed in the card's
  `excluded_languages`; the card also counts the export's sentences of
  languages outside the table (`other_languages`). The trained model's
  outcomes are Tatoeba's codes. The mapping is lenient and can only
  favour the reference.
- Split (`langdetect::SPLIT`). Per language, after the licence check, a
  text that occurs again is left out (the sentence with the smallest id
  is kept), so no text is both trained on and held out. The sentences in
  id order are shuffled as `Collections.shuffle(list, new
  Random(429))` does, each language with its own `Random`; the first
  500 are held out and the next ones, at most 5,000, trained on. A
  language with fewer than 1,500 sentences left (500 held out and at
  least 1,000 to train on) is left out of training and of the gate, and
  listed in the card with its count. 500 sentences per language keep
  the interval of the gate narrow beside its half-point margin over up
  to 50,000 units; the cap of 5,000 keeps the events of 100 languages in
  memory on one machine.
- Training: `LanguageDetectorME.train` with the default factory and the
  parameters `langdetect-183.bin` was most likely trained with. Its
  manifest records only `TrainerType=Event` and an event hash, and the
  command line trainer given no parameters uses
  `ModelUtil.createDefaultTrainingParameters()`: `MAXENT`, 100
  iterations, cutoff 5, one thread; `train` adds the `OnePass` indexer.
  The training script in OpenNLP's corpus repository could not be read
  to confirm this (svn.apache.org could not be reached). The manifest's
  `Timestamp` is fixed, so a run gives the same bytes.
- Gate: every held-out sentence is a unit, right for the trained model
  when its best language (`predictLanguage`) is the sentence's Tatoeba
  code and right for the reference when its best language is one of the
  codes the table gives. Both models are run by turbo-langdetect, whose
  probabilities are the JVM's bit for bit, except for a text with
  equal-hash n-grams in a bucket that became a tree, where they can
  differ in the last bits (docs/langdetect.md). 10,000 resamples, seed 154; a
  gate passes when the difference's interval does not start below -0.5.
- Not used: FineWeb-2 (ODC-By 1.0, permissive), as Hugging Face could
  not be reached; FLORES+ (evaluation only), for the same reason; the
  Leipzig corpora, whose terms are not confirmed (docs/licences.md). The
  gate uses only the Tatoeba held-out split, which neither model trained
  on, so no evaluation-only data is fetched or kept.

### Which way the gate leans

The held-out units are single Tatoeba sentences, which are short (#429
says so; the full run will give their lengths). The
reference was trained on the Leipzig corpora's news and web text, and
its own evaluation (`report.txt` in `langdetect-183.bin`) scored samples
of 123 to 1,668 characters, 533 on average, at 98.81% accuracy over its
103 languages. A detector given one short sentence sees few n-grams,
so both models can be expected to score lower here, and the trained model, which
learned from sentences of the same kind and source as the held-out ones,
has the advantage. It also chooses among fewer languages (only those
with 1,500 sentences), so it has fewer to confuse; the reference keeps
all 103. The table's lenient mapping leans the other way, but only on
the seven pairs. A pass therefore says the trained model is not worse
than `langdetect-183.bin` on short Tatoeba sentences of the shared
languages; it says nothing about longer text, where the reference was
tuned.

### Full run

Not done: downloads.tatoeba.org, tatoeba.org and Hugging Face refused
the connection (403) from the machine the recipe was written on, so the
export could not be fetched, and no export is recorded, no model
trained and no gate run on Tatoeba. On a machine that can reach them:

```
mkdir -p tatoeba/archives tatoeba/export models
curl -fL -o tatoeba/archives/sentences_detailed.tar.bz2 https://downloads.tatoeba.org/exports/sentences_detailed.tar.bz2
curl -fL -o tatoeba/archives/sentences_CC0.tar.bz2 https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2
tar -xjf tatoeba/archives/sentences_detailed.tar.bz2 -C tatoeba/export
tar -xjf tatoeba/archives/sentences_CC0.tar.bz2 -C tatoeba/export
cargo run -p turbo-goldens-tool -- fetch-models models
cargo run --release -p turbo-train-recipes -- tatoeba --export tatoeba/export \
    --references models --out run-a --unrecorded-export
cargo run --release -p turbo-train-recipes -- tatoeba --export tatoeba/export \
    --references models --out run-b --unrecorded-export
cmp run-a/turbo-mul-tatoeba-langdetect-1/turbo-mul-tatoeba-langdetect-1.bin \
    run-b/turbo-mul-tatoeba-langdetect-1/turbo-mul-tatoeba-langdetect-1.bin
```

Each run prints the export's digest, the CC0 file's counts, the
languages trained on and left out, and the gate. To record the run:

1. Re-check the licence values in tatoeba2's current source and record
   the commit checked: `docs/database/tables/sentences.sql` still gives
   `license` the values `CC BY 2.0 FR` and `CC0 1.0` and no other,
   `src/Lib/Licenses.php` lists no other sentence licence, and
   `docs/database/scripts/weekly_exports.sql` still selects the detailed
   file with `license != ''` and the CC0 file with `license = 'CC0 1.0'`,
   in the same columns. They were checked at commit 0fbb4506. If any of
   them changed, the reader's licence classification (`tatoeba`) must be
   revised before a run is recorded.
2. Check that both files are of one weekly export: `tar -tvjf` on each
   archive gives its file's time, and the two must be of one run of
   Tatoeba's export (written one after the other, not days apart).
3. Set `langdetect::EXPORT` to the digest, and check the card's
   `other_languages` and `excluded_languages` for a Tatoeba code the
   table spells differently.
4. Record here, from `card.json` and `gate.json`, the export's digest
   and file sizes, the CC0 file's `cc0_records`, `cc0_differing` and
   `cc0_not_in_detailed`, the languages and sentences on each side
   (`per_language`: `sentences`, `licence_left_out`, `duplicates`,
   `training`, `held_out`, and each model's correct count), the
   languages left out with their counts, the gate's scores and
   interval, and the model's size and SHA-256 and the size and SHA-256
   of `training-sentences.txt` from both runs.

## Tests

`tests/recipes.rs` writes corpora in MASC's layout (`tests/common`, no
MASC text, Penn tags written for the tests) and runs everything on
them: the split in the reader's order; the Twitter genre left out of
both sides without moving another document; the three recipes trained,
scored and gated; the same bytes from a second training; the card, the
model's directory and its zoo entry; the skipped sentences recorded and
bounded on both sides; a sentence with a token inside another skipped
by the token recipe alone; the tagger's tags compared as written and
the reference's through the mapping of its tags, with the tokens of a
tag the reference cannot give left out, and a
reference that cannot give most tags refused; and the binary's refusal
of an unrecorded corpus, of a missing or wrong reference, of an
existing model directory and of held-out data with too many skipped
sentences before it trains anything, and its output, with the lines it
prints of each card (`ModelCard::summary`: the samples and skipped
sentences of both sides, and the held-out tokens left out with their
tags) checked word for word. It
also reads the MASC corpora checked in for the goldens
(`analysis/goldens/data/inputs/formats/masc-*.txt`) with the tagger
recipe's reader: it reads every sample OpenNLP's `MascPOSSampleStream`
read from them in the JVM (the `masc/pos` goldens, whose generator reads
on after an exception), in order, skips one sentence for each exception
the JVM recorded, and refuses the corpus the JVM could not open with the
JVM's exception. The test corpora are made of eight sentences, so a
model trained on one has seen every held-out sentence and its score
there measures nothing. The gate is checked against a weaker model,
trained on plainer text (for the tagger, on the same sentences with
their tags moved one token along): as the reference it loses and the
trained model passes, and as the candidate it loses far more than the
margin and fails, which is the check that tells the gate from one that
passes everything. A model against itself passes with no difference.

The OpenNLP UD references need `TURBO_GOLDENS_MODELS` naming a
directory that holds them (`cargo run -p turbo-goldens-tool --
fetch-models <dir>`). With it, they are checked against the zoo's
digests and scored on the test corpus, the UD tagger asked for Penn tags
is checked to give only its 17 converted tags, and the binary runs end to
end on the recipes whose references are present (`--recipe`); without it
those parts skip, unless `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`, which
makes them fail. CI runs the test with both set, in the step that
requires the published models.

The tagger's reference, `en-pos-maxent.bin`, is local-only, as in
turbo-sequence's tests of the 1.5 models: its training data is not
published, so CI does not fetch it. The parts that need it run when
`TURBO_TEST_OPENNLP_MODELS` names a directory holding it (a missing file
there fails), and skip without it even with
`TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`. With it, it is checked against the
recorded digest, shown to return its 45 Penn outcomes unconverted and
to tag brackets, hyphens, ellipses, semicolons and an opening quotation
mark as `postag::REFERENCE_MAPPINGS` says, and a closing quotation mark
`''`, scored and gated, and the binary runs all three recipes.

On the test corpus, with both directories set, the published references
score 63.16 (sentences, F-measure), 94.34 (tokens, F-measure) and 92.41
(`en-pos-maxent.bin`, accuracy), against 100 for the trained models,
which have seen every held-out sentence; the test checks these numbers,
which show the gate is wired through, not how the recipes do on MASC.

`tests/langdetect.rs` runs the language detector recipe on an export
written for the project (`tests/data/tatoeba`: 90 records in the
layout of the detailed file, not in id order, with ten CC BY 2.0 FR
sentences in each of eight languages, two English CC0 sentences, a
second copy of a German text, three Turkish sentences, two Cantonese
ones and one record with no language; and a CC0 file written as if
after the detailed one, with one of the CC0 sentences edited and a
sentence the detailed file does not hold), with a split scaled to it (four
held out and five trained on per language, eight needed) and the Rust
language fixture (`analysis/goldens/data/fixtures/langdetect-fixture-rust.bin`)
as the reference. It checks the escapes and each sentence's licence, by
id, with the CC0 rows that differ or are not in the detailed file
counted; the licence check, de-duplication and per-language counts;
that the split is the documented shuffle, that the records' order in
the file does not change it and that another seed does; the languages
left out under a threshold; the same bytes from a second training; the
gate over 32 held-out sentences (too few to pass, which the test also
checks), a reference code the reference does not have refused and a
two-code mapping counted; the card, the model's directory, that
`training-sentences.txt` lists exactly the ids trained on, and each
condition a zoo entry needs; the export record; and the binary's
refusal of a missing reference. With
`TURBO_GOLDENS_MODELS` naming a directory holding `langdetect-183.bin`
(required with `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`), it checks the
table's codes against the model's 103 outcomes, and that the binary
refuses the export as unrecorded and, given `--unrecorded-export`,
refuses it for having no language with 1,500 sentences.
