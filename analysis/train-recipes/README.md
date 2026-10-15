# turbo-train-recipes

Recipes that train models only on permissive data, and the
accuracy gate each model passes before it ships (#401). docs/analysis.md
("Models trained on permissive data" and "Model gates") describes both.

| Module | What it holds |
|---|---|
| `masc` | MASC 3.0.0 (CC BY 3.0 US): the release record, its zoo dataset entry and fetch, the digest of an extracted copy, the fixed split, and the Twitter genre left out of it |
| `segmentation` | The English sentence detector and tokenizer recipes: training on the split's training documents, the held-out samples, scoring a model with OpenNLP's evaluators, the gate against the OpenNLP English UD model |
| `postag` | The English part-of-speech tagger recipe, the same way: training on MASC's Penn tags, the held-out samples, scoring with `POSEvaluator`, the gate against OpenNLP's English Penn tagger (`en-pos-maxent.bin`, OpenNLP 1.5) |
| `namefind` | The English name finder recipes, one per MASC entity type (person, location, organization, date): training as `NameFinderME.train` does with its defaults on MASC's named entities as the entity file links them to Penn tokens, the held-out samples, scoring with `TokenNameFinderEvaluator`, the gate against OpenNLP 1.5's English name finder of the same type (`en-ner-<type>.bin`) |
| `spans` | A held-out unit's counts from the `FMeasure` OpenNLP's evaluator keeps for it, both models' counts paired, and the span types a gate of typed spans (names, chunks) scores and renames |
| `card` | The model card, the model's directory (model, `card.json`, `gate.json`) and its permissive zoo entry, made only for a card whose gate against its recipe's reference (by id and SHA-256) passed on the recorded release |

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
`645a094f45a866687a617385233fd23ae8b0f5fa8b1b76996781a50c17bdcf3d`), and
the four name finders of the same page, checked the same way
(`namefind::EN_NER_*`):

| File | Size | SHA-256 |
|---|---|---|
| `en-ner-person.bin` | 5,207,953 bytes | `687a9263d96b37fced707c9f2ac0560f9edaf54658856395555901924f64dbe4` |
| `en-ner-location.bin` | 5,110,658 bytes | `8fe39e48633f4a86c4132d9c54b396a2d8e0460c1d71e3562dacf976984f447b` |
| `en-ner-organization.bin` | 5,297,172 bytes | `0136c12afe1ac357142260c39bb879b7c9d121e41024114db5a6455b4fd5ba00` |
| `en-ner-date.bin` | 5,030,307 bytes | `1207030923852e1c244919d8f15d9e78c217323728fcf909029abd1703967855` |

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
| `turbo-en-masc-3.0.0-ner-person-1` | `opennlp-en-ner-person-1.5` (`en-ner-person.bin`) | MASC Penn tokens and the `person` entities, one sample per sentence | F-measure of `person` names, one unit per sentence |
| `turbo-en-masc-3.0.0-ner-location-1` | `opennlp-en-ner-location-1.5` (`en-ner-location.bin`) | the same, `location` | F-measure of `location` names |
| `turbo-en-masc-3.0.0-ner-organization-1` | `opennlp-en-ner-organization-1.5` (`en-ner-organization.bin`) | the same, MASC's `org` renamed `organization` | F-measure of `organization` names |
| `turbo-en-masc-3.0.0-ner-date-1` | `opennlp-en-ner-date-1.5` (`en-ner-date.bin`) | the same, `date` | F-measure of `date` names |

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
- The name finders: one model per MASC entity type, as OpenNLP 1.5
  published one English name finder per type, so each is gated against
  the reference of its type; MASC's four types (`person`, `location`,
  `org`, `date`) each have one, so no recipe is ungated. They train as
  `NameFinderME.train` does when the trainer tool is given no parameters
  file (`namefind::training_parameters`): the defaults it puts into
  empty parameters, GIS (`MAXENT`), 300 iterations and cutoff 0, with the
  trainer's one thread; `TokenNameFinderFactory` with the default feature
  generator descriptor (`ner-default-features.xml`), no resources and the
  BIO codec; beam size 3; and the same fixed `Timestamp`. Training uses
  no randomness, so there is no seed. The card records the parameters
  (`training.parameters`).
- The name finders' samples: one per sentence of the documents with a
  named entity file (every document the split reads has one), its Penn
  tokens, and the entities of the recipe's type as the entity file links
  them to Penn tokens (`MascSentence::entities`), each the span from its
  first token to one past its last. OpenNLP's
  `MascNamedEntitySampleStream` is not used: it drops the last token of
  every name of two or more tokens, reads a name of one token at a
  sentence's start as empty, keeps only the first name of a sentence of
  two or more (its removal of overlapping names compares a name with
  itself) and marks every sentence to clear the adaptive data
  (`turbo_formats::masc` documents each; `tests/names.rs` shows them).
  Entities of other types are left out of the sample, as
  `NameSampleTypeFilter` leaves them out, and MASC's `org` is renamed
  `organization`, the type the reference finds. Where two entities of the
  type overlap, the first in `Span.compareTo` order (by start, the longer
  first) is kept and the other left out; an entity whose tokens had a
  gap would be kept as its covering span. A document's first sample
  clears the adaptive data, so what the previous-map feature generator
  learns of a document's names reaches only that document's later
  sentences. A sentence of no tokens is skipped, within
  `MAX_SKIPPED_SENTENCES`. The card records for each side the documents
  read, the names, those with a gap, the overlapping entities and the
  entities of other types left out (`training.names`, `held_out.names`).
- The name finders' gate: both models are scored on the same held-out
  samples with `TokenNameFinderEvaluator`, one unit per sentence, over
  the names of the recipe's type; a name is right only with the held-out
  name's tokens and type. Each reference's outcomes are `other`,
  `<type>-start` and `<type>-cont`, so neither model gives another type.
  The gate resamples whole sentences and recomputes precision and recall
  from each draw's summed counts; it never averages the sentences' own
  F-measures. The sentences of a document are not independent: the
  adaptive data links them, a document repeats its names, and a few
  documents hold most of the held-out names (the movie script
  `data/written/movie-script/pirates` holds 787 of the 1,363 held-out
  persons), so an interval from resampling documents would be wider. The
  references' training data is not published, and the zoo does not list
  them: the recipes record their addresses, sizes and SHA-256
  (`namefind::EN_NER_*`), a run reads them from `--references` and checks
  them, and they are only scored. They are never shipped, bundled or
  trained on.
- How MASC's entities differ from what the references find, and which
  way it leans the gate. The references' training data is not published;
  their manifests record OpenNLP 1.5.0, 100 iterations and cutoff 5, and
  they hold no descriptor. Counted over the documents the split reads:
  - `date` in MASC is any expression of time: its `kind` feature is
    `date` (1,769 entities), `dateTime` (393), `time` (357) and
    `duration` (127), and its most frequent dates are `today`, `now`,
    `yesterday`, `tomorrow`, `tonight` and `night`, durations such as
    `three days` and `next four years`, and times such as `6 pm`. On the
    held-out documents the date reference finds calendar dates and
    `today`, but misses `now` (19 times), `tonight`, `night`, `the 1990s`
    and most durations.
  - `person` in MASC includes some pronouns and possessives (`his` 35
    and `her` 24 times in the training documents, and `'s` 7 times
    held out) and leaves out titles (`Turner`, where the reference finds
    `Mister Turner`).
  - `org` in MASC includes common nouns used as names (`court`,
    `government`, `commission`) and government bodies (`Congress`,
    `Senate`), and is not given to every mention (the held-out e-mail
    `data/written/email/enron/175841` has `Enron` 26 times, 13 of them
    labelled an organization on their own).
  - `location` in MASC includes regions, facilities and the like
    (`type` is `country`, `city`, `region`, `province`, `facility`,
    `us_region`, `highway_street`, `continent`, `attraction` and
    others).
  - MASC's genres are fiction, a movie script, spoken transcripts,
    e-mail, blogs, letters and others besides news, its tokens are its
    own Penn tokens, and the held-out names are concentrated in a few
    documents (above).

  Each of these counts against the reference wherever it does not
  follow MASC, and the trained model learned MASC's conventions, so the
  differences lean the gate towards the trained model. That the trained
  models still lose (below) makes the result firmer, not weaker.
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
out) with the references from `--references`. The sentence
detector's, tokenizer's and tagger's gates passed; the four name
finders' gates failed (below). The name finders were run by themselves
(`--recipe` with their four ids), twice, from a release build, which
took 377 and 351 seconds. The scores are in points, rounded to two
places from the gate records' values (`result.candidate.score`,
`result.reference.score` and `result.difference.mean`, `low` and `high`
in `gate.json`; the card repeats all of them but the difference's mean);
the difference is the paired bootstrap's mean with its 95% interval
(10,000 resamples, seed 154), and a gate passes when the interval does
not start below -0.5.

| Model | Units | Measure | Model's score | Reference's score | Difference (95% interval) | Gate |
|---|---|---|---|---|---|---|
| `turbo-en-masc-3.0.0-sentence-1` | 639 samples of 10 sentences | F-measure | 79.94 | 77.67 | +2.28 (+1.66 to +2.92) | pass |
| `turbo-en-masc-3.0.0-tokens-1` | 6,384 sentences | F-measure | 98.83 | 94.61 | +4.21 (+3.97 to +4.47) | pass |
| `turbo-en-masc-3.0.0-pos-1` | 6,384 sentences, 85,286 tokens scored | accuracy | 95.16 | 91.70 | +3.46 (+3.21 to +3.71) | pass |
| `turbo-en-masc-3.0.0-ner-person-1` | 6,384 sentences, 1,363 names | F-measure | 55.33 | 60.82 | -5.48 (-8.43 to -2.38) | fail |
| `turbo-en-masc-3.0.0-ner-location-1` | 6,384 sentences, 436 names | F-measure | 56.39 | 64.10 | -7.72 (-12.23 to -3.14) | fail |
| `turbo-en-masc-3.0.0-ner-organization-1` | 6,384 sentences, 321 names | F-measure | 44.83 | 42.54 | +2.29 (-6.41 to +10.83) | fail |
| `turbo-en-masc-3.0.0-ner-date-1` | 6,384 sentences, 358 names | F-measure | 37.73 | 42.44 | -4.72 (-12.64 to +3.10) | fail |

Both models of a gate are scored on the same held-out samples, so the
held-out sentences skipped are skipped for both.

| Model | Training samples | Training sentences skipped | Events | Held-out sentences skipped | Held-out tokens left out |
|---|---|---|---|---|---|
| `turbo-en-masc-3.0.0-sentence-1` | 2,496 | 0 | 25,490 | 0 | |
| `turbo-en-masc-3.0.0-tokens-1` | 24,957 | 1 | 1,408,312 | 1 | |
| `turbo-en-masc-3.0.0-pos-1` | 24,958 | 0 | 431,637 | 1 | 6 of 85,292: `"` (2), `PRN` (1), ``UH` `` (3) |
| each `turbo-en-masc-3.0.0-ner-*-1` | 24,958 | 0 | 431,637 | 1 | |

The names each name finder's card records (`training.names` and
`held_out.names`); no entity had a gap:

| Type | Training names | Overlapping, left out | Of other types, left out | Held-out names | Overlapping, left out | Of other types, left out |
|---|---|---|---|---|---|---|
| `person` | 4,894 | 6 | 9,692 | 1,363 | 0 | 1,115 |
| `location` | 3,944 | 63 | 10,585 | 436 | 0 | 2,042 |
| `organization` | 2,934 | 36 | 11,622 | 321 | 0 | 2,157 |
| `date` | 2,714 | 1 | 11,877 | 358 | 0 | 2,120 |

The four name finders failed their gates, so they are not a release:
the run wrote their directories, with cards that record the failed gate,
and exited with status 1, and `ModelCard::zoo_entry` refuses them. The
precision and recall behind the scores, from the summed counts of the
gate records' units (names right, names found, held-out names):

| Type | Model's precision | Model's recall | Reference's precision | Reference's recall |
|---|---|---|---|---|
| `person` | 90.94 (542 of 596) | 39.77 (542 of 1,363) | 84.38 (648 of 768) | 47.54 (648 of 1,363) |
| `location` | 87.86 (181 of 206) | 41.51 (181 of 436) | 72.67 (250 of 344) | 57.34 (250 of 436) |
| `organization` | 72.73 (104 of 143) | 32.40 (104 of 321) | 43.37 (134 of 309) | 41.74 (134 of 321) |
| `date` | 68.89 (93 of 135) | 25.98 (93 of 358) | 54.11 (125 of 231) | 34.92 (125 of 358) |

The trained models are more precise than the references and find far
fewer names. In the movie script that holds 787 of the held-out persons,
the trained model finds 319 of them and the reference 400. The
organization model leads by 2.29 points, but the interval does not rule
out a loss of 6.41, and the date model's does not rule out one of 12.64,
so neither passes. The parameters are `NameFinderME.train`'s defaults,
chosen before the run; MASC's 351 training documents give about 2,700
to 4,900 names a type, and with cutoff 0 and 300 iterations GIS fits
the training documents closely (the date model scores 98.50 on them).
The recipes keep the defaults: a recipe's parameters chosen by looking
at its held-out score would make the gate measure that choice.

As a check made after the run, and not a recipe, the same code trained
the four models with the parameters the references' manifests record
(`Iterations` 100 and `Cutoff` 5 in `namefind::training_parameters`) and
gated them the same way. Every gate still failed: person 56.84 against
60.82, -3.98 (-6.80 to -1.12); location 63.26 against 64.10, -0.84
(-5.43 to +3.75); organization 44.44 against 42.54, +1.90 (-6.28 to
+9.75); date 47.53 against 42.44, +5.08 (-1.92 to +12.05).

The trained models are not checked in or published. A second run into
another directory wrote the same bytes (for the name finders, the same
`card.json` and `gate.json` too):

| Model | Size | SHA-256 |
|---|---|---|
| `turbo-en-masc-3.0.0-sentence-1.bin` | 44,601 bytes | `d03253a5559a230d1f42e1cf881119204302fbb7fefb613ff1af11bcbd3c261e` |
| `turbo-en-masc-3.0.0-tokens-1.bin` | 916,027 bytes | `2c5fe73121c41300e6eae4a3c4d7a2247ade20a1e182d8e36d7781124aa9b7db` |
| `turbo-en-masc-3.0.0-pos-1.bin` | 3,671,103 bytes | `391d7f79c7246e9e56ed9d8849fde129c5bed8683fa33af31e532cb0c37f96ea` |
| `turbo-en-masc-3.0.0-ner-person-1.bin` | 17,616,486 bytes | `dbcbf69d2cce4a876024b1b0ce729eff09c3a6c2188945b494860086f7a19878` |
| `turbo-en-masc-3.0.0-ner-location-1.bin` | 17,610,640 bytes | `0a0abf72f2936a672422c3638e074227ea831204affb4d98c582662523205560` |
| `turbo-en-masc-3.0.0-ner-organization-1.bin` | 17,635,648 bytes | `5a20025774ff14e7bbff92c9d3c79f7a1d7a8cd76989b8212a006145a9356fba` |
| `turbo-en-masc-3.0.0-ner-date-1.bin` | 17,584,650 bytes | `1193738ea3b523903bd02a5e441e265a0f27c317054dc5e919c0f9dcbf9b865e` |

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

## Tests

`tests/recipes.rs` writes corpora in MASC's layout (`tests/common`, no
MASC text, Penn tags written for the tests) and runs everything on
them: the split in the reader's order; the Twitter genre left out of
both sides without moving another document; the seven recipes trained,
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
sentences of both sides, the held-out tokens left out with their tags,
and the names of both sides) checked word for word. The test documents
have entity files in MASC's layout, with persons, locations,
organizations and dates in some of their sentences, so the name recipes
train, score and gate there too, with the names they score, those they
leave out and the cards' name counts checked. It
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
their tags moved one token along, and for the name finders, with their
names moved one token along): as the reference it loses and the
trained model passes, and as the candidate it loses far more than the
margin and fails, which is the check that tells the gate from one that
passes everything. A model against itself passes with no difference.

`tests/names.rs` checks the name recipes' reader on documents written
for it: an entity with a gap kept as its covering span, overlapping
entities and entities of other types left out and counted, `org` read as
`organization`, a sentence of no tokens skipped, a document's first
sample clearing the adaptive data, a side with no name of the recipe's
type or too many skipped sentences refused, and, on the same document,
the names OpenNLP's `MascNamedEntitySampleStream` reads (each of its
differences above) against the whole names the recipes read.

The OpenNLP UD references need `TURBO_GOLDENS_MODELS` naming a
directory that holds them (`cargo run -p turbo-goldens-tool --
fetch-models <dir>`). With it, they are checked against the zoo's
digests and scored on the test corpus, the UD tagger asked for Penn tags
is checked to give only its 17 converted tags, and the binary runs end to
end on the recipes whose references are present (`--recipe`); without it
those parts skip, unless `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`, which
makes them fail. CI runs the test with both set, in the step that
requires the published models.

The tagger's reference, `en-pos-maxent.bin`, and the name finders'
references, `en-ner-*.bin`, are local-only, as in turbo-sequence's tests
of the 1.5 models: their training data is not published, so CI does not
fetch them. The parts that need them run when
`TURBO_TEST_OPENNLP_MODELS` names a directory holding them (a missing
file there fails), and skip without it even with
`TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`. With it, each is checked against
its recorded digest; the tagger's is shown to return its 45 Penn
outcomes unconverted and to tag brackets, hyphens, ellipses, semicolons
and an opening quotation mark as `postag::REFERENCE_MAPPINGS` says, and
a closing quotation mark `''`; each name finder's is shown to be an
OpenNLP 1.5.0 model with no descriptor whose outcomes are `other` and
its type's start and continuation; they are scored and gated, and the
binary runs all seven recipes.

On the test corpus, with both directories set, the published references
score 63.16 (sentences, F-measure), 94.34 (tokens, F-measure), 92.41
(`en-pos-maxent.bin`, accuracy), 0.00 (`en-ner-person.bin`,
`en-ner-location.bin` and `en-ner-organization.bin`, F-measure: they
find none of the test corpus's names) and 75.00 (`en-ner-date.bin`,
F-measure), against 100 for the trained models,
which have seen every held-out sentence; the test checks these numbers,
which show the gate is wired through, not how the recipes do on MASC.
