# turbo-train-recipes

Recipes that train models only on permissive data, and the
accuracy gate each model passes before it ships (#401, #415).
docs/analysis.md ("Models trained on permissive data" and "Model gates")
describes both. The MASC recipes come first here; the recipes on the
CC BY 4.0 Universal Dependencies treebanks are under "The UD recipes".

| Module | What it holds |
|---|---|
| `masc` | MASC 3.0.0 (CC BY 3.0 US): the release record, its zoo dataset entry and fetch, the digest of an extracted copy, the fixed split |
| `segmentation` | The English sentence detector and tokenizer recipes: training on the split's training documents, the held-out samples, scoring a model with OpenNLP's evaluators, the gate against the OpenNLP English UD model |
| `postag` | The English part-of-speech tagger recipe, the same way: training on MASC's Penn tags, the held-out samples, scoring with `POSEvaluator`, the gate against OpenNLP's English Penn tagger (`en-pos-maxent.bin`, OpenNLP 1.5) |
| `card` | The model card, the model's directory (model, `card.json`, `gate.json`) and its permissive zoo entry, made only for a card whose gate against its recipe's reference (by id and SHA-256) passed on the recorded release |
| `treebank` | The CC BY 4.0 UD treebanks at release 2.16: each repository's `LICENSE.txt`, `README.md` and `train`, `dev` and `test` files with their sizes and digests, the licence check, the zoo dataset entry and fetch |
| `ud` | The UD recipes: the samples, training on `train`, scoring on `test` with OpenNLP's evaluators, the gate against the OpenNLP UD 1.3 model or the record of a model scored alone |
| `ud_card` | The UD model card, with the treebank's licence, attribution and citation, the model's directory and its zoo entry |

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
reference the run needs is read and checked, and every model directory
it would write is checked not to exist: a run with a reference missing
or wrong, or a model directory already there, names every such problem
and writes nothing. The exit status is 1 when a gate fails and 2 on an
error.

## The recipes

| Model | Reference | Data | Measure |
|---|---|---|---|
| `turbo-en-masc-3.0.0-sentence-1` | `opennlp-en-ud-ewt-sentence-1.3-2.5.4` | MASC sentence samples of 10 sentences | F-measure of sentence spans |
| `turbo-en-masc-3.0.0-tokens-1` | `opennlp-en-ud-ewt-tokens-1.3-2.5.4` | MASC Penn tokens, one sample per sentence | F-measure of token spans |
| `turbo-en-masc-3.0.0-pos-1` | `opennlp-en-pos-maxent-1.5` (`en-pos-maxent.bin`) | MASC Penn tokens and their `msd` tags, one sample per sentence | Word accuracy, one unit per sentence |

- Split: the documents in the reader's order (each directory's headers
  by name); positions 9, 19, 29 and so on are held out, one in ten. No
  randomness, so no seed.
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
  token) alone, where OpenNLP's stops reading the corpus, and the
  part-of-speech stream skips a sentence with a token without a tag
  alone, where OpenNLP's throws and ends the training run; a sentence of
  no tokens gives the tagger no event and no held-out unit, and is
  skipped too. Held-out sentences are skipped the same way. The card
  records how many training sentences (`training.skipped_sentences`) and
  how many held-out sentences (`held_out.skipped_sentences`) were
  skipped, and a recipe fails when its reader skips more than one in a
  thousand of either (`MAX_SKIPPED_SENTENCES`), so a held-out document
  whose tags are missing cannot drop out of the gate.
- The tagger's gate: MASC's tags are Penn tags, so the reference is
  OpenNLP's English maxent tagger of the 1.5 models, `en-pos-maxent.bin`.
  OpenNLP's only English taggers trained on Penn tags are that one and
  the 1.5 perceptron tagger; the maxent one is trained with the
  algorithm the recipe uses. Both models are scored in the reference's
  tag set: the reference asked for Penn tags (`PosTagFormat::Penn`), its
  own outcomes, which come out unconverted, and the trained model with its
  outcomes as they are, MASC's tags. The card records both formats.
  MASC 3.0.0 writes some tags another way than the reference: brackets
  `(` and `)` where the reference has `-LRB-` and `-RRB-`, and tags of its
  own for hyphens and dashes (`-`), ellipses (`...`) and semicolons
  (`;`), tokens the reference tags `:`. The held-out tags and both
  models' tags are compared in the reference's spelling
  (`postag::REFERENCE_SPELLINGS`). A held-out token whose tag the
  reference cannot give even so would count against it whatever it did,
  so it is left out of both models' scores; the card lists those tokens
  by tag (`held_out.tokens.left_out`), and the recipe fails when they are
  more than one in a thousand of the held-out tokens
  (`MAX_TOKENS_LEFT_OUT`). MASC 3.0.0's held-out documents have 6 of
  85,292: ``UH` `` (3), `"` (2) and `PRN` (1). The reference's
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
  MASC's Twitter genre (two documents, 24,180 words) is trained on with
  the rest: neither the release nor anc.org gives it terms of its own
  (docs/licences.md, "Datasets").
  Whether the ANC asks for a particular citation is to be confirmed on
  their site, which could not be reached when the recipes were written.

## The release record

MASC is fetched through the zoo by digest and read from its extracted
directory. It is not checked in: CC BY would allow it with attribution,
but the policy prefers fetching by digest. `masc::RELEASE` is `NotRecorded`:
the release could not be downloaded where the recipes were written
(`anc.org` was not reachable), so its archive's size and digests and the
extracted directory's digest are not known. Until they are recorded:

- `fetch-masc` and `masc::dataset` refuse (`Error::NotRecorded`);
- `masc` refuses a corpus directory unless given `--unrecorded-corpus`,
  prints the directory's digest, and writes cards with
  `release_recorded: false`, which are not a release of the models.

To record it: download the release, record the archive (file name, format,
URL, size, SHA-256, SHA-512) and the digest `masc --unrecorded-corpus`
prints in `masc::RELEASE`. The zoo has no tar format yet, so a release
published as a gzip tar needs one before it can be listed.

## Not here yet

- Scores on MASC: none are recorded until the release is recorded and
  the recipes are run on it. No recipe has been run on MASC itself.

## The UD recipes

The treebanks of Universal Dependencies release 2.16 (tag `r2.16` of
each repository) whose own `LICENSE.txt` and README say CC BY 4.0,
read on 2026-10-14 (#415):

| Treebank | Components | Reference | Notes |
|---|---|---|---|
| `UD_Spanish-AnCora` | sentence, tokens, pos, lemmas | `opennlp-es-ud-gsd-*-1.3-2.5.4` | a README paragraph still says a GNU licence is inherited; `LICENSE.txt`, the metadata and the changelog (release 2.9) say CC BY 4.0 |
| `UD_Catalan-AnCora` | sentence, tokens, pos, lemmas | `opennlp-ca-ud-ancora-*-1.3-2.5.4` | the same README note; the references are trained on this treebank's `train` file |
| `UD_Portuguese-Porttinari` | sentence, tokens, pos, lemmas | `opennlp-pt-ud-gsd-*-1.3-2.5.4` | |
| `UD_Finnish-FTB` | pos, lemmas | `opennlp-fi-ud-tdt-*-1.3-2.5.4` | `LICENSE.txt` offers LGPLv3 or later or, at the user's option, CC BY 4.0; the recipes take CC BY 4.0 |
| `UD_Italian-MarkIT` | pos, lemmas | none: scored alone | OpenNLP's Italian UD models are trained on UD_Italian-VIT, CC BY-NC-SA 3.0 |

- Left out: `UD_Portuguese-DANTEStocks` is CC BY 4.0 too, but its text
  is tweets, which the data policy keeps out with other data under a
  platform's terms (as WNUT-17 is, `docs/licences.md`). FTB and MarkIT
  have no sentence detector or tokenizer: their `text` comments put a
  space between every two tokens and no word has `SpaceAfter=No`, so they
  hold no sentence ends or tokenization of running text.
- Files: each treebank's `LICENSE.txt`, `README.md` and UD's `train`,
  `dev` and `test` files, with their sizes, SHA-256 and SHA-512 recorded
  in `treebank`, fetched through the zoo by digest from
  `raw.githubusercontent.com/UniversalDependencies/<repository>/r2.16/`
  (`fetch-ud`) as permissive datasets (`ud-<language>-<name>-r2.16`).
  They are not checked in.
- Licence check: `LICENSE.txt` must name the Creative Commons
  Attribution 4.0 International licence and its address, name no
  NonCommercial, ShareAlike or NoDerivatives element, and name a GNU
  licence only where the recipe records a choice beside CC BY 4.0, "at
  your option"; the README must have the metadata line `License: CC BY
  4.0`. Then every file must match its recorded size and digests, so a
  changed `LICENSE.txt` is refused too.
- Split: UD's own. Each model is trained on `train` and gated on `test`;
  `dev` is checked and not read.
- Parameters: those of the UD 1.3 models (GIS, 300 iterations, cutoff 5,
  4 threads), the factory defaults for the treebank's language, OpenNLP
  2.5.4's whitespace, 10 sentences per sentence detector sample, the
  universal tags, and the fixed manifest `Timestamp`. No randomness, so
  no seed.
- Samples: those of OpenNLP's CoNLL-U streams (a multiword token is one
  token, its words' tags and lemmas joined by `+`), except that the
  tagger's and the lemmatizer's leave out empty nodes (ids like `6.1`),
  which are not words of the text (AnCora writes their form `_`). A
  sentence left with no word would be skipped; the card records how many
  (none in these treebanks).
- The gate: both models scored on the `test` samples with OpenNLP's
  `SentenceDetectorEvaluator`, `TokenizerEvaluator`, `POSEvaluator` and
  `LemmatizerEvaluator` (gold tags given to the lemmatizer), one unit per
  sample, gated by `turbo_eval::model_gate`. The taggers are scored in
  their own outcomes, the universal tags. A tagger's tokens whose gold
  tag the reference has no outcome for are left out of both models'
  counts, and the card lists those tags; more than 5 percent of the
  tokens and the gate refuses. Every reference is read and checked
  against the zoo's digest, and every treebank against its record,
  before any model is trained.
- A model with no reference (Italian) gets an `UngatedRecord` in
  `gate.json`, its own score with its interval and the reason, and its
  zoo entry has the version `1-experimental`.
- Licence: the models are Apache-2.0 trained on CC BY 4.0 data, so their
  zoo class is permissive. The card carries the attribution (the
  treebank, its release, its contributors, its address and the licence)
  and the citation the treebank's README asks for. `UdModelCard::artifact`
  refuses a card whose gate failed or whose gate's reference is not the
  recipe's, by id and by the zoo's SHA-256.

```
cargo run --release -p turbo-train-recipes -- fetch-ud --home <zoo-home>
cargo run --release -p turbo-train-recipes -- ud --treebanks <zoo-home>/artifacts \
    --references <dir> --out <dir> [--recipe <model id>]...
```

`--references` holds the OpenNLP UD 1.3 models under their published
file names (Maven Central `org.apache.opennlp:opennlp-models-<component>-<language>:1.3.0`,
or `downloads.apache.org/opennlp/models/ud-models-1.3`), checked against
the zoo's SHA-256.

### Results

All sixteen recipes were run once, in release mode, in a run that
started on 2026-10-14 (`ud` with every recipe, 4 training threads on a
4-core machine), with the references from Maven Central checked against
the zoo's SHA-256. The fourteen gated models pass; the two Italian models
are scored alone. Scores are in points: F-measure for the sentence
detector and tokenizer, accuracy for the tagger and lemmatizer, with 95
percent intervals from 10,000 paired bootstrap draws (seed 154). A unit
is a held-out sample: 10 sentences for the sentence detector, one
sentence otherwise. Time is the wall-clock time of a recipe: training,
scoring both models and the gate. The trained models are not checked in.

| Model (`turbo-…-1`) | Reference (`opennlp-…-1.3-2.5.4`) | Units | Candidate | Reference | Difference | Gate | Time |
|---|---|---|---|---|---|---|---|
| `es-ud-ancora-2.16-sentence` | `es-ud-gsd-sentence` | 173 | 97.99 (97.06 to 98.79) | 96.27 (94.93 to 97.50) | +1.73 (+0.85 to +2.72) | pass | 1 s |
| `es-ud-ancora-2.16-tokens` | `es-ud-gsd-tokens` | 1721 | 99.96 (99.94 to 99.98) | 99.89 (99.85 to 99.92) | +0.07 (+0.04 to +0.11) | pass | 26 s |
| `es-ud-ancora-2.16-pos` | `es-ud-gsd-pos` | 1721 | 97.40 (97.22 to 97.57) | 92.73 (92.49 to 92.97) | +4.67 (+4.44 to +4.91) | pass | 119 s |
| `es-ud-ancora-2.16-lemmas` | `es-ud-gsd-lemmas` | 1721 | 91.32 (90.93 to 91.70) | 90.54 (90.16 to 90.92) | +0.77 (+0.67 to +0.87) | pass | 322 s |
| `ca-ud-ancora-2.16-sentence` | `ca-ud-ancora-sentence` | 185 | 99.16 (98.57 to 99.68) | 99.16 (98.57 to 99.68) | +0.00 (+0.00 to +0.00) | pass | 1 s |
| `ca-ud-ancora-2.16-tokens` | `ca-ud-ancora-tokens` | 1846 | 99.92 (99.89 to 99.95) | 99.92 (99.89 to 99.95) | +0.00 (+0.00 to +0.00) | pass | 19 s |
| `ca-ud-ancora-2.16-pos` | `ca-ud-ancora-pos` | 1846 | 97.28 (97.09 to 97.47) | 97.24 (97.05 to 97.43) | +0.04 (-0.01 to +0.08) | pass | 97 s |
| `ca-ud-ancora-2.16-lemmas` | `ca-ud-ancora-lemmas` | 1846 | 89.74 (89.27 to 90.21) | 89.74 (89.26 to 90.21) | +0.00 (-0.01 to +0.01) | pass | 246 s |
| `pt-ud-porttinari-2.16-sentence` | `pt-ud-gsd-sentence` | 169 | 99.11 (98.57 to 99.64) | 98.26 (97.43 to 98.96) | +0.85 (+0.06 to +1.70) | pass | 0 s |
| `pt-ud-porttinari-2.16-tokens` | `pt-ud-gsd-tokens` | 1683 | 99.84 (99.78 to 99.90) | 99.20 (99.06 to 99.33) | +0.64 (+0.52 to +0.77) | pass | 6 s |
| `pt-ud-porttinari-2.16-pos` | `pt-ud-gsd-pos` | 1683 | 95.82 (95.55 to 96.10) | 91.49 (91.14 to 91.84) | +4.33 (+3.99 to +4.68) | pass | 27 s |
| `pt-ud-porttinari-2.16-lemmas` | `pt-ud-gsd-lemmas` | 1683 | 91.28 (90.85 to 91.71) | 81.12 (80.61 to 81.62) | +10.16 (+9.76 to +10.55) | pass | 164 s |
| `fi-ud-ftb-2.16-pos` | `fi-ud-tdt-pos` | 1867 | 89.25 (88.71 to 89.78) | 87.25 (86.67 to 87.83) | +2.00 (+1.47 to +2.53) | pass | 31 s |
| `fi-ud-ftb-2.16-lemmas` | `fi-ud-tdt-lemmas` | 1867 | 88.24 (87.70 to 88.77) | 83.06 (82.41 to 83.71) | +5.18 (+4.62 to +5.74) | pass | 2020 s |
| `it-ud-markit-2.16-pos` | none | 340 | 89.93 (89.19 to 90.67) | | | ungated | 5 s |
| `it-ud-markit-2.16-lemmas` | none | 340 | 82.39 (81.48 to 83.27) | | | ungated | 12 s |

- Tags left out of the taggers' gates, counted in the tagger's tokens
  of the `test` file whose gold tag the reference has no outcome for:
  Porttinari 6 of 31,451 (0.02 percent: `INTJ` 5, `ADP+ADP` 1) and FTB
  635 of 16,261 (3.91 percent: `DET` 398, `PART` 234, `SCONJ+VERB` 3,
  tags the Finnish TDT reference never gives); none for Spanish and
  Catalan. The two taggers' scores above are over the other tokens. The
  lemmatizers' gates leave nothing out.
- The Catalan references are OpenNLP's models of this treebank (the
  zoo records them as trained on `UD_Catalan-AnCora r2.16`), made with
  the settings the recipes take from the UD 1.3 models. The sentence detectors and tokenizers agree on every
  unit. The taggers differ on 127 of the 1,846 units and the lemmatizers
  on 6. One difference in the data is known: the recipes leave the
  treebank's 4,999 empty nodes out of the tagger's and lemmatizer's
  training samples, and OpenNLP's streams keep them. Whether that
  accounts for all of it was not checked.
- The other references are trained on another treebank of the language
  (Spanish GSD, Portuguese GSD, Finnish TDT), so a gap can come from the
  two treebanks' annotation as well as from the models; the widest is
  the Portuguese lemmatizer's 10.16 points.

## Tests

`tests/recipes.rs` writes corpora in MASC's layout (`tests/common`, no
MASC text, Penn tags written for the tests) and runs everything on
them: the split in the reader's order; the three recipes trained,
scored and gated; the same bytes from a second training; the card, the
model's directory and its zoo entry; the skipped sentences recorded and
bounded on both sides; the tagger scored in the reference's spelling
with the tokens of a tag the reference cannot give left out, and a
reference that cannot give most tags refused; and the binary's refusal
of an unrecorded corpus, of a missing or wrong reference and of an
existing model directory before it trains anything, and its output. It
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
to tag brackets, hyphens, ellipses and semicolons as
`postag::REFERENCE_SPELLINGS` says, scored and gated, and the binary
runs all three recipes.

On the test corpus, with both directories set, the published references
score 63.16 (sentences, F-measure), 94.34 (tokens, F-measure) and 92.41
(`en-pos-maxent.bin`, accuracy), against 100 for the trained models,
which have seen every held-out sentence; the test checks these numbers,
which show the gate is wired through, not how the recipes do on MASC.

`tests/ud.rs` does the same for the UD recipes, on a treebank written for
the tests: eight Spanish sentences in CoNLL-U (with multiword tokens, an
empty node and `SpaceAfter=No`; no treebank text), a `LICENSE.txt` and a
README. It checks that the tagger's and lemmatizer's samples leave out
empty nodes and join a multiword token's tags and lemmas with `+`, and
that the held-out digest covers tags and lemmas. It trains, scores,
gates and writes the card and directory of each of the four components
against a reference trained on two of the sentences: each candidate
passes and scores at least as high, except the tagger, whose gate is
refused because that reference has no outcome for most of the gold
tags. A reference trained without the interjection leaves its 40 tokens
out of both taggers' counts and onto the card. A recipe with no
reference gets an `UngatedRecord` and an experimental zoo entry, and a
reference given to it, or missing from a recipe that has one, is an
error. A treebank whose file or licence changed is refused, and the
binary checks every reference and treebank before it trains anything.
With `TURBO_TEST_OPENNLP_MODELS` naming a directory of the OpenNLP UD 1.3
models, each recipe's reference is checked against the zoo's digest and
scored on the test text (Spanish for every language, so the scores only
show that each model loads and is scored); a missing file there is
skipped, or fails with `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`. CI does not
set `TURBO_TEST_OPENNLP_MODELS`, so that part runs only where the models
are present.

