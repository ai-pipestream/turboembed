# turbo-train-recipes

Recipes that train models only on permissive data, and the
accuracy gate each model passes before it ships (#401, #415).
docs/analysis.md ("Models trained on permissive data" and "Model gates")
describes both. The MASC recipes come first here; the recipes on the
CC BY 4.0 Universal Dependencies treebanks are under "The UD recipes".

| Module | What it holds |
|---|---|
| `masc` | MASC 3.0.0 (CC BY 3.0 US): the release record, its zoo dataset entry and fetch, the digest of an extracted copy, the fixed split, and the Twitter genre left out of it |
| `segmentation` | The English sentence detector and tokenizer recipes: training on the split's training documents, the held-out samples, scoring a model with OpenNLP's evaluators, the gate against the OpenNLP English UD model |
| `postag` | The English part-of-speech tagger recipe, the same way: training on MASC's Penn tags, the held-out samples, scoring with `POSEvaluator`, the gate against OpenNLP's English Penn tagger (`en-pos-maxent.bin`, OpenNLP 1.5) |
| `chunk` | The English chunker recipe, the same way: training on MASC's noun and verb chunks as `ChunkerME.train` trains with its defaults, the held-out samples, scoring the chunk types both models give with `ChunkerEvaluator`, the gate against OpenNLP's English chunker (`en-chunker.bin`, OpenNLP 1.5) |
| `spans` | A held-out unit's counts from the `FMeasure` OpenNLP's evaluator keeps for it, both models' counts paired, and the span types a gate of typed spans (names, chunks) scores and renames |
| `card` | The model card, the model's directory (model, `card.json`, `gate.json`) and its permissive zoo entry, made only for a card whose gate against its recipe's reference (by id and SHA-256) passed on the recorded release |
| `treebank` | The CC BY 4.0 UD treebanks at release 2.16: each repository's `LICENSE.txt`, `README.md` and `train`, `dev` and `test` files with their sizes and digests, the licence check, the zoo dataset entry and fetch |
| `ud` | The UD recipes: the samples, training on `train`, the checks run before training, and the gate on `test` against the OpenNLP UD 1.3 model (the zoo's, or the Italian ones recorded by digest), by the `segmentation` and `postag` gates and, for the lemmatizer, `LemmatizerEvaluator` |
| `ud_card` | The UD model card, with the treebank's licence, attribution and citation and the MASC card's held-out and gate parts, the model's directory and its zoo entry |
| `release` | The release of this repository that publishes the MASC models (`models-masc-3.0.0-1`), and their entries in the zoo's built-in index at their addresses there |

```
cargo run --release -p turbo-train-recipes -- masc --corpus <masc-dir> \
    --references <dir> --out <dir> [--unrecorded-corpus] [--recipe <model id>]...
cargo run --release -p turbo-train-recipes -- fetch-masc --home <zoo-home>
cargo run --release -p turbo-train-recipes -- zoo-index --models <dir> --release <tag> [--check]
```

`--references` is a directory holding the reference models under their
published file names: `opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin` and
`opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin`, as `cargo run -p
turbo-goldens-tool -- fetch-models <dir>` writes them and checked against
the zoo's SHA-256, and `en-pos-maxent.bin`, downloaded by hand from
<https://opennlp.sourceforge.net/models-1.5/en-pos-maxent.bin> and checked
against the size and SHA-256 the recipe records (5,696,197 bytes,
`645a094f45a866687a617385233fd23ae8b0f5fa8b1b76996781a50c17bdcf3d`),
and `en-chunker.bin`, downloaded by hand from
<https://opennlp.sourceforge.net/models-1.5/en-chunker.bin> and checked
the same way (2,560,304 bytes,
`7861a0c2f134d9c12a022a1ba501e88bc7039f6db72b4140e1bafd1fb5ef76cc`).
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
| `turbo-en-masc-3.0.0-chunks-1` | `opennlp-en-chunker-1.5` (`en-chunker.bin`) | MASC Penn tokens, their `msd` tags and chunk tags of MASC's noun chunks (`NP`) and verb chunks (`VP`), one sample per sentence | F-measure of `NP` and `VP` chunks, one unit per sentence |

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
  settings together. The chunker is trained as `ChunkerME.train` trains
  with `TrainingParameters.defaultParams()`, which the trainer tool uses
  when no parameters file is named (`chunk::training_parameters`: GIS,
  100 iterations, cutoff 5, one thread), with the `ChunkerFactory`, its
  default context generator and beam size 10. Its reference's manifest
  records OpenNLP 1.5.0, 100 iterations and cutoff 5.
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
- The chunker's samples: MASC's chunk layers, `f.nc` (noun chunks) and
  `f.vc` (verb chunks), which OpenNLP has no reader for, read by
  `turbo-formats`' `MascChunkReader` into one sample per sentence of its
  Penn tokens, their `msd` tags and `B-NP`, `I-NP`, `B-VP`, `I-VP` and
  `O` chunk tags. A chunk links to Penn tokens; it is placed in the first
  sentence that holds all of them. A sentence is skipped when chunk tags
  cannot express its chunks: it holds part of a chunk that crosses into
  another sentence (or names a token no sentence holds), a chunk with a
  gap, or two chunks that share a token (two chunks of one kind over the
  same tokens count as one). It is also skipped for a token without a
  tag or for having no token, as the tagger recipe skips it. MASC's
  resource header marks both chunk layers `automatic-validated`, and the
  release holds many such sentences, so the chunker has a bound of its
  own, one in ten of either side (`chunk::MAX_SKIPPED_SENTENCES`). In
  the recorded release it skips 1,872 of the 24,958 training sentences
  (1,537 with chunks that share a token, 335 with part of a chunk that
  crosses sentences) and 349 of the 6,385 held-out ones (280, 68, and
  the one sentence of no tokens); no chunk in the release has a gap. On
  the recorded release the recipe fails on any other count of skipped
  sentences on either side (`chunk::RECORDED_SKIPPED_SENTENCES`), so a
  change to the reader that skips more or fewer of them is not missed;
  the one in ten bound is the check on any other corpus.
  Over the whole release, the pairs of chunks that share tokens are
  mostly a noun chunk and a verb chunk, one inside the other or over the
  same tokens (2,120 pairs, such as `all those dogs` and `dogs`, `to
  work` and `work`, or two chunks `Posted`), then noun chunks inside noun
  chunks (56) and chunks that cross (68). The same chunk listed twice in
  one layer is common too (1,793 pairs of verb chunks, 63 of noun
  chunks), and is read as one chunk.
- What the chunker's samples do not have beyond the skipped sentences:
  the chunks in no sentence the reader keeps. MASC's sentence layer does
  not cover all of the text (in `data/written/essays/Ant_Robot` only
  5,211 of the 10,064 quarks are inside a sentence region), and the
  reader drops a sentence region that does not end before the next one
  starts, as OpenNLP's MASC reader drops it (`data/written/jokes/jokes16`
  keeps 77 of its 171 sentence regions and 1,111 of its 2,290 Penn
  tokens), so a chunk can name a token no kept sentence has. The card
  counts, on each side (`training.chunks` and `held_out.chunks`), the
  chunks of the chunk files, those outside every kept sentence, those
  whose tokens are in more than one sentence and those with no token.
  In the recorded
  release 16,597 of the 221,931 chunks the split reads are outside every
  kept sentence (13,509 of 183,681 on the training side, 3,088 of 38,250
  held out), about 7.5% of them, beside 95 whose tokens are in more than
  one sentence (83 and 12) and 2 with no token (both training). The
  sentence detector, token and tagger recipes read the same sentences,
  so their samples do not have that text either.
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
- The chunker's gate: the reference is OpenNLP's English chunker of the
  1.5 models, `en-chunker.bin`, OpenNLP's only English chunker. Its
  chunk types are CoNLL-2000's, whose data is drawn from the Wall Street
  Journal part of the Penn Treebank, an LDC corpus; its training data is
  not published, and the zoo does not list it. As with the tagger's
  reference, the recipe records its address, size and SHA-256
  (`chunk::EN_CHUNKER`), a run reads it from `--references` and checks
  it, and it is only scored, never shipped or bundled. Both models are
  scored on the same held-out samples, one `ChunkerEvaluator` per
  sentence, over the chunk types both give (`chunk::SPAN_TYPES`: `NP`
  and `VP`). MASC has chunks of those two types only, so no held-out
  chunk is renamed or left out (`left_out: 0` in the gate record). The
  reference also gives `PP`, `ADVP`, `ADJP`, `SBAR`, `PRT`, `INTJ`,
  `LST`, `CONJP` and `UCP`; its tags of those types are made `O` before
  it is scored (`chunk::scored_tag`), which leaves its `NP` and `VP`
  phrases as they were. The trained model gives only `NP` and `VP`. The
  reference is given `:` for three tags MASC writes and its training
  data does not have (`chunk::REFERENCE_TAGS`: `-`, `...` and `;`), as
  the tagger recipe maps them; the trained model reads the tags as MASC
  writes them, and the gate record lists the mapping
  (`set.scoring.typed_spans.reference_inputs`). The mapping is applied
  to the reference only, and is worth 0.02 points to it on the held-out
  set: it scores 82.46 with it and 82.44 without. The held-out samples
  have 155 `-` and 173 `...` tokens and no `;`. Two more held-out tags
  have no feature in the reference and are given as written: `LS` (9
  tokens) and MASC's ``UH` `` (3). How MASC's chunks differ from
  CoNLL-2000's, and which way that leans the gate, is under "The
  chunker's conventions" below.
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
out) with the four references from `--references`. Every gate passed.
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
| `turbo-en-masc-3.0.0-chunks-1` | 6,036 sentences, 32,252 chunks (`NP` and `VP`) | F-measure | 89.74 | 82.46 | +7.29 (+6.84 to +7.75) | pass |

Both models of a gate are scored on the same held-out samples, so the
held-out sentences skipped are skipped for both.

| Model | Training samples | Training sentences skipped | Events | Held-out sentences skipped | Held-out tokens left out |
|---|---|---|---|---|---|
| `turbo-en-masc-3.0.0-sentence-1` | 2,496 | 0 | 25,490 | 0 | |
| `turbo-en-masc-3.0.0-tokens-1` | 24,957 | 1 | 1,408,312 | 1 | |
| `turbo-en-masc-3.0.0-pos-1` | 24,958 | 0 | 431,637 | 1 | 6 of 85,292: `"` (2), `PRN` (1), ``UH` `` (3) |
| `turbo-en-masc-3.0.0-chunks-1` | 23,086 | 1,872 | 384,930 | 349 | |

The trained models are not checked in; the release below publishes
them. A second run into another directory wrote the same bytes:

| Model | Size | SHA-256 |
|---|---|---|
| `turbo-en-masc-3.0.0-sentence-1.bin` | 44,601 bytes | `d03253a5559a230d1f42e1cf881119204302fbb7fefb613ff1af11bcbd3c261e` |
| `turbo-en-masc-3.0.0-tokens-1.bin` | 916,027 bytes | `2c5fe73121c41300e6eae4a3c4d7a2247ade20a1e182d8e36d7781124aa9b7db` |
| `turbo-en-masc-3.0.0-pos-1.bin` | 3,671,103 bytes | `391d7f79c7246e9e56ed9d8849fde129c5bed8683fa33af31e532cb0c37f96ea` |
| `turbo-en-masc-3.0.0-chunks-1.bin` | 6,042,700 bytes | `8629b99af79f4cac2f997f3f9042ecb0739776f1e49f1b01c713877a1f997e09` |

A run of all four recipes (the release binary, five minutes on a
machine of four cores) wrote these four files, and two runs of the
chunker's recipe alone (`--recipe turbo-en-masc-3.0.0-chunks-1`, about a
minute each) into two other directories wrote the same `.bin`,
`card.json` and `gate.json` bytes as it.

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

### The chunker's conventions

The reference learned CoNLL-2000's chunks, which were made from the Penn
Treebank's parse trees of Wall Street Journal text. The trained model
learned MASC's, made by an automatic chunker and then validated, over
MASC's genres (fiction, e-mail, blogs, letters, spoken transcripts,
journal and technical articles, among others). The gate scores both
against MASC's chunks, so every place where the two conventions differ
counts against the reference, and so do the ways MASC's text differs
from the Wall Street Journal's and the regular mistakes of MASC's
automatic chunker and tagger that the trained model learned. The gate
therefore leans towards the trained model: its lead of 7.29 points
measures how closely each model follows MASC's conventions as well as
how well it chunks, and the gate cannot tell the two apart. A pass
shows that the trained model reproduces MASC's chunks better than the
reference does, not that it would beat the reference on CoNLL-2000's.

The gate resamples sentences, as the other MASC recipes' gates do, so
its interval reflects how chunks cluster in sentences but not how
sentences cluster in documents and genres. Resampled by document (the
39 held-out documents, with the same 10,000 resamples and seed), the
difference's interval is +5.53 to +9.99, 4.46 points wide against the
0.91 of the sentence interval (+6.84 to +7.75), about five times as
wide, and the gate still passes.

Counted over the gate's held-out samples, with the reference's chunks
of other types made `O` as the gate scores them (a program written for
the count, not part of the recipe; the pooled counts give the gate's
two scores), the reference loses most on verb chunks:

| Chunks | Held out | Trained model: precision, recall, F-measure | Reference: precision, recall, F-measure |
|---|---|---|---|
| `NP` | 21,203 | 90.82, 92.01, 91.41 | 83.23, 88.05, 85.57 |
| `VP` | 11,049 | 87.78, 85.22, 86.48 | 79.99, 72.38, 76.00 |

Where the two conventions agree: both start a noun chunk at a
possessive `'s` (of the 338 held-out `POS` tokens, MASC starts a noun
chunk at 268, the reference at 313 and the trained model at 286); both
keep auxiliaries, modals, `to` and the adverbs between them in one verb
chunk (`had to be`, `are slightly offset`); and both put determiners,
numbers and adjectives in the noun chunk of their noun, and give
wh-pronouns (`who`, `which`) noun chunks of their own.

Where they differ, with held-out examples:

- Verb chunks. MASC often splits a verb group that CoNLL-2000 keeps
  whole: a participle or gerund after a verb (`[are] [plotted]`,
  `[suggests] [using]`, `[had to be] [modified]`) and an infinitive
  after a verb (`[are slightly offset] [to permit]`). It leaves an
  adverb or `not` before the main verb out of the chunk (`[were] well
  [searched]`, `not [shown]`), where CoNLL-2000 keeps them in. It puts a
  predicate adjective in the verb chunk (`[is clear]`, `[be arbitrary]`,
  `[got older]`) and a `not` after a copula (`[was not]`), where
  CoNLL-2000 has a verb chunk and an `ADJP`, or leaves `not` out: at 162
  held-out verb chunks that end in an adjective (`JJ` or `JJR`), the
  reference's verb chunk starts with MASC's and stops before the
  adjective.
- Noun chunks. MASC splits a coordination of single nouns or numbers
  (`[life] and [years]`, `[18.5] and [20]`, `[YOL] and [YHL]`), which
  the reference joins, but joins some longer lists (`[their age, sex,
  and health]`), which the reference splits. It splits ranges and
  equations (`[25] to [29.9]`, `[n] = [5,201]`), which the reference
  joins. It sometimes ends a noun chunk with a closing quotation mark or
  a period (`[Don Collins .]`), which the reference leaves out.
- Tags. MASC's Penn tags were made automatically too (by GATE's ANNIE
  tagger and then validated, its resource header says), and the chunks
  follow them: the tagger's gate above lists the punctuation MASC tags
  `NN`, and inside verb chunks it tags `cannot`, `talk` and `harbor` `NN`
  (`[can harbor]`), splits `don't` into `don`, `’` and `t` with `’` and
  `t` tagged `NN` (`[don ’ t worry]`), and tags `plotted` `VBD` after an
  auxiliary. The trained model learned MASC's chunks over these tags;
  the reference chunks `can harbor` as a verb chunk `can` and a noun
  chunk `harbor`.

The reference is given `:` for MASC's `-`, `...` and `;`, and nothing
else is mapped; its chunks of other types are left out, not renamed.

## The models release

The models go out as the assets of this repository's release
`models-masc-3.0.0-1` (`release::MASC_MODELS`), as Kristian Rickert
decided on 2026-10-15, and the zoo's built-in index lists each at its
address there, `https://github.com/ai-pipestream/turboembed/releases/download/models-masc-3.0.0-1/<model id>.bin`,
with the size, SHA-256 and SHA-512 its card records.

`zoo-index --models <dir> --release <tag>` reads the model directories
`masc` wrote into `<dir>`, one for every recipe, checks each model file
against its card, makes each card's zoo entry at its address in release
`<tag>` (`ModelCard::artifact`, which refuses a model that failed its
gate or was not trained on the recorded release), and prints the zoo's
built-in index with those entries added or replaced. That is how
`analysis/zoo/index/zoo.json` got them. With `--check` it prints no
index and exits with 1 unless the built-in index is already that one,
naming the entries that differ.

`.github/workflows/models.yml` makes the release. It fetches MASC
through the zoo and the three references from their published addresses
(`en-pos-maxent.bin` from SourceForge, used only to be scored), caching
both in the repository's Actions cache, runs `masc` on
the recorded release, runs `zoo-index --check`, and uploads each model
with its card and gate record (`<model id>.bin`, `<model id>.card.json`,
`<model id>.gate.json`) and a `SHA256SUMS` file. On a pull request that
changes the workflow it stops there and keeps the files as an artifact;
run by hand on main, it creates the release at that commit with the
files as its assets, and the notes give the models' licence and MASC's
attribution. Creating a release that exists fails, so models trained
otherwise need a new model version and tag.

## The UD recipes

The treebanks of Universal Dependencies release 2.16 (tag `r2.16` of
each repository) whose own `LICENSE.txt` and README say CC BY 4.0,
read on 2026-10-14 (#415):

| Treebank | Components | Reference | Notes |
|---|---|---|---|
| `UD_Spanish-AnCora` | sentence, tokens, pos, lemmas | `opennlp-es-ud-gsd-*-1.3-2.5.4` | a README paragraph still says "The GNU license is inherited from the original dataset"; `LICENSE.txt` and the metadata say CC BY 4.0, and the changelog's entry "2021-11-15 v2.9" says "The license changed to CC BY 4.0 (https://doi.org/10.5281/zenodo.4762030)." |
| `UD_Catalan-AnCora` | sentence, tokens, pos, lemmas | `opennlp-ca-ud-ancora-*-1.3-2.5.4` | the same README paragraph and changelog entry; the references are trained on this treebank's `train` file |
| `UD_Portuguese-Porttinari` | sentence, tokens, pos, lemmas | `opennlp-pt-ud-gsd-*-1.3-2.5.4` | recipe only, no model shipped: the CC BY 4.0 grant is the treebank authors', for their annotations, and the README says the texts are from the Folha de São Paulo newspaper, "publicly available at Kaggle website", with no licence from Folha shown |
| `UD_Finnish-FTB` | pos, lemmas | `opennlp-fi-ud-tdt-*-1.3-2.5.4` | `LICENSE.txt` says "FTB1U is Copyright 2014 FIN-CLARIN" and offers LGPLv3 or later or, at the user's option, CC BY 4.0; the recipes take CC BY 4.0, and the attribution keeps the copyright notice |
| `UD_Italian-MarkIT` | pos, lemmas | `opennlp-it-ud-vit-*-1.3-2.5.4`, recorded by size and SHA-256 | the references are trained on UD_Italian-VIT, CC BY-NC-SA 3.0, so they are local-only: read from `--references`, only scored, not in the zoo |

- Left out: `UD_Portuguese-DANTEStocks` is CC BY 4.0 too, but its text
  is tweets, and the policy refuses platform-terms data, as it does
  MASC's Twitter genre (`docs/licences.md`), so it is refused and has no
  recipe. FTB and MarkIT have no sentence detector or tokenizer: their
  `text` comments put a space between almost every two tokens, and
  almost no word has `SpaceAfter=No` (none in FTB; in MarkIT, 2 in
  `train`, on "1,9", 7 in `dev` and none in `test`), so they hold no
  sentence ends or tokenization of running text.
- Porttinari: the recipes stay, so a user can train the Portuguese
  models locally, but the rights to the treebank's text are not settled
  (`treebank::TextRights::NotSettled`), and `UdModelCard::artifact`
  refuses a Porttinari model whatever its gate says, so none is shipped.
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
  sample, gated by `turbo_eval::model_gate`. The gates are the MASC
  recipes': a sentence detector or tokenizer is gated as a
  `segmentation::SegmentationRecipe`, and a tagger as a
  `postag::PosRecipe` of its own (`UdRecipe::tagger`), with both models'
  tags asked for as their outcomes, the universal tags, and compared
  with the held-out tag as written: no mapping of the reference's tags,
  so the record's and card's `reference_mappings` are empty. A
  lemmatizer is gated the same way on its lemmas, compared with the
  held-out lemma as written.
- Tokens left out: a tagger's tokens whose gold tag the reference has no
  outcome for, and a lemmatizer's tokens whose gold lemma class (the edit
  script from the token to its lemma, as training makes it) gives the
  lemma while no class of the reference does (`ud::ReferenceClasses`),
  are tokens the reference cannot get right and the candidate can be
  taught, so they are left out of both models' scores. Lacking the gold
  class is not enough: OpenNLP's decoder deletes a character without
  checking the letter the class names, so a reference without a token's
  gold class can give its lemma through another class. Of the test
  tokens whose gold class the references lack, they can give the lemma of
  33 of 90 (Spanish), 16 of 34 (Catalan), 8 of 108 (Portuguese), 323 of
  500 (Finnish) and 10 of 43 (Italian), and those are scored. The card
  lists the tokens left out by tag, with the reference's most frequent
  mismatches (tags, or lemmas), and the record counts them; more than 5
  in 100 of the tokens (`ud::MAX_TOKENS_LEFT_OUT`) and the recipe fails.
  A token whose gold class gives another string than its lemma stays in
  the scores, as `LemmatizerEvaluator` counts it (below).
- The limit: one, 5 in 100, for every UD tagger and lemmatizer, a floor
  on how much of the `test` file the gate covers. The MASC tagger's
  limit, 1 in 1,000 (`MAX_TOKENS_LEFT_OUT`), would refuse FTB, where the
  Finnish reference tagger cannot give 3.9 in 100. Every token left out
  is one the reference gets wrong.
  So when the candidate trails, leaving tokens out always lowers its lead
  on the point estimate, and in each of the gate's resamples, which draw
  the same sentences either way; when the two resamples the interval's
  low end is read from have the candidate trailing, the low end is no
  higher with the tokens left out, so leaving them out cannot turn a
  failing gate into a passing one (`ud`'s documentation has the
  arithmetic). When the candidate leads, leaving tokens out raises its
  lead only when its accuracy on them is below its lead on the scored
  tokens (as fractions), and then by at most the left-out share times
  that lead, so at most 5 in 100 of it. The results below give each
  such gate's lead both ways.
- Cross-treebank gates: except for Catalan, the reference was trained
  on another treebank, so it is scored against annotation conventions
  it was not trained on, which favours the candidate (the Finnish TDT
  reference tags 303 of FTB's scored `VERB` tokens `AUX`). The Catalan
  gates are the only ones that compare models trained on the same
  treebank.
- Checks before training: the run reads and checks every reference
  (the zoo's digest, or the recorded size and digest) and checks that no
  model directory exists, then reads every treebank against its record,
  then checks every recipe's samples (`UdRecipe::check_data`, as
  `Recipe::check_data`: each file has a sample, at most 1 in 1,000
  sentences skipped on either side, and the tagger's or lemmatizer's
  left-out tokens within the limit), naming every problem, before any
  model is trained.
- Card and record: `card.json` is a `UdModelCard` (version 3): the
  treebank and its files, licence, attribution and citation where the
  MASC card has the corpus, and the MASC card's `training`, `held_out`
  and `gate` parts; it is read version first, as the MASC card is.
  `gate.json` is the MASC recipes' `ModelGateRecord` (version 2, with the
  held-out sentences skipped and the scoring; `check` refuses a
  lemmatizer's record with a mapping). The run prints each
  card's summary (`UdModelCard::summary`, the lines
  `ModelCard::summary` prints), its gate record and the seconds the
  recipe took.
- The Italian references: OpenNLP's Italian UD models are trained on
  UD_Italian-VIT (CC BY-NC-SA 3.0), so they are local-only
  (`docs/licences.md`, "Non-commercial data") and the zoo's built-in
  index does not list them. `ud::IT_UD_VIT_POS` and
  `ud::IT_UD_VIT_LEMMAS` record each file's name, address, size and
  SHA-256 (the files of Maven Central's
  `org.apache.opennlp:opennlp-models-pos-it:1.3.0` and
  `opennlp-models-lemmatizer-it:1.3.0`, whose SHA-512 is the one Apache
  publishes beside them). A run reads them from `--references`, checks
  them and only scores them, as the MASC tagger's gate does
  `en-pos-maxent.bin`; the trained models are trained on MarkIT alone.
- Licence: the models are Apache-2.0 trained on CC BY 4.0 data, so their
  zoo class is permissive. The card carries the attribution (the
  treebank, its release, its contributors, its address, the licence and
  the copyright notice `LICENSE.txt` gives, FTB's) and the citation the
  treebank's README asks for, and the other licence a treebank offers
  (FTB's `LGPL-3.0-or-later`, a `turbo_zoo::Licence`).
  `UdModelCard::artifact` refuses a card as `ModelCard::artifact` does:
  one no recipe of `UD_RECIPES` trains, one whose class is not the one
  its licences give, one whose gate's reference is not the recipe's (by
  id and by the SHA-256 the zoo lists or the recipe records), and one
  whose gate failed, as the Italian models' do; and also one trained on
  a treebank whose text rights are not settled, Porttinari, whatever its
  gate says.

```
cargo run --release -p turbo-train-recipes -- fetch-ud --home <zoo-home>
cargo run --release -p turbo-train-recipes -- ud --treebanks <zoo-home>/artifacts \
    --references <dir> --out <dir> [--recipe <model id>]...
```

`--references` holds the OpenNLP UD 1.3 models under their published
file names (Maven Central `org.apache.opennlp:opennlp-models-<component>-<language>:1.3.0`,
or `archive.apache.org/dist/opennlp/models/ud-models-1.3`), checked
against the zoo's SHA-256, or for the Italian ones the recorded size and
SHA-256.

### Results

All sixteen recipes were run in release mode on 2026-10-15 on the code
of this change (`ud` with every recipe, 4 training threads on a 4-core
machine), with the references checked against the zoo's SHA-256 (the
Italian ones against the recorded size and SHA-256), and the five
lemmatizer recipes were run again after the last change to which of
their tokens are left out. Every model file has the SHA-256 that the
two runs before the review (on the merge of main at `eecd1eec`)
recorded, and the sentence detectors', tokenizers' and taggers' gate
records are byte-identical to theirs, so their scores are too. The cards are
version 3. The lemmatizers' gates leave out the tokens whose lemma the
reference cannot give (above), so their scores are over fewer tokens
than that run's; the table after this one gives each gate that leaves
tokens out with its lead both ways. The fourteen models gated against
references of the zoo pass; the two Italian models fail. Scores are in
points: F-measure for the sentence detector and tokenizer, accuracy for
the tagger and lemmatizer, with 95 percent intervals from 10,000 paired
bootstrap draws (seed 154). A unit is a held-out sample: 10 sentences
for the sentence detector, one sentence otherwise. Time is the
wall-clock time of a recipe (for a lemmatizer, in the second run):
training, scoring both models and the gate. The trained models are not
checked in.

| Model (`turbo-…-1`) | Reference (`opennlp-…-1.3-2.5.4`) | Units | Candidate | Reference | Difference | Gate | Time |
|---|---|---|---|---|---|---|---|
| `es-ud-ancora-2.16-sentence` | `es-ud-gsd-sentence` | 173 | 97.99 (97.06 to 98.79) | 96.27 (94.93 to 97.50) | +1.73 (+0.85 to +2.72) | pass | 1 s |
| `es-ud-ancora-2.16-tokens` | `es-ud-gsd-tokens` | 1721 | 99.96 (99.94 to 99.98) | 99.89 (99.85 to 99.92) | +0.07 (+0.04 to +0.11) | pass | 18 s |
| `es-ud-ancora-2.16-pos` | `es-ud-gsd-pos` | 1721 | 97.40 (97.22 to 97.57) | 92.73 (92.49 to 92.97) | +4.67 (+4.44 to +4.91) | pass | 96 s |
| `es-ud-ancora-2.16-lemmas` | `es-ud-gsd-lemmas` | 1721 | 91.37 (90.98 to 91.75) | 90.64 (90.26 to 91.02) | +0.73 (+0.63 to +0.82) | pass | 241 s |
| `ca-ud-ancora-2.16-sentence` | `ca-ud-ancora-sentence` | 185 | 99.16 (98.57 to 99.68) | 99.16 (98.57 to 99.68) | +0.00 (+0.00 to +0.00) | pass | 1 s |
| `ca-ud-ancora-2.16-tokens` | `ca-ud-ancora-tokens` | 1846 | 99.92 (99.89 to 99.95) | 99.92 (99.89 to 99.95) | +0.00 (+0.00 to +0.00) | pass | 16 s |
| `ca-ud-ancora-2.16-pos` | `ca-ud-ancora-pos` | 1846 | 97.28 (97.09 to 97.47) | 97.24 (97.05 to 97.43) | +0.04 (-0.01 to +0.08) | pass | 84 s |
| `ca-ud-ancora-2.16-lemmas` | `ca-ud-ancora-lemmas` | 1846 | 89.77 (89.29 to 90.24) | 89.77 (89.29 to 90.24) | +0.00 (-0.01 to +0.01) | pass | 197 s |
| `pt-ud-porttinari-2.16-sentence` | `pt-ud-gsd-sentence` | 169 | 99.11 (98.57 to 99.64) | 98.26 (97.43 to 98.96) | +0.85 (+0.06 to +1.70) | pass | 0 s |
| `pt-ud-porttinari-2.16-tokens` | `pt-ud-gsd-tokens` | 1683 | 99.84 (99.78 to 99.90) | 99.20 (99.06 to 99.33) | +0.64 (+0.52 to +0.77) | pass | 5 s |
| `pt-ud-porttinari-2.16-pos` | `pt-ud-gsd-pos` | 1683 | 95.82 (95.55 to 96.10) | 91.49 (91.14 to 91.84) | +4.33 (+3.99 to +4.68) | pass | 24 s |
| `pt-ud-porttinari-2.16-lemmas` | `pt-ud-gsd-lemmas` | 1683 | 91.37 (90.94 to 91.81) | 81.38 (80.88 to 81.88) | +9.99 (+9.59 to +10.39) | pass | 137 s |
| `fi-ud-ftb-2.16-pos` | `fi-ud-tdt-pos` | 1867 | 89.25 (88.71 to 89.78) | 87.25 (86.67 to 87.83) | +2.00 (+1.47 to +2.53) | pass | 28 s |
| `fi-ud-ftb-2.16-lemmas` | `fi-ud-tdt-lemmas` | 1867 | 88.87 (88.36 to 89.38) | 83.97 (83.36 to 84.61) | +4.90 (+4.35 to +5.45) | pass | 1602 s |
| `it-ud-markit-2.16-pos` | `it-ud-vit-pos` | 340 | 89.96 (89.22 to 90.70) | 92.17 (91.55 to 92.78) | -2.21 (-2.92 to -1.49) | **fail** | 3 s |
| `it-ud-markit-2.16-lemmas` | `it-ud-vit-lemmas` | 340 | 82.57 (81.66 to 83.44) | 83.55 (82.67 to 84.40) | -0.98 (-1.44 to -0.53) | **fail** | 11 s |

The tokens of the `test` file each gate leaves out, and the lead, in
points, with them left out (the gate's) and with them counted (both
models scored on every token):

| Gate | Left out | Candidate right on them | Lead, left out | Lead, counted |
|---|---|---|---|---|
| `pt-ud-porttinari-2.16-pos` | 6 of 31,451 (0.02 percent) | 2 | +4.33 | +4.34 |
| `fi-ud-ftb-2.16-pos` | 635 of 16,261 (3.91 percent) | 428 | +2.00 | +4.55 |
| `it-ud-markit-2.16-pos` | 3 of 9,654 (0.03 percent) | 0 | -2.21 | -2.21 |
| `es-ud-ancora-2.16-lemmas` | 57 of 52,454 (0.11 percent) | 26 | +0.73 | +0.77 |
| `ca-ud-ancora-2.16-lemmas` | 18 of 57,894 (0.03 percent) | 0 | +0.00 | +0.00 |
| `pt-ud-porttinari-2.16-lemmas` | 100 of 31,451 (0.32 percent) | 62 | +9.99 | +10.16 |
| `fi-ud-ftb-2.16-lemmas` | 177 of 16,261 (1.09 percent) | 54 | +4.90 | +5.18 |
| `it-ud-markit-2.16-lemmas` | 33 of 9,654 (0.34 percent) | 10 | -0.98 | -0.87 |

- A tagger's tokens are left out by their gold tag: Porttinari's `INTJ`
  5 and `ADP+ADP` 1, FTB's `DET` 398, `PART` 234 and `SCONJ+VERB` 3
  (tags the Finnish TDT reference never gives), and MarkIT's `NOUN+PRON`
  2 and `DET+DET` 1. A lemmatizer's are listed by tag on its card. The
  reference gets none of them right, and the candidate gets a larger
  share of them right than its lead on the scored tokens in every gate
  but the Catalan lemmatizer's (0 of 18, and no lead), so leaving them
  out lowers each lead (the limit's note above), or leaves it the same
  to two places; FTB's tagger moves most, +2.00 against +4.55. Each card
  lists the reference's ten most frequent mismatches on the scored
  tokens (for FTB's tagger, `VERB` tagged `AUX` 303 times comes first).
- No model gives a lemma with a capital letter: OpenNLP's `decodeLemmas`
  applies a class to the lowercased token, and training makes every
  class from the lowercased lemma. The `test` tokens whose lemma has one
  are 4,093 of 52,454 for Spanish (7.80 percent), 5,323 of 57,894 for
  Catalan (9.19), 2,247 of 31,451 for Portuguese (7.14), 1 of 16,261 for
  Finnish, and 1,170 of 9,654 for Italian (12.12; 927 of them
  `[PUNCT]`, MarkIT's lemma for most punctuation). Without a capital, a
  gold class gives another string than the lemma for 1 Spanish, 2
  Portuguese and 8 Finnish tokens (the Finnish ones through an index of
  ten or more, which OpenNLP reads as one digit). Both models get all of
  these wrong, so they stay in the scores, as `LemmatizerEvaluator`
  counts them, and cap both models' accuracy. Making classes from the
  lemma as written and applying them to the token as written would let
  a model give capitalised lemmas: an improvement over OpenNLP found
  here and not made, so that the recipes train and score as OpenNLP
  does. The Spanish reference's five most frequent mismatches are such
  lemmas (`Gobierno` given as `gobierno` 67 times, `España` 43).
- The Italian models lose to OpenNLP's: the tagger by 2.21 points
  (interval -2.92 to -1.49) and the lemmatizer by 0.98 (-1.44 to
  -0.53), so a loss of more than 0.5 points is not ruled out for
  either, and `UdModelCard::artifact` makes no zoo entry for them.
  MarkIT's `train` file has 611 sentences.
- The Catalan references are OpenNLP's models of this treebank's
  `train` file (the zoo records them as trained on `UD_Catalan-AnCora
  r2.16`), made with the settings the recipes take from the UD 1.3
  models. The recipes make OpenNLP's sentence detector and tokenizer
  events exactly: each trained model's `Training-Eventhash` is its
  reference's (2275230916, from 14,747 events, and 289028499, from
  1,432,196), the model entries are the same byte for byte up to the
  parameters, and the parameters differ only in their last bits (1,107
  of 1,237 in `sent.model` and 39,509 of 55,744 in `token.model`, by at
  most 4.8e-13 of their value). So those two gates check that the
  recipes reproduce OpenNLP's models rather than compare two models, and
  the models agree on every unit. With the empty nodes kept, the recipes
  make OpenNLP's tagger and lemmatizer events exactly too (event hashes
  4145769459, from 422,116 events, and 896960108); the recipes leave the
  treebank's 4,999 empty nodes out (417,117 events), so the tagger and
  lemmatizer are trained on what OpenNLP's were less those nodes. The
  taggers differ on 127 of the 1,846 units and the lemmatizers on 6.
- The other references are trained on another treebank of the language
  (Spanish GSD, Portuguese GSD, Finnish TDT, Italian VIT), so a gap can
  come from the two treebanks' annotation as well as from the models,
  and that favours the candidate (above); the widest is the Portuguese
  lemmatizer's 9.99 points.

## Tests

`tests/recipes.rs` writes corpora in MASC's layout (`tests/common`, no
MASC text, Penn tags written for the tests) and runs everything on
them: the split in the reader's order; the Twitter genre left out of
both sides without moving another document; the four recipes trained,
scored and gated; the same bytes from a second training; the card, the
model's directory and its zoo entry; the zoo entries of a release made
from every recipe's directory and the built-in index with them added,
and their refusal of a missing directory, a card in another model's
directory and a changed model file; the skipped sentences recorded and
bounded on both sides; a sentence with a token inside another skipped
by the token recipe alone; the tagger's tags compared as written and
the reference's through the mapping of its tags, with the tokens of a
tag the reference cannot give left out, and a
reference that cannot give most tags refused; the chunker's samples read
from chunk layers written for the tests, a sentence with part of a chunk
that crosses sentences or with chunks that share a token skipped and
counted on both sides and bounded by its own limit, a chunk of a token
no sentence has, one across sentences and one of no token counted on
the card, the skipped sentences checked against the recorded counts
when the corpus is taken as the recorded release, the gate record's
`reference_inputs`, the reference's
chunks of types the gate does not score left out (a reference trained
with a `PP` chunk scores 100, and would score less if they were
counted), and a corpus read without its chunk files refused; and the
binary's refusal
of an unrecorded corpus, of a missing or wrong reference, of an
existing model directory and of held-out data with too many skipped
sentences before it trains anything, and its output, with the lines it
prints of each card (`ModelCard::summary`: the samples and skipped
sentences of both sides, the held-out tokens left out with their tags,
and the chunker's chunks in no sentence) checked word for word, and
`zoo-index`'s refusal of models trained on a corpus that is not the
recorded release. It
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
their tags moved one token along, and for the chunker with their chunks
moved one token along): as the reference it loses and the
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

The tagger's reference, `en-pos-maxent.bin`, and the chunker's,
`en-chunker.bin`, are local-only, as in turbo-sequence's tests of the
1.5 models: their training data is not published, so CI does not fetch
them. The parts that need them run when `TURBO_TEST_OPENNLP_MODELS`
names a directory holding them (a missing file there fails), and skip
without it even with `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`. With it, the
tagger's reference is checked against the recorded digest, shown to
return its 45 Penn outcomes unconverted and to tag brackets, hyphens,
ellipses, semicolons and an opening quotation mark as
`postag::REFERENCE_MAPPINGS` says, and a closing quotation mark `''`;
the chunker's is checked against its digest and shown to give
CoNLL-2000's eleven chunk types and to have features for `(`, `)` and
`:` and none for the three tags `chunk::REFERENCE_TAGS` maps; both are
scored and gated, and the binary runs all four recipes.

On the test corpus, with both directories set, the published references
score 63.16 (sentences, F-measure), 94.34 (tokens, F-measure), 92.41
(`en-pos-maxent.bin`, accuracy) and 93.33 (`en-chunker.bin`, F-measure
of `NP` and `VP` chunks), against 100 for the trained models,
which have seen every held-out sentence; the test checks these numbers,
which show the gate is wired through, not how the recipes do on MASC.

`tests/ud.rs` does the same for the UD recipes, on a treebank written for
the tests: eight Spanish sentences in CoNLL-U (with multiword tokens, an
empty node and `SpaceAfter=No`; no treebank text), a `LICENSE.txt` and a
README. It checks that the tagger's and lemmatizer's samples leave out
empty nodes and join a multiword token's tags and lemmas with `+`, and
the held-out digests: the tagger's is the MASC tagger's with no mapping,
the lemmatizer's covers the lemmas. It trains, scores, gates and writes
the card and directory of each of the four components against a
reference trained on two of the sentences: each candidate passes and
scores at least as high, with the record's scoring and digest and the
card's summary checked, except the tagger, which the check before
training and the gate both refuse because that reference has no outcome
for most of the gold tags. A reference trained without the interjection
leaves its 40 tokens out of both taggers' scores and onto the card, with
no mapping and both tag formats `custom`. A held-out file with one
sentence of 101 skipped is refused before training. A card gets a zoo
entry only for a recipe of `UD_RECIPES`, gated against that recipe's
reference and passed; an Italian card gated against the recorded
reference and failed gets none, and a card of another version is
refused by its version. A treebank whose file or licence changed is
refused. The binary
checks every reference (the zoo's digest or the recorded one) and that
no model directory exists, naming every problem, and reads every
treebank, before it trains anything; with the Italian tagger's
reference under `TURBO_TEST_OPENNLP_MODELS` it also checks that a
missing treebank is refused.
With `TURBO_TEST_OPENNLP_MODELS` naming a directory of the OpenNLP UD 1.3
models, each recipe's reference is checked against the zoo's digest (the
recorded one for the Italian models) and scored on the test text (Spanish for every language, so the scores only
show that each model loads and is scored); a missing file there is
skipped, or fails with `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`. CI does not
set `TURBO_TEST_OPENNLP_MODELS`, so that part runs only where the models
are present.

