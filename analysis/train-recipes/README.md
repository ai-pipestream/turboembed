# turbo-train-recipes

Recipes that train models only on permissive data, and the
accuracy gate each model passes before it ships (#401). docs/analysis.md
("Models trained on permissive data" and "Model gates") describes both.

| Module | What it holds |
|---|---|
| `masc` | MASC 3.0.0 (CC BY 3.0 US): the release record, its zoo dataset entry and fetch, the digest of an extracted copy, the fixed split, and the Twitter genre left out of it |
| `segmentation` | The English sentence detector and tokenizer recipes: training on the split's training documents, the held-out samples, scoring a model with OpenNLP's evaluators, the gate against the OpenNLP English UD model |
| `postag` | The English part-of-speech tagger recipe, the same way: training on MASC's Penn tags, the held-out samples, scoring with `POSEvaluator`, the gate against OpenNLP's English Penn tagger (`en-pos-maxent.bin`, OpenNLP 1.5) |
| `chunk` | The English chunker recipe, the same way: training on MASC's noun and verb chunks as `ChunkerME.train` trains with its defaults, the held-out samples, scoring the chunk types both models give with `ChunkerEvaluator`, the gate against OpenNLP's English chunker (`en-chunker.bin`, OpenNLP 1.5) |
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

The trained models are not checked in or published. A second run into
another directory wrote the same bytes:

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

## Tests

`tests/recipes.rs` writes corpora in MASC's layout (`tests/common`, no
MASC text, Penn tags written for the tests) and runs everything on
them: the split in the reader's order; the Twitter genre left out of
both sides without moving another document; the four recipes trained,
scored and gated; the same bytes from a second training; the card, the
model's directory and its zoo entry; the skipped sentences recorded and
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
and the chunker's chunks in no sentence) checked word for word. It
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
