# turbo-train-recipes

Recipes that train models only on permissive data, and the
accuracy gate each model passes before it ships (#401, #415).
docs/analysis.md ("Models trained on permissive data" and "Model gates")
describes both. The MASC recipes come first here; the recipes on the
CC BY 4.0 Universal Dependencies treebanks are under "The UD recipes".

The lemma recipes now preserve case by default. The older results below
use the OpenNLP lowercase format. Updated per-language accuracy, hashes,
commands and the Italian gate diagnosis are in
[docs/lemma-case-evaluation.md](../../docs/lemma-case-evaluation.md).

| Module | What it holds |
|---|---|
| `masc` | MASC 3.0.0 (CC BY 3.0 US): the release record, its zoo dataset entry and fetch, the digest of an extracted copy, the fixed split, and the Twitter genre left out of it |
| `segmentation` | The English sentence detector and tokenizer recipes: training on the split's training documents, the held-out samples, scoring a model with OpenNLP's evaluators, the gate against the OpenNLP English UD model |
| `postag` | The English part-of-speech tagger recipe, the same way: training on MASC's Penn tags, the held-out samples, scoring with `POSEvaluator`, the gate against OpenNLP's English Penn tagger (`en-pos-maxent.bin`, OpenNLP 1.5) |
| `chunk` | The English chunker recipe, the same way: training on MASC's noun and verb chunks as `ChunkerME.train` trains with its defaults, the held-out samples, scoring the chunk types both models give with `ChunkerEvaluator`, the gate against OpenNLP's English chunker (`en-chunker.bin`, OpenNLP 1.5) |
| `namefind` | The English name finder recipes, one per MASC entity type (person, location, organization, date): training as `NameFinderME.train` does with its defaults on MASC's named entities as the entity file links them to Penn tokens, the held-out samples, scoring with `TokenNameFinderEvaluator`, the gate against OpenNLP 1.5's English name finder of the same type (`en-ner-<type>.bin`) |
| `spans` | A held-out unit's counts from the `FMeasure` OpenNLP's evaluator keeps for it, both models' counts paired, and the span types a gate of typed spans (names, chunks) scores and renames |
| `card` | The model card, the model's directory (model, `card.json`, `gate.json`) and its permissive zoo entry, made only for a card whose gate against its recipe's reference (by id and SHA-256) passed on the recorded release |
| `treebank` | The CC BY 4.0 UD treebanks at release 2.16: each repository's `LICENSE.txt`, `README.md` and `train`, `dev` and `test` files with their sizes and digests, the licence check, the zoo dataset entry and fetch |
| `ud` | The UD recipes: the samples, training on `train`, the checks run before training, and the gate on `test` against the OpenNLP UD 1.3 model (the zoo's, or the Italian ones recorded by digest), by the `segmentation` and `postag` gates and, for the lemmatizer, `LemmatizerEvaluator` |
| `ud_card` | The UD model card, with the treebank's licence, attribution and citation and the MASC card's held-out and gate parts, the model's directory and its zoo entry |
| `release` | The releases of this repository that publish the recipes' models (`models-masc-3.0.0-1`, the four MASC models, and `models-ud-2.16-1`, ten UD models), each naming the models it publishes, their entries in the zoo's built-in index at their addresses there, and the references its workflow fetches (`masc-references.txt`, `ud-references.txt`) |
| `tatoeba` | Tatoeba's sentence exports: the strict reader of `sentences_detailed.csv` and `sentences_CC0.csv`, each sentence's licence from the two files, and the export's digest |
| `langdetect` | The language detector recipe: the licence check, the table of Tatoeba's and `langdetect-183.bin`'s language codes, the split, training, the gate against `langdetect-183.bin`, its card and the model's directory |
| `fineweb` | FineWeb-2 and FineWeb (ODC-By 1.0) at the revisions their lists record: the files and the ranges of them the web language detector reads, their lists, what to fetch next, the checks of each range, and the choice of web windows |
| `langdetect_web` | The web language detector recipe: the language detector recipe's sentences with FineWeb-2's and FineWeb's windows, training, the gates against `langdetect-183.bin` on held-out sentences and on held-out windows, the comparison with the language detector recipe's model, its card and the model's directory |

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
`en-chunker.bin`, downloaded by hand from
<https://opennlp.sourceforge.net/models-1.5/en-chunker.bin> and checked
the same way (2,560,304 bytes,
`7861a0c2f134d9c12a022a1ba501e88bc7039f6db72b4140e1bafd1fb5ef76cc`), and
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
| `turbo-en-masc-3.0.0-chunks-1` | `opennlp-en-chunker-1.5` (`en-chunker.bin`) | MASC Penn tokens, their `msd` tags and chunk tags of MASC's noun chunks (`NP`) and verb chunks (`VP`), one sample per sentence | F-measure of `NP` and `VP` chunks, one unit per sentence |
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
  `MascNamedEntitySampleStream` is not used. It reads a name of one
  token at a sentence's start as empty, and a sentence of two or more
  names mostly keeps one of them or none (its removal of overlapping
  names compares a name with itself, and can also keep names that
  overlap, which `NameSample` then refuses). What it keeps of a name of
  several tokens depends on the order the entity file lists them
  (`MascSentence.java:217` moves the end to one past a token only when
  the token is past the end so far, which is already one past a token):
  listed last to first, a name is whole; listed first to last, a name of
  an even number of tokens loses its last token and one of an odd number
  is whole, the other way round at a sentence's start. MASC 3.0.0 lists
  8,770 of its 9,068 entities of several tokens last to first, 98 first
  to last (all of an even number of tokens) and 200 in another order, so
  the rule costs few of MASC's names; the test corpus lists them first
  to last. It also marks every sentence to clear the adaptive data
  (`turbo_formats::masc` documents each; `tests/names.rs` shows them).
  Entities of other types are left out of the sample, as
  `NameSampleTypeFilter` leaves them out, and MASC's `org` is renamed
  `organization`, the type the reference finds. Where two entities of the
  type overlap, the first in `Span.compareTo` order (by start, the longer
  first) is kept and the other left out; an entity whose tokens had a
  gap would be kept as its covering span. A sentence of no tokens is
  skipped, within `MAX_SKIPPED_SENTENCES`.

  Some entities reach no sample, for one of two reasons
  (`MascDocument::unplaced_entities`). MASC marks no sentence over some
  text, such as most of the movie script's speaker labels (91 of the
  93 `ELIZABETH SWANN:`), so an entity there is outside every region of
  the sentence file. And the reader drops a sentence region that does
  not end before the next starts, as OpenNLP's `MascDocument` does, so
  an entity whose tokens are in such a region is in no sentence either.
  The Penn tokens of both are in no sample, so these entities are not
  names the training samples mark as `other`: they make no false
  negatives in training, and a held-out one is scored for neither
  model. The card counts them for the recipe's type
  (`outside_sentences`, `in_dropped_sentences`) beside the documents
  read, the names in the samples, those with a gap, the overlapping
  entities and the entities of other types left out (`training.names`,
  `held_out.names`).

  A document's first sample clears the adaptive data, as an empty line
  does in OpenNLP's name format, so what the previous-map feature
  generator learns of a document's names reaches only that document's
  later sentences. This was chosen before any result, and the recipes
  keep it: choosing it from the gates below would tune on the held-out
  set. It is how OpenNLP's `TokenNameFinderTool` runs a name finder over
  text, clearing at each empty line, which starts a document. No
  annotator in this repository runs a trained name finder, so there is
  no annotator's clearing for the gate to match: turbo-namefind's
  dictionary and regex finders keep no adaptive data, and
  turbo-sequence's `NameFinder` carries it from one `find` to the next
  until its caller clears it. `MascNamedEntitySampleStream` clears it at
  every sentence instead; the gates under that setting are recorded
  below as a diagnostic, and the person gate passes with it.
- The name finders' gate: both models are scored on the same held-out
  samples with `TokenNameFinderEvaluator`, one unit per sentence, over
  the names of the recipe's type; a name is right only with the held-out
  name's tokens and type. Each reference's outcomes are `other`,
  `<type>-start` and `<type>-cont`, so neither model gives another type.
  The gate resamples whole sentences and recomputes precision and recall
  from each draw's summed counts; it never averages the sentences' own
  F-measures. The sentences of a document are not independent: the
  adaptive data links them, a document repeats its names, and a few
  documents hold most of the held-out names (787 of the 1,363 persons in
  the held-out samples are in the movie script
  `data/written/movie-script/pirates`, whose entity file labels 1,580
  persons, 791 of them outside every sentence region), so an interval
  from resampling documents is wider (below). The references' training
  data is not published, and the zoo does not list them: the recipes
  record their addresses, sizes and SHA-256
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
    labelled an organization on their own). In the training documents a
    mention left unlabelled is `other` in its sample, which pushes the
    trained model to leave such names out and can cost it recall.
  - `location` in MASC includes regions, facilities and the like
    (`type` is `country`, `city`, `region`, `province`, `facility`,
    `us_region`, `highway_street`, `continent`, `attraction` and
    others).
  - MASC's genres are fiction, a movie script, spoken transcripts,
    e-mail, blogs, letters and others besides news, its tokens are its
    own Penn tokens, and the held-out names are concentrated in a few
    documents (above).

  Most of these count against the reference wherever it does not follow
  MASC, since the trained model learned MASC's conventions; the
  mentions MASC leaves unlabelled count against the trained model, which
  learned to leave them out. Neither effect is measured, and the losses
  below are not evidence that the trained models are worse: resampled
  by document, none of the four differences is clear of zero, and the
  person result turns on how the adaptive data is cleared (below).
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
detector's, tokenizer's, tagger's and chunker's gates passed; the four
name finders' gates failed (below). The name finders were run by
themselves (`--recipe` with their four ids), twice, from a release
build, which took 377 and 351 seconds. The scores are in points, rounded
to two places from the gate records' values (`result.candidate.score`,
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
| `turbo-en-masc-3.0.0-chunks-1` | 6,036 sentences, 32,252 chunks (`NP` and `VP`) | F-measure | 89.74 | 82.46 | +7.29 (+6.84 to +7.75) | pass |
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
| `turbo-en-masc-3.0.0-chunks-1` | 23,086 | 1,872 | 384,930 | 349 | |
| each `turbo-en-masc-3.0.0-ner-*-1` | 24,958 | 0 | 431,637 | 1 | |

The names each name finder's card records (`training.names` and
`held_out.names`). The first three counts are of the samples, an entity
counted once for each sentence it has a token in; no entity had a gap.
The last two count the entities of the type in no sample, once each:
those outside every region of the sentence file, and those in sentence
regions the reader drops.

| Type | Side | Names in the samples | Overlapping, left out | Of other types, left out | Outside every sentence region | In dropped sentence regions |
|---|---|---|---|---|---|---|
| `person` | training | 4,894 | 6 | 9,692 | 831 | 427 |
| `person` | held-out | 1,363 | 0 | 1,115 | 844 | 56 |
| `location` | training | 3,944 | 63 | 10,585 | 199 | 114 |
| `location` | held-out | 436 | 0 | 2,042 | 43 | 30 |
| `organization` | training | 2,934 | 36 | 11,622 | 192 | 143 |
| `organization` | held-out | 321 | 0 | 2,157 | 18 | 25 |
| `date` | training | 2,714 | 1 | 11,877 | 454 | 101 |
| `date` | held-out | 358 | 0 | 2,120 | 71 | 14 |

So 1,258 persons of the training documents and 900 of the held-out ones
are in no sample, 793 of the held-out ones in the movie script
`data/written/movie-script/pirates` (791 outside every sentence region,
2 in dropped ones), whose entity file labels 1,580 persons.

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
fewer names. In the movie script, which holds 787 of the persons in the
held-out samples, the trained model finds 319 of them and the reference
400. The organization model leads by 2.29 points, but the interval does
not rule out a loss of 6.41, and the date model's does not rule out one
of 12.64, so neither passes. The parameters are `NameFinderME.train`'s
defaults, chosen before the run; the training samples hold about 2,700
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

As a second check made after the run, and not a recipe, the same code
trained and scored the four models with every sample clearing the
adaptive data, as `MascNamedEntitySampleStream` marks its samples, the
references scored with the same clearing and nothing else changed:

| Type | Model's score (95% interval) | Reference's score (95% interval) | Difference (95% interval) | Gate |
|---|---|---|---|---|
| `person` | 57.32 (54.35 to 60.24) | 54.59 (51.80 to 57.29) | +2.73 (-0.30 to +5.86) | pass |
| `location` | 58.30 (53.18 to 63.32) | 62.86 (58.31 to 67.20) | -4.56 (-9.24 to +0.13) | fail |
| `organization` | 40.99 (34.11 to 47.52) | 40.19 (34.85 to 45.59) | +0.80 (-7.47 to +8.78) | fail |
| `date` | 43.97 (38.19 to 49.46) | 42.91 (37.46 to 48.19) | +1.06 (-6.60 to +8.79) | fail |

The person gate turns on this choice: its difference moves 8.21 points,
from -5.48 to +2.73, and the gate passes; the other differences move
1.49 to 5.78 points. Clearing per document helps the reference more than
the trained model: the reference's person recall is 47.54 cleared per
document and 40.57 cleared at every sentence, the trained model's 39.77
and 41.23. With the recipes' clearing, per document, the scores are
person 55.33 (52.32 to 58.30) against 60.82 (58.06 to 63.46), location 56.39 (50.90 to 61.66)
against 64.10 (59.63 to 68.35), organization 44.83 (37.58 to 51.59)
against 42.54 (37.11 to 48.00), date 37.73 (31.82 to 43.46) against
42.44 (37.05 to 47.67).

Two held-out documents account for the whole person loss. In the movie
script `data/written/movie-script/pirates` (787 of the persons in the
held-out samples) the trained model finds 319 (P 91.93, R 40.53, F
56.26) and the reference 400 (P 89.89, R 50.83, F 64.94); in the story
`data/written/fiction/Nathans_Bylichka` (151 persons) the trained model
finds 12 (P 92.31, R 7.95, F 14.63) and the reference 54 (P 91.53, R
35.76, F 51.43). On the other 37 held-out documents' 425 persons the
trained model scores P 89.41 (211 of 236), R 49.65, F 63.84, and the
reference P 73.48 (194 of 264), R 45.65, F 56.31; by the gate's
bootstrap over those 3,441 sentences, a subset chosen after seeing the
results, the difference is +7.53 (+1.84 to +13.16).

The gate resamples sentences, as every recipe's gate does. Resampled by
document instead, with the gate's own bootstrap over the 39 held-out
documents as units, each a document's summed counts (10,000 resamples,
seed 154), the person difference is -5.48 (-13.71 to +12.65) cleared
per document and +2.73 (-1.93 to +20.16) cleared at every sentence. The
held-out set is too small to tell the two models apart on person: it is
39 documents, and two of them hold 938 of its 1,363 persons. Resampled
by document, cleared per document, the other types' differences are
location -7.72 (-24.74 to +3.67), organization +2.29 (-17.81 to +30.18)
and date -4.72 (-16.19 to +13.49); none of the four is clear of zero.

The trained models are not checked in. "The models releases" below says
where the sentence detector, tokenizer, tagger and chunker are
published; the name finders, which fail their gates, are not published.
A second run into another directory wrote the same bytes (for the name
finders, the same `card.json` and `gate.json` too):

| Model | Size | SHA-256 |
|---|---|---|
| `turbo-en-masc-3.0.0-sentence-1.bin` | 44,601 bytes | `d03253a5559a230d1f42e1cf881119204302fbb7fefb613ff1af11bcbd3c261e` |
| `turbo-en-masc-3.0.0-tokens-1.bin` | 916,027 bytes | `2c5fe73121c41300e6eae4a3c4d7a2247ade20a1e182d8e36d7781124aa9b7db` |
| `turbo-en-masc-3.0.0-pos-1.bin` | 3,671,103 bytes | `391d7f79c7246e9e56ed9d8849fde129c5bed8683fa33af31e532cb0c37f96ea` |
| `turbo-en-masc-3.0.0-chunks-1.bin` | 6,042,700 bytes | `8629b99af79f4cac2f997f3f9042ecb0739776f1e49f1b01c713877a1f997e09` |
| `turbo-en-masc-3.0.0-ner-person-1.bin` | 17,616,486 bytes | `dbcbf69d2cce4a876024b1b0ce729eff09c3a6c2188945b494860086f7a19878` |
| `turbo-en-masc-3.0.0-ner-location-1.bin` | 17,610,640 bytes | `0a0abf72f2936a672422c3638e074227ea831204affb4d98c582662523205560` |
| `turbo-en-masc-3.0.0-ner-organization-1.bin` | 17,635,648 bytes | `5a20025774ff14e7bbff92c9d3c79f7a1d7a8cd76989b8212a006145a9356fba` |
| `turbo-en-masc-3.0.0-ner-date-1.bin` | 17,584,650 bytes | `1193738ea3b523903bd02a5e441e265a0f27c317054dc5e919c0f9dcbf9b865e` |

A run of the first four recipes (the release binary, five minutes on a
machine of four cores) wrote the first four files, and two runs of the
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

## The models releases

The models go out as the assets of releases of this repository, as
Kristian Rickert decided on 2026-10-15, and the zoo's built-in index
lists each at its address in its release,
`https://github.com/ai-pipestream/turboembed/releases/download/<tag>/<model id>.bin`,
with the size, SHA-256 and SHA-512 its card records. A release names
the models it publishes (`release::MODELS_RELEASES`):

| Release | Models | Workflow | References |
|---|---|---|---|
| `models-masc-3.0.0-1` (`release::MASC_MODELS`) | the MASC sentence detector, tokenizer, tagger and chunker | `.github/workflows/models.yml` | `masc-references.txt` |
| `models-ud-2.16-1` (`release::UD_MODELS`) | the Spanish and Catalan sentence detectors, tokenizers, taggers and lemmatizers, and the Finnish tagger and lemmatizer: every UD recipe's model but Porttinari's, whose text rights are not settled, and the Italian ones, which fail their gates | `.github/workflows/models-ud.yml` | `ud-references.txt` |
| `models-tatoeba-2026-10-10-1` (`release::TATOEBA_MODELS`) | the language detector, with `training-sentences.txt`, the ids of the sentences it was trained on | `.github/workflows/models-tatoeba.yml` | `tatoeba-references.txt` |
| `models-tatoeba-fineweb-2026-10-10-1` (`release::FINEWEB_MODELS`) | the web language detector, with `training-sentences.txt` and `training-documents.txt`, the ids of the sentences and of the documents it was trained on | `.github/workflows/models-fineweb.yml` | `fineweb-references.txt` |

The Tatoeba export the language detector is trained on is kept as well,
as release `data-tatoeba-2026-10-10` (`release::TATOEBA_EXPORT_RELEASE`):
Tatoeba replaces its export every week at the same addresses, so its
two archives, unchanged, are the assets of that release, which
`models-tatoeba.yml` publishes before the model's
(`tatoeba-export.txt` lists their SHA-256s and addresses).

A release does not exist until its workflow has been run by hand on
main; until then its addresses answer 404 and fetching its models
through the zoo fails. A recipe added later is in none of these
releases: its model goes out in a release of its own, with its own
tag, and a release once published keeps its assets.

`zoo-index --models <dir> --release <tag>` reads the directories
`masc`, `ud`, `tatoeba` or `tatoeba-fineweb` wrote into `<dir>` for the
models the release `<tag>` publishes, checks each file of an entry (the
model, the language detectors' `training-sentences.txt` and the web
one's `training-documents.txt`) against its card, each card's gate
against its gate record and, for the web language detector, its second
gate and its two comparisons against theirs, makes each card's zoo
entry at its address in the release, and prints the zoo's built-in
index with those entries added or replaced. That is how
`analysis/zoo/index/zoo.json` got them. The entry is made by
`ModelCard::artifact`, `UdModelCard::artifact`, `LanguageCard::artifact`
or `WebLanguageCard::artifact`, which refuse a model that failed a gate
or was gated against another model than its recipe's reference. The
first also refuses one not trained on the recorded MASC release, the
third one not trained on the recorded Tatoeba export, and the fourth
one not trained on that export and on the recorded web ranges, or not
compared with the Tatoeba language detector; the
second refuses one of Porttinari, and a card that does not give its
recipe's treebank at release 2.16, with the sizes and digests of its
files, or its recipe's component and language.
With `--check` it prints no index and exits with 1 unless the built-in
index is already that one and lists no other model at the release's
addresses, naming the entries that differ with the size, SHA-256 and
address the index lists and the model's, and each other model listed
there.

Each workflow makes its release. Its training job fetches the data
through the zoo (MASC, or the treebanks the release's UD models are
trained on) or, for the language detectors, from the release that keeps
the Tatoeba export ("Full run" below; the web one also from the Hugging
Face hub, "The web language detector recipe"), and the references from
the addresses its list gives, each checked against the SHA-256 listed with
it: for MASC, OpenNLP's UD 1.3 English sentence and token models from
`archive.apache.org` and `en-pos-maxent.bin` and `en-chunker.bin` from
SourceForge; for UD, OpenNLP's UD 1.3 models from `archive.apache.org`;
for the language detectors, `langdetect-183.bin` from
`downloads.apache.org`, and for the web one also the Tatoeba language
detector from its release, to be compared with. The references are
only scored, never shipped: the English, Spanish and Finnish UD models are trained on CC BY-SA
treebanks, the licences of the two 1.5 models are refused, and the
terms of the Leipzig corpora `langdetect-183.bin` was trained on are not
confirmed. It saves each download in the repository's Actions cache as
soon as it is checked, so a run that fails later does not fetch it
again. It then runs `masc` or `ud` for the models the zoo's index lists
at the release's addresses, or `tatoeba` or `tatoeba-fineweb` twice,
checking that both runs write the same files, runs `zoo-index --check`,
and uploads each model with its card and gate record (`<model id>.bin`,
`<model id>.card.json`, `<model id>.gate.json`; the language detector
also with `training-sentences.txt`, and the web one with
`training-sentences.txt`, `training-documents.txt`,
`<model id>.gate-web.json`, `<model id>.compare-tatoeba.json` and
`<model id>.compare-web.json`) and a `SHA256SUMS` file as an artifact.
A pull request that changes the workflow, this crate or the zoo's index runs it
that far. Run by hand on main, a job of its own, which unlike the
training job may write to the repository, runs
`.github/workflows/models-publish.yml`,
which publishes exactly those files: it checks `SHA256SUMS`, stops if
the tag exists, and creates the release at the run's commit, not marked
as the repository's latest release, with notes that give the models'
licence and the attribution of each card (MASC's, each treebank's,
Tatoeba's, or Tatoeba's, FineWeb-2's and FineWeb's). Each `.card.json`
asset carries its attribution as well. A release whose tag exists is not made again, so models trained otherwise
need a new model version and tag. `models-tatoeba.yml` first publishes
the export's release, from another job that may write, with the archives
its run read and `NOTICE.md`, the licence notice of its release notes;
when that release exists, it checks that its archives are the listed
ones instead.

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
  (`fetch-ud`, which with `--recipe` fetches only the treebanks the named
  recipes read) as permissive datasets (`ud-<language>-<name>-r2.16`).
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
- Card and record: `card.json` is a `UdModelCard` (version 4): the
  treebank and its files, licence, attribution and citation where the
  MASC card has the corpus, and the MASC card's `training`, `held_out`
  and `gate` parts, their chunk and name counts `null`; it is read
  version first, as the MASC card is. `gate.json` is the MASC recipes'
  `ModelGateRecord` (version 3; `check` refuses a lemmatizer's record
  with a mapping). The run prints each
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
  whose gate failed, such as the Italian POS model; and also one trained on
  a treebank whose text rights are not settled, Porttinari, whatever its
  gate says.

```
cargo run --release -p turbo-train-recipes -- fetch-ud --home <zoo-home> [--recipe <model id>]...
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
records are byte-identical to theirs, so their scores are too. The cards were
version 3 then. The lemmatizers' gates leave out the tokens whose lemma the
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
- In the historical run below, no model gave a lemma with a capital letter: OpenNLP's `decodeLemmas`
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
  counts them, and capped both models' accuracy. The native format now
  keeps case; the linked evaluation above supersedes that limitation. The Spanish reference's five most frequent mismatches are such
  lemmas (`Gobierno` given as `gobierno` 67 times, `España` 43).
- In that run the Italian models lost to OpenNLP's: the tagger by 2.21 points
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
  files), not by address. `langdetect::EXPORT` records Tatoeba's export
  of 2026-10-10 ("Full run" below), so `tatoeba` refuses any other
  export unless given `--unrecorded-export`, and then writes a card with
  `recorded: false`, which is not a release of the model.
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
  are those of Tatoeba's language list. A code the export does not use
  leaves its language with no sentences, listed in the card's
  `excluded_languages`; in the recorded export every one of the 100
  codes has sentences. The card also counts the export's sentences of
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
  not be reached when this recipe was made ("The web language detector
  recipe" below adds it); FLORES+ (evaluation only), for the same reason; the
  Leipzig corpora, whose terms are not confirmed (docs/licences.md). The
  gate uses only the Tatoeba held-out split, which neither model trained
  on, so no evaluation-only data is fetched or kept.

### Which way the gate leans

The held-out units are single Tatoeba sentences, which are short (#429
says so). The
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

Made on 2026-10-15 on GitHub's runners, since downloads.tatoeba.org,
tatoeba.org and Hugging Face refused the connection (403) from the
machine the recipe was written on: run 38769842277 of the
`models-tatoeba` workflow, from pull request #455, fetched that week's
export and trained on it twice with `--unrecorded-export`. Both runs
wrote the same model, sentence file, card and gate record, in about 190
seconds each. The export was then recorded with the steps below, and the
workflow now trains on it alone.

The record:

- tatoeba2, checked on 2026-10-15: the default branch `dev` was still
  at commit 0fbb4506, and `master` (c3235bf9, 2026-07-16) has the same
  three files. `docs/database/tables/sentences.sql` last changed in
  2019, `src/Lib/Licenses.php` on 2025-10-28 and
  `docs/database/scripts/weekly_exports.sql` in 2022, all before the
  export, so it was written with the two sentence licences the reader
  assumes.
- The export of 2026-10-10. `sentences_detailed.tar.bz2` (303,565,606 bytes,
  SHA-256 `776e298918d984b0e6ea3c8c48eb32ca6f79df624827025d463ce16e019a749f`)
  holds `sentences_detailed.csv`: 1,409,238,272 bytes, SHA-256
  `8ce364127cf6f1ab0387fe75bfa98f7ea767c9a406d39e1d25f66c4d435f58be`,
  written at 06:02. `sentences_CC0.tar.bz2` (7,959,741 bytes, SHA-256
  `74a1577f35d9f11134b281be3125cf17e5531b8a29054f68b3e858418387e17f`)
  holds `sentences_CC0.csv`: 41,860,119 bytes, SHA-256
  `7aa76236dba7c78a4841b7041737c91ef6670bb4fda031e98446d0a52e773198`,
  written at 06:11 the same day. The times are the archives' tar
  headers, so the two files are of one export. Its digest
  (`langdetect::EXPORT`) is
  `4421126b36dc0ce206b222296d696aff04a399620f1ca5dbefbfa50a1890aa92`.
- Records: the detailed file has 13,606,084, 528 of them with no
  language and 2,203,399 of languages outside the table. The CC0 file has 562,242, none
  with another language or text than the detailed file gives and none
  missing from it (`cc0_differing` and `cc0_not_in_detailed` are 0).
- Languages: 71 trained on and held out, with 321,026 training
  sentences and 35,500 held out (500 each). 29 are left out with fewer
  than 1,500 sentences after the licence check and de-duplication, with
  that count: ast 824, bak 434, che 47, fao 427, fry 713, gsw 566,
  guj 168, kan 311, kir 809, lim 47, ltz 1,041, mal 884, min 121,
  mlg 74, mlt 696, mri 503, nan 604, pan 217, pnb 77, pus 66, san 167,
  sin 75, som 164, sun 765, tam 597, tel 318, tgk 81, uzb 1,235,
  zul 89. Every code of the table has sentences in the export.
- Gate, over the 35,500 held-out sentences: the trained model 92.19%
  (91.91 to 92.46), `langdetect-183.bin` 83.91% (83.53 to 84.29), a
  difference of +8.28 points (+7.94 to +8.63): pass.
- Files, the same from both runs: the model, 4,755,070 bytes, SHA-256
  `89f8e570f2f27184267de91ce26433e848ee341b94b7ef9fe623a56236e4e64e`;
  `training-sentences.txt`, 2,648,398 bytes (321,026 ids), SHA-256
  `423b88e8ac2af4f7bd57ef286e019c1b151accf3f50ec0c43618113a64d76ab9`.

Per language (`per_language` in the card), each with 500 sentences held
out; no language has a text twice, so none is left out as a duplicate.
The trained model is right on more held-out sentences than the
reference in 56 languages, as often in 8 and less often in 7: Bosnian
(1 against 58), Norwegian Bokmål (376 against 400), Portuguese (469
against 479), Bengali (497 against 499), and Mandarin, Hebrew and
Albanian (one sentence each). Bosnian is the clear weak spot: the model
was trained on 1,259 Bosnian sentences beside 5,000 each of Croatian and
Serbian, which are written much alike.

<details>
<summary>The 71 languages</summary>

| Language | Sentences | Licence left out | Trained on | Trained model right | `langdetect-183.bin` right |
|---|---:|---:|---:|---:|---:|
| `afr` | 4,940 | 0 | 4,440 | 484 | 482 |
| `ara` | 68,575 | 2 | 5,000 | 500 | 451 |
| `aze` | 13,531 | 0 | 5,000 | 486 | 450 |
| `bel` | 14,427 | 1,929 | 5,000 | 490 | 488 |
| `ben` | 15,812 | 7,151 | 5,000 | 497 | 499 |
| `bos` | 1,759 | 0 | 1,259 | 1 | 58 |
| `bre` | 8,000 | 0 | 5,000 | 468 | 412 |
| `bul` | 25,719 | 0 | 5,000 | 466 | 463 |
| `cat` | 10,963 | 5 | 5,000 | 458 | 386 |
| `ceb` | 1,893 | 0 | 1,393 | 221 | 98 |
| `ces` | 89,990 | 3 | 5,000 | 459 | 447 |
| `cmn` | 89,179 | 1 | 5,000 | 499 | 500 |
| `cym` | 1,803 | 5 | 1,298 | 468 | 460 |
| `dan` | 66,540 | 34 | 5,000 | 431 | 323 |
| `deu` | 783,551 | 46 | 5,000 | 492 | 469 |
| `ell` | 42,774 | 0 | 5,000 | 500 | 500 |
| `eng` | 2,038,719 | 41,512 | 5,000 | 479 | 467 |
| `epo` | 822,798 | 370 | 5,000 | 489 | 484 |
| `est` | 6,358 | 0 | 5,000 | 484 | 451 |
| `eus` | 6,433 | 0 | 5,000 | 491 | 485 |
| `fin` | 154,926 | 1,561 | 5,000 | 493 | 478 |
| `fra` | 728,438 | 14,202 | 5,000 | 489 | 454 |
| `gle` | 2,989 | 0 | 2,489 | 492 | 487 |
| `glg` | 8,212 | 0 | 5,000 | 459 | 396 |
| `heb` | 212,732 | 27 | 5,000 | 499 | 500 |
| `hin` | 16,475 | 57 | 5,000 | 494 | 488 |
| `hrv` | 6,551 | 0 | 5,000 | 457 | 321 |
| `hun` | 442,121 | 1 | 5,000 | 492 | 468 |
| `hye` | 51,922 | 0 | 5,000 | 500 | 500 |
| `ind` | 28,339 | 0 | 5,000 | 342 | 261 |
| `isl` | 13,387 | 0 | 5,000 | 498 | 480 |
| `ita` | 988,826 | 19 | 5,000 | 491 | 447 |
| `jav` | 1,732 | 0 | 1,232 | 390 | 252 |
| `jpn` | 248,924 | 2 | 5,000 | 500 | 500 |
| `kat` | 7,697 | 0 | 5,000 | 500 | 500 |
| `kaz` | 4,575 | 0 | 4,075 | 489 | 479 |
| `kor` | 15,941 | 0 | 5,000 | 500 | 500 |
| `lat` | 56,575 | 31 | 5,000 | 483 | 435 |
| `lit` | 158,709 | 0 | 5,000 | 486 | 479 |
| `lvs` | 19,816 | 0 | 5,000 | 496 | 488 |
| `mar` | 87,415 | 0 | 5,000 | 499 | 495 |
| `mkd` | 78,259 | 0 | 5,000 | 471 | 382 |
| `mon` | 2,791 | 0 | 2,291 | 496 | 488 |
| `nds` | 18,233 | 0 | 5,000 | 473 | 402 |
| `nld` | 201,295 | 50 | 5,000 | 464 | 422 |
| `nno` | 12,493 | 0 | 5,000 | 410 | 359 |
| `nob` | 18,892 | 5 | 5,000 | 376 | 400 |
| `npi` | 4,504 | 0 | 4,004 | 497 | 491 |
| `oci` | 25,170 | 0 | 5,000 | 452 | 382 |
| `pes` | 31,797 | 0 | 5,000 | 496 | 489 |
| `pol` | 137,180 | 57 | 5,000 | 494 | 475 |
| `por` | 445,215 | 17 | 5,000 | 469 | 479 |
| `ron` | 41,613 | 0 | 5,000 | 490 | 478 |
| `rus` | 1,229,481 | 23,075 | 5,000 | 477 | 384 |
| `slk` | 33,264 | 0 | 5,000 | 466 | 390 |
| `slv` | 3,586 | 0 | 3,086 | 393 | 393 |
| `spa` | 442,544 | 37 | 5,000 | 455 | 405 |
| `sqi` | 2,634 | 0 | 2,134 | 460 | 461 |
| `srp` | 52,103 | 0 | 5,000 | 262 | 230 |
| `swe` | 56,846 | 208 | 5,000 | 472 | 453 |
| `swh` | 4,583 | 0 | 4,083 | 487 | 467 |
| `tat` | 100,088 | 0 | 5,000 | 486 | 136 |
| `tgl` | 79,131 | 0 | 5,000 | 493 | 462 |
| `tha` | 6,851 | 0 | 5,000 | 500 | 500 |
| `tur` | 749,693 | 0 | 5,000 | 489 | 482 |
| `ukr` | 188,849 | 393 | 5,000 | 485 | 440 |
| `urd` | 2,851 | 0 | 2,351 | 494 | 447 |
| `vie` | 33,223 | 0 | 5,000 | 498 | 498 |
| `vol` | 6,620 | 3 | 5,000 | 497 | 321 |
| `war` | 2,391 | 0 | 1,891 | 399 | 16 |
| `zsm` | 7,791 | 0 | 5,000 | 414 | 275 |

</details>

The `models-tatoeba` workflow (`.github/workflows/models-tatoeba.yml`)
trains the release's model on the recorded export when it is run by hand
or a pull request changes it or the crates the run mostly rests on: this
one, the trainers (`analysis/train`, `analysis/train-components`), the
gate (`analysis/eval`), `turbo-langdetect` and the zoo's index. It
fetches the two archives from release `data-tatoeba-2026-10-10`, or,
until that release is published, from the artifact run 38769842277 kept
(until 2027-01-13), checks them against `tatoeba-export.txt` before it
caches them, and prints their `tar -tvjf` listings. It trains twice and
fails unless both runs write the same files, prints the card's
per-language counts, and checks with `zoo-index --check` that the
model and `training-sentences.txt` are the bytes the zoo's index lists
("The models releases" above).

A later export gives another model, which needs a model id, a release
and an export release of its own, since a published release keeps its
assets. The recipe trains on one with `--unrecorded-export`; on a
machine that can reach Tatoeba:

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
   in the same columns. If any of them changed, the reader's licence
   classification (`tatoeba`) must be revised before a run is recorded.
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
5. Keep the archives in a release of their own, list them in
   `tatoeba-export.txt`, and name the model's release in
   `release::MODELS_RELEASES`.

## The web language detector recipe

| Model | Reference | Data | Measure |
|---|---|---|---|
| `turbo-mul-tatoeba-fineweb-langdetect-1` | `opennlp-langdetect-183` (`langdetect-183.bin`) | the language detector recipe's Tatoeba sentences, and web text of the same 100 languages from FineWeb-2 and FineWeb (ODC-By 1.0): at most 1,000 windows of at most 300 characters per language | accuracy, one unit per held-out sentence and one per held-out window, in two gates |

```
cargo run --release -p turbo-train-recipes -- fineweb-ranges --paths
cargo run --release -p turbo-train-recipes -- fineweb-ranges --web <dir> [--files <path>] (--need | --record)
cargo run --release -p turbo-train-recipes -- tatoeba-fineweb --export <dir> --web <dir> \
    --references <dir> --out <dir> [--unrecorded-export] [--files <path> --ranges <path> --unrecorded-web]
```

`--export` is the Tatoeba export, read and checked as `tatoeba` reads
it. `--references` holds `langdetect-183.bin` and the language detector
recipe's released model, `turbo-mul-tatoeba-langdetect-1.bin`, each
checked against the digest the recipe records (`fineweb-references.txt`
lists both with their addresses). `--web` holds the ranges of the web
files, fetched as `fineweb-ranges` says (below). `--files` and
`--ranges`, given together, name the two lists to read in place of the
committed ones; a run on lists that are not the committed ones needs
`--unrecorded-web` and writes a card with `recorded: false`, which is
not a release. The model directory, both models, the web directory and
the lists are checked first, with every problem found named; then the
web ranges, against the lists; then the export is read. Both commands
refuse a flag they do not read. The exit status is 1 when a gate fails
and 2 on an error.

The language detector recipe trains on single Tatoeba sentences, which
are short ("Which way the gate leans" above), and leaves out the 29
languages with fewer than 1,500 of them. #429 asks for FineWeb-2 where
it can be fetched; this recipe adds its web text, which is longer and
less tidy, and with it those 29 languages.

- Data and licence. FineWeb-2 (`HuggingFaceFW/fineweb-2` at revision
  `af9c13333eb981300149d5ca60a8e9d659b276b9`) is CommonCrawl text in
  over a thousand language and script configurations. It has no
  English, which comes from FineWeb (`HuggingFaceFW/fineweb` at
  revision `9bb295ddab0e05d785b879661af7260fed5140fc`). Both cards
  publish the datasets under ODC-By 1.0 and say their use is also
  subject to CommonCrawl's terms of use (`fineweb::FINEWEB_2`,
  `fineweb::FINEWEB`, `fineweb::COMMONCRAWL_TERMS`). ODC-By 1.0 is
  permissive and asks for attribution, which the card gives; the zoo
  entry names both datasets and their licence. No web text is shipped: the model's
  directory holds `training-documents.txt`
  (`langdetect_web::DOCUMENTS_FILE`), each document trained on as its
  repository, file, row and FineWeb id (`repo\tpath\trow\tid`), sorted.
- Labels. Neither dataset's languages were given by people: FineWeb-2
  labels each document with GlotLID and FineWeb with fastText's
  language identifier. The recipe trains on those labels, and the web
  gate scores both models against them.
- Languages (`fineweb::WEB_LANGUAGES`): the 100 languages of
  `langdetect::LANGUAGES`, in its order, each with the FineWeb-2
  configuration of the language in the script Tatoeba writes it in.
  Most configurations share Tatoeba's code. Where they do not, the
  configuration is the language Tatoeba's code names or, for a
  macrolanguage, the individual language its written standard is:
  `arb` for `ara`, `azj` for `aze`, `ekk` for `est`, `plt` for `mlg`,
  `khk` for `mon`, `pbt` for `pus`, `als` for `sqi`, `uzn` for `uzb`,
  and `fas` for `pes` and `fil` for `tgl`, which FineWeb-2 has under
  those codes only. Serbian and Uzbek, which Tatoeba writes in two
  scripts, read two configurations each (`srp_Cyrl` and `srp_Latn`,
  `uzn_Latn` and `uzn_Cyrl`), sharing the language's documents (the
  first takes one more when they do not divide evenly). The romanized
  configurations (`hin_Latn` and the like) are left out. English reads
  FineWeb's 10BT sample.
- Files. A source's training documents come from the first file of its
  `train` split (`data/<configuration>/train/000_00000.parquet`) and its
  held-out ones from the first file of its `test` split; FineWeb has no
  test split, so English's come from `sample/10BT/000_00000.parquet` and
  `sample/10BT/001_00000.parquet`. That is 204 files
  (`fineweb-ranges --paths` lists them with their revisions).
- Reading by range. FineWeb-2's files are large (German's first training
  file is about 4.8 GB, with a footer of about 13 MB) and the recipe
  reads a few thousand rows of each, so each file is read from three
  ranges (`fineweb::FileRanges`): its last 8 bytes, which give the
  footer's length; the first 64 KiB of its footer (all of it when
  shorter), which hold its schema and its first row groups' metadata; and its first bytes up to the end
  of the fewest leading row groups that hold two rows for each document
  wanted (`WEB_SPLIT.rows_per_document`), or all of its row groups when
  it has fewer rows. turbo-eval's reader reads those row groups
  (`read::parquet::Leading` and `Parquet::open_head`), and refuses a
  footer prefix that ends before the row groups needed.
- Lists (`fineweb-files.txt` and `fineweb-ranges.txt`, compiled into the
  binary). The first lists each file's repository, revision, path, size
  and SHA-256, as the hub's API gives them at the revision; the second
  lists each range's file, start, end and SHA-256. A run checks the
  files listed are exactly the 204 the recipe reads, that the ranges
  listed are exactly the ones those files give, and every range it reads
  against its SHA-256; a changed byte, a missing range or one not listed
  is refused. The binary checks ranges, not files (only for 16 small
  files do the three ranges cover the whole file), so a file's SHA-256
  is not checked by it; the workflow checks the files list against the
  hub's API on every run (below). Fetched ranges are kept under their
  file's revision (`<dir>/<repo>/<revision>/<path>.<start>-<end>`). The
  card records both lists' digest (`lists_sha256`).
- Windows (`fineweb::WEB_SPLIT`). A document gives its first 300
  characters (Unicode scalar values), or all of them when it has fewer.
  One with fewer than 100 is passed over, and so is one from
  `tatoeba.org` or a subdomain of it (`fineweb::EXCLUDED_HOSTS`), a
  window its language already has or an earlier row of its file gave,
  and a window that holds a Tatoeba
  sentence of its language from the other side of the split, byte for
  byte: a training window a held-out sentence, which would favour the
  trained model on the Tatoeba gate, and a held-out window a training
  sentence (`fineweb::Avoid`). Sentences of fewer than 20 characters are
  not looked for: they are common phrases, found in web text whatever
  its source. A file's documents are taken in row order, at first at
  most one in 20 from one host (the URL's host, lowercased, without
  `www.` or a trailing dot); when that leaves the file short, the
  documents passed over for their host are taken next, in row order.
  Every training file is read before any held-out file, and a held-out
  window that is a training window of any language is passed over. Per
  language, at most 1,000 windows are trained on and 500 held out: a
  file with too few rows left gives fewer, and a language's other source
  does not make up the difference. The card counts, for each file, the
  documents wanted, the rows and row groups read, the documents passed
  over and why, and those chosen (`web.files`), and the binary names each
  file that gave fewer than wanted.
- Training: the language detector recipe's sentences, split as it splits
  them, then the windows, one event each, each language in the order of
  `langdetect::LANGUAGES`, with the same parameters (`MAXENT`, 100
  iterations, cutoff 5, one thread, the `OnePass` indexer) and the same
  fixed `Timestamp`, so a run gives the same bytes. A language left out
  of the Tatoeba split is trained on its windows alone, so the model
  chooses among all 100 languages.
- Gates, both against `langdetect-183.bin`, as the language detector
  recipe gates (the table's codes counted right for the reference, 10,000
  resamples, seed 154, a pass when the difference's interval does not
  start below -0.5): on the language detector recipe's held-out
  sentences, the set its model was gated on, with the same name and
  digest (`gate.json`); and on the held-out windows (`gate-web.json`).
  The model is a release only when it passes both.
- Comparison: the held-out sentences, and the held-out windows of the
  71 languages it has, scored again with `turbo-mul-tatoeba-langdetect-1`
  in the reference's place (`compare-tatoeba.json`, `compare-web.json`),
  so the card says what the web text changed (`compared_on_sentences`
  and `compared_on_windows`, with no pass, and per language in
  `compared_per_language` and `web_per_language`). It is reported, not a
  gate. The windows of the other 29 languages are left out of
  `compare-web.json`, as that model cannot be right on them.

### Which way the web gate leans

The web gate measures agreement with GlotLID's and fastText's labels, on
windows from the test files of the same configurations the training
windows come from, so the trained model, which learned from text the
same classifiers labelled, has the advantage there, as it has on the
Tatoeba sentences. The reference was trained on the Leipzig corpora's
news and web text and is scored against the same labels; where a label
is wrong, a model that is right is counted wrong. Windows of 100 to 300
characters are closer in length to the reference's own evaluation
samples (123 to 1,668 characters) than Tatoeba's sentences are. A pass
says the trained model agrees with those labels at least as often as
`langdetect-183.bin` does, less half a point; it does not say either
model is right where the labels are wrong.

### Full run

Made on 2026-10-16 by run 38848321968 of the `models-fineweb` workflow,
from pull request #474, on the committed lists and the recorded Tatoeba
export. It trained twice, and both runs wrote the same eight files, in
about 1,100 seconds each on a GitHub runner, with a peak of 5.8 GB of
memory. The run then failed its check of the zoo's index, which did not
list the model yet, so it built no release files; the entry it printed
is the one `analysis/zoo/index/zoo.json` now holds.

The `models-fineweb` workflow
(`.github/workflows/models-fineweb.yml`) runs when it is run by hand or
a pull request changes it, `models-publish.yml`, this crate, the
trainers, the gate (`analysis/eval`), the Zstandard decoder,
`turbo-langdetect` or the zoo's index; only a run by hand on main
publishes, as release `models-tatoeba-fineweb-2026-10-10-1` ("The
models releases"). It fetches the Tatoeba export as `models-tatoeba`
does (the same cache), the two models of `fineweb-references.txt`
(checked against their digests before they are cached) and the web
ranges, cached under the digest of both lists. It makes the files list
from the hub's API (`/api/datasets/<repo>/revision/<revision>?blobs=true`,
whose `sha` must be the revision) and fails unless it is the committed
one. It fetches the ranges not in the cache in rounds, each as
`fineweb-ranges --need` says, curl stopping an answer more than 64 KiB
longer than the range (such as the whole file) and the step refusing
any answer that is not `206 Partial Content` of the range's length,
then fails unless `fineweb-ranges --record` gives the committed ranges
list. A cache made under other lists is restored as a start, and what
it holds is checked like anything fetched. It trains twice, fails
unless both runs write the same eight files, prints the card's
per-language and per-file counts, and keeps both runs' directories,
with their logs and digests, as the artifact `models-fineweb-run`. It then checks with `zoo-index
--check` that the model's files are the bytes the zoo's index lists,
printing the entry the run gives when they are not, and uploads the
release's files as the artifact named by the tag. While the committed
lists are empty, the run makes them in place of these checks, keyed
also by `src/fineweb.rs`, prints them and trains with
`--unrecorded-web`.

The record:

- Lists: made by run 38835162381 from pull request #474, on
  2026-10-16 (run 38836303379 printed them again from the cache): 204 files, 612 ranges, 550,033,792 bytes in all.
  `fineweb-files.txt` has SHA-256
  `73b924f48f8fee76019ca92e6c5f5bc1f579a3a16144b33b6c62eb80f114eeec`,
  `fineweb-ranges.txt`
  `9e7fa76c5d6c82ad16da3b4b4a65c168adb3b182b1c687fa6d2f9e4aeba3efea`,
  and their digest (`lists_digest`, the card's `lists_sha256`) is
  `bd599877adb11071caa5c34ff8bf522797469540319d2267b6694b1a1cb74298`.
  16 files have fewer rows than the recipe reads for them and are read
  whole: the test files of `ast`, `bre`, `che`, `gsw`, `lim`, `min`,
  `nan`, `nds`, `oci`, `pnb`, `san`, `sun`, `vol`, `war` and `zul`, and
  `min`'s training file.
- Export: the one the Tatoeba recipe records, digest
  `4421126b36dc0ce206b222296d696aff04a399620f1ca5dbefbfa50a1890aa92`,
  split as that recipe splits it: 71 languages with sentences, 321,026
  trained on and 35,500 held out.
- Languages: 100, the 71 with sentences and 29 trained on their web
  windows alone (the 29 the Tatoeba recipe leaves out). 99,543 windows
  trained on and 47,079 held out. Every language has 1,000 training
  windows but `min` (543), and 500 held out but `che` (204), `lim`
  (251), `min` (11), `nan` (38), `pnb` (256), `san` (233), `vol` (52)
  and `war` (34): of the 16 files read whole, these 9 gave fewer
  windows than wanted, and the binary named them.
- How the windows were chosen, over the 204 files: no document was
  shorter than 100 characters; 3 came from tatoeba.org hosts and 4
  held a Tatoeba sentence of the other side, all passed over; 558
  windows were duplicates; the host cap deferred 52,886 documents, of
  which 9,787 were taken in the second pass. 146,622 windows were
  chosen in all.
- Gate on the held-out sentences, 35,500 units: the trained model
  91.25% (90.96 to 91.54), `langdetect-183.bin` 83.91% (83.53 to
  84.29), a difference of +7.34 points (+6.99 to +7.69): pass.
- Gate on the held-out windows, 47,079 units: the trained model 96.46%
  (96.30 to 96.63), `langdetect-183.bin` 93.45% (93.23 to 93.68), a
  difference of +3.01 points (+2.81 to +3.21): pass.
- Beside the Tatoeba model (not gates): on the 35,500 held-out
  sentences 91.25% against its 92.19%, a difference of -1.12 to -0.75
  points, so this model is about a point worse on Tatoeba's short
  sentences; on the 34,586 held-out windows of the 71 languages the
  Tatoeba model knows, 96.84% against its 90.88%, a difference of +5.70
  to +6.23 points.
- Files, the same from both runs: the model, 16,734,028 bytes, SHA-256
  `6a1b88d5f1b14696810deadf44cec848195f46337dc46e2b0e69023c970af66c`;
  `training-sentences.txt`, 2,648,398 bytes, SHA-256
  `423b88e8ac2af4f7bd57ef286e019c1b151accf3f50ec0c43618113a64d76ab9`,
  the Tatoeba model's file, as the split is the same;
  `training-documents.txt`, 11,345,621 bytes (99,543 lines), SHA-256
  `1f5006aa1e4b44f392731f114c34f83d335dff9753084ea2abdec0c7faaa4959`;
  `card.json`
  `090d221bf9528b006a83e5ac5d4e596e80a6639ed0fcf6578900f54089b3ba4e`,
  `gate.json`
  `aaf7d3bbb3d3b4d748b91124d80e206e999ddef155b641ac1c40869a6d79a611`,
  `gate-web.json`
  `b25ba75e09068a1f635b9aea5296de4c508058ff24dd369ffb9372fad51b4446`,
  `compare-tatoeba.json`
  `2d4ad16881a52277cef706cb95498497a0d6f773293a0a3c0b999e27e5fe143c` and
  `compare-web.json`
  `db40a0c1da0230950e03dc380b59e69c0cf902ae8737fc53fafdb7ae71924522`.

Per language. On the held-out sentences the trained model is right more
often than the reference in 53 languages, as often in 9 and less often
in 9: Bosnian (0 against 58), Slovenian (341 against 393), Bulgarian
(442 against 463), Portuguese (464 against 479), Welsh (447 against
460), Norwegian Bokmål (388 against 400), Afrikaans and Mandarin (two
sentences each) and Albanian (one). Beside the Tatoeba model it is
right more often in 30 languages, as often in 12 and less often in 29,
most of all in Cebuano (106 against 221), Waray (334 against 399),
Slovenian (341 against 393), Indonesian (305 against 342) and Javanese
(356 against 390). On the held-out windows it is right more often than
the reference in 49 languages, as often in 21 and less often in 30,
most of all in Bashkir (454 against 496), Tajik (472 against 495),
English (480 against 500) and Bosnian (319 against 335).

<details>
<summary>The 71 languages' held-out sentences</summary>

| Language | Held out | Trained model right | `langdetect-183.bin` right | Tatoeba model right |
|---|---:|---:|---:|---:|
| `afr` | 500 | 480 | 482 | 484 |
| `ara` | 500 | 499 | 451 | 500 |
| `aze` | 500 | 483 | 450 | 486 |
| `bel` | 500 | 494 | 488 | 490 |
| `ben` | 500 | 499 | 499 | 497 |
| `bos` | 500 | 0 | 58 | 1 |
| `bre` | 500 | 471 | 412 | 468 |
| `bul` | 500 | 442 | 463 | 466 |
| `cat` | 500 | 457 | 386 | 458 |
| `ceb` | 500 | 106 | 98 | 221 |
| `ces` | 500 | 465 | 447 | 459 |
| `cmn` | 500 | 498 | 500 | 499 |
| `cym` | 500 | 447 | 460 | 468 |
| `dan` | 500 | 440 | 323 | 431 |
| `deu` | 500 | 493 | 469 | 492 |
| `ell` | 500 | 500 | 500 | 500 |
| `eng` | 500 | 468 | 467 | 479 |
| `epo` | 500 | 490 | 484 | 489 |
| `est` | 500 | 477 | 451 | 484 |
| `eus` | 500 | 493 | 485 | 491 |
| `fin` | 500 | 493 | 478 | 493 |
| `fra` | 500 | 486 | 454 | 489 |
| `gle` | 500 | 489 | 487 | 492 |
| `glg` | 500 | 465 | 396 | 459 |
| `heb` | 500 | 500 | 500 | 499 |
| `hin` | 500 | 494 | 488 | 494 |
| `hrv` | 500 | 464 | 321 | 457 |
| `hun` | 500 | 491 | 468 | 492 |
| `hye` | 500 | 500 | 500 | 500 |
| `ind` | 500 | 305 | 261 | 342 |
| `isl` | 500 | 495 | 480 | 498 |
| `ita` | 500 | 493 | 447 | 491 |
| `jav` | 500 | 356 | 252 | 390 |
| `jpn` | 500 | 500 | 500 | 500 |
| `kat` | 500 | 500 | 500 | 500 |
| `kaz` | 500 | 489 | 479 | 489 |
| `kor` | 500 | 500 | 500 | 500 |
| `lat` | 500 | 484 | 435 | 483 |
| `lit` | 500 | 491 | 479 | 486 |
| `lvs` | 500 | 498 | 488 | 496 |
| `mar` | 500 | 498 | 495 | 499 |
| `mkd` | 500 | 492 | 382 | 471 |
| `mon` | 500 | 492 | 488 | 496 |
| `nds` | 500 | 476 | 402 | 473 |
| `nld` | 500 | 465 | 422 | 464 |
| `nno` | 500 | 386 | 359 | 410 |
| `nob` | 500 | 388 | 400 | 376 |
| `npi` | 500 | 499 | 491 | 497 |
| `oci` | 500 | 452 | 382 | 452 |
| `pes` | 500 | 498 | 489 | 496 |
| `pol` | 500 | 496 | 475 | 494 |
| `por` | 500 | 464 | 479 | 469 |
| `ron` | 500 | 492 | 478 | 490 |
| `rus` | 500 | 456 | 384 | 477 |
| `slk` | 500 | 465 | 390 | 466 |
| `slv` | 500 | 341 | 393 | 393 |
| `spa` | 500 | 458 | 405 | 455 |
| `sqi` | 500 | 460 | 461 | 460 |
| `srp` | 500 | 254 | 230 | 262 |
| `swe` | 500 | 466 | 453 | 472 |
| `swh` | 500 | 491 | 467 | 487 |
| `tat` | 500 | 488 | 136 | 486 |
| `tgl` | 500 | 496 | 462 | 493 |
| `tha` | 500 | 500 | 500 | 500 |
| `tur` | 500 | 488 | 482 | 489 |
| `ukr` | 500 | 477 | 440 | 485 |
| `urd` | 500 | 495 | 447 | 494 |
| `vie` | 500 | 498 | 498 | 498 |
| `vol` | 500 | 499 | 321 | 497 |
| `war` | 500 | 334 | 16 | 399 |
| `zsm` | 500 | 435 | 275 | 414 |

</details>

<details>
<summary>The 100 languages' windows</summary>

The Tatoeba model's column is empty for the 29 languages it does not
know.

| Language | Trained on | Held out | Trained model right | `langdetect-183.bin` right | Tatoeba model right |
|---|---:|---:|---:|---:|---:|
| `afr` | 1,000 | 500 | 489 | 472 | 427 |
| `ara` | 1,000 | 500 | 499 | 499 | 500 |
| `ast` | 1,000 | 500 | 415 | 417 |  |
| `aze` | 1,000 | 500 | 500 | 500 | 500 |
| `bak` | 1,000 | 500 | 454 | 496 |  |
| `bel` | 1,000 | 500 | 495 | 496 | 491 |
| `ben` | 1,000 | 500 | 500 | 499 | 500 |
| `bos` | 1,000 | 500 | 319 | 335 | 0 |
| `bre` | 1,000 | 500 | 488 | 488 | 479 |
| `bul` | 1,000 | 500 | 495 | 496 | 479 |
| `cat` | 1,000 | 500 | 497 | 498 | 494 |
| `ceb` | 1,000 | 500 | 471 | 9 | 212 |
| `ces` | 1,000 | 500 | 497 | 498 | 497 |
| `che` | 1,000 | 204 | 201 | 203 |  |
| `cmn` | 1,000 | 500 | 499 | 500 | 499 |
| `cym` | 1,000 | 500 | 482 | 479 | 475 |
| `dan` | 1,000 | 500 | 494 | 490 | 487 |
| `deu` | 1,000 | 500 | 491 | 496 | 491 |
| `ell` | 1,000 | 500 | 498 | 498 | 497 |
| `eng` | 1,000 | 500 | 480 | 500 | 496 |
| `epo` | 1,000 | 500 | 487 | 484 | 481 |
| `est` | 1,000 | 500 | 492 | 493 | 488 |
| `eus` | 1,000 | 500 | 481 | 481 | 473 |
| `fao` | 1,000 | 500 | 476 | 488 |  |
| `fin` | 1,000 | 500 | 497 | 496 | 488 |
| `fra` | 1,000 | 500 | 498 | 498 | 498 |
| `fry` | 1,000 | 500 | 469 | 483 |  |
| `gle` | 1,000 | 500 | 476 | 461 | 452 |
| `glg` | 1,000 | 500 | 491 | 488 | 489 |
| `gsw` | 1,000 | 500 | 393 | 353 |  |
| `guj` | 1,000 | 500 | 495 | 495 |  |
| `heb` | 1,000 | 500 | 499 | 499 | 499 |
| `hin` | 1,000 | 500 | 497 | 493 | 476 |
| `hrv` | 1,000 | 500 | 443 | 448 | 487 |
| `hun` | 1,000 | 500 | 494 | 493 | 490 |
| `hye` | 1,000 | 500 | 490 | 488 | 489 |
| `ind` | 1,000 | 500 | 431 | 250 | 255 |
| `isl` | 1,000 | 500 | 497 | 498 | 495 |
| `ita` | 1,000 | 500 | 499 | 498 | 496 |
| `jav` | 1,000 | 500 | 463 | 436 | 30 |
| `jpn` | 1,000 | 500 | 500 | 498 | 499 |
| `kan` | 1,000 | 500 | 494 | 494 |  |
| `kat` | 1,000 | 500 | 497 | 496 | 497 |
| `kaz` | 1,000 | 500 | 490 | 490 | 489 |
| `kir` | 1,000 | 500 | 487 | 496 |  |
| `kor` | 1,000 | 500 | 498 | 495 | 495 |
| `lat` | 1,000 | 500 | 481 | 470 | 470 |
| `lim` | 1,000 | 251 | 210 | 197 |  |
| `lit` | 1,000 | 500 | 496 | 494 | 492 |
| `ltz` | 1,000 | 500 | 481 | 485 |  |
| `lvs` | 1,000 | 500 | 494 | 493 | 492 |
| `mal` | 1,000 | 500 | 496 | 495 |  |
| `mar` | 1,000 | 500 | 496 | 496 | 494 |
| `min` | 543 | 11 | 8 | 1 |  |
| `mkd` | 1,000 | 500 | 496 | 498 | 499 |
| `mlg` | 1,000 | 500 | 490 | 494 |  |
| `mlt` | 1,000 | 500 | 487 | 475 |  |
| `mon` | 1,000 | 500 | 492 | 494 | 494 |
| `mri` | 1,000 | 500 | 492 | 487 |  |
| `nan` | 1,000 | 38 | 33 | 26 |  |
| `nds` | 1,000 | 500 | 407 | 353 | 320 |
| `nld` | 1,000 | 500 | 495 | 497 | 489 |
| `nno` | 1,000 | 500 | 468 | 451 | 314 |
| `nob` | 1,000 | 500 | 477 | 481 | 454 |
| `npi` | 1,000 | 500 | 499 | 499 | 499 |
| `oci` | 1,000 | 500 | 422 | 401 | 380 |
| `pan` | 1,000 | 500 | 490 | 495 |  |
| `pes` | 1,000 | 500 | 497 | 500 | 485 |
| `pnb` | 1,000 | 256 | 244 | 224 |  |
| `pol` | 1,000 | 500 | 498 | 498 | 496 |
| `por` | 1,000 | 500 | 498 | 498 | 495 |
| `pus` | 1,000 | 500 | 484 | 496 |  |
| `ron` | 1,000 | 500 | 499 | 498 | 495 |
| `rus` | 1,000 | 500 | 500 | 500 | 496 |
| `san` | 1,000 | 233 | 222 | 214 |  |
| `sin` | 1,000 | 500 | 484 | 482 |  |
| `slk` | 1,000 | 500 | 494 | 494 | 485 |
| `slv` | 1,000 | 500 | 491 | 491 | 481 |
| `som` | 1,000 | 500 | 496 | 494 |  |
| `spa` | 1,000 | 500 | 499 | 498 | 497 |
| `sqi` | 1,000 | 500 | 497 | 498 | 488 |
| `srp` | 1,000 | 500 | 403 | 249 | 284 |
| `sun` | 1,000 | 500 | 473 | 110 |  |
| `swe` | 1,000 | 500 | 496 | 496 | 489 |
| `swh` | 1,000 | 500 | 495 | 493 | 480 |
| `tam` | 1,000 | 500 | 495 | 496 |  |
| `tat` | 1,000 | 500 | 495 | 471 | 494 |
| `tel` | 1,000 | 500 | 490 | 490 |  |
| `tgk` | 1,000 | 500 | 472 | 495 |  |
| `tgl` | 1,000 | 500 | 473 | 466 | 328 |
| `tha` | 1,000 | 500 | 499 | 496 | 499 |
| `tur` | 1,000 | 500 | 496 | 495 | 495 |
| `ukr` | 1,000 | 500 | 494 | 493 | 492 |
| `urd` | 1,000 | 500 | 496 | 496 | 495 |
| `uzb` | 1,000 | 500 | 494 | 497 |  |
| `vie` | 1,000 | 500 | 499 | 499 | 498 |
| `vol` | 1,000 | 52 | 51 | 50 | 44 |
| `war` | 1,000 | 34 | 33 | 0 | 8 |
| `zsm` | 1,000 | 500 | 453 | 383 | 420 |
| `zul` | 1,000 | 500 | 496 | 493 |  |

</details>

<details>
<summary>The 204 files</summary>

No file had a document shorter than 100 characters, so that column is
left out.

| Language | Side | File | Wanted | Rows in file | Row groups read | Rows read | Tatoeba host | Duplicates | Tatoeba sentence | Host deferred | Taken later | Chosen |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `afr` | training | `data/afr_Latn/train/000_00000.parquet` | 1,000 | 1,992,040 | 2 | 2,000 | 0 | 5 | 0 | 404 | 0 | 1,000 |
| `ara` | training | `data/arb_Arab/train/000_00000.parquet` | 1,000 | 2,693,000 | 2 | 2,000 | 0 | 5 | 0 | 124 | 0 | 1,000 |
| `ast` | training | `data/ast_Latn/train/000_00000.parquet` | 1,000 | 71,329 | 2 | 2,000 | 0 | 0 | 0 | 555 | 0 | 1,000 |
| `aze` | training | `data/azj_Latn/train/000_00000.parquet` | 1,000 | 3,453,000 | 2 | 2,000 | 0 | 2 | 0 | 496 | 0 | 1,000 |
| `bak` | training | `data/bak_Cyrl/train/000_00000.parquet` | 1,000 | 183,068 | 2 | 2,000 | 0 | 5 | 0 | 1,700 | 705 | 1,000 |
| `bel` | training | `data/bel_Cyrl/train/000_00000.parquet` | 1,000 | 2,100,873 | 2 | 2,000 | 0 | 0 | 0 | 1,468 | 468 | 1,000 |
| `ben` | training | `data/ben_Beng/train/000_00000.parquet` | 1,000 | 2,912,000 | 2 | 2,000 | 0 | 3 | 0 | 151 | 0 | 1,000 |
| `bos` | training | `data/bos_Latn/train/000_00000.parquet` | 1,000 | 3,604,000 | 2 | 2,000 | 0 | 2 | 0 | 0 | 0 | 1,000 |
| `bre` | training | `data/bre_Latn/train/000_00000.parquet` | 1,000 | 59,875 | 2 | 2,000 | 0 | 5 | 0 | 1,874 | 879 | 1,000 |
| `bul` | training | `data/bul_Cyrl/train/000_00000.parquet` | 1,000 | 2,647,000 | 2 | 2,000 | 0 | 1 | 0 | 189 | 0 | 1,000 |
| `cat` | training | `data/cat_Latn/train/000_00000.parquet` | 1,000 | 4,106,000 | 2 | 2,000 | 0 | 6 | 0 | 348 | 0 | 1,000 |
| `ceb` | training | `data/ceb_Latn/train/000_00000.parquet` | 1,000 | 204,636 | 2 | 2,000 | 0 | 2 | 0 | 183 | 0 | 1,000 |
| `ces` | training | `data/ces_Latn/train/000_00000.parquet` | 1,000 | 3,007,000 | 2 | 2,000 | 0 | 4 | 0 | 62 | 0 | 1,000 |
| `che` | training | `data/che_Cyrl/train/000_00000.parquet` | 1,000 | 22,345 | 2 | 2,000 | 0 | 2 | 0 | 1,653 | 655 | 1,000 |
| `cmn` | training | `data/cmn_Hani/train/000_00000.parquet` | 1,000 | 1,973,000 | 2 | 2,000 | 0 | 8 | 0 | 33 | 0 | 1,000 |
| `cym` | training | `data/cym_Latn/train/000_00000.parquet` | 1,000 | 831,878 | 2 | 2,000 | 0 | 4 | 0 | 875 | 0 | 1,000 |
| `dan` | training | `data/dan_Latn/train/000_00000.parquet` | 1,000 | 3,515,000 | 2 | 2,000 | 0 | 2 | 0 | 70 | 0 | 1,000 |
| `deu` | training | `data/deu_Latn/train/000_00000.parquet` | 1,000 | 3,320,000 | 2 | 2,000 | 0 | 20 | 0 | 0 | 0 | 1,000 |
| `ell` | training | `data/ell_Grek/train/000_00000.parquet` | 1,000 | 2,759,000 | 2 | 2,000 | 0 | 1 | 0 | 52 | 0 | 1,000 |
| `eng` | training | `sample/10BT/000_00000.parquet` | 1,000 | 1,048,581 | 3 | 2,999 | 0 | 0 | 0 | 0 | 0 | 1,000 |
| `epo` | training | `data/epo_Latn/train/000_00000.parquet` | 1,000 | 335,993 | 2 | 2,000 | 1 | 0 | 0 | 1,838 | 839 | 1,000 |
| `est` | training | `data/ekk_Latn/train/000_00000.parquet` | 1,000 | 2,904,000 | 2 | 2,000 | 0 | 5 | 0 | 393 | 0 | 1,000 |
| `eus` | training | `data/eus_Latn/train/000_00000.parquet` | 1,000 | 1,569,434 | 2 | 2,000 | 0 | 22 | 0 | 394 | 0 | 1,000 |
| `fao` | training | `data/fao_Latn/train/000_00000.parquet` | 1,000 | 291,151 | 2 | 2,000 | 0 | 2 | 0 | 611 | 0 | 1,000 |
| `fin` | training | `data/fin_Latn/train/000_00000.parquet` | 1,000 | 2,994,000 | 2 | 2,000 | 0 | 1 | 0 | 79 | 0 | 1,000 |
| `fra` | training | `data/fra_Latn/train/000_00000.parquet` | 1,000 | 3,314,000 | 2 | 2,000 | 1 | 4 | 0 | 0 | 0 | 1,000 |
| `fry` | training | `data/fry_Latn/train/000_00000.parquet` | 1,000 | 349,743 | 2 | 2,000 | 0 | 0 | 0 | 1,826 | 826 | 1,000 |
| `gle` | training | `data/gle_Latn/train/000_00000.parquet` | 1,000 | 646,842 | 2 | 2,000 | 0 | 29 | 0 | 1,117 | 146 | 1,000 |
| `glg` | training | `data/glg_Latn/train/000_00000.parquet` | 1,000 | 2,522,814 | 2 | 2,000 | 0 | 3 | 0 | 536 | 0 | 1,000 |
| `gsw` | training | `data/gsw_Latn/train/000_00000.parquet` | 1,000 | 58,314 | 2 | 2,000 | 0 | 1 | 0 | 932 | 0 | 1,000 |
| `guj` | training | `data/guj_Gujr/train/000_00000.parquet` | 1,000 | 2,127,094 | 2 | 2,000 | 0 | 4 | 0 | 37 | 0 | 1,000 |
| `heb` | training | `data/heb_Hebr/train/000_00000.parquet` | 1,000 | 2,706,000 | 2 | 2,000 | 0 | 2 | 0 | 265 | 0 | 1,000 |
| `hin` | training | `data/hin_Deva/train/000_00000.parquet` | 1,000 | 2,910,000 | 2 | 2,000 | 0 | 3 | 0 | 390 | 0 | 1,000 |
| `hrv` | training | `data/hrv_Latn/train/000_00000.parquet` | 1,000 | 1,864,000 | 2 | 2,000 | 0 | 2 | 0 | 331 | 0 | 1,000 |
| `hun` | training | `data/hun_Latn/train/000_00000.parquet` | 1,000 | 2,512,000 | 2 | 2,000 | 0 | 4 | 0 | 60 | 0 | 1,000 |
| `hye` | training | `data/hye_Armn/train/000_00000.parquet` | 1,000 | 1,757,415 | 2 | 2,000 | 0 | 0 | 0 | 445 | 0 | 1,000 |
| `ind` | training | `data/ind_Latn/train/000_00000.parquet` | 1,000 | 3,149,000 | 2 | 2,000 | 0 | 3 | 0 | 6 | 0 | 1,000 |
| `isl` | training | `data/isl_Latn/train/000_00000.parquet` | 1,000 | 2,976,000 | 2 | 2,000 | 0 | 5 | 0 | 339 | 0 | 1,000 |
| `ita` | training | `data/ita_Latn/train/000_00000.parquet` | 1,000 | 3,333,000 | 2 | 2,000 | 0 | 0 | 0 | 0 | 0 | 1,000 |
| `jav` | training | `data/jav_Latn/train/000_00000.parquet` | 1,000 | 184,561 | 2 | 2,000 | 0 | 8 | 0 | 521 | 0 | 1,000 |
| `jpn` | training | `data/jpn_Jpan/train/000_00000.parquet` | 1,000 | 2,762,000 | 2 | 2,000 | 0 | 9 | 0 | 63 | 0 | 1,000 |
| `kan` | training | `data/kan_Knda/train/000_00000.parquet` | 1,000 | 2,390,982 | 2 | 2,000 | 0 | 0 | 0 | 528 | 0 | 1,000 |
| `kat` | training | `data/kat_Geor/train/000_00000.parquet` | 1,000 | 2,634,000 | 2 | 2,000 | 0 | 7 | 0 | 757 | 0 | 1,000 |
| `kaz` | training | `data/kaz_Cyrl/train/000_00000.parquet` | 1,000 | 2,360,000 | 2 | 2,000 | 0 | 1 | 0 | 581 | 0 | 1,000 |
| `kir` | training | `data/kir_Cyrl/train/000_00000.parquet` | 1,000 | 1,069,582 | 2 | 2,000 | 0 | 0 | 0 | 1,741 | 741 | 1,000 |
| `kor` | training | `data/kor_Hang/train/000_00000.parquet` | 1,000 | 2,783,000 | 2 | 2,000 | 0 | 2 | 0 | 450 | 0 | 1,000 |
| `lat` | training | `data/lat_Latn/train/000_00000.parquet` | 1,000 | 1,473,541 | 2 | 2,000 | 0 | 19 | 0 | 185 | 0 | 1,000 |
| `lim` | training | `data/lim_Latn/train/000_00000.parquet` | 1,000 | 37,165 | 2 | 2,000 | 0 | 3 | 0 | 1,075 | 78 | 1,000 |
| `lit` | training | `data/lit_Latn/train/000_00000.parquet` | 1,000 | 2,590,000 | 2 | 2,000 | 0 | 4 | 0 | 397 | 0 | 1,000 |
| `ltz` | training | `data/ltz_Latn/train/000_00000.parquet` | 1,000 | 354,553 | 2 | 2,000 | 0 | 0 | 0 | 559 | 0 | 1,000 |
| `lvs` | training | `data/lvs_Latn/train/000_00000.parquet` | 1,000 | 2,752,000 | 2 | 2,000 | 0 | 4 | 0 | 492 | 0 | 1,000 |
| `mal` | training | `data/mal_Mlym/train/000_00000.parquet` | 1,000 | 2,678,000 | 2 | 2,000 | 0 | 0 | 0 | 91 | 0 | 1,000 |
| `mar` | training | `data/mar_Deva/train/000_00000.parquet` | 1,000 | 2,957,000 | 2 | 2,000 | 0 | 0 | 0 | 512 | 0 | 1,000 |
| `min` | training | `data/min_Latn/train/000_00000.parquet` | 1,000 | 557 | 1 | 557 | 0 | 14 | 0 | 60 | 60 | 543 |
| `mkd` | training | `data/mkd_Cyrl/train/000_00000.parquet` | 1,000 | 3,899,000 | 2 | 2,000 | 0 | 4 | 0 | 974 | 0 | 1,000 |
| `mlg` | training | `data/plt_Latn/train/000_00000.parquet` | 1,000 | 272,871 | 2 | 2,000 | 0 | 4 | 0 | 499 | 0 | 1,000 |
| `mlt` | training | `data/mlt_Latn/train/000_00000.parquet` | 1,000 | 489,190 | 2 | 2,000 | 0 | 14 | 0 | 191 | 0 | 1,000 |
| `mon` | training | `data/khk_Cyrl/train/000_00000.parquet` | 1,000 | 1,622,882 | 2 | 2,000 | 0 | 31 | 0 | 88 | 0 | 1,000 |
| `mri` | training | `data/mri_Latn/train/000_00000.parquet` | 1,000 | 166,938 | 2 | 2,000 | 0 | 2 | 0 | 362 | 0 | 1,000 |
| `nan` | training | `data/nan_Latn/train/000_00000.parquet` | 1,000 | 2,891 | 2 | 2,000 | 0 | 3 | 0 | 1,277 | 280 | 1,000 |
| `nds` | training | `data/nds_Latn/train/000_00000.parquet` | 1,000 | 64,394 | 2 | 2,000 | 1 | 1 | 0 | 1,405 | 407 | 1,000 |
| `nld` | training | `data/nld_Latn/train/000_00000.parquet` | 1,000 | 4,004,000 | 2 | 2,000 | 0 | 1 | 0 | 0 | 0 | 1,000 |
| `nno` | training | `data/nno_Latn/train/000_00000.parquet` | 1,000 | 1,214,870 | 2 | 2,000 | 0 | 0 | 0 | 616 | 0 | 1,000 |
| `nob` | training | `data/nob_Latn/train/000_00000.parquet` | 1,000 | 2,891,000 | 2 | 2,000 | 0 | 2 | 0 | 145 | 0 | 1,000 |
| `npi` | training | `data/npi_Deva/train/000_00000.parquet` | 1,000 | 3,496,000 | 2 | 2,000 | 0 | 2 | 0 | 0 | 0 | 1,000 |
| `oci` | training | `data/oci_Latn/train/000_00000.parquet` | 1,000 | 69,376 | 2 | 2,000 | 0 | 2 | 0 | 999 | 1 | 1,000 |
| `pan` | training | `data/pan_Guru/train/000_00000.parquet` | 1,000 | 944,160 | 2 | 2,000 | 0 | 2 | 0 | 52 | 0 | 1,000 |
| `pes` | training | `data/fas_Arab/train/000_00000.parquet` | 1,000 | 2,645,000 | 2 | 2,000 | 0 | 3 | 1 | 29 | 0 | 1,000 |
| `pnb` | training | `data/pnb_Arab/train/000_00000.parquet` | 1,000 | 20,767 | 2 | 2,000 | 0 | 4 | 0 | 1,429 | 433 | 1,000 |
| `pol` | training | `data/pol_Latn/train/000_00000.parquet` | 1,000 | 3,183,000 | 2 | 2,000 | 0 | 1 | 0 | 160 | 0 | 1,000 |
| `por` | training | `data/por_Latn/train/000_00000.parquet` | 1,000 | 3,502,000 | 2 | 2,000 | 0 | 0 | 0 | 0 | 0 | 1,000 |
| `pus` | training | `data/pbt_Arab/train/000_00000.parquet` | 1,000 | 639,983 | 2 | 2,000 | 0 | 5 | 0 | 1,481 | 486 | 1,000 |
| `ron` | training | `data/ron_Latn/train/000_00000.parquet` | 1,000 | 3,143,000 | 2 | 2,000 | 0 | 1 | 0 | 34 | 0 | 1,000 |
| `rus` | training | `data/rus_Cyrl/train/000_00000.parquet` | 1,000 | 1,815,000 | 2 | 2,000 | 0 | 1 | 0 | 28 | 0 | 1,000 |
| `san` | training | `data/san_Deva/train/000_00000.parquet` | 1,000 | 21,453 | 2 | 2,000 | 0 | 16 | 0 | 1,010 | 26 | 1,000 |
| `sin` | training | `data/sin_Sinh/train/000_00000.parquet` | 1,000 | 1,185,323 | 2 | 2,000 | 0 | 1 | 0 | 57 | 0 | 1,000 |
| `slk` | training | `data/slk_Latn/train/000_00000.parquet` | 1,000 | 3,288,000 | 2 | 2,000 | 0 | 8 | 0 | 154 | 0 | 1,000 |
| `slv` | training | `data/slv_Latn/train/000_00000.parquet` | 1,000 | 3,171,000 | 2 | 2,000 | 0 | 8 | 0 | 303 | 0 | 1,000 |
| `som` | training | `data/som_Latn/train/000_00000.parquet` | 1,000 | 1,070,384 | 2 | 2,000 | 0 | 0 | 0 | 77 | 0 | 1,000 |
| `spa` | training | `data/spa_Latn/train/000_00000.parquet` | 1,000 | 3,329,000 | 2 | 2,000 | 0 | 0 | 0 | 0 | 0 | 1,000 |
| `sqi` | training | `data/als_Latn/train/000_00000.parquet` | 1,000 | 4,439,000 | 2 | 2,000 | 0 | 2 | 0 | 214 | 0 | 1,000 |
| `srp` | training | `data/srp_Cyrl/train/000_00000.parquet` | 500 | 2,269,000 | 1 | 1,000 | 0 | 1 | 0 | 625 | 126 | 500 |
| `srp` | training | `data/srp_Latn/train/000_00000.parquet` | 500 | 586,381 | 1 | 1,000 | 0 | 0 | 0 | 19 | 0 | 500 |
| `sun` | training | `data/sun_Latn/train/000_00000.parquet` | 1,000 | 106,542 | 2 | 2,000 | 0 | 8 | 0 | 151 | 0 | 1,000 |
| `swe` | training | `data/swe_Latn/train/000_00000.parquet` | 1,000 | 3,626,000 | 2 | 2,000 | 0 | 2 | 0 | 7 | 0 | 1,000 |
| `swh` | training | `data/swh_Latn/train/000_00000.parquet` | 1,000 | 1,206,300 | 2 | 2,000 | 0 | 8 | 0 | 490 | 0 | 1,000 |
| `tam` | training | `data/tam_Taml/train/000_00000.parquet` | 1,000 | 2,598,000 | 2 | 2,000 | 0 | 1 | 0 | 242 | 0 | 1,000 |
| `tat` | training | `data/tat_Cyrl/train/000_00000.parquet` | 1,000 | 161,354 | 2 | 2,000 | 0 | 2 | 0 | 416 | 0 | 1,000 |
| `tel` | training | `data/tel_Telu/train/000_00000.parquet` | 1,000 | 1,964,395 | 2 | 2,000 | 0 | 29 | 0 | 74 | 0 | 1,000 |
| `tgk` | training | `data/tgk_Cyrl/train/000_00000.parquet` | 1,000 | 688,384 | 2 | 2,000 | 0 | 0 | 0 | 1,684 | 684 | 1,000 |
| `tgl` | training | `data/fil_Latn/train/000_00000.parquet` | 1,000 | 2,349,050 | 2 | 2,000 | 0 | 4 | 1 | 198 | 0 | 1,000 |
| `tha` | training | `data/tha_Thai/train/000_00000.parquet` | 1,000 | 2,363,000 | 2 | 2,000 | 0 | 10 | 0 | 0 | 0 | 1,000 |
| `tur` | training | `data/tur_Latn/train/000_00000.parquet` | 1,000 | 3,381,000 | 2 | 2,000 | 0 | 8 | 0 | 50 | 0 | 1,000 |
| `ukr` | training | `data/ukr_Cyrl/train/000_00000.parquet` | 1,000 | 2,575,000 | 2 | 2,000 | 0 | 5 | 0 | 738 | 0 | 1,000 |
| `urd` | training | `data/urd_Arab/train/000_00000.parquet` | 1,000 | 3,262,000 | 2 | 2,000 | 0 | 3 | 0 | 697 | 0 | 1,000 |
| `uzb` | training | `data/uzn_Latn/train/000_00000.parquet` | 500 | 1,233,463 | 1 | 1,000 | 0 | 0 | 0 | 590 | 90 | 500 |
| `uzb` | training | `data/uzn_Cyrl/train/000_00000.parquet` | 500 | 1,357,811 | 1 | 1,000 | 0 | 1 | 0 | 689 | 190 | 500 |
| `vie` | training | `data/vie_Latn/train/000_00000.parquet` | 1,000 | 2,319,000 | 2 | 2,000 | 0 | 0 | 0 | 0 | 0 | 1,000 |
| `vol` | training | `data/vol_Latn/train/000_00000.parquet` | 1,000 | 4,136 | 2 | 2,000 | 0 | 6 | 1 | 1,639 | 646 | 1,000 |
| `war` | training | `data/war_Latn/train/000_00000.parquet` | 1,000 | 2,223 | 2 | 2,000 | 0 | 10 | 0 | 1,274 | 284 | 1,000 |
| `zsm` | training | `data/zsm_Latn/train/000_00000.parquet` | 1,000 | 3,396,000 | 2 | 2,000 | 0 | 1 | 0 | 0 | 0 | 1,000 |
| `zul` | training | `data/zul_Latn/train/000_00000.parquet` | 1,000 | 127,335 | 2 | 2,000 | 0 | 2 | 0 | 167 | 0 | 1,000 |
| `afr` | held out | `data/afr_Latn/test/000_00000.parquet` | 500 | 15,138 | 1 | 1,000 | 0 | 2 | 0 | 38 | 0 | 500 |
| `ara` | held out | `data/arb_Arab/test/000_00000.parquet` | 500 | 42,080 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `ast` | held out | `data/ast_Latn/test/000_00000.parquet` | 500 | 645 | 1 | 645 | 0 | 0 | 0 | 94 | 0 | 500 |
| `aze` | held out | `data/azj_Latn/test/000_00000.parquet` | 500 | 23,948 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `bak` | held out | `data/bak_Cyrl/test/000_00000.parquet` | 500 | 1,940 | 1 | 1,000 | 0 | 0 | 0 | 171 | 0 | 500 |
| `bel` | held out | `data/bel_Cyrl/test/000_00000.parquet` | 500 | 18,454 | 1 | 1,000 | 0 | 0 | 0 | 97 | 0 | 500 |
| `ben` | held out | `data/ben_Beng/test/000_00000.parquet` | 500 | 59,078 | 1 | 1,000 | 0 | 1 | 0 | 0 | 0 | 500 |
| `bos` | held out | `data/bos_Latn/test/000_00000.parquet` | 500 | 49,273 | 1 | 1,000 | 0 | 1 | 0 | 0 | 0 | 500 |
| `bre` | held out | `data/bre_Latn/test/000_00000.parquet` | 500 | 514 | 1 | 514 | 0 | 0 | 0 | 202 | 188 | 500 |
| `bul` | held out | `data/bul_Cyrl/test/000_00000.parquet` | 500 | 36,933 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `cat` | held out | `data/cat_Latn/test/000_00000.parquet` | 500 | 37,517 | 1 | 1,000 | 0 | 1 | 0 | 0 | 0 | 500 |
| `ceb` | held out | `data/ceb_Latn/test/000_00000.parquet` | 500 | 2,017 | 1 | 1,000 | 0 | 2 | 0 | 45 | 0 | 500 |
| `ces` | held out | `data/ces_Latn/test/000_00000.parquet` | 500 | 28,051 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `che` | held out | `data/che_Cyrl/test/000_00000.parquet` | 500 | 204 | 1 | 204 | 0 | 0 | 0 | 133 | 133 | 204 |
| `cmn` | held out | `data/cmn_Hani/test/000_00000.parquet` | 500 | 33,296 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `cym` | held out | `data/cym_Latn/test/000_00000.parquet` | 500 | 8,395 | 1 | 1,000 | 0 | 1 | 0 | 6 | 0 | 500 |
| `dan` | held out | `data/dan_Latn/test/000_00000.parquet` | 500 | 27,945 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `deu` | held out | `data/deu_Latn/test/000_00000.parquet` | 500 | 65,176 | 1 | 1,000 | 0 | 1 | 0 | 0 | 0 | 500 |
| `ell` | held out | `data/ell_Grek/test/000_00000.parquet` | 500 | 32,936 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `eng` | held out | `sample/10BT/001_00000.parquet` | 500 | 1,046,615 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `epo` | held out | `data/epo_Latn/test/000_00000.parquet` | 500 | 3,375 | 1 | 1,000 | 0 | 0 | 0 | 364 | 0 | 500 |
| `est` | held out | `data/ekk_Latn/test/000_00000.parquet` | 500 | 25,786 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `eus` | held out | `data/eus_Latn/test/000_00000.parquet` | 500 | 15,792 | 1 | 1,000 | 0 | 0 | 0 | 12 | 0 | 500 |
| `fao` | held out | `data/fao_Latn/test/000_00000.parquet` | 500 | 2,991 | 1 | 1,000 | 0 | 2 | 0 | 14 | 0 | 500 |
| `fin` | held out | `data/fin_Latn/test/000_00000.parquet` | 500 | 30,398 | 1 | 1,000 | 0 | 0 | 1 | 0 | 0 | 500 |
| `fra` | held out | `data/fra_Latn/test/000_00000.parquet` | 500 | 44,245 | 1 | 1,000 | 0 | 1 | 0 | 0 | 0 | 500 |
| `fry` | held out | `data/fry_Latn/test/000_00000.parquet` | 500 | 3,383 | 1 | 1,000 | 0 | 1 | 0 | 642 | 143 | 500 |
| `gle` | held out | `data/gle_Latn/test/000_00000.parquet` | 500 | 6,881 | 1 | 1,000 | 0 | 3 | 0 | 68 | 0 | 500 |
| `glg` | held out | `data/glg_Latn/test/000_00000.parquet` | 500 | 3,292 | 1 | 1,000 | 0 | 2 | 0 | 0 | 0 | 500 |
| `gsw` | held out | `data/gsw_Latn/test/000_00000.parquet` | 500 | 513 | 1 | 513 | 0 | 0 | 0 | 81 | 68 | 500 |
| `guj` | held out | `data/guj_Gujr/test/000_00000.parquet` | 500 | 20,231 | 1 | 1,000 | 0 | 0 | 0 | 34 | 0 | 500 |
| `heb` | held out | `data/heb_Hebr/test/000_00000.parquet` | 500 | 26,089 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `hin` | held out | `data/hin_Deva/test/000_00000.parquet` | 500 | 55,242 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `hrv` | held out | `data/hrv_Latn/test/000_00000.parquet` | 500 | 9,248 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `hun` | held out | `data/hun_Latn/test/000_00000.parquet` | 500 | 34,779 | 1 | 1,000 | 0 | 2 | 0 | 0 | 0 | 500 |
| `hye` | held out | `data/hye_Armn/test/000_00000.parquet` | 500 | 16,131 | 1 | 1,000 | 0 | 1 | 0 | 31 | 0 | 500 |
| `ind` | held out | `data/ind_Latn/test/000_00000.parquet` | 500 | 36,821 | 1 | 1,000 | 0 | 1 | 0 | 0 | 0 | 500 |
| `isl` | held out | `data/isl_Latn/test/000_00000.parquet` | 500 | 32,586 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `ita` | held out | `data/ita_Latn/test/000_00000.parquet` | 500 | 41,029 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `jav` | held out | `data/jav_Latn/test/000_00000.parquet` | 500 | 1,867 | 1 | 1,000 | 0 | 5 | 0 | 46 | 0 | 500 |
| `jpn` | held out | `data/jpn_Jpan/test/000_00000.parquet` | 500 | 25,889 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `kan` | held out | `data/kan_Knda/test/000_00000.parquet` | 500 | 24,196 | 1 | 1,000 | 0 | 1 | 0 | 12 | 0 | 500 |
| `kat` | held out | `data/kat_Geor/test/000_00000.parquet` | 500 | 31,598 | 1 | 1,000 | 0 | 1 | 0 | 18 | 0 | 500 |
| `kaz` | held out | `data/kaz_Cyrl/test/000_00000.parquet` | 500 | 31,632 | 1 | 1,000 | 0 | 2 | 0 | 14 | 0 | 500 |
| `kir` | held out | `data/kir_Cyrl/test/000_00000.parquet` | 500 | 10,331 | 1 | 1,000 | 0 | 2 | 0 | 240 | 0 | 500 |
| `kor` | held out | `data/kor_Hang/test/000_00000.parquet` | 500 | 30,074 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `lat` | held out | `data/lat_Latn/test/000_00000.parquet` | 500 | 17,499 | 1 | 1,000 | 0 | 1 | 0 | 0 | 0 | 500 |
| `lim` | held out | `data/lim_Latn/test/000_00000.parquet` | 500 | 251 | 1 | 251 | 0 | 0 | 0 | 15 | 15 | 251 |
| `lit` | held out | `data/lit_Latn/test/000_00000.parquet` | 500 | 27,694 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `ltz` | held out | `data/ltz_Latn/test/000_00000.parquet` | 500 | 3,936 | 1 | 1,000 | 0 | 2 | 0 | 91 | 0 | 500 |
| `lvs` | held out | `data/lvs_Latn/test/000_00000.parquet` | 500 | 22,185 | 1 | 1,000 | 0 | 2 | 0 | 0 | 0 | 500 |
| `mal` | held out | `data/mal_Mlym/test/000_00000.parquet` | 500 | 34,542 | 1 | 1,000 | 0 | 5 | 0 | 0 | 0 | 500 |
| `mar` | held out | `data/mar_Deva/test/000_00000.parquet` | 500 | 38,137 | 1 | 1,000 | 0 | 0 | 0 | 9 | 0 | 500 |
| `min` | held out | `data/min_Latn/test/000_00000.parquet` | 500 | 11 | 1 | 11 | 0 | 0 | 0 | 0 | 0 | 11 |
| `mkd` | held out | `data/mkd_Cyrl/test/000_00000.parquet` | 500 | 41,949 | 1 | 1,000 | 0 | 1 | 0 | 7 | 0 | 500 |
| `mlg` | held out | `data/plt_Latn/test/000_00000.parquet` | 500 | 2,447 | 1 | 1,000 | 0 | 1 | 0 | 178 | 0 | 500 |
| `mlt` | held out | `data/mlt_Latn/test/000_00000.parquet` | 500 | 5,193 | 1 | 1,000 | 0 | 4 | 0 | 29 | 0 | 500 |
| `mon` | held out | `data/khk_Cyrl/test/000_00000.parquet` | 500 | 15,115 | 1 | 1,000 | 0 | 5 | 0 | 0 | 0 | 500 |
| `mri` | held out | `data/mri_Latn/test/000_00000.parquet` | 500 | 1,582 | 1 | 1,000 | 0 | 1 | 0 | 28 | 0 | 500 |
| `nan` | held out | `data/nan_Latn/test/000_00000.parquet` | 500 | 38 | 1 | 38 | 0 | 0 | 0 | 0 | 0 | 38 |
| `nds` | held out | `data/nds_Latn/test/000_00000.parquet` | 500 | 581 | 1 | 581 | 0 | 0 | 0 | 190 | 109 | 500 |
| `nld` | held out | `data/nld_Latn/test/000_00000.parquet` | 500 | 33,283 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `nno` | held out | `data/nno_Latn/test/000_00000.parquet` | 500 | 12,227 | 1 | 1,000 | 0 | 0 | 0 | 39 | 0 | 500 |
| `nob` | held out | `data/nob_Latn/test/000_00000.parquet` | 500 | 43,396 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `npi` | held out | `data/npi_Deva/test/000_00000.parquet` | 500 | 49,462 | 1 | 1,000 | 0 | 1 | 0 | 0 | 0 | 500 |
| `oci` | held out | `data/oci_Latn/test/000_00000.parquet` | 500 | 757 | 1 | 757 | 0 | 5 | 0 | 76 | 0 | 500 |
| `pan` | held out | `data/pan_Guru/test/000_00000.parquet` | 500 | 9,653 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `pes` | held out | `data/fas_Arab/test/000_00000.parquet` | 500 | 43,470 | 1 | 1,000 | 0 | 1 | 0 | 0 | 0 | 500 |
| `pnb` | held out | `data/pnb_Arab/test/000_00000.parquet` | 500 | 256 | 1 | 256 | 0 | 0 | 0 | 59 | 59 | 256 |
| `pol` | held out | `data/pol_Latn/test/000_00000.parquet` | 500 | 36,574 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `por` | held out | `data/por_Latn/test/000_00000.parquet` | 500 | 33,846 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `pus` | held out | `data/pbt_Arab/test/000_00000.parquet` | 500 | 6,522 | 1 | 1,000 | 0 | 1 | 0 | 165 | 0 | 500 |
| `ron` | held out | `data/ron_Latn/test/000_00000.parquet` | 500 | 45,783 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `rus` | held out | `data/rus_Cyrl/test/000_00000.parquet` | 500 | 34,714 | 1 | 1,000 | 0 | 3 | 0 | 0 | 0 | 500 |
| `san` | held out | `data/san_Deva/test/000_00000.parquet` | 500 | 233 | 1 | 233 | 0 | 0 | 0 | 13 | 13 | 233 |
| `sin` | held out | `data/sin_Sinh/test/000_00000.parquet` | 500 | 11,428 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `slk` | held out | `data/slk_Latn/test/000_00000.parquet` | 500 | 32,557 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `slv` | held out | `data/slv_Latn/test/000_00000.parquet` | 500 | 32,447 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `som` | held out | `data/som_Latn/test/000_00000.parquet` | 500 | 10,523 | 1 | 1,000 | 0 | 3 | 0 | 0 | 0 | 500 |
| `spa` | held out | `data/spa_Latn/test/000_00000.parquet` | 500 | 40,970 | 1 | 1,000 | 0 | 2 | 0 | 0 | 0 | 500 |
| `sqi` | held out | `data/als_Latn/test/000_00000.parquet` | 500 | 63,178 | 1 | 1,000 | 0 | 1 | 0 | 11 | 0 | 500 |
| `srp` | held out | `data/srp_Cyrl/test/000_00000.parquet` | 250 | 40,294 | 1 | 1,000 | 0 | 2 | 0 | 13 | 0 | 250 |
| `srp` | held out | `data/srp_Latn/test/000_00000.parquet` | 250 | 5,350 | 1 | 1,000 | 0 | 1 | 0 | 0 | 0 | 250 |
| `sun` | held out | `data/sun_Latn/test/000_00000.parquet` | 500 | 997 | 1 | 997 | 0 | 4 | 0 | 0 | 0 | 500 |
| `swe` | held out | `data/swe_Latn/test/000_00000.parquet` | 500 | 24,692 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `swh` | held out | `data/swh_Latn/test/000_00000.parquet` | 500 | 12,669 | 1 | 1,000 | 0 | 8 | 0 | 26 | 0 | 500 |
| `tam` | held out | `data/tam_Taml/test/000_00000.parquet` | 500 | 41,100 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `tat` | held out | `data/tat_Cyrl/test/000_00000.parquet` | 500 | 1,744 | 1 | 1,000 | 0 | 4 | 0 | 67 | 0 | 500 |
| `tel` | held out | `data/tel_Telu/test/000_00000.parquet` | 500 | 18,933 | 1 | 1,000 | 0 | 3 | 0 | 0 | 0 | 500 |
| `tgk` | held out | `data/tgk_Cyrl/test/000_00000.parquet` | 500 | 6,427 | 1 | 1,000 | 0 | 0 | 0 | 256 | 0 | 500 |
| `tgl` | held out | `data/fil_Latn/test/000_00000.parquet` | 500 | 20,096 | 1 | 1,000 | 0 | 3 | 0 | 0 | 0 | 500 |
| `tha` | held out | `data/tha_Thai/test/000_00000.parquet` | 500 | 24,037 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `tur` | held out | `data/tur_Latn/test/000_00000.parquet` | 500 | 36,146 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `ukr` | held out | `data/ukr_Cyrl/test/000_00000.parquet` | 500 | 40,288 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `urd` | held out | `data/urd_Arab/test/000_00000.parquet` | 500 | 49,136 | 1 | 1,000 | 0 | 0 | 0 | 2 | 0 | 500 |
| `uzb` | held out | `data/uzn_Latn/test/000_00000.parquet` | 250 | 13,420 | 1 | 1,000 | 0 | 0 | 0 | 9 | 0 | 250 |
| `uzb` | held out | `data/uzn_Cyrl/test/000_00000.parquet` | 250 | 13,662 | 1 | 1,000 | 0 | 1 | 0 | 37 | 0 | 250 |
| `vie` | held out | `data/vie_Latn/test/000_00000.parquet` | 500 | 28,276 | 1 | 1,000 | 0 | 0 | 0 | 0 | 0 | 500 |
| `vol` | held out | `data/vol_Latn/test/000_00000.parquet` | 500 | 52 | 1 | 52 | 0 | 0 | 0 | 9 | 9 | 52 |
| `war` | held out | `data/war_Latn/test/000_00000.parquet` | 500 | 35 | 1 | 35 | 0 | 1 | 0 | 0 | 0 | 34 |
| `zsm` | held out | `data/zsm_Latn/test/000_00000.parquet` | 500 | 36,509 | 1 | 1,000 | 0 | 1 | 0 | 0 | 0 | 500 |
| `zul` | held out | `data/zul_Latn/test/000_00000.parquet` | 500 | 762 | 1 | 762 | 0 | 1 | 0 | 13 | 0 | 500 |

</details>

## Tests

`tests/recipes/` (one test binary, a file per part) writes corpora in
MASC's layout (`tests/common`, no MASC text, Penn tags written for the
tests) and runs everything on them: the split in the reader's order,
its paths joined with `/` on every OS; the Twitter genre left out of
both sides without moving another document; the eight recipes trained,
scored and gated; the same bytes from a second training; the card, the
model's directory and its zoo entry; the zoo entries of a release made
from its models' directories and the built-in index with them added,
and their refusal of a missing directory, a card in another model's
directory, a gate record of another model or with another outcome than
the card's and a changed model file; the skipped sentences recorded and
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
and the chunker's chunks in no sentence and the names of both sides)
checked word for word; and `zoo-index` on those directories, which with
`--check` exits with 1 and names each model with the size and SHA-256
the index lists and the model's, and without it prints the index with
them, and its refusal of a tag that names no release, of a missing
`--release` and of models trained on a corpus that is not the recorded
release. The test documents have entity files in MASC's layout, with
persons, locations, organizations and dates in some of their sentences,
so the name recipes train, score and gate there too, with the names they
score, those they leave out and the cards' name counts checked. It
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
their tags moved one token along, for the chunker with their chunks
moved one token along, and for the name finders with their names moved
one token along): as the reference it loses and the
trained model passes, and as the candidate it loses far more than the
margin and fails, which is the check that tells the gate from one that
passes everything. A model against itself passes with no difference.

`tests/names.rs` checks the name recipes' reader on documents written
for it: an entity with a gap kept as its covering span, overlapping
entities and entities of other types left out and counted, `org` read as
`organization`, a sentence of no tokens skipped, a document's first
sample clearing the adaptive data, entities outside every sentence
region and in a region the reader drops counted, a side with no name of
the recipe's type or too many skipped sentences refused, and, on the
same documents, the names OpenNLP's `MascNamedEntitySampleStream` reads
(each of its differences above, with names listed last to first as MASC
lists them, and first to last with an even and an odd number of tokens,
in a sentence and at its start) against the whole names the recipes
read.

The OpenNLP UD references need `TURBO_GOLDENS_MODELS` naming a
directory that holds them (`cargo run -p turbo-goldens-tool --
fetch-models <dir>`). With it, they are checked against the zoo's
digests and scored on the test corpus, the UD tagger asked for Penn tags
is checked to give only its 17 converted tags, and the binary runs end to
end on the recipes whose references are present (`--recipe`); without it
those parts skip, unless `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`, which
makes them fail. CI runs the test with both set, in the step that
requires the published models.

The tagger's reference, `en-pos-maxent.bin`, the chunker's,
`en-chunker.bin`, and the name finders' references, `en-ner-*.bin`, are
not in the zoo, as in turbo-sequence's tests of the 1.5 models: their
training data is not published, so the `ci` workflow (`ci.yml`) does not
fetch them (the `models` workflow fetches the tagger's and the
chunker's, to run the release's gates; see "The models releases"). The
parts that need them run when `TURBO_TEST_OPENNLP_MODELS` names a
directory holding them (a missing file there fails), and skip without it
even with `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`. With it, each is
checked against its recorded digest; the tagger's is shown to return its
45 Penn outcomes unconverted and to tag brackets, hyphens, ellipses,
semicolons and an opening quotation mark as `postag::REFERENCE_MAPPINGS`
says, and a closing quotation mark `''`; the chunker's is shown to give
CoNLL-2000's eleven chunk types and to have features for `(`, `)` and
`:` and none for the three tags `chunk::REFERENCE_TAGS` maps; each name
finder's is shown to be an OpenNLP 1.5.0 model with no descriptor whose
outcomes are `other` and its type's start and continuation; they are
scored and gated, and the binary runs all eight recipes.

On the test corpus, with both directories set, the published references
score 63.16 (sentences, F-measure), 94.34 (tokens, F-measure), 92.41
(`en-pos-maxent.bin`, accuracy), 93.33 (`en-chunker.bin`, F-measure of
`NP` and `VP` chunks), 0.00 (`en-ner-person.bin`, `en-ner-location.bin`
and `en-ner-organization.bin`, F-measure: they find none of the test
corpus's names) and 75.00 (`en-ner-date.bin`, F-measure), against 100
for the trained models,
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
condition a zoo entry needs; the release's entry, made from the model's
directory with both files checked, and refused when
`training-sentences.txt` is changed or missing; the export record; and
the binary's refusal of a missing reference. With
`TURBO_GOLDENS_MODELS` naming a directory holding `langdetect-183.bin`
(required with `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`), it checks the
table's codes against the model's 103 outcomes, and that the binary
refuses the export as not the recorded one and, given `--unrecorded-export`,
refuses it for having no language with 1,500 sentences.

`tests/fineweb.rs` runs the web recipe's reading on Parquet files a
writer in the test makes (`text`, `id` and `url` columns, row groups of
four rows), one for each of the 204 files, served from memory: the
ranges fetched round by round as the workflow fetches them (the last 8
bytes, the footer's first bytes, the head), kept under their file's
revision, recorded, listed and checked against what the files give;
the windows chosen from them, with a document too short, a repeated
window, documents passed over for their host and taken later when a
file is short, held-out windows that are training windows passed over,
two sources sharing a language's documents, and a file with fewer rows
than wanted read whole; and `training-documents.txt`. On files of its
own it checks that a document from `tatoeba.org`, a training window
holding a held-out sentence of its language and a held-out window
holding a training sentence are passed over, while a window holding
another language's sentence, or a sentence too short to be looked for,
is kept. A changed byte, a missing range, a range or a file not
listed, a file the recipe does not read and a footer prefix too short
for the rows wanted are refused. It runs the recipe end to end on the
export of `tests/data/tatoeba` and web files of the nine languages
`tests/langdetect.rs` uses, made of common words of each, with the
language fixture as the reference and as the compared model: the same
bytes from a second training, Turkish trained on its windows alone,
the Tatoeba gate's held-out set the same as the Tatoeba recipe's, both
gates and both comparisons with the trained model's side the same in
each pair, the card's counts, the model's directory, and the card's
zoo entry, refused for a model no recipe trains, one gated against
another reference, a failed gate, an unrecorded export and unrecorded
web lists, and made with its three files once none of those holds. It
also runs the binary: `fineweb-ranges --paths` lists the 204 files,
`--need` with nothing fetched asks for each file's last 8 bytes,
`--record` refuses ranges not fetched, a flag a command does not read
is refused, and `tatoeba-fineweb` without its two models or its web
directory, with `--files` and no `--ranges`, and on lists other than
the committed ones without `--unrecorded-web`, names each problem and
writes nothing. The committed lists are checked to name exactly the
files the recipe reads, three ranges each. turbo-eval's
`tests/parquet.rs` reads the leading row groups of its pyarrow files
from every first part of their footers, byte by byte, and from the head
of the file, refuses a whole footer that ends inside a row group, and
refuses a column chunk that starts inside the magic number or ends past
the footer's start.
